
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess_doom")

# Binary action dims 0..12 -> canonical key names (must match configs/actions/doom.yaml order).
DOOM_BINARY_KEYS: list[str] = [
    "forward", "backward", "strafe_right", "strafe_left",
    "weapon1", "weapon2", "weapon3", "weapon4", "weapon5", "weapon6", "weapon7",
    "attack", "speed",
]  # fmt: skip
TURN_DELTA_IDX = 13  # continuous dim -> mouse channel 0
N_ACTION_DIMS = 14


@dataclass
class SourceEpisode:
    """One source episode's in-tar bytes, keyed by the two players' members."""

    key: str
    video: dict[int, bytes] = field(default_factory=dict)  # player (1-based) -> mp4 bytes
    actions: dict[int, bytes] = field(default_factory=dict)  # player (1-based) -> .npy bytes


def iter_source_episodes(tar_path: Path):
    """Yield complete ``SourceEpisode``s from one source tar (members grouped by episode key)."""
    episodes: dict[str, SourceEpisode] = {}
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = Path(m.name).name
            # {key}.video_p1.mp4 / {key}.actions_p1.npy / {key}.rewards_p1.npy / {key}.meta.json
            key, _, field_name = name.partition(".")
            ep = episodes.setdefault(key, SourceEpisode(key=key))
            data = tar.extractfile(m)
            if data is None:
                continue
            if field_name.startswith("video_p") and field_name.endswith(".mp4"):
                ep.video[_player_num(field_name)] = data.read()
            elif field_name.startswith("actions_p") and field_name.endswith(".npy"):
                ep.actions[_player_num(field_name)] = data.read()
            # rewards / meta ignored
    for ep in episodes.values():
        if set(ep.video) >= {1, 2} and set(ep.actions) >= {1, 2}:
            yield ep
        else:
            logger.warning(
                "Skipping incomplete episode %s (video=%s actions=%s)",
                ep.key,
                sorted(ep.video),
                sorted(ep.actions),
            )


def _player_num(field_name: str) -> int:
    tail = field_name.split("_p", 1)[1]
    num = ""
    for ch in tail:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num)

def decode_all_frames(video_bytes: bytes) -> "tuple[np.ndarray, float]":
    """Decode an mp4 to (T, H, W, 3) uint8 RGB and its average fps, via torchcodec."""
    from torchcodec.decoders import VideoDecoder  # pyright: ignore[reportPrivateImportUsage]

    dec = VideoDecoder(video_bytes, device="cpu")
    n = dec.metadata.num_frames or 0
    fps = float(dec.metadata.average_fps or 35.0)
    frames = dec.get_frames_at(list(range(n))).data  # (T, 3, H, W) uint8
    return frames.permute(0, 2, 3, 1).contiguous().numpy(), fps


def encode_chunk_mp4(frames_thwc: "np.ndarray", fps: float) -> bytes:
    """Encode (T, H, W, 3) uint8 RGB frames to H.264 mp4 bytes via ffmpeg (frame-accurate)."""
    t, h, w, c = frames_thwc.shape
    assert c == 3
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp.name,
        ]  # fmt: skip
        proc = subprocess.run(cmd, input=frames_thwc.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='replace')[-2000:]}")
        return Path(tmp.name).read_bytes()


def actions_to_jsonl(actions_chunk: "np.ndarray") -> bytes:
    """Serialize (T, 14) actions to per-frame jsonl lines: binary dims -> keys, turn delta -> mouse."""
    lines = []
    for row in actions_chunk:
        keys = [DOOM_BINARY_KEYS[j] for j in range(len(DOOM_BINARY_KEYS)) if row[j] > 0.5]
        turn = float(row[TURN_DELTA_IDX])
        lines.append(json.dumps({"keys": keys, "mouse": [turn, 0.0]}, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def is_val(episode_id: int, val_fraction: float) -> bool:
    """Deterministic per-episode train/val routing by hash (stable across runs)."""
    if val_fraction <= 0:
        return False
    h = int(hashlib.sha1(str(episode_id).encode()).hexdigest(), 16) % 10_000
    return h < int(val_fraction * 10_000)


def convert(args: argparse.Namespace) -> None:
    src = Path(args.src)
    tars = sorted(src.rglob("*.tar"))
    if not tars:
        logger.error("No .tar shards found under %s", src)
        sys.exit(1)
    logger.info("Found %d source shards under %s", len(tars), src)

    writers = {
        "train": _ShardWriter(Path(args.out) / "train", args.episodes_per_shard),
        "test": _ShardWriter(Path(args.out) / "test", args.episodes_per_shard),
    }
    id_map: dict[str, str] = {}
    episode_counter = 0

    for tar_path in tars:
        for ep in iter_source_episodes(tar_path):
            split = "test" if is_val(episode_counter, args.val_fraction) else "train"
            match_id = f"ep{episode_counter:06d}"
            id_map[match_id] = ep.key
            try:
                _convert_episode(ep, match_id, writers[split], args)
            except Exception as exc:  # noqa: BLE001 -- one bad episode shouldn't abort the whole run
                logger.warning("Skipping episode %s (%s): %s", ep.key, match_id, exc)
            episode_counter += 1
            if episode_counter % 25 == 0:
                logger.info("Processed %d episodes ...", episode_counter)
            if args.max_episodes and episode_counter >= args.max_episodes:
                logger.info("Reached --max-episodes=%d; stopping.", args.max_episodes)
                _finalize(writers, id_map, args)
                return

    _finalize(writers, id_map, args)


def _convert_episode(
    ep: SourceEpisode, match_id: str, writer: "_ShardWriter", args: argparse.Namespace
) -> None:
    # Decode both players; align lengths across video + both action arrays.
    frames = {}
    fps = 35.0
    for p in (1, 2):
        frames[p], fps = decode_all_frames(ep.video[p])
    acts = {p: np.load(io.BytesIO(ep.actions[p])).astype(np.float32) for p in (1, 2)}
    for p in (1, 2):
        if acts[p].ndim != 2 or acts[p].shape[1] < N_ACTION_DIMS:
            raise ValueError(
                f"player {p} actions have shape {acts[p].shape}, expected (T, >= {N_ACTION_DIMS})"
            )

    n = min(frames[1].shape[0], frames[2].shape[0], acts[1].shape[0], acts[2].shape[0])
    if n < args.chunk_len:
        raise ValueError(f"episode too short: {n} frames < chunk-len {args.chunk_len}")

    chunk_len = args.chunk_len
    n_chunks = n // chunk_len  # drop a trailing partial chunk (keeps every chunk full)
    chunk_frames = [chunk_len] * n_chunks

    writer.begin_match(match_id)
    for ci in range(n_chunks):
        s, e = ci * chunk_len, (ci + 1) * chunk_len
        key = f"{match_id}_c{ci:05d}"
        for p in (1, 2):
            i = p - 1  # 0-based perspective index MIRA expects
            writer.add(f"{key}.p{i}.mp4", encode_chunk_mp4(frames[p][s:e], fps))
            writer.add(f"{key}.p{i}.jsonl", actions_to_jsonl(acts[p][s:e]))
        writer.add(f"{key}.meta.json", json.dumps({"source_key": ep.key, "fps": fps}).encode())

    writer.end_match(
        match_id=match_id,
        chunk_frames=chunk_frames,
        fps=fps,
        n_frames=n_chunks * chunk_len,
    )

class _ShardWriter:
    def __init__(self, out_dir: Path, episodes_per_shard: int):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = episodes_per_shard
        self.shard_idx = 0
        self.eps_in_shard = 0
        self.tar: tarfile.TarFile | None = None
        self.cur_shard_name: str | None = None
        self.entries: list[dict] = []

    def _shard_name(self) -> str:
        return f"doom-{self.shard_idx:05d}.tar"

    def begin_match(self, match_id: str) -> None:
        if self.tar is None or self.eps_in_shard >= self.episodes_per_shard:
            self._rotate()
        self._cur_match = match_id

    def _rotate(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.shard_idx += 1
            self.eps_in_shard = 0
        self.cur_shard_name = self._shard_name()
        self.tar = tarfile.open(self.out_dir / self.cur_shard_name, "w")

    def add(self, name: str, data: bytes) -> None:
        assert self.tar is not None
        _add_bytes(self.tar, name, data)

    def end_match(self, match_id: str, chunk_frames: list[int], fps: float, n_frames: int) -> None:
        duration = n_frames / fps  # so src_fps = frames / duration == fps exactly
        assert self.cur_shard_name is not None
        self.entries.append({
            "match_id": match_id,
            "shard": self.cur_shard_name,
            "n_players": 2,
            "chunk_frames": chunk_frames,
            "arena": "doom",
            "perspectives": [
                {"player_id": 0, "team": 0, "frames": n_frames, "duration": duration,
                 "recording_offset_sec": 0.0, "anchors": []},
                {"player_id": 1, "team": 1, "frames": n_frames, "duration": duration,
                 "recording_offset_sec": 0.0, "anchors": []},
            ],
        })  # fmt: skip
        self.eps_in_shard += 1

    def close_and_write_index(self) -> int:
        if self.tar is not None:
            self.tar.close()
            self.tar = None
        index = {"total_samples": len(self.entries), "entries": self.entries}
        # Validate against the real schema so a malformed index fails here, not at training time.
        from mira.data.schema import Index

        Index.model_validate(index)
        (self.out_dir / "index.json").write_text(json.dumps(index))
        return len(self.entries)


def _finalize(writers: dict[str, "_ShardWriter"], id_map: dict[str, str], args: argparse.Namespace) -> None:
    for split, w in writers.items():
        n = w.close_and_write_index()
        logger.info("[%s] wrote %d matches + index.json to %s", split, n, w.out_dir)
    (Path(args.out) / "id_map.json").write_text(json.dumps(id_map, indent=2))
    logger.info("Wrote id_map.json (match_id -> source episode key). Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--src", required=True, help="Dir containing the downloaded Doom .tar shards (searched recursively)."
    )
    ap.add_argument("--out", required=True, help="Output dir; train/ and test/ subdirs are created.")
    ap.add_argument(
        "--chunk-len",
        type=int,
        default=160,
        help="Frames per chunk. Must be >= the largest world-model clip_len (multiplayer=80). "
        "160 divides both 40 (single) and 80 (multi) with no wasted frames. Default 160.",
    )
    ap.add_argument(
        "--val-fraction", type=float, default=0.02, help="Fraction of episodes routed to test/. Default 0.02."
    )
    ap.add_argument(
        "--episodes-per-shard", type=int, default=8, help="Episodes per output tar shard. Default 8."
    )
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after N episodes (0 = all). Useful for a smoke test.",
    )
    args = ap.parse_args()

    if args.chunk_len < 80:
        logger.warning(
            "chunk-len %d < 80: multiplayer clips (clip_len=80) will not fit in a chunk.", args.chunk_len
        )
    convert(args)

if __name__ == "__main__":
    main()
