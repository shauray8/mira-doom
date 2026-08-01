"""The shuffle-before-decode path: bit-identical output, a pixel-free shuffle buffer, shared bytes.

The buffer holds undecoded clip references and decodes only the sample drawn out of it. These tests
prove that (a) the emitted stream is bit-identical to the previous decode-then-shuffle loader for a
fixed seed, (b) the buffered element carries no pixels, (c) chunk bytes are shared by reference, and
(d) the decoded sample drops the compressed bytes. A real tar + ffmpeg mp4 exercises decode for real.
"""

import io
import json
import shutil
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest
import torch

from mira.data.actions import DEFAULT_KEYS, KeyVocab
from mira.data.dataset import DoomWebDataset
from mira.data.training_loader import _VideoActionIterable, create_loader
from mira.world_model.actions_config import ActionConfig

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg (run via pixi)")

FPS, CHUNK, N_CHUNKS, N_PLAYERS = 20, 20, 2, 4
PLAYER_IDS = [10, 20, 30, 40]


def _chunk_mp4(path: Path) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size=48x32:rate={FPS}",
         "-frames:v", str(CHUNK), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip
    return path.read_bytes()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _two_shard_fixture(out: Path) -> Path:
    """Two matches in two shards, so shard-order shuffle and cross-shard streaming are exercised."""
    out.mkdir(parents=True, exist_ok=True)
    vb = _chunk_mp4(out / "v.mp4")
    (out / "v.mp4").unlink()
    per_actions = [
        "".join(json.dumps({"keys": [DEFAULT_KEYS[i]]}) + "\n" for _ in range(CHUNK)).encode()
        for i in range(N_PLAYERS)
    ]

    entries = []
    for s, match_id in enumerate(("2026-05-10T00-00-00Z-aaaaaa", "2026-05-11T00-00-00Z-bbbbbb")):
        shard = f"dataset_{s:02d}.tar"
        with tarfile.open(out / shard, "w") as tar:
            for c in range(N_CHUNKS):
                key = f"{match_id}_c{c:05d}"
                for i in range(N_PLAYERS):
                    _add(tar, f"{key}.p{i}.mp4", vb)
                    _add(tar, f"{key}.p{i}.jsonl", per_actions[i])
        frames = N_CHUNKS * CHUNK
        entries.append({
            "match_id": match_id,
            "shard": shard,
            "n_players": N_PLAYERS,
            "chunk_frames": [CHUNK] * N_CHUNKS,
            "perspectives": [
                {"player_id": PLAYER_IDS[i], "team": i % 2, "frames": frames,
                 "duration": frames / FPS, "recording_offset_sec": 0.0, "anchors": []}
                for i in range(N_PLAYERS)
            ],
        })  # fmt: skip
    (out / "index.json").write_text(json.dumps({"total_samples": len(entries), "entries": entries}))
    return out


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return _two_shard_fixture(tmp_path_factory.mktemp("rl_shuffle"))


def _load_pre_edit_loader():
    """The loader as committed at HEAD (decode-then-shuffle), imported as a throwaway submodule so
    its relative imports resolve. This is the oracle the new decode-then-shuffle path must match."""
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    src = subprocess.run(
        ["git", "show", "HEAD:src/mira/data/training_loader.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout
    name = "mira.data._pre_edit_training_loader"
    mod = types.ModuleType(name)
    mod.__package__ = "mira.data"
    sys.modules[name] = mod
    exec(compile(src, "pre_edit_training_loader.py", "exec"), mod.__dict__)
    return mod


def _stream(loader) -> list[tuple]:
    """Flatten a finite loader into an ordered list of (provenance, pixel bytes, action bytes)."""
    rows = []
    for batch, metas in loader:
        for i, m in enumerate(metas):
            rows.append(
                (
                    m.match_id,
                    m.chunk_idx,
                    tuple(m.frame_indices),
                    batch.video[i].numpy().tobytes(),
                    batch.actions.key_presses[i].numpy().tobytes(),
                )
            )
    return rows


def test_bit_identical_to_pre_edit_loader(fixture):
    buffer_size = 3

    def build(create):
        return create(
            fixture, clip_len=8, target_fps=10, n_players=1, batch_size=1,
            shuffle=True, infinite=False, shuffle_buffer_size=buffer_size, seed=123, num_workers=0,
        )  # fmt: skip

    new_rows = _stream(build(create_loader))
    old_rows = _stream(build(_load_pre_edit_loader().create_loader))

    assert len(new_rows) > buffer_size  # enough clips to exercise buffer eviction
    assert new_rows == old_rows  # exact order and decoded pixels, not just the set


def test_seed_is_deterministic_and_shuffle_mixes(fixture):
    def build(*, shuffle):
        return create_loader(
            fixture, clip_len=8, target_fps=10, n_players=1, batch_size=1,
            shuffle=shuffle, infinite=False, shuffle_buffer_size=4, seed=7, num_workers=0,
        )  # fmt: skip

    shuffled = _stream(build(shuffle=True))
    again = _stream(build(shuffle=True))
    ordered = _stream(build(shuffle=False))

    assert shuffled == again  # same seed -> same order and pixels
    assert sorted(shuffled) == sorted(ordered)  # same sample set
    assert shuffled != ordered  # buffer_size > 1 actually reorders


def _iterable(fixture) -> _VideoActionIterable:
    action_config = ActionConfig(valid_keys=list(DEFAULT_KEYS), source_fps=FPS, target_fps=10)
    return _VideoActionIterable(
        fixture,
        action_config,
        clip_len=8,
        target_fps=10,
        n_players=1,
        exclude_replays=False,
        frame_size=None,
        shuffle=True,
        infinite=False,
        shuffle_buffer_size=3,
        seed=1,
    )


def _undecoded_clips(fixture, clip_len: int = 8) -> list:
    ds = DoomWebDataset.from_local(fixture, vocab=KeyVocab(tuple(DEFAULT_KEYS)))
    return list(ds.iter_clips(clip_len=clip_len, target_fps=10, decode=False, carry_video=True))


def test_buffered_element_carries_no_pixels(fixture):
    it = _iterable(fixture)
    clip = _undecoded_clips(fixture)[0]
    plan = it._plan_perspective(clip, 0)

    assert clip.frames is None
    assert isinstance(clip.video_bytes[0], bytes)
    assert "video" not in plan  # the pixels do not exist yet
    assert not any(torch.is_tensor(v) and v.ndim == 4 for v in plan.values())  # no (T,C,H,W) tensor


def test_same_chunk_clips_share_bytes_by_reference(fixture):
    by_chunk: dict[tuple, list] = {}
    for clip in _undecoded_clips(fixture, clip_len=4):  # 2 clips per 20-frame chunk
        by_chunk.setdefault((clip.match_id, clip.chunk_idx), []).append(clip)
    multi = [clips for clips in by_chunk.values() if len(clips) > 1]
    assert multi, "fixture must yield multiple clips per chunk to test byte sharing"
    for clips in multi:
        for other in clips[1:]:
            assert other.video_bytes[0] is clips[0].video_bytes[0]  # one bytes object per chunk


def test_decoded_sample_drops_compressed_bytes(fixture):
    it = _iterable(fixture)
    clip = _undecoded_clips(fixture)[0]
    sample = it._decode_sample(it._plan_perspective(clip, 0))

    assert set(sample) == {"video", "actions", "metadata"}  # no clip / video_bytes retained
    assert sample["video"].shape == (8, 3, 32, 48) and sample["video"].dtype == torch.uint8
    # decoding the reference lightweight plan matches a direct eager decode of the same clip
    assert torch.equal(sample["video"], clip.decode_perspective(0, None))
