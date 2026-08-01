"""End-to-end tests of the dataset read path on a self-contained synthetic fixture.

The fixture is a real WebDataset tar built here with stdlib `tarfile` plus an ffmpeg-encoded mp4 (so
the decode path is exercised for real); the fixture is fully self-contained. Covers
random-access and streaming reads, perspective selection, physics surfacing, the within-one-chunk
clip guard, and non-contiguous chunk keys. Needs ffmpeg (run via `pixi run test`).
"""

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from mira.data import DoomWebDataset

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg (run via pixi)")

FPS, CHUNK, N_CHUNKS, N_PLAYERS = 20, 20, 4, 4  # 4 chunks x 20 frames = 80 frames/match
MID, SHARD = "2026-05-10T00-00-00Z-abcdef", "dataset_00.tar"


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


DEFAULT_ANCHORS = [{"event_type": 1, "event_name": "GoalScored", "master_sec": 1.0}]


def _build(
    out: Path,
    *,
    with_physics: bool,
    chunk_indices: list[int] | None = None,
    anchors: list[dict] | None = None,
) -> Path:
    """Build a one-match WebDataset tar + index.json. `chunk_indices` (when given) sets the original,
    non-contiguous source chunk index of each present chunk; otherwise chunks are 0..N_CHUNKS-1.
    `anchors` sets the per-perspective event anchors (defaults to a single GoalScored)."""
    out.mkdir(parents=True, exist_ok=True)
    vb = _chunk_mp4(out / "v.mp4")
    (out / "v.mp4").unlink()
    actions = "".join(json.dumps({"keys": ["W"]}) + "\n" for _ in range(CHUNK)).encode()
    # ball x tags each line with its chunk-local frame index, so tests can assert frame alignment.
    physics = "".join(json.dumps({"ball": {"location": {"x": i}}}) + "\n" for i in range(CHUNK)).encode()

    origs = chunk_indices if chunk_indices is not None else list(range(N_CHUNKS))
    with tarfile.open(out / SHARD, "w") as tar:
        for oi in origs:  # members of one (match, chunk) sample are written contiguously
            key = f"{MID}_c{oi:05d}"
            for i in range(N_PLAYERS):
                _add(tar, f"{key}.p{i}.mp4", vb)
                _add(tar, f"{key}.p{i}.jsonl", actions)
                if with_physics:
                    _add(tar, f"{key}.p{i}.physics.jsonl", physics)

    n = len(origs)
    frames = n * CHUNK
    entry: dict = {
        "match_id": MID,
        "shard": SHARD,
        "n_players": N_PLAYERS,
        "chunk_frames": [CHUNK] * n,
        "perspectives": [
            {
                "player_id": 400 + j,
                "team": j % 2,
                "frames": frames,
                "duration": frames / FPS,
                "recording_offset_sec": 0.0,
                "anchors": anchors if anchors is not None else DEFAULT_ANCHORS,
            }
            for j in range(N_PLAYERS)
        ],
    }
    if chunk_indices is not None:
        entry["chunk_indices"] = chunk_indices
    (out / "index.json").write_text(json.dumps({"total_samples": 1, "entries": [entry]}))
    return out


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("chunks"), with_physics=False)


@pytest.fixture(scope="module")
def physics_fixture(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("chunks_phys"), with_physics=True)


@pytest.fixture(scope="module")
def replay_fixture(tmp_path_factory):
    # A goal replay over master_sec [0.0, 1.0) -> frames [0, 20) at 20fps (offset 0): exactly chunk 0.
    return _build(
        tmp_path_factory.mktemp("replay"),
        with_physics=False,
        anchors=[
            {"event_type": 3, "event_name": "GoalReplayStarted", "master_sec": 0.0},
            {"event_type": 4, "event_name": "GoalReplayEnded", "master_sec": 1.0},
        ],
    )


@pytest.fixture(scope="module")
def gappy_fixture(tmp_path_factory):
    # present chunks 0,1,2,4,6,7 (originals 3 and 5 missing) -> exercises chunk_indices mapping
    return _build(tmp_path_factory.mktemp("gappy"), with_physics=True, chunk_indices=[0, 1, 2, 4, 6, 7])


def test_index_and_chunk_frames(fixture):
    ds = DoomWebDataset.from_local(fixture)
    (entry,) = ds.index.entries
    assert entry.chunk_frames == [CHUNK] * N_CHUNKS
    assert entry.perspectives[0].frames == N_CHUNKS * CHUNK


def test_clip_inside_one_chunk(fixture):
    ds = DoomWebDataset.from_local(fixture)
    # clip_len 8 @ 10fps -> stride 2 -> 15 source-frame span < 20 -> fits in a chunk
    clip = ds.load_match(ds.match_ids()[0], clip_len=8, target_fps=10, decode=True, clip_ids=[0])[0]
    assert clip.frames is not None
    assert clip.frames.shape == (4, 8, 3, 32, 48)
    assert clip.actions.shape == (4, 8, 9)
    assert clip.player_ids == [400, 401, 402, 403] and clip.teams == [0, 1, 0, 1]
    assert clip.chunk_idx == 0
    assert max(clip.frame_indices) < CHUNK  # indices are chunk-local


def test_clip_longer_than_chunk_raises(fixture):
    ds = DoomWebDataset.from_local(fixture)
    # clip_len 12 @ 10fps -> stride 2 -> 23 source-frame span > 20 (chunk) -> too long, must raise
    with pytest.raises(ValueError, match="must fit in one chunk"):
        ds.load_match(ds.match_ids()[0], clip_len=12, target_fps=10, decode=False)


def test_perspective_selection(fixture):
    ds = DoomWebDataset.from_local(fixture)
    one = ds.load_match(
        ds.match_ids()[0], clip_len=8, target_fps=10, decode=True, clip_ids=[0], perspective="player3"
    )[0]
    assert one.frames is not None
    assert one.frames.shape[0] == 1 and one.player_ids == [402]


def test_physics_surfaced_and_frame_aligned(physics_fixture):
    ds = DoomWebDataset.from_local(physics_fixture)
    clip = ds.load_match(ds.match_ids()[0], clip_len=8, target_fps=10, decode=False, clip_ids=[0])[0]
    assert clip.physics is not None
    assert len(clip.physics) == N_PLAYERS  # one per perspective
    assert all(len(p) == 8 for p in clip.physics)  # T per-frame states
    for p in clip.physics:  # sampled at exactly the clip's chunk-local frame indices
        assert [state["ball"]["location"]["x"] for state in p] == clip.frame_indices


def test_physics_none_when_absent(fixture):
    ds = DoomWebDataset.from_local(fixture)
    clip = ds.load_match(ds.match_ids()[0], clip_len=8, target_fps=10, decode=False, clip_ids=[0])[0]
    assert clip.physics is None


def test_physics_surfaced_when_streaming(physics_fixture):
    ds = DoomWebDataset.from_local(physics_fixture)
    clips = list(ds.iter_clips(clip_len=8, target_fps=10, decode=False))
    assert clips and all(c.physics is not None for c in clips)
    for c in clips:
        assert c.physics is not None
        assert [state["ball"]["location"]["x"] for state in c.physics[0]] == c.frame_indices


def test_iter_clips_streams_per_chunk(fixture):
    ds = DoomWebDataset.from_local(fixture)
    # clip_len 8 fits a chunk -> one clip per chunk -> N_CHUNKS clips, each within its own chunk
    clips = list(ds.iter_clips(clip_len=8, target_fps=10, decode=True))
    assert len(clips) == N_CHUNKS
    assert all(c.frames is not None and c.frames.shape == (4, 8, 3, 32, 48) for c in clips)
    assert sorted(c.chunk_idx for c in clips) == list(range(N_CHUNKS))
    assert {c.match_id for c in clips} == set(ds.match_ids())


def test_exclude_replays_drops_overlapping_clips(replay_fixture):
    ds = DoomWebDataset.from_local(replay_fixture)
    mid = ds.match_ids()[0]
    # clip_len 8 @ 10fps -> one clip per chunk; chunk 0's clip spans global frames [0, 16),
    # which overlaps the replay span [0, 20) -> dropped only when exclude_replays=True.
    kept = ds.load_match(mid, clip_len=8, target_fps=10, decode=False, exclude_replays=False)
    pruned = ds.load_match(mid, clip_len=8, target_fps=10, decode=False, exclude_replays=True)
    assert len(kept) == N_CHUNKS
    assert len(pruned) == N_CHUNKS - 1
    assert 0 not in {c.chunk_idx for c in pruned}  # the chunk overlapping the replay is gone
    # the streaming path prunes identically
    streamed = list(ds.iter_clips(clip_len=8, target_fps=10, decode=False, exclude_replays=True))
    assert len(streamed) == N_CHUNKS - 1
    assert 0 not in {c.chunk_idx for c in streamed}


def test_max_clips_caps_count(fixture):
    ds = DoomWebDataset.from_local(fixture)
    clips = ds.load_match(ds.match_ids()[0], clip_len=8, target_fps=10, decode=False, max_clips=2)
    assert len(clips) == 2  # 4 chunks -> 4 clips available, capped to 2


def test_loads_non_contiguous_chunks_by_original_key(gappy_fixture):
    ds = DoomWebDataset.from_local(gappy_fixture)
    mid = ds.match_ids()[0]
    # clip_len 4 @ 20fps -> stride 1 -> fits CHUNK; one clip-set per present chunk position
    clips = ds.load_match(mid, clip_len=4, target_fps=20, decode=False)
    assert clips, "no clips planned for gappy match"
    for c in clips:
        assert c.actions.shape[0] == 4
        assert c.physics is not None and len(c.physics) == 4
    # the loader reaches each of the 6 present chunk positions despite the gaps in original indices
    assert {c.chunk_idx for c in clips} == set(range(6))
    # streaming path resolves the same non-contiguous keys
    assert len(list(ds.iter_clips(clip_len=4, target_fps=20, decode=False))) == len(clips)
