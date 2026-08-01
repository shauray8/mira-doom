"""End-to-end tests of the training DataLoader on a self-contained synthetic WebDataset fixture.

The fixture is a real tar (stdlib ``tarfile`` + an ffmpeg-encoded mp4), so the decode path runs for
real. Each perspective presses a distinct key so row<->perspective ordering can be asserted. Needs
ffmpeg (run via ``pixi run test``).
"""

import io
import itertools
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from mira.data.actions import DEFAULT_KEYS
from mira.data.batch import VideoActionBatch
from mira.data.training_loader import ClipMeta, create_loader

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg (run via pixi)")

FPS, CHUNK, N_CHUNKS, N_PLAYERS = 20, 20, 2, 4  # 2 chunks x 20 frames
PLAYER_IDS = [10, 20, 30, 40]  # ascending == the p0..p3 file order the dataset expects


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


def _build(out: Path, *, match_id: str = "2026-05-10T00-00-00Z-abcdef") -> Path:
    """Build a one-match WebDataset tar + index.json; perspective ``i`` presses ``DEFAULT_KEYS[i]``."""
    out.mkdir(parents=True, exist_ok=True)
    vb = _chunk_mp4(out / "v.mp4")
    (out / "v.mp4").unlink()
    # Perspective i presses exactly key i for every frame, so its multi-hot column i is the only set one.
    per_actions = [
        "".join(json.dumps({"keys": [DEFAULT_KEYS[i]]}) + "\n" for _ in range(CHUNK)).encode()
        for i in range(N_PLAYERS)
    ]

    shard = "dataset_00.tar"
    with tarfile.open(out / shard, "w") as tar:
        for c in range(N_CHUNKS):
            key = f"{match_id}_c{c:05d}"
            for i in range(N_PLAYERS):
                _add(tar, f"{key}.p{i}.mp4", vb)
                _add(tar, f"{key}.p{i}.jsonl", per_actions[i])

    frames = N_CHUNKS * CHUNK
    entry = {
        "match_id": match_id,
        "shard": shard,
        "n_players": N_PLAYERS,
        "chunk_frames": [CHUNK] * N_CHUNKS,
        "perspectives": [
            {
                "player_id": PLAYER_IDS[i],
                "team": i % 2,
                "frames": frames,
                "duration": frames / FPS,
                "recording_offset_sec": 0.0,
                "anchors": [{"event_type": 1, "event_name": "GoalScored", "master_sec": 1.0}],
            }
            for i in range(N_PLAYERS)
        ],
    }
    (out / "index.json").write_text(json.dumps({"total_samples": 1, "entries": [entry]}))
    return out


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("rl_train"))


def _first_batch(loader) -> tuple[VideoActionBatch, list[ClipMeta]]:
    return next(iter(loader))


def test_collate_shapes_and_dtypes_single_player(fixture):
    loader = create_loader(
        fixture,
        clip_len=8,
        target_fps=10,
        n_players=1,
        batch_size=3,
        shuffle=False,
        infinite=False,
        num_workers=0,
    )
    batch, metas = _first_batch(loader)
    assert isinstance(batch, VideoActionBatch)
    b = batch.video.shape[0]
    assert b == 3  # batch_size * n_players
    assert batch.video.shape == (b, 8, 3, 32, 48) and batch.video.dtype == torch.uint8
    assert batch.actions.key_presses.shape == (b, 8, 9) and batch.actions.key_presses.dtype == torch.int32
    assert batch.actions.mouse_movements.shape == (b, 8, 2)
    assert batch.actions.mouse_movements.dtype == torch.float32
    assert torch.count_nonzero(batch.actions.mouse_movements) == 0  # zeros: keyboard-only
    assert batch.actions.game_mouse_sensitivity.shape == (b,)
    assert torch.isnan(batch.actions.game_mouse_sensitivity).all()  # all-NaN: no mouse sensitivity
    assert len(metas) == b and all(isinstance(m, ClipMeta) for m in metas)


def test_action_fps_emits_two_actions_per_frame(fixture):
    """With `action_fps = 2 * target_fps` the batch keeps `T` video frames but `2T` action steps
    (the optional knob; the default is 1:1), and the stored ActionConfig.target_fps follows action_fps."""
    loader = create_loader(
        fixture,
        clip_len=8,
        target_fps=10,
        action_fps=20,
        n_players=1,
        batch_size=3,
        shuffle=False,
        infinite=False,
        num_workers=0,
    )
    batch, _ = _first_batch(loader)
    b = batch.video.shape[0]
    assert b == 3
    assert batch.video.shape == (b, 8, 3, 32, 48)  # T video frames unchanged
    assert batch.actions.key_presses.shape == (b, 16, 9)  # 2T action steps
    assert batch.actions.mouse_movements.shape == (b, 16, 2)
    assert batch.actions.config.target_fps == 20


def test_action_fps_none_keeps_one_action_per_frame(fixture):
    """Regression: omitting action_fps keeps one action step per frame, a (B, T, 9) shape."""
    loader = create_loader(
        fixture, clip_len=8, target_fps=10, n_players=1, batch_size=2,
        shuffle=False, infinite=False, num_workers=0,
    )  # fmt: skip
    batch, _ = _first_batch(loader)
    assert batch.actions.key_presses.shape == (2, 8, 9)
    assert batch.actions.config.target_fps == 10


def test_n_players_grouping_is_contiguous_and_player_id_ordered(fixture):
    loader = create_loader(
        fixture,
        clip_len=8,
        target_fps=10,
        n_players=4,
        batch_size=1,
        shuffle=False,
        infinite=False,
        num_workers=0,
    )
    batch, metas = _first_batch(loader)
    assert batch.video.shape[0] == 4  # batch_size * n_players, one full match group
    # the 4 perspectives of the match are contiguous, in ascending player_id order
    assert [m.perspective for m in metas] == [0, 1, 2, 3]
    assert [m.player_id for m in metas] == PLAYER_IDS
    assert len({m.match_id for m in metas}) == 1
    assert len({m.clip_id for m in metas}) == 1  # same clip, four perspectives
    # row i must be perspective i: perspective i presses exactly key column i
    pressed = batch.actions.key_presses.any(dim=1)  # (4, 9) which keys appear over time
    for i in range(4):
        assert pressed[i, i].item() == 1
        assert torch.count_nonzero(pressed[i]).item() == 1


def test_slice_time_consistent_with_no_audio(fixture):
    loader = create_loader(
        fixture,
        clip_len=8,
        target_fps=10,
        n_players=1,
        batch_size=2,
        shuffle=False,
        infinite=False,
        num_workers=0,
    )
    batch, _ = _first_batch(loader)
    sl = batch.slice_time(1, 4, fps=10)
    assert sl.video.shape == (2, 3, 3, 32, 48)
    assert torch.equal(sl.video, batch.video[:, 1:4])
    assert sl.actions.key_presses.shape == (2, 3, 9)
    assert torch.equal(sl.actions.key_presses, batch.actions.key_presses[:, 1:4, :])
    assert not hasattr(sl, "audio")


def test_infinite_stream_yields_more_than_dataset_size(fixture):
    # 2 chunks x 4 perspectives = 8 single-player rows; infinite must loop past that.
    loader = create_loader(
        fixture,
        clip_len=8,
        target_fps=10,
        n_players=1,
        batch_size=2,
        shuffle=False,
        infinite=True,
        num_workers=0,
    )
    batches = list(itertools.islice(iter(loader), 10))
    assert len(batches) == 10
    assert all(batch.video.shape[0] == 2 for batch, _ in batches)


def test_train_val_split_via_separate_indices(tmp_path_factory):
    train = _build(tmp_path_factory.mktemp("train"), match_id="train-match-aaaaaa")
    val = _build(tmp_path_factory.mktemp("val"), match_id="val-match-bbbbbb")
    train_loader = create_loader(
        train,
        clip_len=8,
        target_fps=10,
        n_players=1,
        batch_size=2,
        shuffle=False,
        infinite=False,
        num_workers=0,
    )
    val_loader = create_loader(
        val,
        clip_len=8,
        target_fps=10,
        n_players=1,
        batch_size=2,
        shuffle=False,
        infinite=False,
        num_workers=0,
    )
    train_ids = {m.match_id for _, metas in train_loader for m in metas}
    val_ids = {m.match_id for _, metas in val_loader for m in metas}
    assert train_ids == {"train-match-aaaaaa"}
    assert val_ids == {"val-match-bbbbbb"}
    assert train_ids.isdisjoint(val_ids)
