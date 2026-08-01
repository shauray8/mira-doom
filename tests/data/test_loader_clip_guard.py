"""The clip-length feasibility guard in ``create_loader`` / ``DoomWebDataset.max_clip_frames``.

When no clip in the dataset is long enough for the requested ``clip_len``, an ``infinite=True`` loader
would skip every match and loop over an empty epoch forever. The guard turns that silent hang into an
up-front error. These tests are metadata-only (they build just an ``index.json``, no tar/video), so
the check fires before any decode -- fast, CPU-only, and it can never actually hang.
"""

import json
from pathlib import Path

import pytest

from mira.data.dataset import DoomWebDataset
from mira.data.training_loader import create_loader

FPS = 20


def _write_index(dir_: Path, chunk_frames_per_match: list[list[int]], *, fps: int = FPS) -> Path:
    """Write a minimal ``index.json`` (no tar) describing one match per ``chunk_frames`` list."""
    dir_.mkdir(parents=True, exist_ok=True)
    entries = []
    for m, chunk_frames in enumerate(chunk_frames_per_match):
        frames = sum(chunk_frames)
        entries.append(
            {
                "match_id": f"2026-01-0{m + 1}T00-00-00Z-match{m:02d}",
                "shard": "dataset_00.tar",
                "n_players": 1,
                "chunk_frames": chunk_frames,
                "perspectives": [{"player_id": 1, "team": 0, "frames": frames, "duration": frames / fps}],
            }
        )
    (dir_ / "index.json").write_text(json.dumps({"total_samples": len(entries), "entries": entries}))
    return dir_


def test_max_clip_frames_at_source_rate(tmp_path):
    # One match, chunks of 80 and 40 frames at 20fps; target_fps == source_fps -> stride 1.
    ds = DoomWebDataset.from_local(_write_index(tmp_path, [[80, 40]]))
    assert ds.max_clip_frames(target_fps=20) == 80  # (max chunk 80 - 1)//1 + 1


def test_max_clip_frames_accounts_for_stride(tmp_path):
    # Downsampling 20fps -> 10fps is stride 2: a clip_len spans (clip_len-1)*2 + 1 source frames,
    # so the longest that fits an 80-frame chunk is (80-1)//2 + 1 = 40.
    ds = DoomWebDataset.from_local(_write_index(tmp_path, [[80]]))
    assert ds.max_clip_frames(target_fps=10) == 40


def test_max_clip_frames_takes_the_longest_across_matches(tmp_path):
    ds = DoomWebDataset.from_local(_write_index(tmp_path, [[40], [80], [20, 20]]))
    assert ds.max_clip_frames(target_fps=20) == 80


def test_create_loader_raises_when_no_clip_fits(tmp_path):
    # Every chunk is 80 frames, so a 200-frame eval clip fits nothing.
    index = _write_index(tmp_path, [[80], [80]])
    with pytest.raises(ValueError, match="longest clip that fits"):
        create_loader(index, clip_len=200, target_fps=20, infinite=True, num_workers=0)


def test_create_loader_raises_before_iterating(tmp_path):
    # The guard fires at construction, so no tar is ever opened (there is none) and nothing can hang.
    index = _write_index(tmp_path, [[80]])
    with pytest.raises(ValueError, match=r"clip_len=81"):
        create_loader(index, clip_len=81, target_fps=20, infinite=True, num_workers=0)


def test_create_loader_ok_when_clip_fits(tmp_path):
    # Exactly-fitting request (clip_len == longest chunk) must not raise.
    index = _write_index(tmp_path, [[80]])
    loader = create_loader(index, clip_len=80, target_fps=20, infinite=True, num_workers=0)
    assert loader is not None


def test_create_loader_ok_when_some_but_not_all_fit(tmp_path):
    # Mixed lengths: the short match is skipped while streaming, but since one match fits the guard
    # must let construction through (it only blocks the all-too-short case).
    index = _write_index(tmp_path, [[20], [80]])
    loader = create_loader(index, clip_len=60, target_fps=20, infinite=True, num_workers=0)
    assert loader is not None
