"""Video helpers: uint8 conversion, grid layout, prediction border, and (gated) ffmpeg writing."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from mira.training.visualization import (
    add_prediction_border,
    video_to_uint8,
    videos_to_grid,
    write_video_ffmpeg,
)


def test_video_to_uint8_scales_floats() -> None:
    out = video_to_uint8(torch.tensor([[0.0, 0.5, 1.0]]))
    assert out.dtype == torch.uint8
    assert out.tolist() == [[0, 127, 255]]  # 0.5 * 255 = 127.5, truncated to 127
    # uint8 input is returned unchanged.
    u = torch.zeros(2, 2, dtype=torch.uint8)
    assert torch.equal(video_to_uint8(u), u)


def test_videos_to_grid_tiles_batch() -> None:
    # 4 videos of (T=2, C=3, H=8, W=8) -> a 2x2 grid: (T, C, 16, 16).
    video = torch.randint(0, 256, (4, 2, 3, 8, 8), dtype=torch.uint8)
    grid = videos_to_grid(video)
    assert grid.shape == (2, 3, 16, 16)


def test_add_prediction_border_marks_later_frames() -> None:
    video = torch.zeros(1, 4, 3, 10, 10, dtype=torch.uint8)
    out = add_prediction_border(video, context=2, color=(255, 0, 0), border=2)
    # Frames before `context` are untouched; frames from `context` on get a red border.
    assert out[0, 0].sum() == 0
    assert out[0, 2, 0, 0, 0] == 255  # red channel set on the border
    assert out[0, 2, 1, 0, 0] == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_write_video_ffmpeg_produces_a_file() -> None:
    video = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "clip.mp4"
        write_video_ffmpeg(out, video, fps=10)
        assert out.is_file() and out.stat().st_size > 0
