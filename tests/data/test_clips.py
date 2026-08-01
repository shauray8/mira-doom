"""Tests for clip enumeration: stride computation and fixed-length window indices."""

from __future__ import annotations

import pytest

from mira.data.clips import compute_clip_frame_indices, compute_stride


def test_compute_stride_integer_multiple():
    assert compute_stride(60.0, 30) == 2
    assert compute_stride(20.0, 10) == 2
    assert compute_stride(20.0, 20) == 1


def test_compute_stride_tolerates_small_slack():
    # Measured fps is sometimes slightly off the nominal value.
    assert compute_stride(29.97, 30) == 1
    assert compute_stride(59.94, 30) == 2


def test_compute_stride_rejects_non_multiple():
    with pytest.raises(ValueError):
        compute_stride(25.0, 10)


def test_compute_stride_rejects_non_positive_target():
    with pytest.raises(ValueError):
        compute_stride(20.0, 0)
    with pytest.raises(ValueError):
        compute_stride(20.0, -10)


def test_clip_indices_non_overlapping_and_strided():
    clips, stride = compute_clip_frame_indices(total_frames=80, source_fps=20.0, clip_len=4, target_fps=10)
    assert stride == 2
    # Indices step by stride; windows are non-overlapping and clip_len long.
    assert clips[0] == [0, 2, 4, 6]
    assert clips[1] == [8, 10, 12, 14]
    assert all(len(c) == 4 for c in clips)


def test_clip_indices_drop_trailing_short_window():
    # 20 frames @ stride 2 -> indices [0,2,...,18] = 10 values; clip_len 4 -> 2 full clips, 2 left over.
    clips, stride = compute_clip_frame_indices(total_frames=20, source_fps=20.0, clip_len=4, target_fps=10)
    assert stride == 2
    assert len(clips) == 2
    assert clips[-1] == [8, 10, 12, 14]
