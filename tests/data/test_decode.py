"""Tests for the resize half of frame decoding (the torchcodec decode itself needs a real video)."""

import torch

from mira.data.decode import _resize


def test_resize_changes_shape_and_keeps_uint8():
    frames = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
    out = _resize(frames, (8, 8))
    assert out.shape == (3, 3, 8, 8)
    assert out.dtype == torch.uint8
    # round/clamp keeps values in the valid uint8 range.
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_resize_is_noop_when_already_target_size():
    frames = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    out = _resize(frames, (8, 8))
    assert out is frames  # same size -> returned untouched, no float round-trip


def test_resize_can_upscale():
    frames = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
    out = _resize(frames, (16, 16))
    assert out.shape == (1, 3, 16, 16)
    assert out.dtype == torch.uint8
