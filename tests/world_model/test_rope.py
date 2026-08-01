"""RoPE parity tests for the temporal and spatial rotary tables.

We guard against drift by recomputing the rotary tables from the documented formula (an independent
reference) and asserting the module reproduces them exactly, plus a couple of pinned golden scalars.
"""

from __future__ import annotations

import math

import torch

from mira.world_model.layers.rope import RoPE, SpatialRoPE2D


def _reference_temporal_rope(dim: int, fps: int, max_period_sec: float, n_frames: int):
    """Independent reference for ``RoPE.forward`` (time-only, fp32)."""
    ds = torch.arange(dim // 2, dtype=torch.float32)
    w = torch.exp(ds * (-math.log(max_period_sec) * 2 / dim))  # (dim // 2)
    grid_t = torch.arange(n_frames, dtype=torch.float32) / fps  # seconds
    freqs = w[None, :] * grid_t[:, None]  # (T, dim // 2)
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # (T, dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def test_temporal_rope_matches_reference() -> None:
    dim, fps, max_period_sec, n_frames = 16, 10, 64.0, 12
    rope = RoPE(dim=dim, fps=fps, max_period_sec=max_period_sec)
    cos, sin = rope(n_frames, torch.device("cpu"))

    ref_cos, ref_sin = _reference_temporal_rope(dim, fps, max_period_sec, n_frames)
    assert cos.shape == (n_frames, dim) and sin.shape == (n_frames, dim)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    torch.testing.assert_close(cos, ref_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, ref_sin, rtol=0, atol=0)


def test_temporal_rope_golden_values() -> None:
    rope = RoPE(dim=16, fps=10, max_period_sec=64.0)
    cos, sin = rope(4, torch.device("cpu"))
    # Frame 0 is at t=0s: every cos is 1, every sin is 0.
    torch.testing.assert_close(cos[0], torch.ones(16))
    torch.testing.assert_close(sin[0], torch.zeros(16))
    # The fastest band (w[0] = exp(0) = 1) at frame 1 (t = 0.1s) rotates by 0.1 rad.
    assert math.isclose(float(cos[1, 0]), math.cos(0.1), rel_tol=1e-6)
    assert math.isclose(float(sin[1, 0]), math.sin(0.1), rel_tol=1e-6)


def test_temporal_rope_odd_dim_padding_branch() -> None:
    # When dim is not a multiple of dim_group (2), the cos/sin tables are left-padded with a
    # ones/zeros column. head_dim is even on the release path so this branch is dead there, but it
    # is part of the verbatim RoPE so we guard it: column 0 is the cos=1 / sin=0 padding.
    dim = 5
    rope = RoPE(dim=dim, fps=10, max_period_sec=64.0)
    cos, sin = rope(4, torch.device("cpu"))
    assert cos.shape == (4, dim) and sin.shape == (4, dim)
    torch.testing.assert_close(cos[:, 0], torch.ones(4))
    torch.testing.assert_close(sin[:, 0], torch.zeros(4))


def test_temporal_rope_offset_shifts_positions() -> None:
    rope = RoPE(dim=16, fps=10, max_period_sec=64.0)
    cos_a, sin_a = rope(3, torch.device("cpu"), offset=torch.tensor(2.0))
    cos_b, _ = rope(5, torch.device("cpu"))
    # An offset of 2 frames reproduces frames [2, 3, 4] of the unshifted table.
    torch.testing.assert_close(cos_a, cos_b[2:5], rtol=0, atol=0)


def _reference_spatial_axis(size: int, dim: int, max_period: float):
    n_freqs = (dim // 2) // 2
    inv_freq_min = 2 * math.pi / max_period
    inv_freq_max = math.pi
    k = torch.arange(n_freqs, dtype=torch.float32)
    inv_freq = inv_freq_min * (inv_freq_max / inv_freq_min) ** (k / max(n_freqs - 1, 1))
    coords = torch.arange(size, dtype=torch.float32)
    freqs = coords[:, None] * inv_freq[None, :]
    return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)


def test_spatial_rope_matches_reference() -> None:
    dim, height, width, max_period = 16, 4, 5, 100.0
    rope = SpatialRoPE2D(dim=dim, max_period=max_period)
    cos, sin = rope(height, width, torch.device("cpu"))
    assert cos.shape == (height * width, dim) and sin.shape == (height * width, dim)
    assert cos.dtype == torch.float32

    cos_y, sin_y = _reference_spatial_axis(height, dim, max_period)
    cos_x, sin_x = _reference_spatial_axis(width, dim, max_period)
    cos_y, sin_y = (t[:, None, :].expand(height, width, -1) for t in (cos_y, sin_y))
    cos_x, sin_x = (t[None, :, :].expand(height, width, -1) for t in (cos_x, sin_x))
    ref_cos = torch.cat([cos_y, cos_x], dim=-1).reshape(height * width, dim)
    ref_sin = torch.cat([sin_y, sin_x], dim=-1).reshape(height * width, dim)
    torch.testing.assert_close(cos, ref_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, ref_sin, rtol=0, atol=0)
