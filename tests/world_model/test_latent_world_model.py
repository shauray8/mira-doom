"""Tests for the LatentWorldModel forward loss, latent encoding, and autoregressive rollout.

All tests use the stubbed codec (see conftest) so they run offline without any checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from mira.world_model.actions_config import ActionConfig
from mira.world_model.config import WorldModelInferenceConfig

from .conftest import (
    KEYS,
    LATENT_DIM,
    SPATIAL_DOWNSAMPLING,
    TEMPORAL_DOWNSAMPLING,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_TIMESTEPS,
    VIDEO_WIDTH,
    build_world_model,
    make_batch,
)


def test_forward_returns_finite_loss_dict(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=2)
    outputs = model(batch)

    assert set(outputs) == {"loss_total", "loss_diffusion"}
    for key, value in outputs.items():
        assert value.shape == (), f"{key} should be a scalar"
        assert torch.isfinite(value), f"{key} should be finite, got {value}"


def test_forward_with_action_fps_knob_atd4(monkeypatch) -> None:
    """The optional action_fps knob (actions at 2x the frame rate) gives atd=4, 2 actions/video-frame,
    and a finite forward loss when the batch carries 2*T action steps (as the knobbed loader emits).

    The released default is 1:1 (atd == codec td); this exercises the decoupled rate explicitly."""
    # actions.target_fps = 2 * video.fps -> action_temporal_downsampling = 20*2//10 = 4.
    model = build_world_model(
        monkeypatch, actions=ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=2 * VIDEO_FPS)
    )
    assert model.action_temporal_downsampling == 4
    assert model.actions_per_video_frame == 2

    # The loader emits 2 action steps per video frame; mirror that shape here.
    batch = make_batch(batch_size=2, n_frames=VIDEO_TIMESTEPS, n_actions=2 * VIDEO_TIMESTEPS)
    outputs = model(batch)

    assert set(outputs) == {"loss_total", "loss_diffusion"}
    assert torch.isfinite(outputs["loss_total"])


def test_forward_loss_backpropagates(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    model.train()
    batch = make_batch(batch_size=2)
    outputs = model(batch)
    outputs["loss_total"].backward()

    # The world model and action encoder receive gradients; the frozen codec must not.
    assert any(p.grad is not None for p in model.world_model.parameters())
    assert all(p.requires_grad is False for p in model.codec.parameters())


def test_encode_video_shapes(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=3)
    model.codec.preprocess_batch(batch)
    z = model.encode_video(batch)

    expected_h = VIDEO_HEIGHT // SPATIAL_DOWNSAMPLING
    expected_w = VIDEO_WIDTH // SPATIAL_DOWNSAMPLING
    expected_t = VIDEO_TIMESTEPS // TEMPORAL_DOWNSAMPLING
    # (b, t, h, w, c) with channels last after encode_video's rearrange.
    assert z.shape == (3, expected_t, expected_h, expected_w, LATENT_DIM)


def test_inference_rollout_shapes(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=2)
    config = WorldModelInferenceConfig(n_diffusion_steps=3)

    outputs = model.inference(batch, config=config, progress_bar=False)

    n_latent_frames = VIDEO_TIMESTEPS // TEMPORAL_DOWNSAMPLING
    assert outputs.z_t.shape == (
        2,
        n_latent_frames,
        VIDEO_HEIGHT // SPATIAL_DOWNSAMPLING,
        VIDEO_WIDTH // SPATIAL_DOWNSAMPLING,
        LATENT_DIM,
    )
    # decode expands the generated latents back to video frames.
    window_size = model.n_context_latents + 1
    n_generated_latents = (n_latent_frames - window_size) + window_size
    assert outputs.output_video.shape == (
        2,
        n_generated_latents * TEMPORAL_DOWNSAMPLING,
        3,
        VIDEO_HEIGHT,
        VIDEO_WIDTH,
    )
    assert torch.isfinite(outputs.z_t).all()


def test_inference_action_offset_alignment(monkeypatch) -> None:
    """The action window selected per step must start at offset ``atd - 1``.

    We replace the action encoder with a recorder that reports the global action indices of each
    slice it receives (encoded in ``mouse_movements[:, :, 0]``), then assert the first slice starts
    at ``off = action_temporal_downsampling - 1`` and the windows are contiguous and atd-aligned.
    """
    model = build_world_model(monkeypatch)
    atd = model.action_temporal_downsampling
    off = atd - 1
    assert atd == 2 and off == 1, "test assumes the tiny config's atd"

    recorded: list[list[float]] = []

    class Recorder(nn.Module):
        def forward(self, actions):
            idx = actions.mouse_movements[:, :, 0]
            recorded.append(idx[0].tolist())
            b, n_in = idx.shape
            return torch.zeros(b, n_in // atd + 1, model.config.hidden_dim)

    model.action_encoder = Recorder()  # type: ignore[assignment]
    model.inference(
        make_batch(batch_size=1), config=WorldModelInferenceConfig(n_diffusion_steps=2), progress_bar=False
    )

    n_latent_frames = VIDEO_TIMESTEPS // TEMPORAL_DOWNSAMPLING
    window_size = model.n_context_latents + 1
    expected_starts = [start * atd + off for start in range(n_latent_frames - window_size + 1)]
    assert [slice_idx[0] for slice_idx in recorded] == expected_starts
    # Each recorded window is contiguous (consecutive global indices).
    for slice_idx in recorded:
        assert slice_idx == list(range(int(slice_idx[0]), int(slice_idx[0]) + len(slice_idx)))
