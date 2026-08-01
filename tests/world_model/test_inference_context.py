"""The inference-time context override (`LatentWorldModel.set_inference_context`) and that the shipped
4 s eval defaults roll out cleanly inside an 80-frame chunk.

These use the stubbed codec (no DINO/Inception weights needed), so they always run and directly guard
the rollout sizing / encode-divisibility that the eval-clip default depends on.
"""

from __future__ import annotations

import pytest

from mira.ml.image_config import ImageConfig
from mira.world_model.config import WorldModelInferenceConfig

from .conftest import build_world_model, make_batch

# Multiplayer-sized window (video.timesteps=80) so the 38-frame eval context is valid; fps matches
# the stub's action rate (VIDEO_FPS) so action_temporal_downsampling stays 2 (= temporal_downsampling).
WINDOW_80 = ImageConfig(height=64, width=64, channels=3, timesteps=80, fps=10)
WINDOW_40 = ImageConfig(height=64, width=64, channels=3, timesteps=40, fps=10)


def test_set_inference_context_updates_latents_together(monkeypatch) -> None:
    model = build_world_model(monkeypatch, video=WINDOW_80)
    model.set_inference_context(38)
    assert model.config.n_context_frames == 38
    assert model.n_context_frames == 38
    assert model.n_context_latents == 38 // model.temporal_downsampling  # 19


def test_set_inference_context_rejects_non_multiple_of_downsampling(monkeypatch) -> None:
    model = build_world_model(monkeypatch, video=WINDOW_80)
    with pytest.raises(ValueError, match="multiple of"):
        model.set_inference_context(39)  # odd, temporal_downsampling is 2


def test_set_inference_context_rejects_context_at_or_above_window(monkeypatch) -> None:
    # video.timesteps=40 forbids a 40-frame context: the window n_context_latents+1 would overflow
    # the 20-latent trained window. 38 is allowed.
    model = build_world_model(monkeypatch, video=WINDOW_40)
    with pytest.raises(ValueError, match="video.timesteps"):
        model.set_inference_context(40)
    model.set_inference_context(38)  # window = 38//2 + 1 = 20 = trained window, valid
    assert model.n_context_latents == 19


def test_shipped_default_rollout_fits_80_frame_chunk(monkeypatch) -> None:
    # The shipped eval defaults: n_context_frames=38, num_unrolled_frames=20, temporal_downsampling=2
    # -> clip_len = 38 + 20*2 = 78 <= 80. The rollout must produce 78//2 = 39 latent frames with no
    # size-mismatch, i.e. the clip length stays divisible by the codec temporal_downsampling.
    model = build_world_model(monkeypatch, video=WINDOW_80)
    model.set_inference_context(38)
    stride = model.temporal_downsampling
    n_frames = model.config.n_context_frames + 20 * stride
    assert n_frames == 78 and n_frames <= 80 and n_frames % stride == 0
    batch = make_batch(batch_size=1, n_frames=n_frames, n_actions=2 * n_frames)
    out = model.inference(
        batch, WorldModelInferenceConfig(n_diffusion_steps=2, noise_level=0.0), progress_bar=False
    )
    assert out.z_t.shape[1] == n_frames // stride  # 39 latent frames rolled out, no size mismatch
