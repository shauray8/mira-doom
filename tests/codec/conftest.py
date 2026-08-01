"""Shared fixtures for the codec tests.

The RAE encoder needs the DINOv3 backbone, which is loaded through ``torch.hub`` and requires
network access (or a populated hub cache) plus a few transitive deps. Tests that build the full
codec go through :func:`build_codec_or_skip`, which skips gracefully when the backbone can't be
constructed (offline / no hub cache / missing deps).
"""

from __future__ import annotations

import pytest
import torch

from mira.codec import (
    RAEEncoderConfig,
    StridedConvBottleneckConfig,
    VideoCodec,
    VideoCodecConfig,
    ViTDecoderConfig,
)
from mira.ml import ImageConfig

# Matches config/model/raev2_codec_tdown.yaml: 40 frames @ 20 fps, 288x512, td=2.
VIDEO = ImageConfig(height=288, width=512, channels=3, timesteps=40, fps=20)


def raev2_tdown_config(*, vit_depth: int = 2, vit_width: int = 256) -> VideoCodecConfig:
    """The release RAEv2-tdown architecture, with a shallow/narrow ViT decoder for test speed.

    Depth and width do not change any latent or video shape, only the parameter count, so the
    round-trip and downsampling-factor assertions are unaffected.
    """
    encoder = RAEEncoderConfig(
        latent_dim=32,
        rae_model="dinov3_vitl16",
        aggregation_layers=[11, 13, 15, 17, 19, 21, 23],
        bottleneck=StridedConvBottleneckConfig(stride=2, temporal_stride=2, noise_tau=0.0),
        compile_dino=False,
        video=VIDEO,
    )
    decoder = ViTDecoderConfig(
        video=VIDEO,
        latent_dim=32,
        activation_checkpointing=False,
        bottleneck=StridedConvBottleneckConfig(stride=2),
        vit_width=vit_width,
        vit_depth=vit_depth,
        vit_num_heads=16,
        mlp_dim_multiplier=4,
        qk_norm="layernorm",
        patch_size=16,
        patch_size_t=2,
    )
    return VideoCodecConfig(encoder=encoder, decoder=decoder)


def build_codec_or_skip(config: VideoCodecConfig | None = None) -> VideoCodec:
    """Build a codec, skipping the test if the DINOv3 backbone can't be loaded.

    The backbone load can fail offline (no hub cache / network), without local pretrained weights,
    or with missing transitive deps of the dinov3 hub repo; all of these skip rather than fail.
    """
    config = config or raev2_tdown_config()
    try:
        return VideoCodec(config, require_dino_weights=True).eval()
    except Exception as exc:  # noqa: BLE001 -- any backbone-load failure should skip, not fail
        pytest.skip(f"DINOv3 backbone unavailable, skipping: {type(exc).__name__}: {exc}")


@pytest.fixture
def codec() -> VideoCodec:
    return build_codec_or_skip()


def random_video(batch: int = 1, frames: int = 40, height: int = 288, width: int = 512) -> torch.Tensor:
    """A ``(B, T, 3, H, W)`` uint8 video, as produced by the data loader."""
    return torch.randint(0, 256, (batch, frames, 3, height, width), dtype=torch.uint8)


def tiny_raev2_config() -> VideoCodecConfig:
    """A tiny RAEv2-tdown config (64x64, 4 frames, shallow decoder) for fast CPU forward/backward.

    Keeps the release architecture (DINOv3-L backbone, layer aggregation, td=2 strided bottleneck,
    ViT decoder) so the loss and training paths are exercised, but shrinks every shape so a full
    forward+backward is cheap enough to run in a unit test.
    """
    video = ImageConfig(height=64, width=64, channels=3, timesteps=4, fps=20)
    encoder = RAEEncoderConfig(
        latent_dim=8,
        rae_model="dinov3_vitl16",
        aggregation_layers=[11, 13, 15, 17, 19, 21, 23],
        bottleneck=StridedConvBottleneckConfig(stride=2, temporal_stride=2, noise_tau=0.0),
        compile_dino=False,
        video=video,
    )
    decoder = ViTDecoderConfig(
        video=video,
        latent_dim=8,
        activation_checkpointing=False,
        bottleneck=StridedConvBottleneckConfig(stride=2),
        vit_width=64,
        vit_depth=2,
        vit_num_heads=4,
        mlp_dim_multiplier=2,
        qk_norm="layernorm",
        patch_size=16,
        patch_size_t=2,
    )
    return VideoCodecConfig(encoder=encoder, decoder=decoder)


def tiny_batch(batch: int = 1):
    """A ``VideoActionBatch`` with a tiny random video (actions are unused by the codec)."""
    from mira.data.actions import DEFAULT_KEYS
    from mira.data.batch import VideoActionBatch
    from mira.world_model.actions_config import ActionConfig, ActionTensors

    actions = ActionTensors(
        config=ActionConfig(valid_keys=list(DEFAULT_KEYS), source_fps=20, target_fps=20),
        batch_size=batch,
    )
    return VideoActionBatch(video=random_video(batch=batch, frames=4, height=64, width=64), actions=actions)
