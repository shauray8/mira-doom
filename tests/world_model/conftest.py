"""Shared fixtures for the world-model tests.

The world model needs a frozen codec, but the real codec requires DINOv3 weights and a trained
checkpoint. These tests build a tiny :class:`StubCodec` with the same interface (encode / decode /
preprocess_batch / downsampling factors / latent_dim) so the world-model forward, encoding and
rollout logic can be exercised offline without any checkpoint or network access.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from mira.codec.rae_encoder import RAEEncoderOutputs
from mira.data.batch import VideoActionBatch
from mira.ml.image_config import ImageConfig
from mira.world_model.actions_config import ActionConfig, ActionTensors
from mira.world_model.config import LatentWorldModelConfig

KEYS = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]

# Tiny shapes that still exercise every code path (patchify, spatial+temporal attention, kv-cache).
LATENT_DIM = 8
TEMPORAL_DOWNSAMPLING = 2
SPATIAL_DOWNSAMPLING = 16
VIDEO_HEIGHT = 64
VIDEO_WIDTH = 64
VIDEO_TIMESTEPS = 8
VIDEO_FPS = 10


class StubCodec(nn.Module):
    """A frozen-codec stand-in with the interface :class:`LatentWorldModel` reads.

    It downsamples the input video by fixed temporal/spatial factors to a random latent of the right
    shape and decodes back to a video of the matching shape; the numbers are arbitrary but the shapes
    match a real RAE codec so the world-model wiring is exercised faithfully.
    """

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.temporal_downsampling = TEMPORAL_DOWNSAMPLING
        self.spatial_downsampling = SPATIAL_DOWNSAMPLING
        self.info_from_checkpoint: dict | None = None
        # Mirrors codec.config.encoder.video.{height,width}; the world model mutates these.
        self.config = SimpleNamespace(
            encoder=SimpleNamespace(
                video=SimpleNamespace(height=VIDEO_HEIGHT, width=VIDEO_WIDTH, timesteps=VIDEO_TIMESTEPS)
            )
        )

    def preprocess_batch(self, batch: VideoActionBatch) -> None:
        batch.video = batch.video.float()

    def encode(self, video: Tensor, trim_video: bool = True) -> tuple[Tensor, RAEEncoderOutputs]:
        b, t, _, h, w = video.shape
        t_lat = t // self.temporal_downsampling
        h_lat = h // self.spatial_downsampling
        w_lat = w // self.spatial_downsampling
        z = torch.randn(b, t_lat, self.latent_dim, h_lat, w_lat, device=video.device)
        return video, RAEEncoderOutputs(z=z)

    def decode(self, z: Tensor) -> Tensor:
        b, t, _, h, w = z.shape
        out_t = t * self.temporal_downsampling
        out_h = h * self.spatial_downsampling
        out_w = w * self.spatial_downsampling
        return torch.zeros(b, out_t, 3, out_h, out_w, device=z.device)


def tiny_config(**overrides) -> LatentWorldModelConfig:
    """A minimal valid :class:`LatentWorldModelConfig` for the stubbed codec."""
    config = LatentWorldModelConfig(
        actions=ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=VIDEO_FPS),
        video=ImageConfig(
            height=VIDEO_HEIGHT,
            width=VIDEO_WIDTH,
            channels=3,
            timesteps=VIDEO_TIMESTEPS,
            fps=VIDEO_FPS,
        ),
        codec_checkpoint="stub.pth",
        latent_mean_std=[0.0, 1.0],
        use_clean_past=True,
        learned_temporal_pool=True,
        use_codec_posterior_mean=True,
        ada_attn_ln=True,
        attention_gating=True,
        n_context_frames=4,
        hidden_dim=64,
        n_head=4,
        n_kv_head=2,
        n_layers=2,
        time_attention_every=1,
    )
    return config.model_copy(update=overrides) if overrides else config


def build_world_model(monkeypatch, **config_overrides):
    """Build a :class:`LatentWorldModel` whose codec is the :class:`StubCodec`."""
    import mira.world_model.latent_world_model as lwm

    monkeypatch.setattr(lwm.VideoCodec, "load_from_checkpoint", staticmethod(lambda *a, **k: StubCodec()))
    model = lwm.LatentWorldModel(tiny_config(**config_overrides))
    model.eval()
    return model


def make_batch(batch_size: int = 2, n_frames: int = VIDEO_TIMESTEPS, n_actions: int = 16):
    """A toy batch: random uint8 video and keyboard-only (all-NaN sensitivity) actions.

    ``mouse_movements[:, t, 0]`` is set to the global action index ``t`` so tests can recover which
    actions a slice selected (used by the action-offset alignment test).
    """
    video = torch.randint(0, 256, (batch_size, n_frames, 3, VIDEO_HEIGHT, VIDEO_WIDTH), dtype=torch.uint8)
    actions = ActionTensors(ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=VIDEO_FPS), batch_size)
    actions.key_presses = torch.randint(0, 2, (batch_size, n_actions, len(KEYS)), dtype=torch.int32)
    mouse = torch.zeros(batch_size, n_actions, 2, dtype=torch.float32)
    mouse[:, :, 0] = torch.arange(n_actions, dtype=torch.float32)
    actions.mouse_movements = mouse
    # game_mouse_sensitivity left as all-NaN: the keyboard-only contract.
    return VideoActionBatch(video=video, actions=actions)
