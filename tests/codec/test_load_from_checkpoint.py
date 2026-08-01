"""Saving and loading a codec checkpoint with its ``codec_config.yaml``.

The tiny round-trip test builds a small codec (skipping when the DINOv3 backbone is unavailable).
The tests gated on ``RS_REF_CKPT`` exercise a released RAEv2-tdown checkpoint (path to a checkpoint
with a ``codec_config.yaml`` in a parent directory); no checkpoint ships in CI / offline, so they
are opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mira.codec import VideoCodec, VideoCodecConfig
from mira.codec.codec_model import _find_codec_config

REF_CKPT = os.environ.get("RS_REF_CKPT")
requires_ref_ckpt = pytest.mark.skipif(
    not REF_CKPT, reason="set RS_REF_CKPT to a released RAEv2-tdown codec checkpoint to run"
)


def _write_codec_config(config: VideoCodecConfig, directory: Path) -> None:
    """Write ``config`` in the ``model.architecture.config`` layout load_from_checkpoint reads."""
    from omegaconf import OmegaConf  # noqa: PLC0415 -- optional dep, used only here

    payload = {"model": {"architecture": {"config": config.model_dump()}}}
    OmegaConf.save(OmegaConf.create(payload), directory / VideoCodec.CONFIG_FILENAME)


def test_save_then_load_tiny_round_trip(tmp_path: Path) -> None:
    """save_checkpoint requires codec_config.yaml alongside; reload restores identical weights."""
    from tests.codec.conftest import build_codec_or_skip, tiny_raev2_config

    config = tiny_raev2_config()
    codec = build_codec_or_skip(config).eval()

    _write_codec_config(config, tmp_path)
    ckpt = tmp_path / "codec.pt"
    codec.save_checkpoint(ckpt)

    reloaded = VideoCodec.load_from_checkpoint(ckpt, device="cpu").eval()
    assert reloaded.info_from_checkpoint == {}
    for (k1, v1), (k2, v2) in zip(codec.state_dict().items(), reloaded.state_dict().items(), strict=True):
        assert k1 == k2
        assert torch.equal(v1, v2), k1


@requires_ref_ckpt
def test_find_codec_config_walks_parents() -> None:
    assert REF_CKPT is not None
    config_path = _find_codec_config(REF_CKPT)
    assert config_path is not None, "codec_config.yaml not found in any parent dir of RS_REF_CKPT"
    assert config_path.name == VideoCodec.CONFIG_FILENAME


@requires_ref_ckpt
def test_load_from_checkpoint_runs_and_is_frozen_compatible() -> None:
    assert REF_CKPT is not None
    codec = VideoCodec.load_from_checkpoint(REF_CKPT, device="cpu").eval()

    assert codec.info_from_checkpoint is not None
    assert (codec.temporal_downsampling, codec.spatial_downsampling) == (2, 32)

    video = (torch.rand(1, 40, 3, 288, 512) * 2) - 1
    with torch.no_grad():
        _, encoder_output = codec.encode(video, trim_video=True)
        decoded = codec.decode(encoder_output.z)
    assert encoder_output.z.shape[0] == 1 and encoder_output.z.shape[2] == codec.latent_dim
    assert decoded.shape == (1, 40, 3, 288, 512)
