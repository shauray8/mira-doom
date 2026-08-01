"""Config schema tests: the release codec YAML validates into ``VideoCodecConfig``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mira.codec import (
    RAEEncoderConfig,
    StridedConvBottleneckConfig,
    VideoCodecConfig,
    ViTDecoderConfig,
)
from mira.codec.codec_model import REMOVED_CONFIG_FIELDS
from mira.ml.config_loading import drop_removed_fields, strip_hydra_targets

FIXTURE = Path(__file__).parent / "fixtures" / "raev2_codec_tdown.yaml"


def _load_architecture_config() -> dict:
    """Load the fixture through the same strip + drop-removed-fields path as load_from_checkpoint.

    The fixture mirrors a released config, which still carries the since-removed ``is_audio_model``
    field; this exercises that the load path tolerates and drops it.
    """
    omegaconf = pytest.importorskip("omegaconf")
    raw = omegaconf.OmegaConf.load(FIXTURE)
    # The model fragment interpolates ${run.compile}; supply it so resolution succeeds.
    raw = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.create({"run": {"compile": False}}), raw)
    container = omegaconf.OmegaConf.to_container(raw.architecture.config, resolve=True)
    cleaned = drop_removed_fields(strip_hydra_targets(container), REMOVED_CONFIG_FIELDS)
    assert isinstance(cleaned, dict)
    return cleaned


def test_release_yaml_validates() -> None:
    """The release codec config (mirroring configs/model/raev2_codec_tdown.yaml) parses into
    VideoCodecConfig with the expected RAEv2-tdown shape."""
    config = VideoCodecConfig.model_validate(_load_architecture_config())

    assert isinstance(config.encoder, RAEEncoderConfig)
    assert isinstance(config.decoder, ViTDecoderConfig)
    assert config.encoder.latent_dim == 32
    assert config.encoder.rae_model == "dinov3_vitl16"
    assert config.encoder.aggregation_layers == [11, 13, 15, 17, 19, 21, 23]
    assert config.encoder.bottleneck.stride == 2
    assert config.encoder.bottleneck.temporal_stride == 2

    assert config.decoder.vit_width == 1152
    assert config.decoder.vit_depth == 28
    assert config.decoder.vit_num_heads == 16
    assert config.decoder.patch_size == 16
    assert config.decoder.patch_size_t == 2
    assert config.decoder.qk_norm == "layernorm"
    assert isinstance(config.decoder.bottleneck, StridedConvBottleneckConfig)
    assert config.decoder.bottleneck.stride == 2
