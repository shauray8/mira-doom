"""Shared fixtures for the inference tests.

The single-player builders (``StubCodec``, ``tiny_config``, ``build_world_model``, ``make_batch``)
are reused from :mod:`tests.world_model.conftest`; this module only adds a multiplayer builder so the
rollout / eval tests can exercise both the single and the 4-player paths offline, with no
checkpoint or network access.
"""

from __future__ import annotations

from tests.world_model.conftest import StubCodec, build_world_model, make_batch, tiny_config

__all__ = ["StubCodec", "build_multi_wrapper", "build_world_model", "make_batch", "tiny_config"]


def _td1_codec(*args, **kwargs) -> StubCodec:
    """A :class:`StubCodec` with ``temporal_downsampling == 1``."""
    codec = StubCodec()
    codec.temporal_downsampling = 1
    return codec


def build_multi_wrapper(monkeypatch, n_players: int = 4, **wm_overrides):
    """Build a :class:`MultiWrapperWorldModel` whose inner codec is a td==1 :class:`StubCodec`."""
    import mira.world_model.latent_world_model as lwm
    from mira.world_model.multi_wrapper_world_model import (
        MultiWrapperWorldModel,
        MultiWrapperWorldModelConfig,
    )

    monkeypatch.setattr(lwm.VideoCodec, "load_from_checkpoint", staticmethod(_td1_codec))
    config = MultiWrapperWorldModelConfig(n_players=n_players, wm_config=tiny_config(**wm_overrides))
    model = MultiWrapperWorldModel(config)
    model.eval()
    return model
