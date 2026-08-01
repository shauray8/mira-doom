"""Tests for the PSD-M self-distillation loss on the LatentWorldModel and the multiplayer wrapper.

All tests use the stubbed codec (see conftest) so they run offline without any checkpoint. The two
mixing paths under test are ``psd_weight`` (deterministic) and ``psd_loss_prob`` (stochastic); the
default (both 0.0 => ``psd_enabled`` False) must be exactly behavior-preserving.
"""

from __future__ import annotations

import pytest
import torch

from mira.world_model.actions_config import ActionConfig
from mira.world_model.config import LatentWorldModelConfig

from .conftest import KEYS, VIDEO_FPS, build_world_model, make_batch, tiny_config
from .test_multi_wrapper_world_model import build_multi_wrapper_td2

DELTA_KEY = "world_model.diffusion_time_embedding_delta"


def test_psd_disabled_is_behavior_preserving(monkeypatch) -> None:
    """Default config: no delta embedding, PSD branch never runs, exactly the two base keys."""
    model = build_world_model(monkeypatch)
    assert model.config.psd_enabled is False
    assert model.world_model.diffusion_time_embedding_delta is None
    assert all(DELTA_KEY not in k for k in model.state_dict())

    outputs = model(make_batch(batch_size=2))
    assert set(outputs) == {"loss_total", "loss_diffusion"}


def test_psd_stochastic_fires_at_prob_one(monkeypatch) -> None:
    """psd_loss_prob=1.0 forces the branch every step: loss_psd finite, delta created, and the
    logged value is the raw loss (importance weight 1/1.0). loss_total = loss_diffusion + loss_psd."""
    model = build_world_model(monkeypatch, psd_loss_prob=1.0)
    model.train()
    assert model.config.psd_enabled is True
    assert model.world_model.diffusion_time_embedding_delta is not None

    outputs = model(make_batch(batch_size=2))
    assert set(outputs) == {"loss_total", "loss_diffusion", "loss_psd"}
    assert torch.isfinite(outputs["loss_psd"]) and outputs["loss_psd"] > 0
    # At p=1.0 the (unweighted) PSD loss is added to the total and logged as-is (raw / 1.0).
    assert torch.allclose(outputs["loss_total"], outputs["loss_diffusion"] + outputs["loss_psd"])

    outputs["loss_total"].backward()
    assert any(p.grad is not None for p in model.world_model.parameters())


def test_psd_stochastic_importance_weighting(monkeypatch) -> None:
    """When the stochastic branch fires at p<1, the logged loss is importance-weighted by 1/p so its
    expectation over steps equals the true PSD loss."""
    p = 0.5
    model = build_world_model(monkeypatch, psd_loss_prob=p)
    batch = make_batch(batch_size=2)

    fired = False
    for _ in range(50):
        outputs = model(batch)
        assert set(outputs) == {"loss_total", "loss_diffusion", "loss_psd"}
        if outputs["loss_psd"] > 0:  # the branch fired this step
            fired = True
            # Logged value is importance-weighted by 1/p, so the raw (unweighted) contribution
            # added to loss_total is p * loss_psd. Check via the forward identity to avoid the
            # float32 cancellation of (loss_total - loss_diffusion) when the raw loss is tiny.
            assert torch.allclose(outputs["loss_total"], outputs["loss_diffusion"] + p * outputs["loss_psd"])
        else:  # skipped step: logged 0, total unchanged
            assert outputs["loss_psd"] == 0
            assert outputs["loss_total"] == outputs["loss_diffusion"]
    assert fired, "psd_loss_prob=0.5 never fired in 50 steps"


def test_psd_weight_deterministic_scales_total(monkeypatch) -> None:
    """psd_weight>0 adds psd_weight * loss_psd to loss_total every step (no skipping)."""
    weight = 0.3
    model = build_world_model(monkeypatch, psd_weight=weight)
    model.train()
    assert model.config.psd_enabled is True
    assert model.world_model.diffusion_time_embedding_delta is not None

    outputs = model(make_batch(batch_size=2))
    assert set(outputs) == {"loss_total", "loss_diffusion", "loss_psd"}
    assert torch.isfinite(outputs["loss_psd"]) and outputs["loss_psd"] > 0
    assert torch.allclose(outputs["loss_total"], outputs["loss_diffusion"] + weight * outputs["loss_psd"])

    outputs["loss_total"].backward()
    assert any(p.grad is not None for p in model.world_model.parameters())


def test_psd_loss_matches_two_hop_recompute(monkeypatch) -> None:
    """_compute_psdm_loss regresses the student velocity (s -> t) onto the stop-grad midpoint
    two-hop teacher target. Recompute that target here from the model's own forward passes on the
    (z_s, tau_s, tau_t) it sampled and check the loss matches exactly (a self-consistency check of
    the wiring, not a comparison to any external implementation)."""
    model = build_world_model(monkeypatch, psd_weight=1.0)
    model.eval()

    b, h, w, c = 2, model.world_model.latent_height, model.world_model.latent_width, model.latent_dim
    z = torch.randn(b, 4, h, w, c)
    a = torch.randn(b, 4, model.config.hidden_dim)

    # Capture the (z_s, tau_s, tau_t) that the loss actually samples this call.
    captured: dict[str, torch.Tensor] = {}
    real_prep = model.prepare_psdm_inputs

    def spy_prep(z_1):
        z_s, tau_s, tau_t = real_prep(z_1)
        captured.update(z_s=z_s, tau_s=tau_s, tau_t=tau_t)
        return z_s, tau_s, tau_t

    model.prepare_psdm_inputs = spy_prep  # type: ignore[method-assign]
    loss = model._compute_psdm_loss(z, a, shifted_z=None)

    z_s, tau_s, tau_t = captured["z_s"], captured["tau_s"], captured["tau_t"]
    assert (tau_s < tau_t).all(), "upper-triangle invariant 0 <= s < t <= 1 violated"
    tau_u = (tau_s + tau_t) / 2
    with torch.no_grad():
        v_st = model.world_model(z_s, a, tau_s, tau_delta=tau_t - tau_s)
        v_su = model.world_model(z_s, a, tau_s, tau_delta=tau_u - tau_s)
        x_su = z_s + (tau_u - tau_s) * v_su
        v_ut = model.world_model(x_su, a, tau_u, tau_delta=tau_t - tau_u)
        target = 0.5 * v_su + 0.5 * v_ut
        ref = torch.nn.functional.mse_loss(v_st.float(), target.float())
    assert torch.allclose(loss, ref, atol=1e-6)


def test_psd_validator_forbids_both(monkeypatch) -> None:
    """The config validator rejects setting both mixing knobs > 0."""
    with pytest.raises(ValueError, match="at most one of psd_loss_prob"):
        LatentWorldModelConfig(
            actions=ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=VIDEO_FPS),
            video=tiny_config().video,
            codec_checkpoint="stub.pth",
            latent_mean_std=[0.0, 1.0],
            psd_loss_prob=0.5,
            psd_weight=0.5,
        )


def test_config_validates_with_psd_fields_at_zero() -> None:
    """Regression: a saved config carrying both knobs at 0.0 validates natively under extra=forbid
    (they are real fields now, not dropped via REMOVED_CONFIG_FIELDS)."""
    config = tiny_config().model_dump()
    config["psd_loss_prob"] = 0.0
    config["psd_weight"] = 0.0
    validated = LatentWorldModelConfig.model_validate(config)
    assert validated.psd_enabled is False


def test_psd_state_dict_delta_presence(monkeypatch) -> None:
    """The delta embedding is in the state dict iff PSD is enabled; a strict self-load round-trips
    in both cases."""
    off = build_world_model(monkeypatch)
    assert all(DELTA_KEY not in k for k in off.state_dict())
    off.load_state_dict(off.state_dict(), strict=True)

    on = build_world_model(monkeypatch, psd_loss_prob=1.0)
    assert any(DELTA_KEY in k for k in on.state_dict())
    on.load_state_dict(on.state_dict(), strict=True)


def test_psd_denoise_streaming_passes_no_delta_when_disabled(monkeypatch) -> None:
    """With PSD off, denoise_streaming (via inference) passes tau_delta=None to the transformer."""
    from mira.world_model.config import WorldModelInferenceConfig

    model = build_world_model(monkeypatch)
    seen: list[object] = []
    real_forward = model.world_model.forward

    def spy(*args, tau_delta=None, **kwargs):
        seen.append(tau_delta)
        return real_forward(*args, tau_delta=tau_delta, **kwargs)

    model.world_model.forward = spy  # type: ignore[method-assign]
    model.inference(
        make_batch(batch_size=1), config=WorldModelInferenceConfig(n_diffusion_steps=2), progress_bar=False
    )
    assert seen and all(td is None for td in seen)


def test_psd_through_multi_wrapper(monkeypatch) -> None:
    """PSD is inherited for free through the multiplayer wrapper's swm.diffusion_loss delegation."""
    model = build_multi_wrapper_td2(monkeypatch, n_players=2, psd_loss_prob=1.0)
    model.train()
    assert model.single_world_model.world_model.diffusion_time_embedding_delta is not None

    batch = make_batch(batch_size=4)  # 2 players x 2 groups
    outputs = model(batch)
    assert set(outputs) == {"loss_total", "loss_diffusion", "loss_psd"}
    assert torch.isfinite(outputs["loss_psd"]) and outputs["loss_psd"] > 0
