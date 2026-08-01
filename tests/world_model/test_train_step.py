"""World-model training-loop building blocks: one train step + EMA, a validation-loss loop, and the
EMA swap/restore used around evaluation.

All offline via the stubbed codec (see conftest) — no checkpoint, no DINO weights.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from mira.training.ema import ModelEMA
from mira.training.metrics.distributed_metric import DistributedMetric

from .conftest import build_world_model, make_batch


def test_one_train_step_is_finite_and_steps_optimizer_and_ema(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    model.train()
    ema = ModelEMA(model, decay=0.99)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)

    # EMA only tracks trainable (world-model / action-encoder) params, never the frozen codec.
    assert ema.state and all(not name.startswith("codec.") for name in ema.state)
    name, param = next(iter(ema.state.items()))
    ema_before = ema.state[name].detach().clone()
    weight_before = dict(model.named_parameters())[name].detach().clone()

    optimizer.zero_grad(set_to_none=True)
    losses = model(make_batch(batch_size=2))
    assert set(losses) == {"loss_total", "loss_diffusion"}
    assert torch.isfinite(losses["loss_total"])

    losses["loss_total"].backward()
    optimizer.step()
    ema.step()

    # The optimizer moved the live weights and the EMA tracked toward the new value.
    assert ema.count == 1.0
    assert not torch.equal(dict(model.named_parameters())[name].detach(), weight_before)
    assert not torch.equal(ema.state[name], ema_before)


def test_validation_loss_loop_averages_finite_losses(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    model.eval()
    trackers: dict[str, DistributedMetric] = defaultdict(DistributedMetric)

    for _ in range(3):
        with torch.no_grad():
            for key, value in model(make_batch(batch_size=2)).items():
                trackers[key].update(value)

    metrics = {k: t.compute_and_reset().item() for k, t in trackers.items()}
    assert set(metrics) == {"loss_total", "loss_diffusion"}
    assert all(math.isfinite(v) for v in metrics.values())


def test_ema_average_parameters_swaps_then_restores_around_eval(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    ema = ModelEMA(model, decay=0.9)

    # Diverge every tracked live weight from the EMA snapshot taken at construction.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ema.state:
                p.add_(1.0)

    name, param = next((n, p) for n, p in model.named_parameters() if n in ema.state)
    live = param.detach().clone()

    with ema.average_parameters():
        # Inside the context the live weights are the EMA values (the un-incremented snapshot).
        assert not torch.equal(param.detach(), live)
        assert torch.equal(param.detach(), ema.state[name])
    # After the context the live weights are restored exactly.
    assert torch.equal(param.detach(), live)
