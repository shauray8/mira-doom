"""WarmupConstantCosineDecayLR: warmup ramp, constant plateau, and cosine decay."""

from __future__ import annotations

import math

import torch
from torch import nn

from mira.training.lr_schedule import WarmupConstantCosineDecayLR

BASE_LR = 1.0
MIN_LR = 0.1


def _scheduler(decay_steps: int = 10) -> WarmupConstantCosineDecayLR:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR)
    return WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=10, constant_steps=5, decay_steps=decay_steps, min_lr=MIN_LR
    )


def _lr_at(scheduler: WarmupConstantCosineDecayLR, step: int) -> float:
    # Drive the scheduler to `step` (it starts at last_epoch=0 after construction).
    while scheduler.last_epoch < step:
        scheduler.optimizer.step()
        scheduler.step()
    return float(scheduler.get_last_lr()[0])


def test_linear_warmup() -> None:
    sched = _scheduler()
    assert _lr_at(sched, 0) == 0.0
    assert math.isclose(_lr_at(sched, 5), BASE_LR * 5 / 10)


def test_constant_plateau() -> None:
    sched = _scheduler()
    assert math.isclose(_lr_at(sched, 10), BASE_LR)  # warmup just finished
    assert math.isclose(_lr_at(sched, 14), BASE_LR)  # still in the constant window


def test_cosine_decay() -> None:
    sched = _scheduler()
    # Decay starts at step 15 (warmup 10 + constant 5); half-way (step 20) is the cosine midpoint.
    assert math.isclose(_lr_at(sched, 15), BASE_LR)
    assert math.isclose(_lr_at(sched, 20), MIN_LR + (BASE_LR - MIN_LR) * 0.5, rel_tol=1e-6)
    # After the decay window the LR holds at min_lr.
    assert math.isclose(_lr_at(sched, 25), MIN_LR)
    assert math.isclose(_lr_at(sched, 40), MIN_LR)


def test_no_decay_phase_holds_base_lr() -> None:
    sched = _scheduler(decay_steps=0)
    assert math.isclose(_lr_at(sched, 100), BASE_LR)
