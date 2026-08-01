"""Learning-rate schedule used by the trainers."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupConstantCosineDecayLR(LRScheduler):
    """Linear warmup, then a constant plateau, then optional cosine decay to ``min_lr``.

    Set ``decay_steps=0`` to disable the decay phase (the LR stays at ``base_lr`` after warmup +
    constant). After the decay phase completes the LR stays at ``min_lr``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        constant_steps: int,
        decay_steps: int,  # Set to 0 to disable decay phase
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps

        self.min_lr = min_lr

        assert self.warmup_steps >= 0
        assert self.constant_steps >= 0
        assert self.decay_steps >= 0
        assert self.min_lr >= 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]  # noqa: ANN201 -- base stub annotates -> float
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        elif step >= self.warmup_steps and step < self.warmup_steps + self.constant_steps:
            # Constant phase
            return self.base_lrs
        elif (
            step >= self.warmup_steps + self.constant_steps
            and step < self.warmup_steps + self.constant_steps + self.decay_steps
        ):
            # Cosine decay
            decay_step = step - self.warmup_steps - self.constant_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / self.decay_steps))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
        elif self.decay_steps > 0:
            # After decay
            return [self.min_lr for _ in self.base_lrs]
        else:
            # No decay phase
            return self.base_lrs
