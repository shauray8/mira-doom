"""Exponential moving averages used during training.

:class:`ModelEMA` tracks an EMA of a model's floating-point parameters and can temporarily swap them
in for evaluation/checkpointing via :meth:`ModelEMA.average_parameters`. :class:`DistributedEMA`
tracks an EMA of a scalar (e.g. the latent mean/std) and averages it across ranks on ``compute``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


class ModelEMA:
    """Exponential moving average over model parameters.

    Only tracks floating-point parameters with ``requires_grad=True``. Supports bias correction
    (``unbias=True``) for early-training stability.

    Can be disabled with ``decay=0.0``, which is useful so you don't have to add extra ``if``s in
    the training loop when you want to turn off EMA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, unbias: bool = True):
        self.decay = decay
        self.unbias = unbias
        self.model = model
        self.state: dict[str, torch.Tensor] = {}
        self.count: float = 0.0
        # State tensors are cloned on each parameter's current device; construct this EMA *after*
        # moving the model to its training device, or step() will hit a device mismatch.
        for name, param in model.named_parameters():
            if param.requires_grad and param.is_floating_point():
                self.state[name] = param.data.detach().clone()

    @torch.no_grad()
    def step(self) -> None:
        if self.decay == 0.0:  # decay=0.0 disables EMA
            return

        if self.unbias:
            self.count = self.count * self.decay + 1
            w = 1.0 / self.count
        else:
            w = 1.0 - self.decay

        for name, param in self.model.named_parameters():
            if name not in self.state:
                continue
            self.state[name].mul_(1 - w).add_(param.data.detach(), alpha=w)

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily swap model parameters with EMA values."""
        if self.decay == 0.0:  # decay=0.0 disables EMA
            yield
            return

        original: dict[str, torch.Tensor] = {}
        try:
            for name, param in self.model.named_parameters():
                if name in self.state:
                    original[name] = param.data.detach().clone()
                    param.data.copy_(self.state[name])
            yield
        finally:
            for name, param in self.model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])

    def state_dict(self) -> dict[str, Any]:
        return {"state": self.state, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.count = state_dict["count"]
        for name, val in state_dict["state"].items():
            if name in self.state:
                self.state[name].copy_(val)


class DistributedEMA(nn.Module):
    """Exponential moving average of a scalar, averaged across distributed ranks on ``compute``."""

    _ema: Tensor

    def __init__(self, decay: float, initial_value: float = 0.0, device: str | int | torch.device = "cpu"):
        super().__init__()
        self.decay = decay
        # Defaults to CPU; pass the training device when ``update`` will be fed CUDA tensors, else the
        # `_ema * decay + batch_mean` below hits a device mismatch.
        self.register_buffer("_ema", torch.tensor(initial_value, dtype=torch.double, device=device))

    def update(self, values: Tensor) -> None:
        """Update EMA with the mean of ``values`` (local to this rank)."""
        batch_mean = values.detach().to(dtype=torch.double).mean()
        self._ema = self.decay * self._ema + (1 - self.decay) * batch_mean

    def all_reduce(self) -> None:
        """Average the EMA in-place across ranks."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._ema, op=dist.ReduceOp.AVG)

    def compute(self) -> float:
        """Average the EMA across ranks and return the value."""
        self.all_reduce()
        return self._ema.item()

    @property
    def value(self) -> float:
        return self._ema.item()
