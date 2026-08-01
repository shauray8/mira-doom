"""A scalar metric accumulated across steps and averaged across distributed ranks."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


class DistributedMetric(nn.Module):
    """Accumulates a scalar metric (sum / count) across steps and distributed ranks."""

    _sum: Tensor
    _n: Tensor

    def __init__(self, device: str | int | torch.device = "cpu"):
        super().__init__()
        self.register_buffer("_sum", torch.zeros((), dtype=torch.double, device=device))
        self.register_buffer("_n", torch.zeros((), dtype=torch.long, device=device))

    def reset(self) -> None:
        self._sum.zero_()
        self._n.zero_()

    def update(self, values: Tensor) -> None:
        """Accumulate all elements of ``values``."""
        self._sum += values.detach().sum().to(dtype=torch.double)
        self._n += values.numel()

    def all_reduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self._sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._n, op=dist.ReduceOp.SUM)

    def compute(self) -> Tensor:
        """Return mean across all accumulated values (and ranks), or 0 if nothing was accumulated."""
        self.all_reduce()
        if self._n == 0:  # no update() yet — avoid a 0/0 nan
            return torch.zeros_like(self._sum)
        return self._sum / self._n

    def compute_and_reset(self) -> Tensor:
        """Return mean across all accumulated values (and ranks), then reset."""
        output = self.compute()
        self.reset()
        return output
