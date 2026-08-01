"""Sliced Frechet distance: one Gaussian per unrolled-prediction window plus an aggregate.

The world-model metrics unroll a prediction over many frames and report the Frechet distance between
the predicted and target feature distributions both per window (a curve over "frames unrolled") and
pooled over all windows. The per-window Gaussians reuse :class:`OnlineGaussian` from
``image_metrics`` so the sufficient-statistics / distributed-reduce logic lives in one place.
"""

from __future__ import annotations

from typing import cast

import torch

from .image_metrics import OnlineGaussian, frechet_distance


def _pool(gaussians: list[OnlineGaussian]) -> OnlineGaussian:
    """Sum per-slice sufficient statistics into one Gaussian over all slices' frames."""
    pooled = OnlineGaussian(dim=gaussians[0].sum_x.shape[0]).to(gaussians[0].sum_x.device)
    for g in gaussians:
        pooled.sum_x += g.sum_x
        pooled.sum_xxT += g.sum_xxT
        pooled.n += g.n
    return pooled


def _frechet(target: OnlineGaussian, pred: OnlineGaussian) -> float:
    t_mean, t_cov = target.compute()
    p_mean, p_cov = pred.compute()
    return float(frechet_distance(t_mean, t_cov, p_mean, p_cov).item())


class SlicedFrechetMetric(torch.nn.Module):
    """Frechet distance between a target and a predicted feature distribution, split into per-slice
    windows of the unrolled prediction plus the aggregate over all slices.

    Holds one :class:`OnlineGaussian` per slice for both target and prediction. :meth:`update`
    accumulates one slice's features; :meth:`compute` returns ``(aggregate, per_slice_curve)``.
    """

    def __init__(self, dim: int, num_slices: int):
        super().__init__()
        # Stored in ModuleLists so .to(device)/state_dict recurse into the per-slice Gaussians.
        self.target = torch.nn.ModuleList(OnlineGaussian(dim) for _ in range(num_slices))
        self.pred = torch.nn.ModuleList(OnlineGaussian(dim) for _ in range(num_slices))

    @staticmethod
    def _gaussians(module_list: torch.nn.ModuleList) -> list[OnlineGaussian]:
        # nn.ModuleList types its members as Module; the metric only ever stores OnlineGaussians.
        return [cast(OnlineGaussian, g) for g in module_list]

    def reset(self) -> None:
        for g in self._gaussians(self.target) + self._gaussians(self.pred):
            g.reset()

    def update(self, slice_idx: int, target_feats: torch.Tensor, pred_feats: torch.Tensor) -> None:
        cast(OnlineGaussian, self.target[slice_idx]).update(target_feats)
        cast(OnlineGaussian, self.pred[slice_idx]).update(pred_feats)

    def compute(self) -> tuple[float, list[float]]:
        """Return ``(aggregate over all slices, per-slice curve)``. Must run on all ranks.

        Pools the per-slice stats BEFORE the per-slice ``compute()`` calls: ``OnlineGaussian.compute()``
        all_reduces its buffers in place, so pooling afterwards would double-count across ranks.
        """
        targets, preds = self._gaussians(self.target), self._gaussians(self.pred)
        aggregate = _frechet(_pool(targets), _pool(preds))
        per_slice = [_frechet(t, p) for t, p in zip(targets, preds)]
        return aggregate, per_slice
