"""DistributedMetric and PSNR accumulate correctly single-process."""

from __future__ import annotations

import math

import torch

from mira.training.metrics.distributed_metric import DistributedMetric
from mira.training.metrics.image_metrics import PSNRMetric


def test_distributed_metric_means_over_all_elements() -> None:
    metric = DistributedMetric()
    metric.update(torch.tensor([1.0, 3.0]))
    metric.update(torch.tensor([[2.0, 2.0]]))
    assert math.isclose(metric.compute().item(), 2.0)  # (1+3+2+2)/4
    metric.reset()
    assert metric._n.item() == 0


def test_psnr_is_finite_and_higher_for_closer_videos() -> None:
    torch.manual_seed(0)
    target = torch.rand(1, 2, 3, 8, 8)

    close = PSNRMetric()
    close.update((target + 0.01).clamp(0, 1), target)
    far = PSNRMetric()
    far.update((target + 0.3).clamp(0, 1), target)

    close_val = close.compute_and_reset().item()
    far_val = far.compute_and_reset().item()
    assert math.isfinite(close_val) and math.isfinite(far_val)
    assert close_val > far_val
