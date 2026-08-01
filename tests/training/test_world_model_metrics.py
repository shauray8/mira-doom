"""Offline-safe world-model metric math: SlicedFrechetMetric, the config defaults, and the curve
plots. The full DINO/Inception-driven metrics smoke lives in tests/world_model/.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from mira.training.metrics.frechet import SlicedFrechetMetric
from mira.training.metrics.image_metrics import OnlineGaussian
from mira.training.metrics.world_model_metrics import (
    WorldModelMetricsConfig,
    build_frechet_curve_plots,
)


def test_sliced_frechet_is_finite_and_returns_one_value_per_slice() -> None:
    torch.manual_seed(0)
    dim, num_slices = 16, 3
    metric = SlicedFrechetMetric(dim, num_slices)
    for s in range(num_slices):
        target = torch.randn(64, dim)
        pred = torch.randn(64, dim) + 0.7  # shifted mean -> non-trivial distance
        metric.update(s, target, pred)

    aggregate, curve = metric.compute()
    assert math.isfinite(aggregate) and aggregate > 0
    assert len(curve) == num_slices
    assert all(math.isfinite(v) and v >= 0 for v in curve)


def test_sliced_frechet_near_zero_for_identical_distributions() -> None:
    torch.manual_seed(1)
    metric = SlicedFrechetMetric(dim=16, num_slices=2)
    for s in range(2):
        x = torch.randn(128, 16)
        metric.update(s, x, x)

    aggregate, curve = metric.compute()
    assert aggregate < 1e-3
    assert all(v < 1e-3 for v in curve)


def test_sliced_frechet_reset_clears_state() -> None:
    metric = SlicedFrechetMetric(dim=4, num_slices=1)
    metric.update(0, torch.randn(8, 4), torch.randn(8, 4) + 1.0)
    metric.reset()
    for g in (*metric.target, *metric.pred):
        assert int(cast(OnlineGaussian, g).n.item()) == 0


def test_world_model_metrics_config_defaults() -> None:
    config = WorldModelMetricsConfig(num_unrolled_frames=120, drift_metric_frames=20)
    assert config.fdd_slice_frames == 20
    assert config.eval_temporal_downsampling is None
    # The inference rollout config defaults are carried through.
    assert config.inference.n_diffusion_steps == 10
    assert config.inference.schedule_type == "linear_quadratic"


def test_build_frechet_curve_plots_one_plot_per_curve() -> None:
    pytest.importorskip("plotly")
    pytest.importorskip("wandb")

    curves = {"fdd_at": [3.0, 2.0, 1.0], "fid_at": [0.5, 0.4, 0.3]}
    plots = build_frechet_curve_plots(curves, slice_frames=20)
    assert set(plots) == {"viz/fdd_at", "viz/fid_at"}
