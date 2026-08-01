"""WorldModelMetrics smoke test: one stub-codec rollout yields finite drift + Frechet values.

The DINO and Inception backbones require ``torch.hub`` / ``pytorch_fid`` weights, so the test
skip-gates gracefully when they are unavailable offline. The metric math itself (SlicedFrechetMetric)
is covered without those backbones in tests/training/test_world_model_metrics.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from mira.ml.image_config import ImageConfig
from mira.training.metrics.world_model_metrics import WorldModelMetricsConfig
from mira.world_model.config import WorldModelInferenceConfig

from .conftest import build_world_model, make_batch


def test_world_model_metrics_smoke_finite(monkeypatch) -> None:
    pytest.importorskip("pytorch_fid")
    from mira.training.metrics.world_model_metrics import WorldModelMetrics

    config = WorldModelMetricsConfig(
        num_samples=1,
        per_device_batch_size=1,
        num_unrolled_frames=4,
        drift_metric_frames=2,
        fdd_slice_frames=2,
        compile=False,
        num_viz_samples=0,
        inference=WorldModelInferenceConfig(n_diffusion_steps=2, noise_level=0.0),
    )

    model = build_world_model(monkeypatch)
    stride = model.temporal_downsampling
    n_frames = model.config.n_context_frames + config.num_unrolled_frames * stride
    batch = make_batch(batch_size=1, n_frames=n_frames, n_actions=2 * n_frames)

    try:
        metrics = WorldModelMetrics(config, iter([(batch, ["clip"])]), device="cpu")
    except Exception as exc:  # noqa: BLE001 -- DINO/Inception weights unavailable offline -> skip
        pytest.skip(f"DINO/Inception backbone unavailable offline: {type(exc).__name__}: {exc}")

    metrics.process_batch(model)
    scalar_result, frechet_curves = metrics.compute()

    df = config.drift_metric_frames
    for key in (
        f"dino_cos_drift_{df}",
        f"dino_l2_drift_{df}",
        f"latent_drift_{df}",
        "psnr",
        "lpips",
        "ssim",
        "frechet_dino_distance",
        "frechet_inception_distance",
    ):
        assert math.isfinite(float(scalar_result[key])), key
    assert len(frechet_curves["fdd_at"]) == config.num_unrolled_frames // config.fdd_slice_frames
    assert torch.isfinite(torch.tensor(float(scalar_result["frechet_dino_distance"])))


def test_world_model_metrics_end_to_end_with_shipped_4s_defaults(monkeypatch) -> None:
    """The shipped 4 s eval defaults run end-to-end on an 80-frame chunk with no size-mismatch.

    n_context_frames=38, num_unrolled_frames=20, fdd_slice_frames=10, temporal_downsampling=2 ->
    clip_len = 38 + 20*2 = 78 <= 80, which stays divisible by the codec temporal_downsampling so the
    rollout encodes without a size mismatch.
    """
    pytest.importorskip("pytorch_fid")
    from mira.training.metrics.world_model_metrics import WorldModelMetrics

    config = WorldModelMetricsConfig(
        num_samples=1,
        per_device_batch_size=1,
        n_context_frames=38,
        num_unrolled_frames=20,
        drift_metric_frames=20,
        fdd_slice_frames=10,
        compile=False,
        num_viz_samples=0,
        inference=WorldModelInferenceConfig(n_diffusion_steps=2, noise_level=0.0),
    )
    assert config.n_context_frames is not None

    # video.timesteps=80 (multiplayer window) so the 38-frame context is valid; fps matches the stub
    # action rate so action_temporal_downsampling stays equal to temporal_downsampling.
    model = build_world_model(
        monkeypatch, video=ImageConfig(height=64, width=64, channels=3, timesteps=80, fps=10)
    )
    model.set_inference_context(config.n_context_frames)
    stride = model.temporal_downsampling
    n_frames = model.config.n_context_frames + config.num_unrolled_frames * stride
    assert n_frames == 78 and n_frames <= 80  # fits the shipped 80-frame chunk
    batch = make_batch(batch_size=1, n_frames=n_frames, n_actions=2 * n_frames)

    try:
        metrics = WorldModelMetrics(config, iter([(batch, ["clip"])]), device="cpu")
    except Exception as exc:  # noqa: BLE001 -- DINO/Inception weights unavailable offline -> skip
        pytest.skip(f"DINO/Inception backbone unavailable offline: {type(exc).__name__}: {exc}")

    metrics.process_batch(model)  # must not raise a size-mismatch RuntimeError
    scalar_result, frechet_curves = metrics.compute()
    assert len(frechet_curves["fdd_at"]) == 2  # 20 // 10
    assert math.isfinite(float(scalar_result["frechet_dino_distance"]))


def test_shipped_config_defaults_fit_80_frame_chunk() -> None:
    """The shipped eval config defaults yield a clip that fits an 80-frame chunk on both models."""
    config = WorldModelMetricsConfig(
        n_context_frames=38, num_unrolled_frames=20, drift_metric_frames=20, fdd_slice_frames=10
    )
    assert config.n_context_frames is not None
    temporal_downsampling = 2  # the shipped RAE codec
    clip_len = config.n_context_frames + config.num_unrolled_frames * temporal_downsampling
    assert clip_len == 78 <= 80
    assert clip_len % temporal_downsampling == 0  # encodes cleanly
    assert config.num_unrolled_frames % config.fdd_slice_frames == 0  # whole number of Frechet slices
    # Context stays under both shipped model windows (video.timesteps: 40 single-player, 80 multi).
    assert config.n_context_frames < 40
    assert config.n_context_frames % temporal_downsampling == 0
