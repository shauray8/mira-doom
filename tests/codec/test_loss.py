"""CodecLoss: a training step produces finite gradients for each active term.

Builds the full codec (frozen DINOv3 backbone), so it skips gracefully when the backbone can't be
loaded offline (see conftest.build_codec_or_skip).
"""

from __future__ import annotations

import torch

from mira.codec import CodecLoss, CodecLossWeights
from tests.codec.conftest import build_codec_or_skip, tiny_batch, tiny_raev2_config


def _build_loss(codec, *, auto_weight: bool) -> CodecLoss:
    weights = CodecLossWeights(
        loss_mae=1.0,
        loss_lpips_perceptual=1.0,
        loss_dino_latent_consistency=1.0,
        compile_dino=False,
        auto_weight=auto_weight,
    )
    loss = CodecLoss(weights)
    if auto_weight:
        loss.bind_last_layer(codec.decoder.last_layer_weight)
    loss.bind_encoder_dino(codec.encoder.rae_dino)
    return loss


def test_train_step_produces_finite_grads() -> None:
    codec = build_codec_or_skip(tiny_raev2_config()).train()
    loss = _build_loss(codec, auto_weight=True)

    batch = tiny_batch()
    outputs = codec(batch)
    losses = loss(outputs, global_step=0)

    # Each active term is present, finite, and contributes to the total.
    for name in ("loss_mae", "loss_lpips_perceptual", "loss_dino_latent_consistency"):
        assert name in losses, name
        assert torch.isfinite(losses[name]), name
    assert torch.isfinite(losses["loss_total"])
    # auto_weight records a (detached, finite, non-negative) factor per perceptual term.
    for name in ("loss_lpips_perceptual", "loss_dino_latent_consistency"):
        factor = losses[f"{name}_auto_w"]
        assert torch.isfinite(factor) and factor >= 0

    losses["loss_total"].backward()

    # The decoder (the only trainable part — the DINO backbone is frozen) gets finite, non-zero grads.
    grads = [p.grad for p in codec.decoder.parameters() if p.grad is not None]
    assert grads, "decoder received no gradients"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)

    # The frozen backbone stays frozen.
    assert all(not p.requires_grad for p in codec.encoder.rae_dino.parameters())


def test_each_term_individually_has_finite_grads() -> None:
    """Isolate each reconstruction term and confirm it alone yields finite decoder gradients."""
    codec = build_codec_or_skip(tiny_raev2_config()).train()

    for term in ("loss_mae", "loss_lpips_perceptual", "loss_dino_latent_consistency"):
        kwargs = {
            "loss_mae": 0.0,
            "loss_lpips_perceptual": 0.0,
            "loss_dino_latent_consistency": 0.0,
            "compile_dino": False,
            "auto_weight": False,
        }
        kwargs[term] = 1.0
        loss = CodecLoss(CodecLossWeights(**kwargs))
        loss.bind_encoder_dino(codec.encoder.rae_dino)

        codec.zero_grad(set_to_none=True)
        losses = loss(codec(tiny_batch()), global_step=0)
        assert term in losses and torch.isfinite(losses[term]), term
        losses["loss_total"].backward()

        grads = [p.grad for p in codec.decoder.parameters() if p.grad is not None]
        assert grads, f"{term}: decoder received no gradients"
        assert all(torch.isfinite(g).all() for g in grads), term
