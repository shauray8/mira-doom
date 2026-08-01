"""The codec training loss: L1 + LPIPS + DINO latent-consistency, with auto-weight balancing.

:class:`CodecLossWeights` configures which terms are active and their weights; :class:`CodecLoss`
computes them from a :class:`~mira.codec.codec_model.VideoCodecOutputs`. The release codec
uses three reconstruction terms — pixel L1 (``loss_mae``), an LPIPS perceptual loss
(``loss_lpips_perceptual``), and a DINO-feature latent-consistency loss
(``loss_dino_latent_consistency``) — with optional VQ-GAN-style ``auto_weight`` balancing of the
perceptual terms against the L1 anchor.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from mira.codec.codec_model import VideoCodecOutputs
from mira.codec.dino import DinoModel, DinoPerceptualLoss


def calculate_adaptive_weight(
    anchor_loss: Tensor,
    other_loss: Tensor,
    last_layer: Tensor,
    max_weight: float = 1e4,
) -> Tensor:
    """VQ-GAN-style adaptive weight (Esser et al., arXiv:2012.09841 §3.3).

    Balances ``other_loss`` against ``anchor_loss`` by the ratio of their gradient norms w.r.t. the
    decoder's last layer, so each perceptual term contributes a comparable gradient magnitude to the
    L1 anchor. The returned factor is detached (it scales the loss but carries no gradient itself).
    """
    anchor_grads = torch.autograd.grad(anchor_loss, last_layer, retain_graph=True)[0]
    other_grads = torch.autograd.grad(other_loss, last_layer, retain_graph=True)[0]
    return (anchor_grads.norm() / (other_grads.norm() + 1e-6)).clamp(0.0, max_weight).detach()


class CodecLossWeights(BaseModel):
    """Weights and per-term options for :class:`CodecLoss`.

    A term is active iff its weight is ``> 0``. ``auto_weight`` rescales each active perceptual term
    (LPIPS, DINO latent consistency) by :func:`calculate_adaptive_weight` against the L1 anchor.
    """

    model_config = ConfigDict(extra="forbid")

    loss_mae: float = Field(..., ge=0)

    loss_lpips_perceptual: float = Field(default=0.0, ge=0)
    lpips_perceptual_frame_frac: float = Field(default=0.25, gt=0, le=1.0)

    # RAEEncoder only: consistency between DINO features of the reconstruction and the encoder's.
    loss_dino_latent_consistency: float = Field(default=0.0, ge=0)
    dino_latent_consistency_frame_frac: float = Field(default=0.25, gt=0, le=1.0)

    compile_dino: bool = True

    auto_weight: bool = False
    max_auto_weight: float = Field(default=1e4, gt=0)


# Perceptual terms rescaled by auto_weight against the L1 (loss_mae) anchor.
_AUTO_WEIGHT_LOSSES: Tuple[str, ...] = (
    "loss_lpips_perceptual",
    "loss_dino_latent_consistency",
)


class CodecLoss(nn.Module):
    """Computes the codec's reconstruction loss terms and their weighted total.

    The DINO latent-consistency term needs the encoder's frozen DINO backbone, which is bound after
    construction via :meth:`bind_encoder_dino`; ``auto_weight`` needs the decoder's last-layer weight,
    bound via :meth:`bind_last_layer`.
    """

    def __init__(self, weights: CodecLossWeights):
        super().__init__()
        self.weights = weights

        self.lpips_perceptual_loss: nn.Module | None = None
        if self.weights.loss_lpips_perceptual > 0:
            import lpips  # noqa: PLC0415 -- optional dep, loaded only when in use

            self.lpips_perceptual_loss = lpips.LPIPS(net="vgg", verbose=False).eval()
            for p in self.lpips_perceptual_loss.parameters():
                p.requires_grad = False

        self.backward_metrics: dict[str, Tensor] = {}
        self._last_layer: Tensor | None = None
        # Bound post-init by the trainer (needs the encoder's DINO).
        self.dino_latent_consistency_loss: DinoPerceptualLoss | None = None

    def bind_last_layer(self, param: Tensor) -> None:
        self._last_layer = param

    def bind_encoder_dino(self, dino: DinoModel) -> None:
        """Build the latent-consistency loss sharing the encoder's already-loaded DINO backbone."""
        if self.weights.loss_dino_latent_consistency > 0:
            self.dino_latent_consistency_loss = DinoPerceptualLoss(
                dino_model=dino.dino_model_name,
                preloaded_dino_module=dino.dino_model,
                layer_indices=dino.layers if isinstance(dino.layers, tuple) else None,
                last_layer_only=isinstance(dino.layers, int),
                compile=False,
                normalize=True,
            ).to(next(dino.parameters()).device)

    def _hook_clone(self, tensor: Tensor, loss_name: str) -> Tensor:
        """Clone ``tensor`` and register a backward hook recording its per-term gradient norm."""
        if not tensor.requires_grad:
            return tensor
        clone = tensor.clone()
        clone.register_hook(
            lambda grad: self.backward_metrics.update({loss_name: grad.data.norm(p=2, dim=-1).mean()})
        )
        return clone

    def forward(self, model_outputs: VideoCodecOutputs, global_step: int = 0) -> dict[str, Tensor]:
        self.backward_metrics.clear()  # Clear metrics from previous backward hooks

        # Those are in range [-1, 1].
        predicted = model_outputs.output_video.float()
        predicted = self._hook_clone(predicted, "loss_total_video")
        target = model_outputs.input_video.float()

        loss: dict[str, Tensor] = {}

        if self.weights.loss_mae > 0:
            loss["loss_mae"] = F.l1_loss(self._hook_clone(predicted, "loss_mae"), target)

        t_total = predicted.shape[1]

        if self.lpips_perceptual_loss is not None:
            # LPIPS expects (N, 3, H, W) in [-1, 1] (the codec's native range).
            t_lpips_k = max(1, round(t_total * self.weights.lpips_perceptual_frame_frac))
            t_lpips = torch.randperm(t_total, device=predicted.device)[:t_lpips_k].sort().values
            predicted_lpips = self._hook_clone(predicted, "loss_lpips_perceptual")
            pred_2d = rearrange(predicted_lpips[:, t_lpips], "b t c h w -> (b t) c h w")
            tgt_2d = rearrange(target[:, t_lpips], "b t c h w -> (b t) c h w")
            loss["loss_lpips_perceptual"] = self.lpips_perceptual_loss(pred_2d, tgt_2d).mean()

        if self.dino_latent_consistency_loss is not None:
            assert model_outputs.dino_features is not None
            t_lc_k = max(1, round(t_total * self.weights.dino_latent_consistency_frame_frac))
            t_lc = torch.randperm(t_total, device=predicted.device)[:t_lc_k].sort().values
            pred_lc = (self._hook_clone(predicted, "loss_dino_latent_consistency") + 1) / 2
            real_lc = tuple(f[:, t_lc].detach() for f in model_outputs.dino_features)
            loss["loss_dino_latent_consistency"], _ = self.dino_latent_consistency_loss(
                pred_lc[:, t_lc], target_features=real_lc
            )

        if (
            self.weights.auto_weight
            and self._last_layer is not None
            and "loss_mae" in loss
            and loss["loss_mae"].requires_grad
            and torch.is_grad_enabled()
        ):
            for name in _AUTO_WEIGHT_LOSSES:
                if name not in loss or not loss[name].requires_grad:
                    continue
                factor = calculate_adaptive_weight(
                    loss["loss_mae"],
                    loss[name],
                    self._last_layer,
                    max_weight=self.weights.max_auto_weight,
                )
                loss[name] = factor * loss[name]
                loss[f"{name}_auto_w"] = factor

        weighted = [getattr(self.weights, k) * v for k, v in loss.items() if hasattr(self.weights, k)]
        # `weighted` is non-empty for any sensible config (loss_mae is required and > 0); guard the
        # degenerate all-zero-weights case so it returns a finite zero rather than raising on stack.
        loss["loss_total"] = torch.stack(weighted).sum() if weighted else predicted.new_zeros(())
        return loss
