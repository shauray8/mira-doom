"""Pluggable attention backend: FlashAttention-3 (Hopper) -> FlashAttention-2 -> PyTorch SDPA.

The world-model transformer's self-attention (``mira.ml.attention.SelfAttention``) runs the same
scaled-dot-product op every layer. On H100 the FlashAttention-3 kernels (fused fwd+bwd, warp-
specialized, FP8-capable) are materially faster and lower-memory than the default SDPA math/cuDNN
path -- and crucially FA's autograd function fuses the forward AND backward, so selecting it here
speeds up training in both directions with no separate backward wiring.

Selection is controlled by ``MIRA_ATTN_BACKEND`` = ``auto`` (default) | ``fa3`` | ``fa2`` | ``sdpa``.
``auto`` prefers FA3, then FA2, then SDPA. A hard name (``fa3``/``fa2``) raises if that package is
missing so a misconfigured "fast" run fails loudly instead of silently falling back.

Scope: this is used only on the *training / full-sequence* path (no KV-cache, bf16/fp16 q,k,v).
The incremental-decoding path (KV-cache, q_len=1) and any float32 tensors stay on SDPA, which is
the correctness-critical inference path and not throughput-bound.

Tensor layout: this module takes and returns ``(B, S, H, D)`` (heads-last), which is FlashAttention's
native layout. GQA is expressed by passing ``k``/``v`` with fewer heads than ``q`` (FA supports it
natively; SDPA gets ``enable_gqa``). RoPE and QK-norm are applied by the caller beforehand.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)

_Backend = Literal["fa3", "fa2", "sdpa"]
_RESOLVED: tuple[_Backend, Callable | None] | None = None


def _resolve_backend() -> tuple[_Backend, Callable | None]:
    """Pick and cache the attention backend from ``MIRA_ATTN_BACKEND`` (see module docstring)."""
    pref = os.environ.get("MIRA_ATTN_BACKEND", "auto").lower()
    if pref not in ("auto", "fa3", "fa2", "sdpa"):
        raise ValueError(f"MIRA_ATTN_BACKEND must be auto|fa3|fa2|sdpa, got {pref!r}")

    if pref in ("auto", "fa3"):
        try:
            # FA3 Hopper build installs as `flash_attn_interface`.
            from flash_attn_interface import flash_attn_func as fa3  # type: ignore

            logger.info("Attention backend: FlashAttention-3 (flash_attn_interface).")
            return "fa3", fa3
        except Exception as exc:  # noqa: BLE001 -- any import/ABI failure -> try the next backend
            if pref == "fa3":
                raise RuntimeError(
                    "MIRA_ATTN_BACKEND=fa3 but flash_attn_interface (FA3 Hopper build) is not "
                    "importable. Build it from the flash-attention repo's `hopper/` dir on an H100."
                ) from exc

    if pref in ("auto", "fa2"):
        try:
            from flash_attn import flash_attn_func as fa2  # type: ignore

            logger.info("Attention backend: FlashAttention-2 (flash_attn).")
            return "fa2", fa2
        except Exception as exc:  # noqa: BLE001
            if pref == "fa2":
                raise RuntimeError("MIRA_ATTN_BACKEND=fa2 but `flash_attn` is not importable.") from exc

    logger.info("Attention backend: PyTorch scaled_dot_product_attention (SDPA).")
    return "sdpa", None


def get_backend() -> tuple[_Backend, Callable | None]:
    global _RESOLVED
    if _RESOLVED is None:
        _RESOLVED = _resolve_backend()
    return _RESOLVED


def _force_sdpa() -> None:
    """Pin the backend to SDPA for the rest of the process (after a runtime flash failure)."""
    global _RESOLVED
    _RESOLVED = ("sdpa", None)


def _sdpa(q: Tensor, k: Tensor, v: Tensor, causal: bool, context: int | None, gqa: bool) -> Tensor:
    """Reference path: (B,S,H,D) -> SDPA in (B,H,S,D) -> (B,S,H,D). Mirrors the original code."""
    from mira.ml.attention import local_causal_mask

    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, S, D)
    attn_mask = None
    # A single query with an unbounded causal window attends to every cached key, so the mask
    # local_causal_mask() would build is entirely True -- a no-op. Materialising it is not free:
    # any explicit attn_mask disqualifies SDPA's flash and cuDNN kernels and forces the much slower
    # cutlass mem-efficient path. This is exactly the shape incremental decoding uses every step, so
    # skipping the mask is both bit-exact and squarely on the hot path. A bounded `context` window
    # is a real mask, so it still gets built.
    vacuous_causal = causal and context is None and qt.shape[-2] == 1
    if causal and not vacuous_causal:
        attn_mask = local_causal_mask(
            q_len=qt.shape[-2], k_len=kt.shape[-2], context=context, device=q.device
        )
    y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask, enable_gqa=gqa)
    return y.transpose(1, 2)  # (B, S, H, D)


def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
    context: int | None = None,
    gqa: bool = False,
    prefer_flash: bool = True,
) -> Tensor:
    """Scaled-dot-product attention over ``(B, S, H, D)`` tensors, returning ``(B, S, H, D)``.

    Uses FlashAttention (FA3/FA2) when available, ``prefer_flash`` is set, and the inputs are
    half-precision; otherwise SDPA. ``context`` (a local causal window) maps to FlashAttention's
    ``window_size=(context-1, 0)``; ``None`` means an unbounded (fully) causal or bidirectional op.

    A runtime FlashAttention failure logs once and permanently falls back to SDPA, so a kernel/ABI
    problem degrades to correct-but-slower rather than crashing a long run.
    """
    if q.shape[1] == 1 and k.shape[1] == 1:
        n_head, n_kv_head = q.shape[2], v.shape[2]
        if n_head != n_kv_head:  # GQA: broadcast each kv head over its query-head group
            v = v.repeat_interleave(n_head // n_kv_head, dim=2)
        return v.expand(-1, q.shape[1], -1, -1) if v.shape[1] != q.shape[1] else v

    backend, fn = get_backend()
    half = q.dtype in (torch.float16, torch.bfloat16)

    if backend == "sdpa" or fn is None or not prefer_flash or not half or not q.is_cuda:
        return _sdpa(q, k, v, causal, context, gqa)

    window = (context - 1, 0) if context is not None else (-1, -1)
    try:
        out = fn(q, k, v, causal=causal, window_size=window)
        # FA3's interface returns (out, softmax_lse); FA2 returns just out.
        return out[0] if isinstance(out, tuple) else out
    except Exception as exc:  # noqa: BLE001 -- fall back permanently, keep the run alive
        logger.warning(
            "FlashAttention (%s) failed at runtime (%s: %s); falling back to SDPA for the rest of the run.",
            backend,
            type(exc).__name__,
            exc,
        )
        _force_sdpa()
        return _sdpa(q, k, v, causal, context, gqa)
