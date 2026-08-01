"""Optional FSDP2 (`torch.distributed.fsdp.fully_shard`) wrapping for multi-GPU world-model training.

DDP (the trainer default) replicates the full model on every rank -- perfect for the 1B world model
on a single H100 or a small node. FSDP2 shards parameters, gradients, and optimizer state across
ranks, which is what you want once the model/optimizer state or the multiplayer tiled activations
stop fitting comfortably, or when scaling to many GPUs. This is the primitive worth taking from the
torchtitan playbook; the surrounding trainer stays as-is.

This is a thin, optional path (`run.parallelism=fsdp2`). It shards each transformer block plus the
top-level module with a bf16 mixed-precision policy. The frozen codec's parameters
(`requires_grad=False`) are left unsharded -- they are not trained and are cheap to replicate.

Validate a short run before committing to a long one: FSDP2 interacts with `torch.compile`,
activation checkpointing, and EMA, and the right sharding granularity is model-specific.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_fsdp2(model: nn.Module, *, param_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Shard `model` in place with FSDP2. Returns the same module (wrapped), for symmetry with DDP.

    Shards every `AdaSTBlock` (the trainable transformer blocks) individually so each block's
    parameters are gathered just-in-time for its forward/backward, then the root module. Requires an
    initialized process group (i.e. launched under `torchrun`).
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from mira.world_model.layers.transformer import AdaSTBlock

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError("apply_fsdp2 requires an initialized process group (launch with torchrun).")

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)

    n_sharded = 0
    for module in model.modules():
        if isinstance(module, AdaSTBlock):
            fully_shard(module, mp_policy=mp_policy)
            n_sharded += 1

    fully_shard(model, mp_policy=mp_policy)
    logger.info(
        "FSDP2: sharded %d transformer blocks + root module (param_dtype=%s).", n_sharded, param_dtype
    )
    return model
