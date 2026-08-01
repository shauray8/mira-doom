"""Smoke tests for `mira.ml`: the public building blocks import and construct."""

from __future__ import annotations

import torch

from mira.ml import SelfAttention, SelfAttentionConfig, init_weights


def test_self_attention_forward_shape() -> None:
    cfg = SelfAttentionConfig(embed_dim=32, num_heads=4, num_kv_heads=2)
    assert cfg.head_dim == 8

    attn = SelfAttention(cfg)
    attn.apply(init_weights)

    x = torch.randn(2, 5, 32)
    y = attn(x)
    assert y.shape == x.shape


def test_self_attention_kv_heads_default() -> None:
    cfg = SelfAttentionConfig(embed_dim=32, num_heads=4, num_kv_heads=None)
    assert cfg.num_kv_heads == 4
