"""ViT decoder shape tests. These run fully offline (no DINOv3 backbone needed)."""

from __future__ import annotations

import torch

from mira.codec import StridedConvBottleneckConfig, ViTDecoderConfig, ViTVideoDecoder
from mira.ml import ImageConfig

VIDEO = ImageConfig(height=288, width=512, channels=3, timesteps=40, fps=20)


def _decoder(*, stride: int = 2, patch_size: int = 16, patch_size_t: int = 2) -> ViTVideoDecoder:
    config = ViTDecoderConfig(
        video=VIDEO,
        latent_dim=32,
        bottleneck=StridedConvBottleneckConfig(stride=stride),
        vit_width=128,
        vit_depth=2,
        vit_num_heads=16,
        mlp_dim_multiplier=4,
        qk_norm="layernorm",
        patch_size=patch_size,
        patch_size_t=patch_size_t,
    )
    return ViTVideoDecoder(config).eval()


def test_decoder_round_trip_shape() -> None:
    """A (B, T_lat, C, H_lat, W_lat) latent decodes to (B, T_lat*patch_t, 3, H, W)."""
    decoder = _decoder()
    z = torch.randn(2, 20, 32, 9, 16)
    with torch.no_grad():
        out = decoder(z)
    assert out.shape == (2, 40, 3, 288, 512)


def test_decoder_temporal_expansion() -> None:
    """patch_size_t expands the latent time axis by exactly that factor."""
    for patch_size_t in (1, 2):
        decoder = _decoder(patch_size_t=patch_size_t)
        z = torch.randn(1, 10, 32, 9, 16)
        with torch.no_grad():
            out = decoder(z)
        assert out.shape[1] == 10 * patch_size_t


def test_decoder_spatial_expansion() -> None:
    """ConvTranspose stride and patch size together upsample the latent grid by stride * patch."""
    decoder = _decoder(stride=2, patch_size=16)  # 32x spatial expansion
    z = torch.randn(1, 5, 32, 9, 16)
    with torch.no_grad():
        out = decoder(z)
    assert out.shape[-2:] == (9 * 32, 16 * 32)


def test_decoder_output_is_tanh_bounded() -> None:
    decoder = _decoder()
    z = 10.0 * torch.randn(1, 4, 32, 9, 16)
    with torch.no_grad():
        out = decoder(z)
    assert out.min() >= -1.0 and out.max() <= 1.0
