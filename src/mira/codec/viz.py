"""Codec-specific visualizations: reconstruction side-by-side and latent diagnostics.

These build W&B-loggable artifacts from a :class:`~mira.codec.codec_model.VideoCodecOutputs`:
a ground-truth/reconstruction side-by-side video, and interactive plots of the latent's correlation
structure and per-position standard deviation.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from einops import rearrange

from mira.codec.codec_model import VideoCodecOutputs


def visualize_side_by_side(model_outputs: VideoCodecOutputs) -> dict:
    """Concatenate the ground-truth and reconstructed videos side by side (uint8, [0, 255])."""
    pred_video = model_outputs.output_video * 0.5 + 0.5
    gt_video = model_outputs.input_video * 0.5 + 0.5
    viz_video = torch.cat([gt_video, pred_video], dim=-1)
    viz_video = (255 * viz_video).to(torch.uint8)
    return {"viz_video": viz_video, "pred_video": pred_video}


def visualize_latent_correlation(
    model_outputs: VideoCodecOutputs,
    correlation_dim: Literal["patch", "channels", "time"],
    caption: str | None = None,
) -> Any:
    """Pearson correlation matrix of the video latent as an interactive plotly heatmap.

    "channels" -> (C, C) correlation, samples = (B, T, H, W).
    "patch"    -> (H*W, H*W) correlation, samples = (B, T, C).
    "time"     -> per-patch (T, T) correlation (samples = (B, C)), tiled into an (H*T, W*T) grid laid
                  out by patch coordinates (i=row, j=col).
    """
    import numpy as np  # noqa: PLC0415 -- optional viz deps, used only here
    import plotly.graph_objects as go
    import wandb

    z = model_outputs.z.float()  # (B, T, C, H, W)
    customdata = None
    hovertemplate = "row=%{y}, col=%{x}<br>corr=%{z:.4f}<extra></extra>"
    if correlation_dim == "channels":
        z_flat = rearrange(z, "b t c h w -> (b t h w) c")
        z_flat = z_flat - z_flat.mean(dim=0, keepdim=True)
        z_flat = z_flat / z_flat.std(dim=0, keepdim=True).clamp_min(1e-8)
        corr = (z_flat.T @ z_flat) / (z_flat.shape[0] - 1)
        title, x_title, y_title = "Latent Correlation (Channels)", "Channel", "Channel"
    elif correlation_dim == "patch":
        z_flat = rearrange(z, "b t c h w -> (b t c) (h w)")
        z_flat = z_flat - z_flat.mean(dim=0, keepdim=True)
        z_flat = z_flat / z_flat.std(dim=0, keepdim=True).clamp_min(1e-8)
        corr = (z_flat.T @ z_flat) / (z_flat.shape[0] - 1)
        title = "Latent Correlation (Patch)"
        x_title, y_title = "Patch (h*W + w)", "Patch (h*W + w)"
    elif correlation_dim == "time":
        # Per-patch (T, T) correlation, with B*C samples, then tiled by (h, w).
        z_patch = rearrange(z, "b t c h w -> h w (b c) t")
        z_patch = z_patch - z_patch.mean(dim=2, keepdim=True)
        z_patch = z_patch / z_patch.std(dim=2, keepdim=True).clamp_min(1e-8)
        n_samples = z_patch.shape[2]
        corr_per_patch = torch.einsum("hwnt,hwns->hwts", z_patch, z_patch) / (n_samples - 1)
        corr = rearrange(corr_per_patch, "h w t1 t2 -> (h t1) (w t2)")
        title = "Latent Correlation (Time, per patch)"
        x_title, y_title = "w*T + t", "h*T + t"
        H, W, T = z_patch.shape[0], z_patch.shape[1], z_patch.shape[3]
        ys, xs = np.arange(H * T), np.arange(W * T)
        i_grid = np.broadcast_to((ys // T)[:, None], (H * T, W * T))
        t1_grid = np.broadcast_to((ys % T)[:, None], (H * T, W * T))
        j_grid = np.broadcast_to((xs // T)[None, :], (H * T, W * T))
        t2_grid = np.broadcast_to((xs % T)[None, :], (H * T, W * T))
        customdata = np.stack([i_grid, j_grid, t1_grid, t2_grid], axis=-1)
        hovertemplate = (
            "i=%{customdata[0]}, j=%{customdata[1]}<br>"
            "t1=%{customdata[2]}, t2=%{customdata[3]}<br>"
            "corr=%{z:.4f}<extra></extra>"
        )
    else:
        raise ValueError(f"Invalid correlation_dim {correlation_dim}, must be 'channels', 'patch', or 'time'")

    corr_np = corr.detach().cpu().float().numpy()
    heatmap_kwargs: dict[str, Any] = dict(
        z=corr_np, colorscale="Greens", zmin=0, zmax=1, hovertemplate=hovertemplate
    )
    if customdata is not None:
        heatmap_kwargs["customdata"] = customdata
    fig = go.Figure(data=[go.Heatmap(**heatmap_kwargs)])
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
    )
    return wandb.Plotly(fig)


def visualize_latent_std(
    model_outputs: VideoCodecOutputs,
    axis: Literal["patch", "channels"],
) -> Any:
    """Per-position std of the video latent as an interactive plotly chart.

    "patch" computes std over (B, T, C) for each spatial position and renders a heatmap with i (row)
    and j (col) patch coordinates on the axes. "channels" computes std over (B, T, H, W) for each
    channel as a bar chart.
    """
    import plotly.graph_objects as go  # noqa: PLC0415 -- optional viz deps, used only here
    import wandb

    z = model_outputs.z.float()  # (B, T, C, H, W)
    _, _, _, H, W = z.shape

    if axis == "patch":
        grid = rearrange(z, "b t c h w -> h w (b t c)").std(dim=2).cpu().numpy()
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=grid,
                    x=list(range(W)),
                    y=list(range(H)),
                    colorscale="Greens",
                    hovertemplate="i=%{y}, j=%{x}<br>std=%{z:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Latent Std per Patch",
            xaxis_title="j (col)",
            yaxis_title="i (row)",
            yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
        )
    elif axis == "channels":
        stds = rearrange(z, "b t c h w -> c (b t h w)").std(dim=1).cpu().tolist()
        table = wandb.Table(data=[[i, v] for i, v in enumerate(stds)], columns=["channel", "std"])
        return wandb.plot_table(
            "wandb/bar/v0",
            table,
            {"label": "channel", "value": "std"},
            {"title": "Latent Std per Channel"},
        )
    else:
        raise ValueError(f"Invalid axis {axis!r}, must be 'patch' or 'channels'")

    return wandb.Plotly(fig)
