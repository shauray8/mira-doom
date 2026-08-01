"""LatentWorldModel.visualize() renders a HUD-annotated, ground-truth-stacked rollout (video-only).

Uses the stubbed codec (see conftest) so it runs offline without any checkpoint or DINO weights.
"""

from __future__ import annotations

import torch

from mira.world_model.config import WorldModelInferenceConfig

from .conftest import build_world_model, make_batch


def test_visualize_returns_hud_video_of_expected_shape(monkeypatch) -> None:
    model = build_world_model(monkeypatch)
    # The HUD draws one keyboard frame per video frame (actions_per_video_frame == 1 in this seam),
    # so the rollout batch carries one action per video frame.
    assert model.actions_per_video_frame == 1
    batch = make_batch(batch_size=2, n_actions=model.config.video.timesteps)
    outputs = model.inference(
        batch, config=WorldModelInferenceConfig(n_diffusion_steps=2), progress_bar=False
    )

    viz = model.visualize(outputs)

    assert set(viz) == {"viz_video", "pred_video"}
    pred_video, viz_video = viz["pred_video"], viz["viz_video"]
    assert pred_video.dtype == torch.uint8 and viz_video.dtype == torch.uint8

    b, t, c, h, w = pred_video.shape
    assert (b, c) == (2, 3)
    assert t == outputs.output_video.shape[1]
    # viz_video stacks the HUD-annotated prediction over the ground truth vertically (double height).
    assert viz_video.shape == (b, t, c, 2 * h, w)


def test_visualize_overlay_modifies_predicted_frames(monkeypatch) -> None:
    """The keyboard HUD + prediction border must actually change the predicted frames' pixels."""
    model = build_world_model(monkeypatch)
    batch = make_batch(batch_size=1, n_actions=model.config.video.timesteps)
    outputs = model.inference(
        batch, config=WorldModelInferenceConfig(n_diffusion_steps=2), progress_bar=False
    )
    raw = outputs.output_video[:, : model.n_context_frames + 1]
    pred_video = model.visualize(outputs)["pred_video"]

    from mira.training.visualization import video_to_uint8

    # The HUD draws over a top-right corner, so the overlaid clip differs from the raw decoded video.
    assert not torch.equal(pred_video[:, : raw.shape[1]], video_to_uint8(raw.cpu()))
