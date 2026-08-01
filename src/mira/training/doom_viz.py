"""Doom input HUD for training visualizations (GT vs. reconstruction / generation).

The Rocket League keyboard overlay in :mod:`mira.training.visualization` is layout-specific, so it
does not render Doom's action names or the continuous turn-delta. This module draws a compact
Doom-specific strip -- movement / fire / speed / weapon chips plus a centered turn-delta bar -- and
appends it under a video, so any GT/recon/generation clip logged during training shows the inputs
that conditioned it.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

# Column order MUST match configs/actions/doom.yaml (the multi-hot key_presses columns).
DOOM_KEYS: list[str] = [
    "forward", "backward", "strafe_right", "strafe_left",
    "weapon1", "weapon2", "weapon3", "weapon4", "weapon5", "weapon6", "weapon7",
    "attack", "speed",
]  # fmt: skip
_CHIPS = [
    ("FWD", "forward"), ("BCK", "backward"), ("◄STRAFE", "strafe_left"),
    ("STRAFE►", "strafe_right"), ("FIRE", "attack"), ("SPD", "speed"),
]  # fmt: skip
TURN_MAX = 12.5  # observed |turn-delta| range in the dataset


def _hud_strip(width: int, hud_h: int, keys_on: set[str], turn: float, weapon: int | None) -> np.ndarray:
    """Render one HUD strip (H=hud_h, W=width, RGB uint8) for a frame's Doom inputs."""
    img = Image.new("RGB", (width, hud_h), (14, 14, 20))
    d = ImageDraw.Draw(img)
    cw, ch, gap = 78, 26, 8
    x, y = 8, 6
    for label, key in _CHIPS:
        on = key in keys_on
        d.rectangle([x, y, x + cw, y + ch], fill=(90, 190, 255) if on else (40, 40, 48), outline=(70, 70, 80))
        d.text((x + 6, y + 7), label, fill=(10, 10, 15) if on else (150, 150, 160))
        x += cw + gap
    wlabel = f"W{weapon}" if weapon is not None else "W-"
    d.rectangle([x, y, x + 46, y + ch], fill=(60, 60, 30) if weapon else (40, 40, 48), outline=(70, 70, 80))
    d.text((x + 8, y + 7), wlabel, fill=(255, 230, 120) if weapon else (150, 150, 160))
    # Turn bar: centered zero, fills toward the turn direction.
    bx0, bx1, by, bh = 8, width - 8, y + ch + 12, 14
    cx = (bx0 + bx1) // 2
    d.rectangle([bx0, by, bx1, by + bh], fill=(30, 30, 38), outline=(70, 70, 80))
    d.line([cx, by, cx, by + bh], fill=(120, 120, 130))
    frac = max(-1.0, min(1.0, turn / TURN_MAX))
    ex = int(cx + frac * (bx1 - cx - 2))
    lo, hi = (cx, ex) if ex >= cx else (ex, cx)
    d.rectangle([lo, by, hi, by + bh], fill=(255, 140, 80))
    d.text((bx0, by + bh + 2), f"TURN {turn:+.1f}", fill=(210, 210, 220))
    return np.asarray(img, dtype=np.uint8)


def append_input_hud(
    video: Tensor,
    key_presses: Tensor,
    mouse: Tensor,
    top_labels: tuple[str, str] | None = None,
) -> Tensor:
    """Append a per-frame Doom input HUD strip under a ``(T, C, H, W)`` uint8 video.

    Args:
        video: ``(T, C, H, W)`` uint8 frames (e.g. a GT|RECON side-by-side, or a rollout).
        key_presses: ``(T', K)`` multi-hot Doom keys (columns aligned to :data:`DOOM_KEYS`).
        mouse: ``(T', 2)`` mouse/analog deltas; channel 0 is the turn-delta.
        top_labels: optional ``(left, right)`` labels drawn on the top corners (for side-by-sides).

    Returns:
        ``(T, C, H + hud_h, W)`` uint8 video with the HUD strip stacked below.
    """
    t, c, h, w = video.shape
    n = min(t, key_presses.shape[0])
    frames = video.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    hud_h = max(70, h // 3)
    out = []
    for i in range(t):
        j = min(i, n - 1)
        on = {DOOM_KEYS[k] for k in range(len(DOOM_KEYS)) if key_presses[j, k] > 0}
        weapon = next((wi + 1 for wi in range(7) if key_presses[j, 4 + wi] > 0), None)
        hud = _hud_strip(w, hud_h, on, float(mouse[j, 0]), weapon)
        img = Image.fromarray(np.concatenate([frames[i], hud], axis=0))
        if top_labels is not None:
            dd = ImageDraw.Draw(img)
            dd.text((6, 4), top_labels[0], fill=(80, 255, 120))
            dd.text((w // 2 + 6, 4), top_labels[1], fill=(255, 200, 80))
        out.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1))
    return torch.stack(out, dim=0)
