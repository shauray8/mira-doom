
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from mira.codec.codec_model import VideoCodec
from mira.codec.viz import visualize_side_by_side
from mira.data.training_loader import create_loader
from mira.training.visualization import VideoForWandb, videos_for_wandb

DOOM_KEYS = [
    "forward", "backward", "strafe_right", "strafe_left",
    "weapon1", "weapon2", "weapon3", "weapon4", "weapon5", "weapon6", "weapon7",
    "attack", "speed",
]  # fmt: skip
# HUD chips: (label, key-name, lit-colour). Movement + fire + speed.
CHIPS = [
    ("FWD", "forward"), ("BCK", "backward"), ("◄STRAFE", "strafe_left"),
    ("STRAFE►", "strafe_right"), ("FIRE", "attack"), ("SPD", "speed"),
]  # fmt: skip
TURN_MAX = 12.5  # observed |turn-delta| range in the dataset


def _hud_strip(width: int, hud_h: int, keys_on: set[str], turn: float, weapon: int | None) -> np.ndarray:
    """Render one HUD strip (H=hud_h, W=width, RGB uint8) for a frame's Doom inputs."""
    img = Image.new("RGB", (width, hud_h), (14, 14, 20))
    d = ImageDraw.Draw(img)
    pad, cw, ch, gap = 8, 78, 26, 8
    x, y = pad, 6
    for label, key in CHIPS:
        on = key in keys_on
        fill = (90, 190, 255) if on else (40, 40, 48)
        txt = (10, 10, 15) if on else (150, 150, 160)
        d.rectangle([x, y, x + cw, y + ch], fill=fill, outline=(70, 70, 80))
        d.text((x + 6, y + 7), label, fill=txt)
        x += cw + gap
    # Weapon chip
    wlabel = f"W{weapon}" if weapon is not None else "W-"
    d.rectangle([x, y, x + 46, y + ch], fill=(60, 60, 30) if weapon else (40, 40, 48), outline=(70, 70, 80))
    d.text((x + 8, y + 7), wlabel, fill=(255, 230, 120) if weapon else (150, 150, 160))
    # Turn bar (centered zero, fills left/right with the turn delta).
    bx0, bx1 = pad, width - pad
    by = y + ch + 12
    bh = 14
    cx = (bx0 + bx1) // 2
    d.rectangle([bx0, by, bx1, by + bh], fill=(30, 30, 38), outline=(70, 70, 80))
    d.line([cx, by, cx, by + bh], fill=(120, 120, 130))
    frac = max(-1.0, min(1.0, turn / TURN_MAX))
    ex = int(cx + frac * (bx1 - cx - 2))
    lo, hi = (cx, ex) if ex >= cx else (ex, cx)
    d.rectangle([lo, by, hi, by + bh], fill=(255, 140, 80))
    d.text((bx0, by + bh + 2), f"TURN {turn:+.1f}", fill=(210, 210, 220))
    return np.asarray(img, dtype=np.uint8)


def _annotate(side_by_side: torch.Tensor, key_presses: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
    """Stack a per-frame Doom-input HUD under a (T, C, H, 2W) uint8 GT|RECON video."""
    t, c, h, w = side_by_side.shape
    vid = side_by_side.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    hud_h = max(70, h // 3)
    out = []
    for i in range(t):
        on = {DOOM_KEYS[j] for j in range(len(DOOM_KEYS)) if key_presses[i, j] > 0}
        weapon = next((wi + 1 for wi in range(7) if key_presses[i, 4 + wi] > 0), None)
        hud = _hud_strip(w, hud_h, on, float(mouse[i, 0]), weapon)
        # GT / RECON labels on the first frame's top corners.
        frame = vid[i].copy()
        img = Image.fromarray(np.concatenate([frame, hud], axis=0))
        dd = ImageDraw.Draw(img)
        dd.text((6, 4), "GT", fill=(80, 255, 120))
        dd.text((w // 2 + 6, 4), "RECON", fill=(255, 200, 80))
        out.append(torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1))
    return torch.stack(out, dim=0)  # (T, C, H+hud_h, 2W)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", required=True, help="Path to a converted Doom split index.json.")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--wandb-project", default="mira-codec")
    ap.add_argument("--wandb-name", default="doom-recon")
    ap.add_argument("--wandb-mode", default="online")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoCodec.load_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    v = model.config.encoder.video

    loader = create_loader(
        index_path=args.index,
        clip_len=v.timesteps,
        target_fps=v.fps,
        action_fps=v.fps,
        batch_size=args.n_clips,
        n_players=1,
        num_workers=2,
        shuffle_buffer_size=8,
        frame_size=(v.height, v.width),
        valid_keys=DOOM_KEYS,
        infinite=True,
        seed=0,
        exclude_replays=False,
    )
    batch, _ = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        outputs = model(batch)

    side = visualize_side_by_side(outputs)["viz_video"].cpu()  # (B, T, C, H, 2W) uint8
    kp = batch.actions.key_presses.cpu()
    mv = batch.actions.mouse_movements.cpu()
    annotated = [_annotate(side[b], kp[b], mv[b]) for b in range(side.shape[0])]  # each (T, C, H+hud, 2W)
    video = torch.cat(annotated, dim=0)  # concat clips along time

    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode)
    with videos_for_wandb(
        {"videos/gt_vs_recon_inputs": VideoForWandb(video=video, caption="GT | RECON with Doom inputs")},
        fps=float(v.fps),
    ) as wv:
        wandb.log(dict(wv))
    wandb.finish()
    print(f"Logged {side.shape[0]} clips ({video.shape[0]} frames) to wandb project {args.wandb_project}.")


if __name__ == "__main__":
    main()
