import argparse
import copy
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MIRA_ATTN_BACKEND", "sdpa")

import numpy as np  # noqa: E402
import play_app  # noqa: E402
import serve_local  # noqa: E402  -- sets the attn backend and the codec symlink
import torch  # noqa: E402
import worldkv  # noqa: E402

TURN_PER_STEP = 12.0  # the server's per-step clamp (Session.consume_turn)
SPIN_STEPS = int(round(worldkv.REV / TURN_PER_STEP))  # 30 steps = one revolution
SETTLE = 6  # steps of standing still before and after the spin


def score(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation of two frames, HUD cropped, grayscale, 8x downsampled."""
    def prep(f):
        f = f.astype(np.float32).mean(-1)[: int(f.shape[0] * 0.78), :][::8, ::8].ravel()
        return (f - f.mean()) / (f.std() + 1e-6)
    x, y = prep(a), prep(b)
    return float((x * y).mean())


def load(args):
    model, loader, ckpt = play_app._load_model_and_seed(args.stage, "doom")
    if args.ctx:
        import time as _t

        from mira.data.training_loader import create_loader

        model.set_inference_context(args.ctx)
        loader = create_loader(
            index_path=f"{serve_local.DATA_DIR}/doom_mira/test",
            clip_len=model.config.n_context_frames + 4 * model.temporal_downsampling,
            target_fps=int(model.config.video.fps),
            batch_size=1, num_workers=2, frame_size=(384, 512),
            valid_keys=list(play_app.DOOM_KEYS), seed=int(_t.time()) % 100000, infinite=True,
        )
    return model, loader, ckpt


def auto_dwell(model, sweep_steps: int) -> int:
    W = model.n_context_latents
    return max(10, W + 2 - 2 * sweep_steps + 20)


def trial(model, seed_batch, mode: str, rng: int, steps: int, args) -> dict:
    """One rollout. mode in {still, off, on}. Returns the return-consistency scores."""
    torch.manual_seed(rng)
    player = play_app.InteractivePlayer(
        model, copy.deepcopy(seed_batch), n_diffusion_steps=args.diffusion_steps,
        noise_level=play_app.NOISE_LEVEL, device="cuda",
    )
    keys = [0] * player.n_keys
    keys[play_app.DOOM_KEYS.index("speed")] = 1  # the model's normal state (88% of training data)

    player.step(keys, 0.0)
    bank = None
    if mode == "on":
        bank = worldkv.attach(player, slots=args.slots, gb=args.gb, tol_deg=args.tol,
                              spread=args.spread, refresh=args.refresh, chunk=args.chunk)
        bank.armed = True  # eager path: there is no setup_graph() to arm it

    for _ in range(SETTLE):
        f0 = player.step(keys, 0.0)
    ref = f0[-1]  # the view we must come back to

    sweep = steps // 2
    if mode == "still":
        script = [0.0] * (2 * sweep + args.dwell)
    elif args.protocol == "spin":
        script = [TURN_PER_STEP] * steps + [0.0] * args.dwell
    else:
        script = [TURN_PER_STEP] * sweep + [0.0] * args.dwell + [-TURN_PER_STEP] * sweep

    mid_at = sweep + args.dwell // 2  # facing away: the control, which should score LOW
    mid = None
    for i, turn in enumerate(script):
        f = player.step(keys, turn)
        if i == mid_at:
            mid = f[-1]
    for _ in range(SETTLE):
        f = player.step(keys, 0.0)

    out = {"return": score(ref, f[-1]), "control_180": score(ref, mid)}
    if bank is not None:
        out["summary"] = bank.summary()
    del player, bank
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--rng", type=int, default=2, help="RNG seeds per clip")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--gb", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=20.0)
    ap.add_argument("--spread", type=int, default=8)
    ap.add_argument("--refresh", type=int, default=4)
    ap.add_argument("--diffusion-steps", type=int, default=6)
    ap.add_argument("--spin-steps", type=int, default=SPIN_STEPS)
    ap.add_argument("--protocol", choices=("awayback", "spin"), default="awayback")
    ap.add_argument("--dwell", type=int, default=-1, help="steps facing away; -1 = scale to the window")
    ap.add_argument("--chunk", type=int, default=1, help="1 = retrieve a contiguous run of R frames")
    ap.add_argument("--stage", default="wm", choices=("wm", "wm_long"))
    ap.add_argument("--ctx", type=int, default=0, help="inference context frames; 0 = checkpoint default")
    args = ap.parse_args()

    model, loader, ckpt = load(args)
    if args.dwell < 0:
        args.dwell = auto_dwell(model, args.spin_steps // 2)
    it = iter(loader)
    seeds = [next(it)[0] for _ in range(args.clips)]
    away = (args.spin_steps + args.dwell) if args.protocol == "awayback" else args.spin_steps
    print(f"[bench] {ckpt}")
    print(f"[bench] protocol={args.protocol}: {args.spin_steps//2 if args.protocol=='awayback' else args.spin_steps}"
          f" steps x {TURN_PER_STEP} deg each way, dwell {args.dwell}"
          f"  -> away for {away/17.5:.2f}s vs a {model.n_context_latents/17.5:.2f}s window")
    print(f"[bench] slots={args.slots} tol={args.tol} chunk={args.chunk} refresh={args.refresh}\n")

    res = {m: {"return": [], "control_180": []} for m in ("still", "off", "on")}
    summary = ""
    t0 = time.time()
    for ci, sb in enumerate(seeds):
        line = []
        for r in range(args.rng):
            for mode in ("still", "off", "on"):
                o = trial(model, sb, mode, 1000 + r, args.spin_steps, args)
                res[mode]["return"].append(o["return"])
                res[mode]["control_180"].append(o["control_180"])
                summary = o.get("summary", summary)
                line.append(f"{mode}={o['return']:+.3f}")
        print(f"[bench] clip {ci}: " + "  ".join(line), flush=True)

    print(f"\n[bench] {summary}")
    print(f"[bench] {time.time()-t0:.0f}s for {args.clips*args.rng*3} rollouts\n")
    print("condition   return-consistency (mean / min / max)      control@180deg")
    for m in ("still", "off", "on"):
        a = np.array(res[m]["return"]); c = np.array(res[m]["control_180"])
        print(f"  {m:6s}    {a.mean():+.3f} / {a.min():+.3f} / {a.max():+.3f}"
              f"                 {c.mean():+.3f}")
    off, on = np.array(res["off"]["return"]), np.array(res["on"]["return"])
    ceil = np.array(res["still"]["return"]).mean()
    print(f"\n  WorldKV delta: {on.mean()-off.mean():+.3f} "
          f"({100*(on.mean()-off.mean())/max(1e-6, ceil-off.mean()):.0f}% of the "
          f"still-ceiling gap of {ceil-off.mean():+.3f})")
    wins = int((on > off).sum())
    print(f"  won {wins}/{len(on)} paired rollouts")

if __name__ == "__main__":
    main()
