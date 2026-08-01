"""Sweep how much of the window the memory gets: `on-R` reserves R of the 19 latent slots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import time

import bench_worldkv as B
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--clips", type=int, default=8)
ap.add_argument("--rng", type=int, default=2)
ap.add_argument("--slots-sweep", default="2,4,8")
ap.add_argument("--tol", type=float, default=30.0)
ap.add_argument("--gb", type=float, default=1.0)
ap.add_argument("--spread", type=int, default=8)
ap.add_argument("--refresh", type=int, default=4)
ap.add_argument("--chunk", type=int, default=1)
ap.add_argument("--diffusion-steps", type=int, default=6)
ap.add_argument("--protocol", default="awayback")
ap.add_argument("--dwell", type=int, default=-1)
ap.add_argument("--stage", default="wm", choices=("wm", "wm_long"))
ap.add_argument("--ctx", type=int, default=0)
ap.add_argument("--spin-steps", type=int, default=B.SPIN_STEPS)
args = ap.parse_args()
SLOTS = [int(x) for x in args.slots_sweep.split(",")]

model, loader, ckpt = B.load(args)
if args.dwell < 0:
    args.dwell = B.auto_dwell(model, args.spin_steps // 2)
it = iter(loader)
seeds = [next(it)[0] for _ in range(args.clips)]
print(f"[sweep] {args.stage} ctx={model.n_context_latents} latents "
      f"({model.n_context_latents/17.5:.2f}s) | {args.clips} clips x {args.rng} rng, slots {SLOTS}, "
      f"tol {args.tol}, dwell {args.dwell} -> away {(args.spin_steps+args.dwell)/17.5:.2f}s\n", flush=True)

conds = ["off"] + [f"on{r}" for r in SLOTS]
res = {c: [] for c in conds}
summ = {}
t0 = time.time()
for ci, sb in enumerate(seeds):
    for r in range(args.rng):
        row = []
        for c in conds:
            if c != "off":
                args.slots = int(c[2:])
            o = B.trial(model, sb, "off" if c == "off" else "on", 1000 + r, args.spin_steps, args)
            res[c].append(o["return"])
            if "summary" in o:
                summ[c] = o["summary"]
            row.append(f"{c}={o['return']:+.3f}")
        print(f"[sweep] clip {ci} rng {r}: " + "  ".join(row), flush=True)

print(f"\n[sweep] {time.time()-t0:.0f}s")
for c in conds:
    if c in summ:
        print(f"[sweep] {c}: {summ[c]}")
off = np.array(res["off"])
print("\ncondition   mean return   delta vs off   paired wins")
print(f"  off       {off.mean():+.3f}          --            --")
for c in conds[1:]:
    a = np.array(res[c])
    d = a - off
    print(f"  {c:8s}  {a.mean():+.3f}        {d.mean():+.3f}        {int((d>0).sum())}/{len(d)}"
          f"   (paired sd {d.std():.3f})")
