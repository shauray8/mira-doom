# Doom on MIRA — full training run guide (with measured budgets)

This is the follow-along for the **main run** on `chrisxx/doom-2players-mp4` (~2,600 episodes, ~167 h,
~21 M frames, 480×640 @ 35 fps, 2 players). It answers: train order, per-stage steps, and GPU-hours /
wall-clock, all grounded in **overfit runs we actually executed on an H100** (not guesses).

> TL;DR order: **preprocess → codec → single-player world model → 2-player world model (warm-started)**.
> You must train the codec first — the world model trains on the *frozen* codec's latents.

---

## 0. Validation we ran (why you can trust the setup)

Both stages were overfit end-to-end on real Doom data with the **real gated DINOv3** weights, on one H100:

| Stage | What ran | Result | W&B |
|---|---|---|---|
| Codec | real DINOv3-L + RAEv2, 384×512×40, real clips | total loss 1.60 → 0.21; `loss_mae` 0.82 → 0.60 monotonic | `mira-codec/doom-overfit-codec-384` |
| World model | 1.19 B diffusion transformer + frozen codec, single-player | flow-matching loss 11.1 → ~1.4 (≈8× drop) | `mira-world-model/doom-overfit-wm-384` |
| Recon viz | GT ∣ RECON + Doom input HUD | logged video | `mira-codec/doom-gt-vs-recon` |

Loss collapses on both → the released hyperparameters + our data wiring train stably. No NaNs, no OOM
at batch 2 / 384×512.

**One-time setup** (see `docs/doom_training.md` for detail):
```bash
huggingface-cli login                                    # gated dataset + DINOv3
python scripts/convert_dinov3_hf_to_hub.py \             # gated HF safetensors -> hub .pth
    --hf-repo facebook/dinov3-vitl16-pretrain-lvd1689m --model dinov3_vitl16 --out-dir /data/dino
export RS_DINO_WEIGHTS_DIR=/data/dino                     # needed only for CODEC training
```
The converter self-checks (`strict=True` load + feature match vs the HF model, rel ≈ 1e-6).

---

## 1. Preprocess  (CPU, one-time)

```bash
huggingface-cli download chrisxx/doom-2players-mp4 --repo-type dataset --local-dir /data/doom_src
python scripts/preprocess_doom.py --src /data/doom_src --out /data/doom_mira \
    --chunk-len 160 --val-fraction 0.02 --episodes-per-shard 8
```
- Measured **~1.8 min/shard** (2 episodes) single-process → **~40 CPU-hours** for the full set.
- Embarrassingly parallel across shards — run N processes over disjoint `--src` subsets and it drops to
  ~40/N hours. It re-encodes ~21 M frames, so use fast disk.

---

## 2. Codec  (needs `RS_DINO_WEIGHTS_DIR`)

The codec is video-only; `n_players=1` treats every perspective as an independent frame source. It's
data-efficient (per-frame reconstruction) — a **diverse 15–25 % subset** of episodes is enough; you do
not need all 21 M frames here.

```bash
python scripts/train_codec.py \
    dataset=doom dataset.n_players=1 dataset.frame_size=[384,512] dataset.target_fps=35 \
    model.architecture.config.encoder.video.fps=35 \
    model.architecture.config.encoder.video.height=384 \
    model.architecture.config.encoder.video.width=512 \
    model.architecture.config.encoder.video.timesteps=40 \
    run.batch_size=4 run.compile=true run.steps=60000 \
    run.checkpoint_every=5000 run.checkpoint_keep_recent=3 run.checkpoint_keep_permanent_every=20000 \
    dataset.train_index=/data/doom_mira/train dataset.test_index=/data/doom_mira/test \
    run.output_dir=runs/doom_codec
```
- **Checkpointing:** `checkpoint_every` takes a **step count** (`5000`) or a percent (`"5%"`). `keep_recent=3`
  keeps a rolling last-3 for crash recovery; `keep_permanent_every=20000` also keeps an un-deletable
  milestone every 20 k steps. Auto-resume is automatic — re-run the same command and it continues from the
  latest checkpoint in `output_dir` (or pass `run.continue_from=<path>`).
- **Batch size:** at 384×512 × 40 frames the codec is memory-heavy (the LPIPS pass on 40 frames spikes) —
  batch **4** fits an 80 GB H100 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; batch 8 OOMs.
  Raise it until you OOM, or drop `encoder.video.timesteps` to 24 to fit a bigger batch.
- **Stop on plateau**, not step count — see *"Is the codec trained enough?"* below. Codecs usually plateau
  in **~40–80 k steps**.
- Resolution must be **÷32** (DINOv3 patch 16 × spatial bottleneck stride 2). 384×512 keeps 4:3; 480×640 is
  native but ~1.7× the tokens.
- A GT∣RECON video (+ Doom input HUD) logs to W&B every `run.viz_every` steps (default 1000) so you watch
  reconstructions sharpen.

Keep the resulting `runs/doom_codec/checkpoint-XXXX/checkpoint.pth` — every later stage points at it.

---

## 3. Single-player world model  (no DINOv3 needed — codec ships frozen in its checkpoint)

Each of the 2 perspectives is an independent training row (~2× the clips). **Use the full dataset here** —
this is where scale matters. FA3 accelerates this stage the most (attention-heavy 1 B transformer).

```bash
MIRA_ATTN_BACKEND=fa3 python scripts/train_world_model.py \
    dataset=doom dataset.n_players=1 dataset.frame_size=[384,512] dataset.target_fps=35 \
    model.architecture.config.codec_checkpoint=runs/doom_codec/checkpoint-XXXX/checkpoint.pth \
    model.architecture.config.video.fps=35 \
    model.architecture.config.video.height=384 model.architecture.config.video.width=512 \
    model.architecture.config.video.timesteps=40 \
    run.batch_size=8 run.compile=true run.steps=250000 \
    run.checkpoint_every=5000 run.checkpoint_keep_recent=3 run.checkpoint_keep_permanent_every=25000 \
    dataset.train_index=/data/doom_mira/train dataset.test_index=/data/doom_mira/test \
    run.output_dir=runs/doom_wm_sp
```
- Target **~200–300 k steps** (matches the released default); stop when val loss + rollout metrics plateau
  — see *"Is the world model trained enough?"* below.
- The WM is lighter per-clip than the codec (no LPIPS/decoder-at-full-res in the step); start batch **8** and
  raise until you OOM. A GT-vs-generated rollout logs to W&B every `run.viz_every` steps (default 1000).
- Multi-GPU: `torchrun --nproc_per_node=N ...`; add `run.parallelism=fsdp2` only if you outgrow DDP memory.

---

## 4. Two-player world model  (warm-started — short)

```bash
MIRA_ATTN_BACKEND=fa3 torchrun --nproc_per_node=N scripts/train_world_model.py \
    model=multi_wrapper_world_model dataset=doom \
    dataset.n_players=2 model.architecture.config.n_players=2 \
    dataset.frame_size=[384,512] dataset.target_fps=35 \
    model.architecture.config.wm_config.codec_checkpoint=runs/doom_codec/checkpoint-XXXX/checkpoint.pth \
    model.architecture.config.wm_config.video.fps=35 \
    model.architecture.config.wm_config.video.height=384 \
    model.architecture.config.wm_config.video.width=512 \
    run.finetune_from=runs/doom_wm_sp/checkpoint-XXXX/checkpoint.pth \
    run.batch_size=4 run.compile=true run.steps=40000 \
    run.checkpoint_every=5000 run.checkpoint_keep_recent=3 run.checkpoint_keep_permanent_every=20000 \
    dataset.train_index=/data/doom_mira/train dataset.test_index=/data/doom_mira/test \
    run.output_dir=runs/doom_wm_mp
```
- RoPE makes the single→multi warm-start clean, so this needs only **~30–50 k steps**. The tiled 2-player
  frame is ~2× the tokens per step (uses `timesteps=80`, activation checkpointing already on).

---

## Measured throughput (1× H100, bf16, batch 2, 384×512, 40-frame clips)

| Stage | s/step (no compile) | s/clip | notes |
|---|---|---|---|
| Codec | 2.1 | 1.05 | DINOv3-L encode + ViT decoder + LPIPS + DINO-consistency |
| WM single | 1.55 | 0.78 | includes the codec encode each step; SDPA (no FA3 here) |

`torch.compile` (`run.compile=true`, the released default) gives roughly **1.6×**; FA3 adds ~**1.2–1.4×**
on the WM's attention. The budget below applies those factors (labelled "compiled").

## Budget & wall-clock (this dataset)

| Stage | Rec. steps | Batch | Clips seen | GPU-h (compiled) | 1× H100 | 8× H100 |
|---|---|---|---|---|---|---|
| Preprocess | — | — | — | ~40 **CPU**-h | — | (parallel) |
| Codec | 60 k | 8 | 0.48 M | **~90** | ~3.6 d | ~11 h |
| WM single | 250 k | 8 | 2.0 M | **~200** | ~8.4 d | ~25 h |
| WM 2-player | 40 k | 8 rows | 0.32 M | **~80** | ~3.4 d | ~10 h |
| **Total (GPU)** | | | | **~370 GPU-h** | ~15 d | **~46 h** |

These scale ~linearly — halve the steps → halve the hours. Numbers are extrapolated from the measured
per-clip times above; treat ±30 % as the confidence band (batch-size utilization, disk, exact plateau).

**Biggest lever — cache codec latents.** The WM step re-runs the frozen DINOv3 encode every iteration.
Because the codec is frozen and deterministic here (`noise_tau=0`, `use_codec_posterior_mean=true`),
precomputing latents once is **lossless** and cuts the WM stages by roughly **1.5–1.8×** (≈ **110 GPU-h**
for the single-player stage instead of ~200). It ties the cache to a specific codec checkpoint. Ask if you
want the precompute script + the "latents-provided" WM fast path — it's a clean add-on.

## Is the codec trained enough?

The codec is a reconstruction model, so its "done" signal is reconstruction quality plateauing. Logged
to W&B each `validation.val_every`:

| Signal (W&B key) | Direction | "Trained enough" |
|---|---|---|
| `test/psnr` (dB) | ↑ | rises then **flattens** (game frames typically plateau ~30–34 dB) |
| `test/ssim` | ↑ | flattens (typically ~0.9+) |
| `test/loss_lpips_perceptual` | ↓ | flattens (perceptual similarity) |
| `test/loss_mae` | ↓ | flattens |
| `train/latent_std` | — | stabilizes (latent scale settled) |

**Decision rule:** stop when PSNR/SSIM/LPIPS stop improving for a sustained window **and** the
`videos/recon_inputs` clip is visually faithful (HUD text legible, no smearing on motion). PSNR/SSIM are the
interpretable numbers; the loss components are auto-weight-balanced so their *total* is not a clean progress
signal — read the components.

## Is the world model trained enough?

Two layers. (1) The **training/val flow-matching loss** should fall then plateau. (2) The **rollout metrics**
(logged every `validation.downstream_val_every`; needs `mira[eval]` + reachable DINO/Inception weights) are
what actually say the *generated video* is good:

| Signal (W&B key) | What it measures | "Trained enough" |
|---|---|---|
| `metrics/dino_frechet` | sliced **Fréchet DINO distance** (FDD) — headline video quality | falls and **flattens**; approaches the codec floor below |
| `metrics/dino_frechet_codec_floor` | the codec's own reconstruction FDD | the WM's FDD **cannot beat this** — "done" ≈ WM FDD near the floor |
| `metrics/inception_frechet` | FVD-like Inception distance | falls and flattens |
| `metrics/dino_cos_drift`, `dino_l2_drift`, `latent_drift` | how far the rollout drifts from GT over the horizon | low and **does not blow up** with horizon (stability) |
| per-frame PSNR / LPIPS / SSIM | rollout vs GT, per frame | plateau |

**Decision rule:** stop when FDD + drift plateau (FDD near the codec floor) **and** the `videos/rollout_inputs`
clip stays coherent to the end of the horizon and responds to the input HUD (e.g. the view turns when the
turn-delta bar is large). Drift that grows with the rollout horizon = not trained enough (or needs more
context/diffusion-forcing), even if the 1-step loss looks fine.

## Making training faster

Already wired in (on by default where safe): bf16 autocast, TF32, fused AdamW, `run.compile=true`, and the
FlashAttention backend. Beyond those, in rough order of payoff:

1. **Cache the codec latents for the WM stages** — the single biggest lever. Each WM step re-runs the frozen
   DINOv3 encode; precomputing latents once (lossless here — `noise_tau=0`, `use_codec_posterior_mean=true`)
   cuts the WM stages **~1.5–1.8×**. Ask for the precompute script + "latents-provided" fast path.
2. **FlashAttention-3** on the WM: `MIRA_ATTN_BACKEND=fa3` (build once, see setup). ~1.2–1.4× on the
   attention-heavy transformer; fuses fwd+bwd. **Check the startup log line** —
   `Attention backend: FlashAttention-3` means it's active; `... SDPA` means neither `flash_attn_interface`
   (FA3) nor `flash_attn` (FA2) is installed, so it fell back (correct, just slower). Use
   `MIRA_ATTN_BACKEND=fa3` to make a missing FA3 fail loudly instead of silently falling back.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — cuts fragmentation, often lets you fit a bigger
   batch with no code change.
4. **Bigger batch** — raise until you OOM (better GPU utilization). Codec is the memory-tight one (LPIPS on 40
   frames); the WM has more headroom.
5. **Avoid data starvation** — video decode is CPU-bound on the loader workers. If `System/step_ms` is well
   above pure compute, raise `dataloader.num_workers`.
6. **Multi-GPU** — `torchrun --nproc_per_node=N` (DDP); `run.parallelism=fsdp2` for large scale-out.
7. **Cheaper codec encoder** — `encoder.rae_model=dinov3_vitb16` (and convert the ViT-B weights) is faster than
   ViT-L at some quality cost; or drop to 288×512 / fewer `timesteps` for fewer tokens.
8. **Advanced:** FP8 training (torchao float8) on the transformer linears — a further ~1.3× on H100; validate
   quality before committing.

## Quick sanity before the big run

Reproduce the overfit (should collapse in a few hundred steps) on a 1–2 shard slice:
```bash
python scripts/preprocess_doom.py --src <one_src_shard_dir> --out /tmp/doom_mini --val-fraction 0 --max-episodes 4
# then the codec command with run.steps=400 run.batch_size=2 on /tmp/doom_mini, and log recon:
python scripts/log_codec_recon.py --checkpoint runs/.../checkpoint.pth --index /tmp/doom_mini/train/index.json
```
