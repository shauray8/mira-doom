# Training MIRA on the Doom 2-player dataset

This guide adapts MIRA (built for Rocket League) to
[`chrisxx/doom-2players-mp4`](https://huggingface.co/datasets/chrisxx/doom-2players-mp4). The model
is game-agnostic — the work is a data adapter plus a few config overrides. The recommended path is
**codec → single-player world model → 2-player world model (warm-started)**, exactly how MIRA is
designed.

## 0. Environment

```bash
pixi run setup                       # or your own env with torch>=2.8 + torchcodec + FFmpeg
huggingface-cli login                # do NOT paste your token into code/chat; login stores it
```

The DINOv3-L/16 weights (gated by Meta) are needed **only for codec training**. Download
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` from the DINOv3 page and
`export RS_DINO_WEIGHTS_DIR=/path/to/weights`. World-model training/inference don't need it.

## 1. Download + preprocess

The Doom data is WebDataset tars with a different member layout (`{key}.video_p1.mp4`,
`{key}.actions_p1.npy`, …) and no `index.json`. `scripts/preprocess_doom.py` converts it into MIRA's
layout (`{match}_c{chunk:05d}.p{i}.mp4`/`.jsonl` + `index.json`), splitting episodes into fixed-length
chunks, and mapping the 14-dim action vector: **13 binary dims → keyboard `keys`, the continuous turn
delta (dim 13) → the mouse channel** (channel 0).

```bash
huggingface-cli download chrisxx/doom-2players-mp4 --repo-type dataset --local-dir /data/doom_src
python scripts/preprocess_doom.py \
    --src /data/doom_src --out /data/doom_mira \
    --chunk-len 160 --val-fraction 0.02 --episodes-per-shard 8
# -> /data/doom_mira/{train,test}/index.json + doom-*.tar  (+ id_map.json)
```

`--chunk-len 160` divides both the single-player clip length (40) and the multiplayer one (80), so no
frames are wasted; it must be ≥ 80 or multiplayer clips won't fit in a chunk. This is a big one-time
job (~950 GB source, ~21M frames re-encoded); run it on a machine with fast disk. Start with
`--max-episodes 20` for a smoke test.

## 2. Resolution / fps (already wired in the configs)

- **Resolution**: source is 480×640 (4:3). `configs/dataset/doom.yaml` sets `frame_size: [384, 512]`
  — 16-divisible and **4:3-preserving** (avoids the geometry distortion of squashing to 16:9). This
  requires matching the codec + world-model `video.height/width` (overrides below). If you'd rather
  reuse the released configs unchanged, set `frame_size: [288, 512]` and drop the height overrides
  (accepting some horizontal squash).
- **fps**: kept native at **35** end-to-end. 35→20 is not an integer action stride, so `target_fps`
  and every `video.fps` are 35. (Overrides below.)

## 3. Train the codec (needs DINOv3 weights)

The codec is video-only; `n_players=1` treats each perspective as an independent frame source. Use a
diverse subset (~10–20% of episodes is plenty for a strong reconstruction codec — it's data-efficient
per-frame). Point `train_index`/`test_index` at the split **directories or their `index.json`**.

```bash
python scripts/train_codec.py \
    dataset=rocket_league dataset/actions@dataset.actions=doom \
    dataset.train_index=/data/doom_mira/train dataset.test_index=/data/doom_mira/test \
    dataset.n_players=1 dataset.frame_size=[384,512] dataset.target_fps=35 \
    model.architecture.config.encoder.video.fps=35 \
    model.architecture.config.encoder.video.height=384 \
    model.architecture.config.encoder.video.width=512
```

Watch val PSNR / LPIPS plateau. Note the resulting `checkpoint.pth` path.

## 4. Train the single-player world model (no DINOv3 needed)

Uses the frozen codec. Each of the 2 perspectives is an independent training row → ~2× the clips
(~334 perspective-hours from 167h). Train until val loss / rollout metrics stop improving. **Use the
full dataset here** — this is where scale matters.

```bash
torchrun --nproc_per_node=1 scripts/train_world_model.py \
    dataset=doom dataset.n_players=1 dataset.frame_size=[384,512] dataset.target_fps=35 \
    model.architecture.config.codec_checkpoint=/path/to/codec/checkpoint.pth \
    model.architecture.config.video.fps=35 \
    model.architecture.config.video.height=384 \
    model.architecture.config.video.width=512 \
    dataset.train_index=/data/doom_mira/train dataset.test_index=/data/doom_mira/test \
    run.output_dir=runs/doom_sp
```

## 5. Finetune the 2-player world model (warm-started)

`MultiWrapperWorldModel` tiles the 2 players and warm-starts from the single-player checkpoint via
RoPE (resolution-independent). Much shorter than stage 4 (tens of thousands of steps).

```bash
torchrun --nproc_per_node=1 scripts/train_world_model.py \
    model=multi_wrapper_world_model dataset=doom \
    dataset.n_players=2 model.architecture.config.n_players=2 \
    dataset.frame_size=[384,512] dataset.target_fps=35 \
    model.architecture.config.wm_config.codec_checkpoint=/path/to/codec/checkpoint.pth \
    model.architecture.config.wm_config.video.fps=35 \
    model.architecture.config.wm_config.video.height=384 \
    model.architecture.config.wm_config.video.width=512 \
    run.finetune_from=runs/doom_sp/checkpoint-XXXX/checkpoint.pth \
    dataset.train_index=/data/doom_mira/train dataset.test_index=/data/doom_mira/test \
    run.output_dir=runs/doom_mp
```

## 6. H100 performance

Wired in and on by default where safe:

- **FlashAttention on the transformer** (`mira.ml.flash_attention`). `MIRA_ATTN_BACKEND` selects the
  kernel: `auto` (default: FA3→FA2→SDPA), `fa3`, `fa2`, `sdpa`. FA's autograd function fuses the
  **forward and backward**, so selecting FA3 accelerates both — no separate wiring. The world model's
  head_dim is 128 (ideal for FA3) and GQA (16 q / 4 kv heads) is supported natively. The KV-cache
  inference path stays on SDPA (correctness-critical, not throughput-bound).

  Install FA3 (Hopper) once on the H100:
  ```bash
  git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention/hopper
  python setup.py install     # provides `flash_attn_interface`
  ```
  Verify it's picked up: the trainer logs `Attention backend: FlashAttention-3` on startup. If FA3
  isn't installed, `auto` silently uses FA2 or SDPA — set `MIRA_ATTN_BACKEND=fa3` to fail loudly.

- **bf16 autocast** (already in the trainer), **TF32** matmul/cudnn + `float32_matmul_precision=high`,
  **fused AdamW**.
- **`torch.compile`**: opt-in with `run.compile=true` (compiles codec encode, the diffusion
  transformer, and codec decode separately). Big throughput win once warm; validate a short run first.
- **Activation checkpointing**: already on for the multiplayer model (`wm_config`).

Multi-GPU / scale-out: DDP is the default and ideal for the 1B model on one H100 or a small node. For
large scale-out there's an opt-in FSDP2 path (`run.parallelism=fsdp2`, `torchrun --nproc_per_node=N`)
that shards params/grads/optimizer state — validate a short run before a long one. (torchtitan the
*framework* isn't used; FSDP2 is the primitive worth taking from it.)

## How much data — summary

| Stage | Data | Notes |
|---|---|---|
| Codec | ~10–20% of episodes, diverse | per-frame reconstruction is data-efficient; train to PSNR/LPIPS plateau |
| Single-player WM | **all** (~334 perspective-hours) | where scale matters; train to metric plateau |
| 2-player finetune | all paired 2-player groups (~167h) | warm-started → short (tens of thousands of steps) |

## Evaluation

```bash
python scripts/eval_world_model_offline.py runs/doom_mp/checkpoint-XXXX/checkpoint.pth
```
