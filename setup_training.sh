#!/usr/bin/env bash
# Deps, DINOv3 weights, and the dataset converted into MIRA's layout. Idempotent.
#
#   ./setup_training.sh                  small trial slice (8 shards)
#   DOOM_SHARDS=0 ./setup_training.sh    the FULL dataset (~950 GB source, ~21M frames)
#
# Needs: NVIDIA GPU, python>=3.10 with torch>=2.8 ALREADY installed, ffmpeg, a HuggingFace login,
# and -- for CODEC training only -- access to the gated DINOv3 weights.
#
#   DOOM_SHARDS=8   shards to download; 0 = all 1323      DOOM_SRC=./data_src
#   VAL_FRACTION=0.02  episodes routed to test/           DOOM_MIRA=./data/doom_mira
#   DINO_DIR=./dino_weights
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DOOM_SHARDS=${DOOM_SHARDS:-8}
DOOM_SRC=${DOOM_SRC:-$REPO_DIR/data_src}
DOOM_MIRA=${DOOM_MIRA:-$REPO_DIR/data/doom_mira}
DINO_DIR=${DINO_DIR:-$REPO_DIR/dino_weights}
VAL_FRACTION=${VAL_FRACTION:-0.02}
CHUNK_LEN=${CHUNK_LEN:-160}
DINO_HF=facebook/dinov3-vitl16-pretrain-lvd1689m
DINO_FILE=dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

log() { printf '\033[1;36m[setup-train]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup-train] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found on PATH (needed to re-encode chunks)."
python3 - <<'PY' || die "torch with CUDA is required. Install a torch >= 2.8 build matching your CUDA, then re-run."
import torch
assert torch.cuda.is_available(), "torch is installed but CUDA is not available"
major = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert major >= (2, 8), f"torch >= 2.8 required, found {torch.__version__}"
print(f"[setup-train] torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
PY

log "Installing training dependencies (train + eval + decode + HF tooling) ..."
pip install -q -e '.[doom]' hf_transfer
python3 - <<'PY'
import subprocess, sys, torch
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
except Exception:
    pkg = "torchcodec==0.7.0" if torch.__version__.startswith("2.8") else "torchcodec"
    print(f"[setup-train] installing {pkg} (the installed one does not load)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
huggingface-cli whoami >/dev/null 2>&1 || die "not logged in to HuggingFace — run: huggingface-cli login"

# Codec only: the backbone travels inside the codec checkpoint for everything downstream.
if [ -f "$DINO_DIR/$DINO_FILE" ]; then
    log "DINOv3 weights already at $DINO_DIR — skipping."
else
    log "Converting gated HF DINOv3 -> hub .pth in $DINO_DIR ..."
    python3 scripts/convert_dinov3_hf_to_hub.py --hf-repo "$DINO_HF" --model dinov3_vitl16 --out-dir "$DINO_DIR" \
        || die "DINOv3 conversion failed — the repo is gated; accept the licence at https://huggingface.co/$DINO_HF"
fi

if find "$DOOM_SRC" -name '*.tar' -print -quit 2>/dev/null | grep -q .; then
    log "Source shards already under $DOOM_SRC ($(find "$DOOM_SRC" -name '*.tar' | wc -l) tars) — skipping download."
else
    log "Downloading chrisxx/doom-2players-mp4 -> $DOOM_SRC (shards: ${DOOM_SHARDS:-all}) ..."
    DOOM_SHARDS="$DOOM_SHARDS" DOOM_SRC="$DOOM_SRC" python3 - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
n, dest = int(os.environ["DOOM_SHARDS"]), os.environ["DOOM_SRC"]
if n <= 0:
    snapshot_download("chrisxx/doom-2players-mp4", repo_type="dataset", local_dir=dest,
                      allow_patterns=["data/*.tar"], max_workers=8)
else:
    for i in range(n):
        print("[setup-train] got", hf_hub_download(
            "chrisxx/doom-2players-mp4", f"data/mp-0000-{i:06d}.tar",
            repo_type="dataset", local_dir=dest), flush=True)
PY
fi

if [ -f "$DOOM_MIRA/train/index.json" ]; then
    log "Preprocessed data already at $DOOM_MIRA — skipping."
else
    # 13 binary action dims -> keys, the continuous turn delta -> the mouse channel. chunk-len 160
    # divides both the single-player clip length (40) and the 2-player one (80).
    log "Preprocessing -> $DOOM_MIRA (chunk-len $CHUNK_LEN, val-fraction $VAL_FRACTION) ..."
    log "NOTE: this holds one whole episode in RAM at a time (~12 GB peak) and is CPU/disk bound."
    python3 scripts/preprocess_doom.py \
        --src "$DOOM_SRC" --out "$DOOM_MIRA" \
        --chunk-len "$CHUNK_LEN" --val-fraction "$VAL_FRACTION" --episodes-per-shard 8
fi

log "Done."
cat <<EOF

  Data:    $DOOM_MIRA/{train,test}/index.json
  DINOv3:  $DINO_DIR/$DINO_FILE

The configs already default to Doom geometry (384x512, 35 fps, 13-key action vocabulary), so the
commands below need only paths. Full guide with budgets and stopping criteria: docs/training_run_guide.md

  # 1. Codec (frozen DINOv3 encoder + ViT decoder). Needs RS_DINO_WEIGHTS_DIR.
  export RS_DINO_WEIGHTS_DIR=$DINO_DIR
  python3 scripts/train_codec.py \\
      dataset.train_index=$DOOM_MIRA/train dataset.test_index=$DOOM_MIRA/test \\
      run.batch_size=4 run.steps=60000 run.output_dir=ckpts/doom_codec

  # 2. Single-player world model (the playable one). The codec ships frozen inside its checkpoint.
  python3 scripts/train_world_model.py \\
      dataset.train_index=$DOOM_MIRA/train dataset.test_index=$DOOM_MIRA/test \\
      model.architecture.config.codec_checkpoint=ckpts/doom_codec/checkpoint-XXXX/checkpoint.pth \\
      run.batch_size=8 run.steps=250000 run.output_dir=ckpts/doom_wm

  # 3. Two-player world model (warm-started from step 2; optional).
  torchrun --nproc_per_node=N scripts/train_world_model.py model=multi_wrapper_world_model \\
      dataset.n_players=2 \\
      dataset.train_index=$DOOM_MIRA/train dataset.test_index=$DOOM_MIRA/test \\
      model.architecture.config.wm_config.codec_checkpoint=ckpts/doom_codec/checkpoint-XXXX/checkpoint.pth \\
      run.finetune_from=ckpts/doom_wm/checkpoint-XXXX/checkpoint.pth \\
      run.batch_size=4 run.steps=40000 run.output_dir=ckpts/doom_wm_mp

Then play your own checkpoint with ./infer.sh (it picks the newest checkpoint under ckpts/doom_wm).
EOF
