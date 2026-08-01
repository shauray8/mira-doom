#!/usr/bin/env bash
# Configure via env vars:
#   DOOM_SRC=/data/doom_src        # raw dataset download dir
#   DOOM_MIRA=/data/doom_mira      # converted (MIRA-format) output dir
#   DINO_DIR=/data/dino_weights    # DINOv3 hub .pth output dir  (-> RS_DINO_WEIGHTS_DIR)
#   DOOM_SHARDS=""                 # e.g. "8" to fetch only the first 8 shards for a quick trial ("" = all)
#   CHUNK_LEN=160  VAL_FRACTION=0.02  EPISODES_PER_SHARD=8
set -euo pipefail

DOOM_SRC=${DOOM_SRC:-/data/doom_src}
DOOM_MIRA=${DOOM_MIRA:-/data/doom_mira}
DINO_DIR=${DINO_DIR:-/data/dino_weights}
DOOM_SHARDS=${DOOM_SHARDS:-}
CHUNK_LEN=${CHUNK_LEN:-160}
VAL_FRACTION=${VAL_FRACTION:-0.02}
EPISODES_PER_SHARD=${EPISODES_PER_SHARD:-8}

HF_DATASET=chrisxx/doom-2players-mp4
DINO_HF=facebook/dinov3-vitl16-pretrain-lvd1689m
DINO_FILE=dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\033[1;36m[setup-doom]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup-doom] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found on PATH (needed to re-encode chunks)."
command -v huggingface-cli >/dev/null 2>&1 || die "huggingface-cli not found (pip install -e '.[doom]')."
python -c "import torch, torchcodec, numpy" 2>/dev/null || die "python deps missing — run: pip install -e '.[doom]'"
huggingface-cli whoami >/dev/null 2>&1 || die "not logged in — run: huggingface-cli login (needs gated access)."
log "Preflight OK (ffmpeg, deps, HF auth as $(huggingface-cli whoami 2>/dev/null | head -1))."

if [ -f "$DINO_DIR/$DINO_FILE" ]; then
    log "DINOv3 weights already at $DINO_DIR/$DINO_FILE — skipping conversion."
else
    log "Converting gated HF DINOv3 -> hub .pth in $DINO_DIR ..."
    python "$SCRIPT_DIR/convert_dinov3_hf_to_hub.py" --hf-repo "$DINO_HF" --model dinov3_vitl16 --out-dir "$DINO_DIR"
fi

if ls "$DOOM_SRC"/**/*.tar >/dev/null 2>&1 || ls "$DOOM_SRC"/*.tar >/dev/null 2>&1; then
    n=$(find "$DOOM_SRC" -name '*.tar' | wc -l)
    log "Dataset already present under $DOOM_SRC ($n .tar shards) — skipping download."
else
    log "Downloading $HF_DATASET -> $DOOM_SRC ${DOOM_SHARDS:+(first $DOOM_SHARDS shards)} ..."
    if [ -n "$DOOM_SHARDS" ]; then
        patterns=()
        for i in $(seq 0 $((DOOM_SHARDS - 1))); do patterns+=(--include "data/*$(printf '%06d' "$i").tar"); done
        huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DOOM_SRC" "${patterns[@]}"
    else
        huggingface-cli download "$HF_DATASET" --repo-type dataset --local-dir "$DOOM_SRC"
    fi
fi

if [ -f "$DOOM_MIRA/train/index.json" ]; then
    log "Preprocessed data already at $DOOM_MIRA/train/index.json — skipping preprocessing."
else
    log "Preprocessing $DOOM_SRC -> $DOOM_MIRA (chunk_len=$CHUNK_LEN, val=$VAL_FRACTION) ..."
    python "$SCRIPT_DIR/preprocess_doom.py" \
        --src "$DOOM_SRC" --out "$DOOM_MIRA" \
        --chunk-len "$CHUNK_LEN" --val-fraction "$VAL_FRACTION" --episodes-per-shard "$EPISODES_PER_SHARD"
fi

log "Setup complete."
cat <<EOF

  Data:   $DOOM_MIRA/{train,test}/index.json
  DINOv3: $DINO_DIR/$DINO_FILE

Next:
  export RS_DINO_WEIGHTS_DIR=$DINO_DIR        # needed for CODEC training only
  #   git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention/hopper && python setup.py install
  # then follow docs/doom_full_training_run.md  (codec -> single-player WM -> 2-player WM)
EOF
