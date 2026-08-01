#!/usr/bin/env bash
# Deps, checkpoints and seed clips for playing. Idempotent.  ./setup_inference.sh && ./infer.sh
#
# Needs: NVIDIA GPU, python>=3.10 with torch>=2.8 ALREADY installed (CUDA wheels are environment
# specific, so this never touches yours), ffmpeg, and a HuggingFace login.
#
#   MIRA_HF_REPO=shauray/mira-doom   model repo holding codec/ + wm/ (+ wm_long/)
#   MIRA_STAGES="wm"                 or "wm wm_long" for the long-memory finetune
#   DOOM_SHARDS=2                    source shards for seed clips (~570 MB each)
#   DOOM_EPISODES=4                  episodes to preprocess
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MIRA_HF_REPO=${MIRA_HF_REPO:-shauray/mira-doom}
MIRA_STAGES=${MIRA_STAGES:-wm}
DOOM_SHARDS=${DOOM_SHARDS:-2}
DOOM_EPISODES=${DOOM_EPISODES:-4}
DOOM_SRC=${DOOM_SRC:-$REPO_DIR/data_src}

log()  { printf '\033[1;36m[setup-infer]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup-infer] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found on PATH (needed to build the seed clips)."
python3 - <<'PY' || die "torch with CUDA is required. Install a torch >= 2.8 build matching your CUDA, then re-run."
import sys, torch
assert torch.cuda.is_available(), "torch is installed but CUDA is not available"
major = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert major >= (2, 8), f"torch >= 2.8 required, found {torch.__version__}"
print(f"[setup-infer] torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
      f"| sm_{''.join(str(x) for x in torch.cuda.get_device_capability())}")
PY

log "Installing inference dependencies ..."
pip install -q -e '.[hydra,decode,viz]' 'fastapi' 'uvicorn[standard]' 'huggingface_hub[cli]' hf_transfer

# torchcodec must match the torch/CUDA pair. 0.15 is a CUDA 13 build that fails to load under
# torch 2.8+cu128, and preprocess_doom then silently writes an index with zero matches.
python3 - <<'PY'
import subprocess, sys, torch
want = "0.7.0" if torch.__version__.startswith("2.8") else None
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    ok = True
except Exception:
    ok = False
if not ok:
    pkg = f"torchcodec=={want}" if want else "torchcodec"
    print(f"[setup-infer] installing {pkg} (the installed one does not load)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

# DINOv3's hubconf.py imports these at module scope, and the codec builds its architecture through
# torch.hub even when the weights come from the checkpoint. Without them the WM fails to load.
pip install -q torchmetrics termcolor

export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
huggingface-cli whoami >/dev/null 2>&1 || die "not logged in to HuggingFace — run: huggingface-cli login"

log "Fetching checkpoints from $MIRA_HF_REPO (stages: codec $MIRA_STAGES) ..."
MIRA_HF_REPO="$MIRA_HF_REPO" MIRA_STAGES="$MIRA_STAGES" python3 - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo = os.environ["MIRA_HF_REPO"]
stages = ["codec", *os.environ["MIRA_STAGES"].split()]
ckpts = Path("ckpts")
files = HfApi().list_repo_files(repo)

for stage in stages:
    # newest checkpoint-<step>/ for this stage, numerically (checkpoint-9000 must not beat 75000)
    steps = sorted(
        (int(f.split("/")[1].split("-")[1]), f) for f in files
        if f.startswith(f"{stage}/checkpoint-") and f.endswith("checkpoint.pth")
    )
    if not steps:
        print(f"[setup-infer] no checkpoint for stage '{stage}' in {repo}; skipping")
        continue
    _, remote = steps[-1]
    cfg = f"{stage}/codec_config.yaml" if stage == "codec" else f"{stage}/world_model_config.yaml"
    for src in (remote, cfg):
        if src not in files:
            continue
        dest = ckpts / f"doom_{stage}" / Path(src).relative_to(stage)
        if dest.exists():
            print(f"[setup-infer] have {dest}")
            continue
        print(f"[setup-infer] downloading {src} ...", flush=True)
        got = hf_hub_download(repo, src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(got, dest)
        print(f"[setup-infer] -> {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
PY

if [ -f data/doom_mira/test/index.json ]; then
    log "Seed clips already at data/doom_mira/test — skipping."
else
    log "Fetching $DOOM_SHARDS source shard(s) for seed clips ..."
    DOOM_SHARDS="$DOOM_SHARDS" DOOM_SRC="$DOOM_SRC" python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
for i in range(int(os.environ["DOOM_SHARDS"])):
    p = hf_hub_download("chrisxx/doom-2players-mp4", f"data/mp-0000-{i:06d}.tar",
                        repo_type="dataset", local_dir=os.environ["DOOM_SRC"])
    print("[setup-infer] got", p, flush=True)
PY
    log "Converting to MIRA layout (val-fraction 1.0 -> everything lands in test/) ..."
    python3 scripts/preprocess_doom.py \
        --src "$DOOM_SRC" --out data/doom_mira \
        --chunk-len 160 --val-fraction 1.0 --episodes-per-shard 4 --max-episodes "$DOOM_EPISODES"
fi

log "Done."
echo
echo "  play:   ./infer.sh                 -> http://<host>:3754"
