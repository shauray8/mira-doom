#!/usr/bin/env bash
# Run the world model as a browser-playable game. Defaults are the measured ones (docs/inference.md).
#
#   ./infer.sh                                  http://<host>:3754
#   MIRA_STAGE=wm_long MIRA_CTX=78 ./infer.sh   2x memory horizon, weaker action control
#   MIRA_STEPS=5 ./infer.sh                     cheaper frames
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Fail early and legibly, rather than 90 seconds into a model load.
[ -d ckpts/doom_codec ] || { echo "no ckpts/doom_codec — run ./setup_inference.sh first" >&2; exit 1; }
[ -f data/doom_mira/test/index.json ] || { echo "no seed clips at data/doom_mira/test — run ./setup_inference.sh first" >&2; exit 1; }

export MIRA_HOST=${MIRA_HOST:-0.0.0.0}
export MIRA_PORT=${MIRA_PORT:-3754}
export MIRA_STAGE=${MIRA_STAGE:-wm}          # wm (action control) | wm_long (2x memory, use MIRA_CTX=78)
export MIRA_STEPS=${MIRA_STEPS:-6}           # diffusion steps per frame
export MIRA_NOISE=${MIRA_NOISE:-0.0}         # context noise; 0 is optimal at 6 steps (0.45 is for 2)
export MIRA_WORLDKV=${MIRA_WORLDKV:-0}       # 1 = episodic memory; measured no better than off (docs §5d)

echo "[infer] stage=$MIRA_STAGE steps=$MIRA_STEPS ctx=${MIRA_CTX:-default} worldkv=$MIRA_WORLDKV"
echo "[infer] first run compiles the decoder and captures a CUDA graph (~2 min; ~20 s once cached)"
echo "[infer] -> http://localhost:$MIRA_PORT"
exec python3 serve/serve_local.py
