#!/usr/bin/env bash
# Steps D1-D2 of C2_pi05_Training_Process.md: norm stats, then LoRA training, detached with a log.
# Usage: pi05/train_v0.sh [exp_name] [extra train.py args, e.g. --num-train-steps=20]
set -euo pipefail

CONFIG=pi05_fr3duo_handover_lora
EXP="${1:-v0}"
shift || true
OPENPI="${OPENPI:-$HOME/openpi}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
LOG="$OPENPI/checkpoints/$CONFIG/$EXP.log"

cd "$OPENPI"
mkdir -p "checkpoints/$CONFIG"
# Recompute every time: stats depend on data and config (never reuse the cube set's norm_stats.json).
"$UV" run scripts/compute_norm_stats.py --config-name "$CONFIG"

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" setsid nohup \
    "$UV" run scripts/train.py "$CONFIG" --exp-name="$EXP" --overwrite "$@" > "$LOG" 2>&1 < /dev/null &
echo "[train] started $CONFIG/$EXP (pid $!), log: $LOG"
