#!/usr/bin/env bash
# Data loop: generate expert demos with W parallel workers, one Isaac Sim process per episode.
# Usage: sim/gen_demos.sh [WORKERS=3] [EPISODES_PER_WORKER=8] [OUT=sim/out/expert_demos] [SEED_BASE=0] [EXTRA expert_demos.py args...]
# Worker w writes OUT/w<w>/batch_NN (seed = SEED_BASE + 1000*w + episode), live view on port 8090+w.
set -u
cd "$(dirname "$0")/.."
W=${1:-3}
N=${2:-8}
OUT=${3:-sim/out/expert_demos}
SEED_BASE=${4:-0}
shift 4 2>/dev/null || shift $#
EXTRA=("$@")
PY=${ISAAC_PY:-$HOME/isaac-env-6/bin/python}
mkdir -p "$OUT"
for w in $(seq 0 $((W - 1))); do
    (
        for e in $(seq 0 $((N - 1))); do
            dir="$OUT/w$w/batch_$(printf %02d "$e")"; mkdir -p "$OUT/w$w"
            PYTHONPATH=${ISAAC_PYDEPS:-$HOME/.isaac-pydeps} $PY sim/expert_demos.py --episodes 1 --seed $((SEED_BASE + 1000 * w + e)) \
                --stream-port $((8090 + w)) --out "$dir" "${EXTRA[@]}" > "$dir.log" 2>&1 || true
            echo "[gen] $(date +%T) worker $w episode $e: $(grep -oE '\[expert\] ep 0: (SUCCESS|FAILED)[^|]*\|[^|]*\|[^|]*' "$dir.log" | head -1)"
        done
    ) &
    sleep 20  # stagger the Isaac Sim start-ups
done
wait
echo "[gen] done"
