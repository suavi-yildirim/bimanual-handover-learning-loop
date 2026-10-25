#!/usr/bin/env bash
# Closed-loop comparison of policy checkpoints on identical box poses: raw policy, real-time rendering.
# Usage: sim/eval_models.sh OUT N_NORMAL N_WIDE NAME=CONFIG:DIR [NAME=CONFIG:DIR ...]
#   e.g. sim/eval_models.sh sim/out/eval_v0_v1 8 4 \
#          v0=pi05_fr3duo_6a5f7c87:checkpoints/fr3duo_6a5f7c87 v1=pi05_fr3duo_v1:checkpoints/pi05_fr3duo_v1/v1/2999
# Seeds: normal set 20000.. (box +-15 mm, the expert data's range), wide set 21000.. (+-25 mm). Never used for data.
# Each model gets its own policy server on port 8000 (stopped before the next). Then: python3 sim/eval_summary.py OUT
set -u
cd "$(dirname "$0")/.."
OUT=$1; NN=$2; NW=$3; shift 3
PY=${ISAAC_PY:-$HOME/isaac-env-6/bin/python}
mkdir -p "$OUT"
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}; cfg=${rest%%:*}; dir=${rest#*:}
    echo "[eval] $(date +%T) $name: starting policy server ($cfg, $dir)"
    (cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec ~/.local/bin/uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config="$cfg" --policy.dir="$dir") > "$OUT/server_$name.log" 2>&1 &
    srv=$!
    until ss -ltn | grep -q ":8000 "; do
        kill -0 $srv 2>/dev/null || { echo "[eval] $name: server died, see $OUT/server_$name.log"; continue 2; }
        sleep 3
    done
    for set_ in normal wide; do
        if [ $set_ = normal ]; then n=$NN; base=20000; range=15; else n=$NW; base=21000; range=25; fi
        for i in $(seq 0 $((n - 1))); do
            seed=$((base + i))
            log="$OUT/$name/${set_}_$seed.log"; mkdir -p "$OUT/$name"
            PYTHONPATH=${ISAAC_PYDEPS:-$HOME/.isaac-pydeps} $PY sim/policy_rollout.py --headless --raw --max-frames 2300 \
                --seed $seed --box-range-mm $range --out "$OUT/$name" > "$log" 2>&1
            echo "[eval] $(date +%T) $name $set_ seed $seed: $(grep -oE '"(abort_reason|placed_near_recorded_pose)": [^,]*' "$log" | tr '\n' ' ')"
        done
    done
    kill $srv; wait $srv 2>/dev/null
    until ! ss -ltn | grep -q ":8000 "; do sleep 1; done
done
echo "[eval] done"
