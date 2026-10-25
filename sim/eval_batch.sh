#!/usr/bin/env bash
# Closed-loop evaluation: N episodes per setting, settings interleaved (A B C A B C ...), one sim process each.
#   retry0:      failure detection only (abort on failure), learned pre-shape on approach
#   retry2:      supervisor with up to 2 retries per arm, learned pre-shape on approach
#   retry2_open: supervisor with retries, grippers fully open on approach (--open-approach)
# Usage: sim/eval_batch.sh [N=10] [OUT=sim/out/eval_<stamp>]   then: python sim/eval_summary.py OUT
set -u
cd "$(dirname "$0")/.."
N=${1:-10}
OUT=${2:-sim/out/eval_$(date +%Y%m%d_%H%M)}
PY=${ISAAC_PY:-$HOME/isaac-env-6/bin/python}
mkdir -p "$OUT"
echo "[eval] $N episodes per setting -> $OUT"
for i in $(seq 1 "$N"); do
    for setting in retry0 retry2 retry2_open; do
        case $setting in
            retry0) flags="--max-retries 0 --no-open-approach" ;;
            retry2) flags="--max-retries 2 --no-open-approach" ;;
            retry2_open) flags="--max-retries 2 --open-approach" ;;
        esac
        log="$OUT/$setting/run_$(printf %02d "$i").log"
        mkdir -p "$OUT/$setting"
        echo "[eval] $(date +%T) $setting run $i/$N"
        PYTHONPATH=${ISAAC_PYDEPS:-$HOME/.isaac-pydeps} $PY sim/policy_rollout.py --headless --max-frames 2300 \
            $flags --out "$OUT/$setting" > "$log" 2>&1
        echo "[eval] $(date +%T) $setting run $i/$N exit $? : $(grep -oE '"(abort_reason|placed_near_recorded_pose)": [^,]*' "$log" | tr '\n' ' ')"
    done
done
echo "[eval] done"
