"""Summarise a closed-loop evaluation written by sim/eval_batch.sh.

For every run (result.json + trace.json) it scores sub-task milestones from the trace, then per setting:
success rate with a Wilson 95% interval, milestone rates, failure reasons, retries used, run length.
Writes OUT/summary.json and OUT/summary.md.

    python sim/eval_summary.py sim/out/eval_<stamp>        (any python3 with numpy-free stdlib is fine)

Milestones (a gripper "holds" when commanded closed < 0.25 with its state > 0.3, for >= 30 frames):
  left_grasp   left holds while the box is >= 50 mm above its start height (lifted, not just squeezed)
  right_grasp  right holds while the box is lifted
  handover     after right_grasp, the left gripper opens while the right still holds and the box stays up
  placed       final box position within 30 mm of the recorded final pose and the run was not aborted
"""

import glob
import json
import math
import os
import sys

HOLD_FRAMES = 30
LIFT_M = 0.05


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def milestones(trace, result):
    act, st, box = trace["action"], trace["state"], trace["box_centre"]
    n = min(len(act), len(st), len(box))
    y0 = box[0][1]
    run = {"L": 0, "R": 0}
    held_lifted = {"L": None, "R": None}
    handover = False
    for k in range(n):
        lifted = box[k][1] >= y0 + LIFT_M
        for side, g in (("L", 7), ("R", 15)):
            run[side] = run[side] + 1 if (act[k][g] < 0.25 and st[k][g] > 0.3) else 0
            if run[side] >= HOLD_FRAMES and lifted and held_lifted[side] is None:
                held_lifted[side] = k
        if held_lifted["R"] is not None and run["R"] > 0 and act[k][7] > 0.25 and lifted:
            handover = True
    placed = bool(result.get("placed_near_recorded_pose")) and not result.get("abort_reason")
    return {"left_grasp": held_lifted["L"] is not None, "right_grasp": held_lifted["R"] is not None,
            "handover": handover, "placed": placed,
            "left_grasp_frame": held_lifted["L"], "right_grasp_frame": held_lifted["R"]}


def main(out):
    report = {}
    for setting_dir in sorted(d for d in glob.glob(os.path.join(out, "*")) if os.path.isdir(d)):
        setting = os.path.basename(setting_dir)
        runs = []
        for rdir in sorted(glob.glob(os.path.join(setting_dir, "rollout_*"))):
            try:
                result = json.load(open(os.path.join(rdir, "result.json")))
                trace = json.load(open(os.path.join(rdir, "trace.json")))
            except (OSError, json.JSONDecodeError):
                runs.append({"run": os.path.basename(rdir), "crashed": True})
                continue
            m = milestones(trace, result)
            kinds = [e["kind"] for e in result.get("supervisor_events", [])]
            runs.append({"run": os.path.basename(rdir), "crashed": False, **m,
                         "abort_reason": result.get("abort_reason"), "frames": result.get("frames"),
                         "wall_time_s": result.get("wall_time_s"), "retries": result.get("retries"),
                         "events": kinds, "final_box_pos_error_mm": result.get("final_box_pos_error_mm")})
        n = len(runs)
        ok = [r for r in runs if not r["crashed"]]
        stats = {}
        for key in ("left_grasp", "right_grasp", "handover", "placed"):
            k = sum(r[key] for r in ok)
            lo, hi = wilson(k, n)
            stats[key] = {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}
        reasons = {}
        for r in ok:
            reason = r["abort_reason"] or ("success" if r["placed"] else "ran out of frames / not placed")
            reason = reason.split(";")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
        retried = [r for r in ok if r["retries"] and sum(r["retries"].values()) > 0]
        report[setting] = {
            "runs": n, "crashed": n - len(ok), "milestones": stats, "outcomes": reasons,
            "runs_with_retries": len(retried),
            "successes_after_retry": sum(r["placed"] for r in retried),
            "mean_frames": round(sum(r["frames"] or 0 for r in ok) / max(1, len(ok)), 1),
            "mean_wall_time_s": round(sum(r["wall_time_s"] or 0 for r in ok) / max(1, len(ok)), 1),
            "per_run": runs,
        }
    json.dump(report, open(os.path.join(out, "summary.json"), "w"), indent=2)

    lines = [f"# Closed-loop evaluation: {out}", "",
             "| setting | runs | left grasp | right grasp | handover | placed (success) | 95% CI | runs with retries | successes after retry |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, r in report.items():
        m = r["milestones"]
        cell = lambda key: f"{m[key]['k']}/{m[key]['n']}"
        lines.append(f"| {s} | {r['runs']} | {cell('left_grasp')} | {cell('right_grasp')} | {cell('handover')} | "
                     f"{cell('placed')} | {m['placed']['ci95'][0]:.2f}-{m['placed']['ci95'][1]:.2f} | "
                     f"{r['runs_with_retries']} | {r['successes_after_retry']} |")
    lines += ["", "## Outcomes"]
    for s, r in report.items():
        lines.append(f"- **{s}**: " + ", ".join(f"{k} x{v}" for k, v in sorted(r["outcomes"].items(), key=lambda kv: -kv[1])))
    lines += ["", "## Runs"]
    for s, r in report.items():
        for x in r["per_run"]:
            if x["crashed"]:
                lines.append(f"- {s} {x['run']}: crashed (no result)")
                continue
            lines.append(f"- {s} {x['run']}: L {'Y' if x['left_grasp'] else '-'} R {'Y' if x['right_grasp'] else '-'} "
                         f"hand {'Y' if x['handover'] else '-'} placed {'Y' if x['placed'] else '-'} | "
                         f"{x['frames']} frames | retries {x['retries']} | {x['abort_reason'] or ''}")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
