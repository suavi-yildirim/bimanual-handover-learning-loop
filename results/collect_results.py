"""Collect every run in sim/out into one index and one success-rate table per model.

    python3 results/collect_results.py            (run from the repo root, stdlib only)

Writes results/runs_index.csv (one row per run, with video path), results/success_rates.json and
results/SUCCESS_RATES.md. Re-run after every new evaluation: runs under sim/out/eval_v0_v1/<model>/ are
picked up automatically (that folder is written by sim/eval_models.sh).

Model attribution:
  expert       sim/out/expert_demos*/, expert_test   (scripted IK expert in Isaac Sim, meta.json "success")
  <name>       sim/out/eval_v0_v1/<name>/            (controlled eval: raw policy, seeded box poses)
  v0           every other rollout_* folder          (provided fine-tuned checkpoint 6a5f7c87, 23 Oct
                                                      development runs, before v1 existed; mixed settings)
  base         no closed-loop runs; offline only (pi05_base, arm MAE 6.79 deg vs 0.39 deg fine-tuned)
Milestones and success use sim/eval_summary.py (success = "placed": box within 30 mm of the recorded
final pose and the run not aborted).
"""

import csv
import glob
import importlib.util
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "sim", "out")
HERE = os.path.join(ROOT, "results")

spec = importlib.util.spec_from_file_location("eval_summary", os.path.join(ROOT, "sim", "eval_summary.py"))
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)

MILESTONES = ("left_grasp", "right_grasp", "handover", "placed")
SPLIT = json.load(open(os.path.join(OUT, "expert_split_v1.json")))


def rel(p):
    return os.path.relpath(p, ROOT) if p else ""


def first_video(d):
    for name in ("video.mp4", "external.mp4"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    vids = sorted(glob.glob(os.path.join(d, "*.mp4")))
    return vids[0] if vids else None


def settings_of(result):
    s = []
    if result.get("supervisor"):
        s.append(f"supervisor(retries={result.get('max_retries')})")
    if result.get("align_assist"):
        s.append("align-assist")
    if result.get("open_approach"):
        s.append("open-approach")
    if result.get("box_lifted_mm_at_start"):
        s.append(f"box-lift {result['box_lifted_mm_at_start']:.0f}mm")
    return " + ".join(s) or "raw"


def policy_runs():
    rows = []
    for rdir in sorted(glob.glob(os.path.join(OUT, "**", "rollout_*"), recursive=True)):
        if not os.path.isdir(rdir):
            continue
        parts = os.path.relpath(rdir, OUT).split(os.sep)
        group = parts[0]
        if group == "eval_v0_v1":
            model, protocol = parts[1], "controlled (raw policy, seeded box)"
        else:
            model, protocol = "v0", "development (mixed settings)"
        row = {"model": model, "protocol": protocol, "group": "/".join(parts[:-1]), "run": parts[-1],
               "video": rel(first_video(rdir)), "folder": rel(rdir)}
        seed = None
        for log in glob.glob(os.path.join(os.path.dirname(rdir), "*_2*.log")):
            m = re.search(r"_(\d{5})\.log$", log)
            seed = m.group(1) if m else seed
        row["seed"] = seed or ""
        try:
            result = json.load(open(os.path.join(rdir, "result.json")))
            trace = json.load(open(os.path.join(rdir, "trace.json")))
        except (OSError, json.JSONDecodeError):
            row.update(status="crashed / no result", settings="", **{k: "" for k in MILESTONES})
            rows.append(row)
            continue
        if "action" not in trace:  # earliest run: trace without state/action, milestones not scorable
            m = {"left_grasp": False, "right_grasp": False, "handover": False,
                 "placed": bool(result.get("placed_near_recorded_pose")) and not result.get("abort_reason")}
        else:
            m = es.milestones(trace, result)
        row.update({k: m[k] for k in MILESTONES})
        row.update(status="success" if m["placed"] else (result.get("abort_reason") or "ran out of frames / not placed").split(";")[0],
                   settings=settings_of(result), frames=result.get("frames"),
                   final_box_err_mm=result.get("final_box_pos_error_mm"),
                   latency_p50_ms=(result.get("latency_ms") or {}).get("p50"),
                   latency_p95_ms=(result.get("latency_ms") or {}).get("p95"))
        rows.append(row)
    return rows


def expert_runs():
    rows = []
    metas = sorted(glob.glob(os.path.join(OUT, "expert_demos*", "*", "*", "ep_*", "meta.json"))) + \
        sorted(glob.glob(os.path.join(OUT, "expert_test", "ep_*", "meta.json")))
    for f in metas:
        meta = json.load(open(f))
        ep = os.path.dirname(f)
        batch = rel(os.path.dirname(ep)) if "expert_test" not in f else rel(os.path.dirname(ep))
        batch_key = os.path.relpath(os.path.dirname(ep), ROOT) if "expert_test" not in f else "sim/out/expert_test"
        if "DISCARDED" in f:
            split = "excluded (corrupted, two writers)"
        elif batch_key in SPLIT["train"]:
            split = "train"
        elif batch_key in SPLIT["val"]:
            split = "val (held out)"
        else:
            split = "excluded (failed)"
        off = meta.get("box_offset_dx_dy_dz_yaw") or [0, 0, 0, 0]
        ok = bool(meta.get("success"))
        rows.append({"model": "expert", "protocol": "data generation (box offset +-15 mm, +-5 deg)",
                     "group": batch, "run": os.path.basename(ep), "seed": "",
                     "video": rel(first_video(ep)), "folder": rel(ep),
                     "status": "success" if ok else "failed", "settings": f"render {'rt' if '_rt' in f else 'pt'}; {split}",
                     "left_grasp": "", "right_grasp": "", "handover": "", "placed": ok,
                     "frames": meta.get("frames"), "final_box_err_mm": "",
                     "box_dx_mm": round(off[0] * 1000, 1), "box_dz_mm": round(off[2] * 1000, 1),
                     "box_yaw_deg": round(off[3], 1)})
    return rows


def scored(rows):
    """Runs that produced a result: crashed runs and the corrupted expert batch are listed, not scored."""
    return [r for r in rows if r.get("placed") not in ("", None) and "DISCARDED" not in r["group"]]


def rate(rows, key):
    ok = scored(rows)
    n = len(ok)
    k = sum(1 for r in ok if r[key])
    lo, hi = es.wilson(k, n)
    return {"k": k, "n": n, "ci95": [round(lo, 2), round(hi, 2)]}


def main():
    rows = expert_runs() + policy_runs()
    fields = ["model", "protocol", "group", "run", "seed", "settings", "status", "left_grasp", "right_grasp",
              "handover", "placed", "frames", "final_box_err_mm", "box_dx_mm", "box_dz_mm", "box_yaw_deg",
              "latency_p50_ms", "latency_p95_ms", "video", "folder"]
    with open(os.path.join(HERE, "runs_index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for r in rows:
        summary.setdefault((r["model"], r["protocol"]), []).append(r)
    out = {}
    md = ["# Success rates per model (auto-generated by results/collect_results.py, do not edit by hand)", "",
          "Success = box placed within 30 mm of the recorded final pose, run not aborted. 95% CI = Wilson.",
          "n = scored runs. Crashed runs (sim error, no result.json) are not counted; see column 'crashed'.", "",
          "| model | protocol | runs scored | crashed | left grasp | right grasp | handover | **placed (success)** | 95% CI |",
          "|---|---|---|---|---|---|---|---|---|",
          "| base (pi05_base, zero-shot) | offline only, 9 val episodes | 0 | - | - | - | - | - | arm MAE 6.79 deg, grip MAE 0.248 |"]
    order = {"expert": 0, "v0": 1}
    for (model, protocol), rs in sorted(summary.items(), key=lambda kv: (order.get(kv[0][0], 2), kv[0][0], kv[0][1])):
        st = {k: rate(rs, k) for k in MILESTONES}
        crashed = len(rs) - len(scored(rs))
        out[f"{model} | {protocol}"] = {"runs_scored": len(scored(rs)), "crashed_or_excluded": crashed, **st}
        cell = lambda k: "-" if model == "expert" and k != "placed" else f"{st[k]['k']}/{st[k]['n']}"
        md.append(f"| {model} | {protocol} | {len(scored(rs))} | {crashed} | {cell('left_grasp')} | {cell('right_grasp')} | "
                  f"{cell('handover')} | **{st['placed']['k']}/{st['placed']['n']}** | "
                  f"{st['placed']['ci95'][0]:.2f}-{st['placed']['ci95'][1]:.2f} |")
    if not any(m not in ("expert", "v0") for m, _ in summary):
        md.append("| v1 | controlled | 0 | 0 | - | - | - | - | not evaluated yet |")
    md += ["", "Expert: the DISCARDED batch (two writers, corrupted) is not scored.", "",
           "## v0 development runs by setting", "",
           "| folder | settings | runs | left grasp | right grasp | handover | placed |", "|---|---|---|---|---|---|---|"]
    dev = {}
    for r in rows:
        if r["model"] == "v0":
            dev.setdefault((r["group"], r["settings"]), []).append(r)
    for (g, s), rs in sorted(dev.items()):
        md.append(f"| {g} | {s or 'no result'} | {len(rs)} | " +
                  " | ".join(f"{rate(rs, k)['k']}/{rate(rs, k)['n']}" for k in MILESTONES) + " |")
    md += ["", "## Best run per model (most milestones, then smallest final box error)", "",
           "Copied to PRESENTATION_VIDEOS/auto/<model>_best.mp4 (refreshed on every run of this script).", "",
           "| model | run | milestones | status | video |", "|---|---|---|---|---|"]
    auto = os.path.join(ROOT, "PRESENTATION_VIDEOS", "auto")
    os.makedirs(auto, exist_ok=True)
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in scored([r for r in rows if r["model"] == model]) if r["video"]]
        if not rs:
            continue
        best = max(rs, key=lambda r: (sum(bool(r[k]) for k in MILESTONES), -(r.get("final_box_err_mm") or 0)))
        got = "placed" if model == "expert" else ", ".join(k for k in MILESTONES if best[k]) or "none"
        md.append(f"| {model} | {best['folder']} | {got} | {best['status']} | {best['video']} |")
        shutil.copy2(os.path.join(ROOT, best["video"]), os.path.join(auto, f"{model}_best.mp4"))
    json.dump(out, open(os.path.join(HERE, "success_rates.json"), "w"), indent=2)
    open(os.path.join(HERE, "SUCCESS_RATES.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
