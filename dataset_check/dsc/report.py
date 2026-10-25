"""Write report.json, episodes.csv, manifest.json and a self-contained-folder report.html."""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

import numpy as np

from .checks import Result
from .io_lerobot import Dataset

SEV_ORDER = {"critical": 0, "warning": 1, "info": 2, "ok": 3}
SEV_COL = {"critical": "#D55E00", "warning": "#E69F00", "info": "#0072B2", "ok": "#009E73"}
VERDICT_COL = {"READY": "#009E73", "CONDITIONALLY READY": "#E69F00", "NOT READY": "#D55E00"}

CAPTIONS = {
    "fig01_overview": "Episode length, share of idle frames, longest pause, and sub-task phase durations derived from gripper events.",
    "fig02_joint_trajectories": "All 14 arm joints after piecewise-linear time warping between gripper events. A narrow band means every demonstration does the same motion.",
    "fig03_timing_and_pauses": "(a) When gripper events happen. (b) Share of episodes idle at each point of the task. (c) Joint-speed profile. (d) Action-state lag: the lag k with minimal error tells what the action means.",
    "fig04_motion_diversity": "Pairwise distance between phase-aligned trajectories, nearest-neighbour distances (redundancy), PCA embedding coloured by episode verdict, and how many components explain 90% of the motion variance.",
    "fig05_quality_and_safety": "Outlier score (max robust z over length, idle share, jerk, distance to medoid, action rate), smoothness, joint range used relative to the sim robot's joint limits, and minimum safety margin.",
    "fig06_visual_diversity": "Head-camera first frames, lighting coverage (luminance vs colour balance), visual vs motion novelty per episode, and a check that the requested randomisation shows up in the images.",
    "fig07_generator_metrics": "Generator metadata: placement accuracy, tilt, slip, grip force, outcome vs lighting, and which variation factors were actually randomised.",
    "fig07_coverage": "Which variation factors change across episodes.",
    "fig00_dataset_comparison": "Size versus effective diversity across datasets.",
}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def write(ds: Dataset, r: Result, out: Path, figs: list[str]) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().isoformat(timespec="seconds")
    findings = sorted(r.findings, key=lambda f: SEV_ORDER[f.severity])

    rep = dict(generated=now, tool="dataset_check 0.1", summary=r.summary, verdict=r.verdict,
               verdict_reason=r.verdict_reason, findings=[f.__dict__ for f in findings],
               checklist=[dict(item=a, ok=bool(b), detail=c) for a, b, c in r.checklist],
               coverage=r.coverage, data_plan=[dict(priority=a, action=b, detail=c) for a, b, c in r.data_plan],
               params=r.params.__dict__, figures=figs)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=_jsonable))

    r.episodes.to_csv(out / "episodes.csv")

    ep = r.episodes
    manifest = dict(
        generated=now, dataset=ds.name, lineage=r.lineage,
        fixes=[dict(id=f.id, fix=f.fix) for f in findings if f.auto_fixable and f.fix],
        episodes={int(e): dict(verdict=row.verdict, reason=row.reason, motion_cluster=int(row.motion_cluster),
                               split="val" if int(e) in r.lineage["split"]["val"] else "train")
                  for e, row in ep.iterrows()},
        trim_idle={int(e): row.pause_starts for e, row in ep.iterrows() if row.pause_starts},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_jsonable))

    s = r.summary
    esc = html.escape

    def kv(k, v):
        return f"<div class='kv'><span>{esc(k)}</span><b>{esc(str(v))}</b></div>"

    cards = "".join([
        kv("Episodes", s["episodes"]), kv("Frames", f"{s['frames']:,}"), kv("Rate", f"{s['fps']:g} Hz"),
        kv("Total time", f"{s['hours'] * 60:.1f} min"), kv("Distinct motions", s["distinct_motions"]),
        kv("Idle frames", f"{s['idle_frac_mean']:.0%}"), kv("Action-state lag", f"{s['action_state_lag']} frame(s)"),
        kv("Keep / review / drop", f"{s['keep']} / {s['review']} / {s['drop']}"),
    ])
    frows = "".join(
        f"<tr><td><span class='sev' style='background:{SEV_COL[f.severity]}'>{f.severity}</span></td>"
        f"<td><b>{esc(f.title)}</b><br><small>{esc(f.detail)}</small></td>"
        f"<td><small>{esc(f.fix)}{' <i>(auto-fix in manifest)</i>' if f.auto_fixable and f.fix else ''}</small></td></tr>"
        for f in findings)
    crow = "".join(f"<tr><td>{'&#10004;' if b else '&#10008;'}</td><td>{esc(a)}</td><td><small>{esc(c)}</small></td></tr>"
                   for a, b, c in r.checklist)
    prow = "".join(f"<tr><td><b>{esc(a)}</b></td><td>{esc(b)}</td><td><small>{esc(c)}</small></td></tr>"
                   for a, b, c in r.data_plan)
    figs_html = "".join(
        f"<figure><img src='figures/{esc(f)}' alt='{esc(f)}'><figcaption><b>Figure {i + 1}.</b> "
        f"{esc(CAPTIONS.get(f[:-4], ''))} <a href='figures/{esc(f[:-4])}.pdf'>PDF</a></figcaption></figure>"
        for i, f in enumerate(figs))
    flagged = ep[ep.verdict != "keep"].sort_values("outlier_score", ascending=False).head(30)
    erow = "".join(f"<tr><td>{e}</td><td>{row.verdict}</td><td>{row.outlier_score:.1f}</td><td><small>{esc(row.reason)}</small></td></tr>"
                   for e, row in flagged.iterrows()) or "<tr><td colspan=4>none</td></tr>"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dataset Check: {esc(ds.name)}</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#e3e3e3;--card:#f6f7f9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#ececec;--mut:#9a9a9a;--line:#33353a;--card:#202227}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;max-width:1150px;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}
.mut{{color:var(--mut)}} .verdict{{display:inline-block;color:#fff;font-weight:700;padding:6px 12px;border-radius:6px;margin:10px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}}
.kv{{background:var(--card);border-radius:6px;padding:8px 10px}} .kv span{{display:block;color:var(--mut);font-size:12px}} .kv b{{font-size:17px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}}
.sev{{color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;text-transform:uppercase}}
figure{{margin:16px 0}} figure img{{width:100%;background:#fff;border-radius:4px}} figcaption{{font-size:12.5px;color:var(--mut)}}
.wrap{{overflow-x:auto}} a{{color:#0a84ff}}
</style></head><body>
<h1>Dataset Check: {esc(ds.name)}</h1>
<div class="mut">Stage 02 Dataset Intelligence &middot; source <code>{esc(s['source'])}</code> &middot; LeRobot {esc(str(s['codebase_version']))} &middot; generated {now}</div>
<div class="verdict" style="background:{VERDICT_COL[r.verdict]}">{r.verdict}</div>
<div>{esc(r.verdict_reason)}</div>
<p class="mut">Task: &ldquo;{esc('; '.join(s['tasks']))}&rdquo; &middot; cameras: {esc(', '.join(c.split('.')[-1] for c in s['cameras']))}</p>
<div class="grid">{cards}</div>
<h2>Findings</h2><div class="wrap"><table><tr><th></th><th>Finding</th><th>Fix</th></tr>{frows}</table></div>
<h2>VLA readiness checklist</h2><div class="wrap"><table>{crow}</table></div>
<h2>Data plan (input to the next loop iteration)</h2><div class="wrap"><table><tr><th>Priority</th><th>Action</th><th>Detail</th></tr>{prow}</table></div>
<h2>Figures</h2>{figs_html}
<h2>Flagged episodes (top 30)</h2><div class="wrap"><table><tr><th>Episode</th><th>Verdict</th><th>Score</th><th>Reason</th></tr>{erow}</table></div>
<h2>Files</h2><p><a href="report.json">report.json</a> (machine-readable) &middot; <a href="episodes.csv">episodes.csv</a> (per-episode metrics) &middot;
<a href="manifest.json">manifest.json</a> (keep/review/drop, seeded split, data hashes, fixes)</p>
<p class="mut">Thresholds: idle speed {r.params.idle_speed} rad/s, duplicate RMS {r.params.dup_threshold} rad, robust z {r.params.robust_z}. Joint limits from the scene USD (fr3v2).</p>
</body></html>"""
    (out / "report.html").write_text(page)
    return dict(html=out / "report.html", json=out / "report.json", csv=out / "episodes.csv",
                manifest=out / "manifest.json")
