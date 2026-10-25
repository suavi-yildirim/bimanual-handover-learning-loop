#!/usr/bin/env python3
"""Combine reports/<name>/report.json + figures into one PDF (A4 landscape).

Usage: python make_pdf.py [--reports reports] [--out ../C2_Dataset_Intelligence_Report.pdf] [NAME ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
SEV = {"critical": "#D55E00", "warning": "#E69F00", "info": "#0072B2", "ok": "#009E73"}
VERD = {"READY": "#009E73", "CONDITIONALLY READY": "#E69F00", "NOT READY": "#D55E00"}
CAPTIONS = {
    "fig01_overview": "Episode length, share of idle frames, longest pause, and sub-task phase durations derived from gripper events.",
    "fig02_joint_trajectories": "All 14 arm joints after piecewise-linear time warping between gripper events (median, 5-95% band, 25 random episodes). A narrow band means every demonstration performs the same motion.",
    "fig03_timing_and_pauses": "(a) Gripper event times. (b) Share of episodes idle along the task. (c) Joint-speed profile; dashed line = idle threshold. (d) Action-state lag: the k with minimal error defines what the action means.",
    "fig04_motion_diversity": "(a) Pairwise RMS distance of phase-aligned trajectories. (b) Nearest-neighbour distance (redundancy). (c) PCA embedding coloured by episode verdict. (d) Components needed to explain 90% of motion variance.",
    "fig05_quality_and_safety": "(a) Outlier score = max robust z over length, idle share, jerk, distance to medoid and action rate. (b) Smoothness vs idling. (c) Joint range used relative to the sim robot joint limits (scene USD). (d) Minimum margin to a joint limit.",
    "fig06_visual_diversity": "(a) Head-camera first frames. (b) Lighting coverage. (c) Visual vs motion novelty per episode. (d) Randomisation check or brightness distribution.",
    "fig07_generator_metrics": "Organizer generator metadata: placement accuracy, tilt, slip, grip force, outcome vs lighting, and which variation factors were randomised (green) or held constant (red).",
    "fig07_coverage": "Variation factors: green = varied across episodes, red = constant.",
}

ss = getSampleStyleSheet()
S = dict(
    title=ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=26, alignment=TA_LEFT, spaceAfter=4),
    sub=ParagraphStyle("s", parent=ss["Normal"], fontSize=10, textColor=colors.HexColor("#555555")),
    h1=ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=18, spaceBefore=4, spaceAfter=6),
    h2=ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4),
    body=ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12),
    small=ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.8, leading=10),
    cap=ParagraphStyle("c", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#444444")),
    cell=ParagraphStyle("cell", parent=ss["Normal"], fontSize=8, leading=10),
    cellb=ParagraphStyle("cellb", parent=ss["Normal"], fontSize=8, leading=10, fontName="Helvetica-Bold"),
)
W = landscape(A4)[0] - 3 * cm


def P(t, st="body"):
    return Paragraph(t, S[st])


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(rows, widths, header=True, zebra=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F5F6F8")))
    t.setStyle(TableStyle(st))
    return t


def fig(path: Path, caption: str, num: int, max_h=12.5 * cm):
    w, h = PILImage.open(path).size
    width = W
    height = width * h / w
    if height > max_h:
        height = max_h
        width = height * w / h
    return KeepTogether([Image(str(path), width=width, height=height),
                         P(f"<b>Figure {num}.</b> {esc(caption)}", "cap"), Spacer(1, 8)])


def badge(verdict):
    t = Table([[P(f"<font color='white'><b>{verdict}</b></font>", "body")]])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(VERD[verdict])),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    t.hAlign = "LEFT"
    return t


def dataset_section(name: str, rep: dict, fig_dir: Path, fignum: list[int]):
    s = rep["summary"]
    out = [P(f"Dataset: {esc(name)}", "h1"), badge(rep["verdict"]), Spacer(1, 4),
           P(esc(rep["verdict_reason"])),
           P(f"Task: <i>&ldquo;{esc('; '.join(s['tasks']))}&rdquo;</i> &middot; source <font face='Courier'>{esc(s['source'])}</font>"
             f" &middot; cameras: {esc(', '.join(c.split('.')[-1] for c in s['cameras']))}", "small"), Spacer(1, 6)]
    kv = [("Episodes", s["episodes"]), ("Frames", f"{s['frames']:,}"), ("Rate", f"{s['fps']:g} Hz"),
          ("Total time", f"{s['hours'] * 60:.1f} min"), ("Distinct motions", s["distinct_motions"]),
          ("Idle frames", f"{s['idle_frac_mean']:.0%}"), ("Action-state lag", f"{s['action_state_lag']} fr."),
          ("Keep / review / drop", f"{s['keep']} / {s['review']} / {s['drop']}")]
    out.append(table([[P(k, "small") for k, _ in kv], [P(f"<b>{v}</b>", "body") for _, v in kv]],
                     [W / len(kv)] * len(kv), header=False, zebra=False))
    out.append(P("Findings", "h2"))
    rows = [["Severity", "Finding", "Recommended fix"]]
    for f in rep["findings"]:
        fix = esc(f["fix"]) + (" <i>(auto-fix in manifest)</i>" if f["auto_fixable"] and f["fix"] else "")
        rows.append([P(f"<font color='{SEV[f['severity']]}'><b>{f['severity'].upper()}</b></font>", "cell"),
                     P(f"<b>{esc(f['title'])}</b><br/>{esc(f['detail'])}", "cell"), P(fix, "cell")])
    out.append(table(rows, [2.2 * cm, W * 0.58, W - 2.2 * cm - W * 0.58]))
    left = [["", "VLA readiness check", "Detail"]] + [
        [P("<font color='#009E73'>&#10004;</font>" if c["ok"] else "<font color='#D55E00'>&#10008;</font>", "cell"),
         P(esc(c["item"]), "cell"), P(esc(c["detail"]), "cell")] for c in rep["checklist"]]
    right = [["Prio", "Data plan (next loop iteration)"]] + [
        [P(f"<b>{d['priority']}</b>", "cell"), P(f"<b>{esc(d['action'])}</b>: {esc(d['detail'])}", "cell")]
        for d in rep["data_plan"]]
    half = (W - 0.6 * cm) / 2
    two = Table([[table(left, [0.6 * cm, half * 0.45, half * 0.55 - 0.6 * cm]),
                  table(right, [1.0 * cm, half - 1.0 * cm])]], colWidths=[half + 0.3 * cm, half + 0.3 * cm])
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    out += [Spacer(1, 6), KeepTogether([P("Readiness checklist and data plan", "h2"), two])]
    out.append(PageBreak())
    out.append(P(f"Figures: {esc(name)}", "h1"))
    for f in rep["figures"]:
        p = fig_dir / f
        if p.exists():
            fignum[0] += 1
            out.append(fig(p, CAPTIONS.get(p.stem, ""), fignum[0]))
    out.append(PageBreak())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--reports", default=str(HERE / "reports"))
    ap.add_argument("--summary-only", nargs="*", default=[], help="datasets listed in the summary table only")
    ap.add_argument("--out", default=str(HERE.parent / "C2_Dataset_Intelligence_Report.pdf"))
    a = ap.parse_args(argv)
    root = Path(a.reports)
    names = a.names or sorted(p.parent.name for p in root.glob("*/report.json"))
    reps = {n: json.loads((root / n / "report.json").read_text()) for n in names}

    story = [P("Dataset Intelligence Report", "title"),
             P("Siemens Healthineers Hackathon 2026 &middot; Challenge 2 &ldquo;Cathlab Assistant Robot&rdquo; &middot; "
               f"Stage 02 of the learning loop &middot; generated {dt.date.today().isoformat()}", "sub"),
             Spacer(1, 10),
             P("<b>Question (from the challenge slides):</b> Is the dataset ready for training? Which episodes are "
               "invalid, redundant or unrepresentative? What is still missing? Is it VLA-ready? "
               "This report answers these automatically with <font face='Courier'>dataset_check</font>; the same "
               "command runs on every new dataset, and its JSON output feeds the next loop iteration "
               "(episode selection, held-out split, data generation plan)."),
             P("Summary", "h2")]
    rows = [["Dataset", "Verdict", "Episodes", "Frames @ rate", "Motions", "Idle", "Lag", "Keep/rev./drop",
             "Top issue"]]
    for n, r in reps.items():
        s = r["summary"]
        top = next((f["title"] for f in r["findings"] if f["severity"] == "critical"),
                   next((f["title"] for f in r["findings"] if f["severity"] == "warning"), "-"))
        rows.append([P(f"<b>{esc(n)}</b>", "cell"),
                     P(f"<font color='{VERD[r['verdict']]}'><b>{r['verdict']}</b></font>", "cell"),
                     s["episodes"], f"{s['frames']:,} @ {s['fps']:g} Hz", s["distinct_motions"],
                     f"{s['idle_frac_mean']:.0%}", s["action_state_lag"], f"{s['keep']}/{s['review']}/{s['drop']}",
                     P(esc(top), "cell")])
    story.append(table(rows, [5.2 * cm, 3.2 * cm, 1.6 * cm, 3.3 * cm, 2.2 * cm, 1.2 * cm, 1.0 * cm, 2.4 * cm,
                              W - 20.1 * cm]))
    story += [P("Key findings", "h2")]
    bullets = [
        "<b>Handover (the task that matches the Isaac Sim scene) contains a single motion.</b> All 84 episodes are "
        "identical in joint space (nearest-neighbour distance 0); start pose, box pose (dx, dz, yaw) and outcome "
        "metrics (placement offset, tilt, slip, grip force) are bit-identical. Only lighting varies, and the image "
        "statistics confirm it (requested colour temperature vs measured R/B ratio: Spearman &rho; = -1.00). "
        "Expected consequence: a policy trained on it will fail when the box moves. This is the v0 -&gt; v1 test.",
        "<b>Cube set (rand500) is diverse:</b> 490 distinct motions, strong visual randomisation (table colour, texture, "
        "lighting), ~5 principal components explain 90% of motion variance. Blocking bug: "
        "<font face='Courier'>splits = 0:41</font> would make loaders use 41 of 490 episodes.",
        "<b>All sets:</b> success label is True on every frame (no failure data), one language instruction, no held-out "
        "split, 17-27% idle frames, identical robot start pose. Sub-tasks are recovered reliably from gripper events "
        "(100% of episodes follow the canonical sequence), giving progress labels for evaluation.",
        "<b>Action semantics differ:</b> rand500 action = next state (absolute joint target, error 0); handover action "
        "leads the measured state by 4 frames, and the gripper command has 3 levels (open / pre-shape 0.48 / closed).",
        "<b>Safety:</b> no episode leaves the sim robot's joint limits (taken from the scene USD); our minimum "
        "margins (left 13.07 deg, right 37.70 deg) match the generator's metadata exactly.",
    ]
    story += [P("&bull; " + b) for b in bullets]
    if a.summary_only:
        story.append(P("Detailed sections omitted for: " + ", ".join(esc(n) for n in a.summary_only)
                       + " (same content as a larger set; full HTML report in dataset_check/reports/).", "small"))
    comp = root / "_comparison" / "fig00_dataset_comparison.png"
    fignum = [0]
    if comp.exists():
        fignum[0] += 1
        story += [Spacer(1, 2), fig(comp, "Dataset size versus effective diversity. Panel (a) is log scale: the "
                                    "handover sets have 1 distinct motion.", fignum[0], max_h=3.9 * cm)]
    story.append(PageBreak())
    for n, r in reps.items():
        if n in a.summary_only:
            continue
        story += dataset_section(n, r, root / n / "figures", fignum)

    story += [P("Method", "h1")]
    meth = [
        ("Input", "LeRobot v3.0 read directly from .zip or folder (parquet + meta + AV1 video)."),
        ("Schema", "info.json counts vs data, split coverage, channel names, timestamps, NaN/Inf, referenced video "
                   "files, norm stats vs data mean."),
        ("Sub-tasks", "Gripper grasp/release events by hysteresis thresholds near the closed level; canonical event "
                      "sequence = most frequent; phase durations between events."),
        ("Idle", "Max arm-joint speed < 0.02 rad/s and gripper command unchanged; pauses reported if >= 1 s."),
        ("Action semantics", "Lag k minimising RMS(action[t] - state[t+k]) over arm joints, k = 0..8."),
        ("Motion diversity", "Trajectories time-warped piecewise-linearly between gripper events (25 points per "
                             "phase); pairwise RMS distance; nearest neighbour; duplicate clusters at < 1e-3 rad; PCA."),
        ("Outliers", "Robust z (median/MAD) on length, idle share, RMS jerk, distance to medoid, peak action rate; "
                     "|z| > 3.5 = review. Hard rules: NaN/Inf or outside joint limits (scene USD) = drop; velocity limit, "
                     "unusual event sequence = review."),
        ("Visual", "Head-camera frame at start and middle of every episode; luminance, R/B colour ratio, 16x16 "
                   "grey-level distance; check of requested lighting vs measured image statistics."),
        ("Lineage", "manifest.json: sha256 of every data file, parameters, seeded 10% validation split, "
                    "per-episode verdict, idle runs to trim."),
        ("Limits", "Joint limits from the scene USD (fr3v2); velocity limits from the FR3 datasheet. Object poses are "
                   "not stored in the cube set, so diversity is measured from robot motion only."),
    ]
    story.append(table([["Step", "Description"]] + [[P(f"<b>{a}</b>", "cell"), P(esc(b), "cell")] for a, b in meth],
                       [3.5 * cm, W - 3.5 * cm]))
    story += [Spacer(1, 8), P("Reproduce: <font face='Courier'>python dataset_check/check_dataset.py &lt;dataset.zip&gt; "
                              "...</font> then <font face='Courier'>python dataset_check/make_pdf.py</font>.", "small")]

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawString(1.5 * cm, 0.8 * cm, "C2 Cathlab Assistant Robot - Dataset Intelligence (Stage 02)")
        c.drawRightString(landscape(A4)[0] - 1.5 * cm, 0.8 * cm, f"page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(a.out, pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.4 * cm, title="Dataset Intelligence Report",
                            author="Team C2")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
