"""Presentation figures for Challenge 2 (Teach the Robot), Siemens Healthineers colour code.

Run with the Isaac env python (has matplotlib, numpy, PIL):
    $ISAAC_PY presentation/make_figures.py [--only F07]
Writes presentation/figures/<name>.png (200 dpi, 16:9) and .pdf.
All numbers come from sim/out, dataset_check/reports and presentation/data; nothing is typed in by hand
except values documented in CLAUDE.md / pi05/README.md (marked SRC_DOC below).
"""
import argparse
import glob
import json
import math
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = os.path.join(ROOT, "presentation")
OUT = os.path.join(PRES, "figures")
ASSET = os.path.join(PRES, "assets")
DATA = os.path.join(PRES, "data")
SIM_OUT = os.path.join(ROOT, "sim", "out")
EXPERT_EP = os.path.join(SIM_OUT, "expert_demos_rt", "w0", "batch_02", "ep_000")

# ---------------------------------------------------------------- colour code
# Primary: Healthy Orange on black or white, Siemens Petrol as accent.
ORANGE, PETROL, BLACK, WHITE = "#EC6602", "#009999", "#000000", "#FFFFFF"
INK, GREY, GREY2, LINE, PAPER = "#1B1B1B", "#5A5A5A", "#929292", "#D0D0D0", "#ECECEC"
ORANGE_M, ORANGE_L = "#EFB997", "#F8DECD"
PETROL_M, PETROL_L = "#98D0D1", "#CFE4E5"
# Additional / functional colours (palette page of the colour code)
BERRY, RED, YELLOW, SKY, NAVY, GREEN = "#71202F", "#D52D2C", "#F9D447", "#64BDE9", "#2A247E", "#449745"
RED_L, GREEN_L, YELLOW_L, SKY_L, NAVY_L = "#F0C1C7", "#C7E4CE", "#FEF3C5", "#D4EFFA", "#CAC8DE"

plt.rcParams.update({
    "font.family": ["Lato", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": LINE,
    "axes.labelcolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.titleweight": "bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

W, H = 1333, 750          # canvas units: 100 per inch, y grows downward
PT = 100 / 72             # canvas units per typographic point
FOOTER = "Challenge 2 · Teach the Robot · Physical AI learning loop for a cathlab companion"


# ---------------------------------------------------------------- drawing kit
def canvas(title, subtitle=None, source=None, dark=False):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    bg = BLACK if dark else WHITE
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_facecolor(bg)
    ax.text(48, 62, title, fontsize=27, fontweight="bold", color=ORANGE, va="baseline")
    if subtitle:
        ax.text(48, 94, subtitle, fontsize=13.5, color=WHITE if dark else INK, va="baseline")
    fg = GREY2 if dark else GREY
    ax.plot([48, W - 48], [712, 712], color="#333333" if dark else LINE, lw=0.8)
    ax.text(48, 732, FOOTER, fontsize=8.5, color=fg, va="center")
    if source:
        ax.text(W - 48, 732, source, fontsize=8.5, color=fg, va="center", ha="right")
    return fig, ax


def overlay(fig):
    """Transparent full-canvas axes drawn above inset images."""
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, fc=PAPER, ec="none", r=8, lw=1.2, z=1, ls="-", alpha=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, ls=ls, alpha=alpha)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=11, color=INK, weight="normal", ha="left", va="top", wrap=None, z=5, **kw):
    if wrap:
        s = "\n".join(textwrap.fill(par, wrap) for par in s.split("\n"))
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z,
                   linespacing=1.3, **kw)


def arrow(ax, p0, p1, color=GREY2, lw=2, rad=0.0, head=12, z=3, ls="-", style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=f"{style},head_length={head * 0.6},head_width={head * 0.35}",
                        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=z, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def pill(ax, x, y, s, fc=ORANGE, tc=WHITE, size=10, weight="bold", h=24, pad=12, ha="left", z=6):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    fig = ax.figure
    fig.canvas.draw() if not hasattr(fig.canvas, "renderer") else None
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    w = bb.width / fig.dpi * 100 + 2 * pad
    x0 = x - w if ha == "right" else (x - w / 2 if ha == "center" else x)
    box(ax, x0, y, w, h, fc=fc, r=h / 2, z=z)
    ax.text(x0 + w / 2, y + h / 2 + 0.5, s, fontsize=size, fontweight=weight, color=tc, ha="center",
            va="center", zorder=z + 1)
    return x0 + w


def num_badge(ax, x, y, s, fc=ORANGE, tc=WHITE, r=17, size=13, z=6):
    ax.add_patch(patches.Circle((x, y), r, fc=fc, ec="none", zorder=z))
    ax.text(x, y + 0.5, s, fontsize=size, fontweight="bold", color=tc, ha="center", va="center", zorder=z + 1)


def fit_crop(im, w, h):
    iw, ih = im.size
    target = w / h
    if iw / ih > target:
        nw = int(ih * target)
        x0 = (iw - nw) // 2
        return im.crop((x0, 0, x0 + nw, ih))
    nh = int(iw / target)
    y0 = (ih - nh) // 2
    return im.crop((0, y0, iw, y0 + nh))


def place(fig, im, x, y, w, h, border=None, lw=1.5, crop=None):
    if isinstance(im, str):
        im = Image.open(im).convert("RGB")
    if crop:
        im = im.crop(crop)
    im = fit_crop(im, w, h)
    a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    a.imshow(im, interpolation="lanczos")
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(bool(border))
        if border:
            s.set_edgecolor(border)
            s.set_linewidth(lw)
    return a


def chart(fig, x, y, w, h):
    a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color(LINE)
    a.tick_params(length=0, labelsize=10)
    a.set_facecolor("none")
    return a


def person(ax, x, y, s=1.0, color=PETROL, z=6):
    """Nurse icon: head + shoulders; (x, y) is the bottom centre."""
    ax.add_patch(patches.Wedge((x, y), 26 * s, 180, 360, fc=color, ec="none", zorder=z))
    ax.add_patch(patches.Circle((x, y - 40 * s), 13 * s, fc=color, ec="none", zorder=z))
    # nurse cap cross
    ax.add_patch(patches.Rectangle((x - 2 * s, y - 20 * s), 4 * s, 12 * s, fc=WHITE, zorder=z + 1))
    ax.add_patch(patches.Rectangle((x - 6 * s, y - 16 * s), 12 * s, 4 * s, fc=WHITE, zorder=z + 1))


def robot(ax, x, y, s=1.0, color=ORANGE, z=6):
    """Robot icon; (x, y) is the bottom centre."""
    box(ax, x - 22 * s, y - 30 * s, 44 * s, 30 * s, fc=color, r=6 * s, z=z)
    box(ax, x - 17 * s, y - 58 * s, 34 * s, 24 * s, fc=color, r=6 * s, z=z)
    for dx in (-7, 7):
        ax.add_patch(patches.Circle((x + dx * s, y - 46 * s), 3.5 * s, fc=WHITE, zorder=z + 1))
    ax.plot([x, x], [y - 58 * s, y - 67 * s], color=color, lw=2.5 * s, zorder=z)
    ax.add_patch(patches.Circle((x, y - 69 * s), 3.5 * s, fc=color, zorder=z))


def bubble(ax, x, y, w, h, s, tail, fc=WHITE, ec=PETROL, size=12, tc=INK, weight="bold", wrap=None, z=8):
    box(ax, x, y, w, h, fc=fc, ec=ec, r=12, lw=2, z=z)
    tx, ty = tail
    bx = min(max(tx, x + 20), x + w - 40)
    by = y + h if ty > y + h else y
    ax.add_patch(patches.Polygon([[bx, by], [bx + 22, by], [tx, ty]], closed=True, fc=fc, ec=ec, lw=2, zorder=z))
    ax.plot([bx + 1.5, bx + 20.5], [by, by], color=fc, lw=3.5, zorder=z + 0.5)
    txt(ax, x + w / 2, y + h / 2, s, size=size, color=tc, weight=weight, ha="center", va="center", wrap=wrap, z=z + 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(os.path.join(OUT, name + ".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name)


def frame(cam, i):
    return os.path.join(ASSET, "frames", f"{cam}_{i:02d}.png")


# frame file index -> episode frame (extracted from EXPERT_EP at these frames)
FRAME_AT = {1: 0, 2: 250, 3: 345, 4: 700, 5: 1020, 6: 1100, 7: 1500, 8: 2000, 9: 2060}


# ---------------------------------------------------------------- data
def load_data():
    d = {}
    # dataset intelligence
    d["datasets"] = {}
    for n in ("fr3_duo_rand500", "fr3_duo_handover_v9_night_84", "FR3Duo_Handover_Dataset_lerobot3"):
        p = os.path.join(ROOT, "dataset_check", "reports", n, "report.json")
        if os.path.exists(p):
            d["datasets"][n] = json.load(open(p))["summary"]
    # expert demos (v1 data)
    rows = []
    for f in sorted(glob.glob(os.path.join(SIM_OUT, "expert*", "**", "meta.json"), recursive=True)):
        if "DISCARDED" in f:
            continue
        m = json.load(open(f))
        o = m.get("box_offset_dx_dy_dz_yaw") or [0, 0, 0, 0]
        errs = {}
        for e in m.get("events", []):
            if "TCP error" in e["event"]:
                errs[e["event"][0]] = float(e["event"].split("TCP error")[1].split("mm")[0])
        rows.append({"run": os.path.relpath(f, SIM_OUT), "success": m["success"], "dx_mm": o[0] * 1000,
                     "dz_mm": o[2] * 1000, "yaw_deg": o[3], "err_L": errs.get("L"), "err_R": errs.get("R"),
                     "render": m.get("render_mode", "pt"), "wall_s": m.get("wall_s"),
                     "tray_mm": (m.get("box_to_tray_centre_m") or 0) * 1000})
    if rows:
        json.dump(rows, open(os.path.join(DATA, "expert_demos.json"), "w"), indent=1)
    else:
        rows = json.load(open(os.path.join(DATA, "expert_demos.json")))
    d["expert"] = rows
    # v0 closed-loop development runs (scored with sim/eval_summary.milestones)
    try:
        sys.path.insert(0, os.path.join(ROOT, "sim"))
        from eval_summary import milestones
        runs = []
        for f in sorted(glob.glob(os.path.join(SIM_OUT, "*", "**", "result.json"), recursive=True)):
            r = json.load(open(f))
            try:
                t = json.load(open(os.path.join(os.path.dirname(f), "trace.json")))
                m = milestones(t, r)
            except Exception:
                continue
            runs.append({"run": os.path.relpath(os.path.dirname(f), SIM_OUT), "frames": r["frames"],
                         "abort": r.get("abort_reason"), "lat": r["latency_ms"],
                         "assist": bool(r.get("align_assist")), "sup": bool(r.get("supervisor")),
                         "lift0": r.get("box_lifted_mm_at_start", 0) or 0, "sig": r.get("stopped_by_signal"),
                         "attempt": not r.get("stopped_by_signal") and not (r.get("box_lifted_mm_at_start", 0) or 0),
                         **{k: m[k] for k in ("left_grasp", "right_grasp", "handover", "placed")}})
        if runs:
            json.dump(runs, open(os.path.join(DATA, "v0_runs.json"), "w"), indent=1)
    except Exception as e:
        print("v0 scoring skipped:", e)
    d["v0"] = json.load(open(os.path.join(DATA, "v0_runs.json")))
    # optional controlled v0 vs v1 evaluation (fill presentation/data/v0_v1_eval.json when available)
    p = os.path.join(DATA, "v0_v1_eval.json")
    d["v0v1"] = json.load(open(p)) if os.path.exists(p) else None
    # expert episode arrays
    npz = os.path.join(EXPERT_EP, "data.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        np.savez_compressed(os.path.join(DATA, "expert_episode.npz"), state=z["state"], action=z["action"])
    z = np.load(os.path.join(DATA, "expert_episode.npz"))
    d["ep_state"], d["ep_action"] = z["state"], z["action"]
    return d


def v0_failure_class(r):
    a = r["abort"] or ""
    if "frame 77" in a:
        return "Waits forever at the\ndemo's 2.5 s pause"
    if "stalled" in a:
        return "Stalls while carrying\n(drift, box slips)"
    if "R grasp missed" in a:
        return "Right grasp missed\n(handover fails)"
    if "tipped" in a:
        return "Knocks the box over\n(2-4 cm drift)"
    return "Ends before a lifted\ngrasp (short dev runs)"


# ================================================================ F00 one-image story
def f00_story(d):
    fig, ax = canvas("One robot skill is not the goal. The learning loop is.",
                     "A cathlab robot companion that learns new tasks from demonstrations, measured in simulation, "
                     "with the nurse in charge",
                     source="Isaac Sim 6.0.1 · FR3 Duo · π0.5 VLA")
    cols = [(48, 380), (458, 420), (908, 377)]
    heads = [("THE PROBLEM", BLACK), ("OUR ROBOT COMPANION", ORANGE), ("THE LEARNING LOOP", PETROL)]
    for (x, w), (h, c) in zip(cols, heads):
        pill(ax, x, 118, h, fc=c, size=10.5)
    # --- problem column
    place(fig, os.path.join(ASSET, "sim_room_gui.png"), 48, 154, 380, 262, crop=(300, 236, 1150, 748))
    probs = [("Cathlab nurses prepare patients and equipment and assist the cardiologist, "
              "under acute staff shortage.", ORANGE),
             ("Fetch, hand over and place: routine tasks take nurse time away from the patient.", ORANGE),
             ("Classic robots are programmed per object and per scene: every change costs an engineer.", GREY2)]
    y = 440
    for s, c in probs:
        ax.add_patch(patches.Rectangle((48, y + 2), 4, 50, fc=c, zorder=3))
        txt(ax, 62, y, s, size=11.5, wrap=44)
        y += 84 if len(s) > 88 else 66
    # --- companion column (hero)
    place(fig, os.path.join(ASSET, "sim_robot_closeup.png"), 458, 154, 420, 392, crop=(0, 0, 578, 492))
    ov = overlay(fig)
    person(ov, 506, 532, s=1.05, color=PETROL)
    bubble(ov, 540, 444, 318, 66, "“Hand me the box and place it\non the serving tray.”", tail=(526, 486),
           size=12.5)
    box(ov, 470, 166, 230, 30, fc=BLACK, r=6, z=8, alpha=0.85)
    txt(ov, 480, 181, "Isaac Sim · FR3 Duo · dual arm", size=10.5, color=WHITE, va="center", weight="bold", z=9)
    box(ax, 458, 560, 420, 132, fc=PAPER, r=8)
    txt(ax, 474, 572, "π0.5 vision-language-action policy", size=13, weight="bold")
    txt(ax, 474, 597, "3 cameras + spoken/typed instruction  →  16 joint targets at 30 Hz, "
        "re-planned every 10 steps (~95 ms per plan).", size=11, wrap=58)
    txt(ax, 474, 646, "Autonomy level 2: the nurse starts, watches and can stop; no patient contact.", size=11,
        color=PETROL, weight="bold", wrap=58)
    # --- loop column
    cx, cy, R = 1096, 330, 120
    stages = [("01", "Ingest &\nTransform"), ("02", "Dataset\nIntelligence"), ("03", "Fine-Tune\n(VLA)"),
              ("04", "Evaluate\nin Sim"), ("05", "Version &\nServe")]
    ang = [-90 + i * 72 for i in range(5)]
    pts = [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in ang]
    ax.add_patch(patches.Circle((cx, cy), R, fc="none", ec=PETROL_M, lw=10, zorder=2))
    for i in range(5):
        a0, a1 = ang[i] + 17, ang[(i + 1) % 5] - 17 + (360 if i == 4 else 0)
        mid = math.radians(a1)
        p0 = (cx + R * math.cos(math.radians(a1 - 6)), cy + R * math.sin(math.radians(a1 - 6)))
        p1 = (cx + R * math.cos(mid), cy + R * math.sin(mid))
        arrow(ax, p0, p1, color=PETROL, lw=0.1, head=16, z=3)
    for (n, lab), (px, py), a in zip(stages, pts, ang):
        num_badge(ax, px, py, n, fc=ORANGE if n == "04" else PETROL, r=24, size=14)
        ox, oy = math.cos(math.radians(a)), math.sin(math.radians(a))
        ha = "center" if abs(ox) < 0.3 else ("left" if ox > 0 else "right")
        lx = px + ox * 34
        ly = py + oy * 34
        txt(ax, lx, ly, lab, size=10.5, weight="bold", ha=ha, va="center" if abs(oy) < 0.9 else
            ("bottom" if oy < 0 else "top"))
    txt(ax, cx, cy - 10, "Stage 5 feeds", size=11, color=GREY, ha="center", va="center")
    txt(ax, cx, cy + 10, "the next cycle", size=11, color=GREY, ha="center", va="center")
    txt(ax, 908, 520, "Raw demonstrations in; a served, evaluated, reproducible policy out. "
        "Failures in simulation become the next training data.", size=11.5, wrap=48)
    # --- evidence band
    hd = d["datasets"].get("fr3_duo_handover_v9_night_84", {})
    v0 = d["v0"]
    ex = d["expert"]
    k_ok = sum(r["success"] for r in ex)
    tiles = [(f"{hd.get('episodes', 84)} \u2192 {hd.get('distinct_motions', 1)}",
              f"{hd.get('episodes', 84)} handover demos hold one motion", BLACK),
             (f"{sum(r['placed'] for r in v0)} / {len(v0)}", "v0 sim runs complete the task", RED),
             (f"{k_ok} / {len(ex)}", "expert demos succeed with the box moved", PETROL)]
    x = 908
    for i, (big, small, c) in enumerate(tiles):
        tx = 908 + i * 128
        box(ax, tx, 600, 120, 92, fc=WHITE, ec=LINE, r=8)
        ax.add_patch(patches.Rectangle((tx, 600), 120, 4, fc=c, zorder=3))
        txt(ax, tx + 10, 612, big, size=17, weight="bold", color=c)
        txt(ax, tx + 10, 646, small, size=9, color=GREY, wrap=19)
    save(fig, "F00_one_image_story")


# ================================================================ F01 problem
def f01_problem(d):
    fig, ax = canvas("The problem we are solving",
                     "Robots cannot be programmed for every object, variation and situation: "
                     "they need a repeatable learning pipeline",
                     source="Problem framing: challenge brief (Siemens Healthineers, C2)")
    xs = [48, 470, 892]
    wcol = 393
    heads = [("THE BACKGROUND", "Cardiac cath lab nurse", BLACK), ("TODAY", "Traditional robot automation", GREY),
             ("WHAT WE BUILD", "Physical AI companion", ORANGE)]
    body = [
        ["Prepares the patient and monitors", "Prepares equipment and supplies", "Assists the cardiologist",
         "Impulse: acute staff shortage, less experienced staff"],
        ["Task-specific programming", "Fixed conditions: object, pose, light", "High adaptation effort per change",
         "One engineer per new skill"],
        ["Learns from demonstrations", "Adapts to variable objects and scenes", "Reusable AI-driven skills",
         "A loop that creates, evaluates and deploys them"],
    ]
    marks = [("•", ORANGE), ("✕", RED), ("✓", GREEN)]
    for i, x in enumerate(xs):
        k, h, c = heads[i]
        box(ax, x, 122, wcol, 344, fc=PAPER if i < 2 else ORANGE_L, r=10)
        pill(ax, x + 18, 138, k, fc=c, size=10)
        txt(ax, x + 18, 176, h, size=17, weight="bold", color=INK)
        y = 222
        for s in body[i]:
            m, mc = marks[i]
            txt(ax, x + 22, y, m, size=15, color=mc, weight="bold", family="DejaVu Sans")
            txt(ax, x + 48, y + 2, s, size=12.5, wrap=36)
            y += 56
        if i < 2:
            arrow(ax, (x + wcol + 6, 294), (xs[i + 1] - 6, 294), color=ORANGE, lw=3, head=18)
    # bottom: concrete first skill
    box(ax, 48, 488, W - 96, 204, fc=WHITE, ec=LINE, r=10)
    txt(ax, 70, 506, "Our first skill, end to end", size=15, weight="bold", color=ORANGE)
    txt(ax, 70, 534, "“Hand the box from the left gripper to the right gripper and place it on the serving "
        "tray.”", size=12.5, weight="bold", wrap=46)
    txt(ax, 70, 596, "A typical cathlab fetch-and-pass motion: two arms, two grasps, one handover, a placement. "
        "Simulation first, then the same loop toward the real FR3 Duo.", size=11, color=GREY, wrap=56)
    fw, fh, gx = 116, 116, 470
    for j, i in enumerate((1, 3, 4, 5, 6, 9)):
        place(fig, frame("external", i), gx + j * (fw + 18), 532, fw, fh, border=LINE, lw=1)
    ov = overlay(fig)
    labs = ["start", "left grasp", "lift & carry", "handover", "right carries", "placed"]
    for j, s in enumerate(labs):
        txt(ov, gx + j * (fw + 18) + fw / 2, 654, s, size=10, color=GREY, ha="center")
        if j < 5:
            arrow(ov, (gx + j * (fw + 18) + fw + 2, 590), (gx + (j + 1) * (fw + 18) - 2, 590), color=ORANGE,
                  lw=1.5, head=9)
    txt(ov, gx, 516, "Head camera of the robot, one simulated episode (69 s)", size=9.5, color=GREY)
    save(fig, "F01_problem")


# ================================================================ F02 interaction
def f02_interaction(d):
    fig, ax = canvas("Nurse and robot work as one team",
                     "The nurse asks and stays in charge; the robot executes and checks itself; "
                     "every run teaches the next model version",
                     source="Concept for the clinic; supervisor and logging exist in our sim stack (sim/recovery.py)")
    lanes = [("Nurse", "asks, watches,\nconfirms", PETROL, PETROL_L),
             ("Robot", "π0.5 VLA\npolicy", ORANGE, ORANGE_L),
             ("Safety supervisor", "runtime checks", BLACK, PAPER),
             ("Learning loop", "offline, sandboxed", NAVY, NAVY_L)]
    x0, x1 = 212, W - 48
    ly0, lh, gap = 140, 112, 8
    ncol = 6
    cw = (x1 - x0) / ncol
    steps = ["1  Request", "2  Plan", "3  Execute", "4  Monitor", "5  Hand over", "6  Learn"]
    for i, s in enumerate(steps):
        txt(ax, x0 + cw * i + 10, 132, s, size=11, weight="bold", color=GREY, va="bottom")
    for k, (n, sub, c, cl) in enumerate(lanes):
        y = ly0 + k * (lh + gap)
        box(ax, 48, y, W - 96, lh, fc=cl, r=8, z=0, alpha=0.55)
        ax.add_patch(patches.Rectangle((48, y), 6, lh, fc=c, zorder=1))
        txt(ax, 66, y + 30, n, size=13, weight="bold", color=c if c != BLACK else INK)
        txt(ax, 66, y + 54, sub, size=10, color=GREY)
    person(ax, 182, ly0 + lh - 8, s=0.62, color=PETROL)
    robot(ax, 182, ly0 + 2 * lh + gap - 8, s=0.6, color=ORANGE)

    def cell(lane, col, s, fc=WHITE, ec=None, span=1, bold_first=True, h=None):
        y = ly0 + lane * (lh + gap) + 12
        x = x0 + col * cw + 6
        w = cw * span - 12
        hh = h or lh - 24
        box(ax, x, y, w, hh, fc=fc, ec=ec or LINE, r=8, z=3, lw=1.2)
        head, _, rest = s.partition("|")
        txt(ax, x + 10, y + 9, head, size=10.8, weight="bold", wrap=int(w / 7.3), z=6)
        nlines = len(textwrap.wrap(head, int(w / 7.3)))
        if rest:
            txt(ax, x + 10, y + 13 + 15 * nlines, rest, size=9.6, color=GREY, wrap=int(w / 6.2), z=6)
        return (x, y, w, hh)

    c_req = cell(0, 0, "“Prepare the tray: box to the serving tray.”|Voice or tablet, in plain language.")
    c_watch = cell(0, 2, "Watches, can stop at any time|E-stop, voice “stop”, pause on the tablet.", span=1)
    c_ask = cell(0, 3, "Asked to help only when needed|“Box tipped. Re-place it?”", ec=RED)
    c_take = cell(0, 4, "Takes the item or confirms ✓|One tap: success / failure label.")
    c_sign = cell(0, 5, "Signs off a new version|Only approved versions reach the real robot.", ec=PETROL)
    c_plan = cell(1, 1, "Skill orchestrator picks the skill|Instruction → skill → policy version.")
    c_exec = cell(1, 2, "Executes the skill|3 cameras + instruction → joint targets, 30 Hz, "
                  "new plan every 333 ms.")
    c_place = cell(1, 4, "Places on the tray, returns home|Release only over the tray (interlock).")
    c_mon = cell(2, 3, "Checks every frame|Grasp missed · stalled 5 s · box tipped > 35° · "
                 "box dropped.")
    c_rec = cell(2, 2, "Auto-recover|Rewind the arm 3 s along its path, retry (max 2).", ec=ORANGE)
    c_log = cell(3, 4, "Logs every run|Video, joint trace, result, nurse label.")
    c_learn = cell(3, 5, "Failures become data|Expert corrections, retrain, evaluate in sim (N runs, 95% CI).")
    c_gate = cell(3, 3, "Sandbox only|Pipeline runs automatically in simulation.", fc=PAPER)

    def mid(c, side):
        x, y, w, h = c
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[side]

    A = dict(color=GREY, lw=1.8, head=11, z=4)
    arrow(ax, mid(c_req, "b"), mid(c_plan, "l"), rad=0.25, **A)
    arrow(ax, mid(c_plan, "r"), mid(c_exec, "l"), **A)
    arrow(ax, (c_exec[0] + c_exec[2] - 4, c_exec[1] + c_exec[3]), (c_mon[0] + 36, c_mon[1]), **A)
    arrow(ax, mid(c_mon, "t"), mid(c_place, "l"), rad=-0.25, color=GREEN, lw=1.8, head=11, z=4)
    txt(ax, mid(c_mon, "t")[0] + 34, mid(c_mon, "t")[1] - 22, "all clear", size=9, color=GREEN, weight="bold")
    arrow(ax, mid(c_mon, "l"), mid(c_rec, "r"), color=ORANGE, lw=1.8, head=11, z=4)
    arrow(ax, (mid(c_rec, "t")[0] + 30, mid(c_rec, "t")[1]), (mid(c_exec, "b")[0] + 30, mid(c_exec, "b")[1]),
          color=ORANGE, lw=1.8, head=11, z=4)
    arrow(ax, (mid(c_mon, "t")[0] - 30, mid(c_mon, "t")[1]), (mid(c_ask, "b")[0] - 30, mid(c_ask, "b")[1]),
          color=RED, lw=1.8, head=11, z=4)
    txt(ax, mid(c_mon, "t")[0] - 24, mid(c_ask, "b")[1] + 16, "retries used up:\nask the nurse", size=8.8,
        color=RED, weight="bold")
    arrow(ax, mid(c_place, "t"), mid(c_take, "b"), **A)
    arrow(ax, (mid(c_take, "b")[0] + 50, mid(c_take, "b")[1]),
          (mid(c_log, "t")[0] + 50, mid(c_log, "t")[1]), color=NAVY, lw=1.6, head=11, z=2, ls="--")
    arrow(ax, mid(c_log, "r"), mid(c_learn, "l"), color=NAVY, lw=1.8, head=11, z=4)
    arrow(ax, (mid(c_learn, "t")[0], mid(c_learn, "t")[1]), (mid(c_sign, "b")[0], mid(c_sign, "b")[1]),
          color=NAVY, lw=1.8, head=11, z=2)
    # bottom message band
    yb = ly0 + 4 * (lh + gap) + 2
    box(ax, 48, yb, W - 96, 712 - yb - 14, fc=BLACK, r=8)
    txt(ax, 66, yb + (712 - yb - 14) / 2, "Robot: task autonomy (level 2), no patient contact.   "
        "Pipeline: automatic inside the simulation sandbox.   Human sign-off for anything that moves a real robot.",
        size=11.5, color=WHITE, va="center", weight="bold")
    save(fig, "F02_nurse_robot_interaction")


# ================================================================ F03 pipeline flowchart
def f03_pipeline(d):
    fig, ax = canvas("The autonomous learning loop, as we built it",
                     "Five stages, each with a tool, an output and a number; the simulator closes the loop",
                     source="Tools in this repo: dataset_check/, pi05/, sim/")
    ds = d["datasets"]
    n_prov = sum(v["episodes"] for v in ds.values()) if ds else 584
    ex = d["expert"]
    v0 = d["v0"]
    k_ok = sum(r["success"] for r in ex)
    hd = ds.get("fr3_duo_handover_v9_night_84", {})
    cards = [
        ("01", "Ingest &\nTransform", ["LeRobot v3 → openpi", "Sim demos → LeRobot", "Action↔state lag "
                                       "found: 4 frames"], f"{n_prov} provided\n+ {k_ok} generated episodes"),
        ("02", "Dataset\nIntelligence", ["Diversity, idle time, events", "Visual near-duplicates", "Ready / not-ready "
                                         "verdict"], f"Handover: NOT READY\n{hd.get('distinct_motions', 1)} distinct "
                                                     f"motion in {hd.get('episodes', 84)}"),
        ("03", "Fine-Tune\n(VLA)", ["π0.5, LoRA (openpi)", "Absolute joints, horizon 10", "v1: + sim expert demos, "
                                    "3k steps"], "v0 offline arm error\n0.39° (base model 6.79°)"),
        ("04", "Evaluate", ["Offline: held-out episodes", "Closed loop in Isaac Sim", "4 milestones, 95% CI, "
                            "failure reasons"], f"v0 closed loop:\n{sum(r['placed'] for r in v0)} / {len(v0)} "
                                                f"tasks completed"),
        ("05", "Version &\nServe", ["Policy server (websocket)", "Fixed split + config per version",
                                    "Run logs: video, trace, result"], "p50 95 ms per\n10-step plan (RTX 4090)"),
    ]
    x0, cw, g, y0, ch = 60, 226, 21, 158, 262
    pill(ax, 60, 120, "IN  raw demonstrations: 3 cameras, 16-D state + action, instruction", fc=BLACK, size=10)
    pill(ax, W - 60, 120, "OUT  served, evaluated, versioned policy", fc=ORANGE, size=10, ha="right")
    for i, (n, name, items, res) in enumerate(cards):
        x = x0 + i * (cw + g)
        hl = n == "04"
        box(ax, x, y0, cw, ch, fc=WHITE, ec=ORANGE if hl else LINE, lw=2 if hl else 1.2, r=10)
        box(ax, x, y0, cw, 62, fc=ORANGE if hl else BLACK, r=10, z=2)
        ax.add_patch(patches.Rectangle((x, y0 + 40), cw, 22, fc=ORANGE if hl else BLACK, zorder=2))
        txt(ax, x + 14, y0 + 31, n, size=22, weight="bold", color=WHITE if hl else ORANGE, va="center", z=6)
        txt(ax, x + 62, y0 + 31, name, size=12, weight="bold", color=WHITE, va="center", z=6)
        y = y0 + 76
        for s in items:
            txt(ax, x + 14, y, "•", size=12, color=ORANGE)
            txt(ax, x + 28, y + 1, s, size=10.8, wrap=27)
            y += 34 if len(s) > 27 else 26
        box(ax, x + 10, y0 + ch - 70, cw - 20, 58, fc=PETROL_L, r=6, z=2)
        txt(ax, x + 20, y0 + ch - 41, res, size=10.8, weight="bold", color="#006B6B", va="center", z=6)
        if i < 4:
            arrow(ax, (x + cw + 2, y0 + 31), (x + cw + g - 2, y0 + 31), color=ORANGE, lw=2.5, head=12)
    # feedback loop
    yl = y0 + ch + 24
    xa, xb = x0 + 4 * (cw + g) + cw / 2, x0 + cw / 2
    ax.plot([xa, xa, xb], [y0 + ch, yl, yl], color=ORANGE, lw=3, zorder=2, solid_capstyle="round")
    arrow(ax, (xb, yl), (xb, y0 + ch + 2), color=ORANGE, lw=3, head=16)
    box(ax, 380, yl - 14, 572, 28, fc=WHITE, r=4, z=3)
    txt(ax, W / 2 - 10, yl, "Stage 5 feeds the next cycle: sim failures → new demos → v0 → v1 → v2",
        size=11.5, weight="bold", color=ORANGE, ha="center", va="center", z=6)
    # simulator strip
    ys = yl + 30
    hs = 712 - ys - 14
    place(fig, os.path.join(ASSET, "sim_robot_gui.png"), 60, ys, 330, hs, border=LINE, crop=(190, 236, 1340, 900))
    sx = 410
    txt(ax, sx, ys + 2, "Isaac Sim closes the loop", size=14, weight="bold")
    replay = "0.72°"
    sim_items = [("Replay test", f"Recorded actions replayed open loop: worst joint error {replay} over 2,247 "
                  "frames, so action space and timing are right."),
                 ("Expert demo generator", f"Scripted expert with the box moved ±15 mm / ±5° and "
                  f"0.6° noise: {k_ok} of {len(ex)} episodes succeed."),
                 ("Closed-loop evaluation", "Policy drives the robot from its own camera images; milestones, "
                  "failure reason, video and trace per run.")]
    for j, (h, s) in enumerate(sim_items):
        xx = sx + j * 300
        num_badge(ax, xx + 12, ys + 46, str(j + 1), fc=PETROL, r=12, size=11)
        txt(ax, xx + 32, ys + 37, h, size=11.5, weight="bold", color=PETROL)
        txt(ax, xx + 32, ys + 60, s, size=10.2, color=INK, wrap=40)
    save(fig, "F03_pipeline_flowchart")


# ================================================================ F04 policy I/O
def f04_policy_io(d):
    fig, ax = canvas("What goes in, what comes out",
                     "One policy query: three camera images, the instruction and the joint state in; "
                     "a 10-step plan for both arms out",
                     source="pi05/fr3duo_policy.py · checkpoint 6a5f7c87 · latency on RTX 4090")
    pill(ax, 48, 116, "IN", fc=BLACK, size=11)
    pill(ax, 520, 116, "MODEL", fc=ORANGE, size=11)
    pill(ax, 868, 116, "OUT", fc=PETROL, size=11)
    cams = [("external", "Head camera"), ("left_wrist", "Left wrist"), ("right_wrist", "Right wrist")]
    for j, (c, lab) in enumerate(cams):
        place(fig, frame(c, 3), 48 + j * 140, 152, 130, 130, border=LINE, lw=1)
        txt(ax, 48 + j * 140, 290, lab, size=10, color=GREY)
    txt(ax, 48, 316, "224 × 224 RGB, 30 Hz", size=9.5, color=GREY2)
    box(ax, 48, 340, 410, 58, fc=WHITE, ec=BLACK, r=12, lw=2)
    txt(ax, 253, 369, "\u201cHand the box from the left gripper to the right\ngripper and place it on "
        "the serving tray.\u201d", size=11, ha="center", va="center")
    txt(ax, 48, 410, "Instruction (text)", size=10, color=GREY)
    # state vector cells
    st = d["ep_state"][345]
    labs = [f"q{i}" for i in range(1, 8)] + ["grip"]
    y = 448
    for side, off, c in (("left arm", 0, ORANGE), ("right arm", 8, PETROL)):
        for i in range(8):
            v = st[off + i]
            fc = c if i < 7 else BLACK
            box(ax, 48 + i * 51, y, 47, 40, fc=fc, r=4, alpha=0.18 + 0.5 * (i == 7) if i < 7 else 0.85)
            txt(ax, 48 + i * 51 + 23.5, y + 12, labs[i], size=9, ha="center", va="center", weight="bold",
                color=WHITE if i == 7 else INK)
            txt(ax, 48 + i * 51 + 23.5, y + 29, f"{v:+.2f}" if i < 7 else f"{v:.2f}", size=8.5, ha="center",
                va="center", color=WHITE if i == 7 else GREY)
        y += 48
    txt(ax, 48, y + 4, "Joint state: 16 numbers. Orange row left arm, petrol row right arm (rad);\ngripper 0 closed \u2026 1 open",
        size=10, color=GREY)
    # model block
    mx, my, mw, mh = 520, 152, 316, 390
    box(ax, mx, my, mw, mh, fc=BLACK, r=12)
    txt(ax, mx + 20, my + 22, "π0.5", size=30, weight="bold", color=ORANGE)
    txt(ax, mx + 20, my + 70, "Vision-language-action model", size=12.5, color=WHITE, weight="bold")
    layers = [("Vision-language backbone", "reads images + instruction"),
              ("Action expert (flow matching)", "turns context into smooth motion"),
              ("LoRA adapters", "fine-tuned on FR3 Duo demonstrations")]
    yy = my + 110
    for h, s in layers:
        box(ax, mx + 20, yy, mw - 40, 64, fc="#262626", r=8, z=2)
        txt(ax, mx + 34, yy + 12, h, size=11.5, weight="bold", color=WHITE, z=6)
        txt(ax, mx + 34, yy + 36, s, size=10, color=GREY2, z=6)
        yy += 76
    txt(ax, mx + 20, my + mh - 30, "~95 ms per query (p50), RTX 4090", size=10.5, color=ORANGE, weight="bold")
    for yy in (217, 369, 490):
        arrow(ax, (472, yy), (mx - 6, my + mh / 2 + (yy - 360) * 0.25), color=GREY2, lw=2, head=12)
    arrow(ax, (mx + mw + 6, my + mh / 2), (838, my + mh / 2), color=ORANGE, lw=3, head=16)
    # output chunk plot
    a = d["ep_action"]
    t0, t1, c0 = 300, 360, 322
    t = np.arange(t0, t1) / 30.0
    ac = chart(fig, 896, 160, 390, 220)
    cols = [ORANGE, "#F08A3E", ORANGE_M, PETROL, PETROL_M, NAVY, SKY]
    for j in range(7):
        ac.plot(t, np.degrees(a[t0:t1, j] - a[t0, j]), color=cols[j], lw=1.8)
    ac.axvspan(c0 / 30, (c0 + 10) / 30, color=ORANGE_L, zorder=0)
    lo_, hi_ = ac.get_ylim()
    ac.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
    ac.text((c0 + 5) / 30, ac.get_ylim()[1], "one plan\n10 steps = 333 ms", ha="center", va="top", fontsize=9.5,
            color=ORANGE, fontweight="bold")
    ac.set_ylabel("left arm joints, Δ deg", fontsize=10)
    ac.set_xticklabels([])
    ac.grid(axis="y", color=PAPER)
    ag = chart(fig, 896, 392, 390, 96)
    ag.step(t, a[t0:t1, 7], color=BLACK, lw=2, where="post")
    ag.axvspan(c0 / 30, (c0 + 10) / 30, color=ORANGE_L, zorder=0)
    ag.set_yticks([0, 0.48, 1])
    ag.set_yticklabels(["closed", "pre-shape", "open"], fontsize=9)
    ag.set_ylim(-0.1, 1.1)
    ag.set_xticks(np.arange(10, 12.01, 0.5))
    ag.set_xlabel("time in episode (s)", fontsize=10)
    ag.text(t[-1], 0.62, "left gripper command", ha="right", fontsize=9.5, color=GREY)
    ov = overlay(fig)
    txt(ov, 896, 540, "Action chunk: 10 × 16 joint position targets", size=12, weight="bold", color=PETROL)
    txt(ov, 896, 564, "Both arms and both grippers, executed at 30 Hz; the policy re-plans every 10 steps "
        "from fresh images.", size=10.5, wrap=56)
    # bottom: timing bar
    yb = 616
    box(ax, 48, yb, W - 96, 76, fc=PAPER, r=8)
    txt(ax, 66, yb + 14, "Real-time budget per plan", size=11.5, weight="bold")
    bx, bw = 300, 940
    box(ax, bx, yb + 36, bw, 22, fc=PETROL_L, r=4, z=2)
    box(ax, bx, yb + 36, bw * 95 / 333, 22, fc=ORANGE, r=4, z=3)
    txt(ax, bx + 8, yb + 47, "inference 95 ms (p50)", size=9.5, color=WHITE, va="center", weight="bold", z=6)
    txt(ax, bx + bw - 8, yb + 47, "333 ms of motion per plan (10 steps at 30 Hz)", size=9.5, color="#006B6B",
        va="center", ha="right", weight="bold", z=6)
    txt(ax, 66, yb + 40, "inference uses 29% of it", size=10.5, color=GREY)
    save(fig, "F04_policy_inputs_outputs")


# ================================================================ F05 task storyboard
def f05_storyboard(d):
    fig, ax = canvas("The task, step by step",
                     "Bimanual handover in the provided Isaac Sim scene: two grasps, one handover, one placement",
                     source=f"Expert demo {os.path.relpath(EXPERT_EP, ROOT)} (box moved +4 mm / -7.5 mm)")
    s = d["ep_state"]
    fps = 30.0
    idx = [1, 2, 3, 4, 5, 6, 7, 9]
    labels = ["Start", "Approach", "Left grasp", "Lift & carry", "Right grasp", "Left releases", "Right carries",
              "Placed on tray"]
    fw, g = 146, 10.6
    x0 = 48
    for j, i in enumerate(idx):
        x = x0 + j * (fw + g)
        place(fig, frame("external", i), x, 118, fw, fw, border=LINE, lw=1)
    ov = overlay(fig)
    for j, i in enumerate(idx):
        x = x0 + j * (fw + g)
        num_badge(ov, x + 16, 134, str(j + 1), fc=ORANGE, r=11, size=10)
        txt(ov, x, 272, labels[j], size=11, weight="bold")
        txt(ov, x, 291, f"t = {FRAME_AT[i] / fps:.0f} s", size=9.5, color=GREY)
    # gripper traces
    ac = chart(fig, 100, 340, W - 160, 230)
    t = np.arange(len(s)) / fps
    ac.plot(t, s[:, 7], color=ORANGE, lw=2.4, label="left gripper")
    ac.plot(t, s[:, 15], color=PETROL, lw=2.4, label="right gripper")
    phases = [(0, 329, "approach", PAPER), (329, 1004, "left holds, carries", ORANGE_L),
              (1004, 1060, "handover", YELLOW_L), (1060, 1800, "right holds, carries", PETROL_L),
              (1800, len(s), "place", PAPER)]
    for a0, a1, lab, c in phases:
        ac.axvspan(a0 / fps, a1 / fps, color=c, zorder=0, lw=0)
        ac.text((a0 + a1) / 2 / fps, 1.12, lab, ha="center", fontsize=9.5, color=GREY, fontweight="bold")
    for i in idx:
        ac.axvline(FRAME_AT[i] / fps, color=LINE, lw=0.8, ls=":", zorder=1)
    ac.set_ylim(0.3, 1.18)
    ac.set_xlim(0, len(s) / fps)
    ac.set_yticks([0.43, 0.55, 0.99])
    ac.set_yticklabels(["holding", "pre-shape", "open"])
    ac.set_xlabel("time (s)")
    ac.legend(loc="center right", frameon=False, fontsize=10.5, ncol=2, bbox_to_anchor=(1, 0.62))
    ov2 = overlay(fig)
    txt(ov2, 48, 330, "Gripper state", size=11, weight="bold", color=GREY)
    # milestones band
    yb = 632
    box(ov2, 48, yb, W - 96, 66, fc=BLACK, r=8)
    txt(ov2, 66, yb + 33, "Scored milestones", size=12, weight="bold", color=ORANGE, va="center")
    ms = ["Left grasp (box lifted ≥ 50 mm)", "Right grasp", "Handover (left opens, box stays up)",
          "Placed (≤ 30 mm of goal)"]
    xx = 240
    for k, m in enumerate(ms):
        num_badge(ov2, xx, yb + 33, str(k + 1), fc=ORANGE, r=12, size=10)
        txt(ov2, xx + 18, yb + 33, m, size=11, color=WHITE, va="center")
        xx += 18 + len(m) * 6.4 + 50
    save(fig, "F05_task_storyboard")


# ================================================================ F06 dataset intelligence
def f06_dataset(d):
    fig, ax = canvas("Dataset intelligence: 84 demonstrations, one movement",
                     "Our stage-02 checker reads every episode before training and says what is missing",
                     source="dataset_check/reports/*/report.json")
    ds = d["datasets"]
    names = [("fr3_duo_rand500", "Cube stacking", "490 ep · 6 fps"),
             ("fr3_duo_handover_v9_night_84", "Box handover (84)", "84 ep · 30 fps"),
             ("FR3Duo_Handover_Dataset_lerobot3", "Box handover (10)", "10 ep · 30 fps")]
    verdict_c = {"NOT READY": RED, "CONDITIONALLY READY": YELLOW, "READY": GREEN}
    for j, (k, n, sub) in enumerate(names):
        s = ds.get(k, {})
        x = 48 + j * 210
        box(ax, x, 122, 198, 116, fc=PAPER, r=8)
        txt(ax, x + 14, 136, n, size=12.5, weight="bold")
        txt(ax, x + 14, 158, sub, size=10, color=GREY)
        v = s.get("verdict", "?")
        pill(ax, x + 14, 196, v, fc=verdict_c.get(v, GREY), tc=WHITE if v != "CONDITIONALLY READY" else INK,
             size=9)
    # panel A: distinct motions
    aa = chart(fig, 48 + 160, 290, 440, 150)
    vals = [ds.get(k, {}).get("distinct_motions", 0) for k, _, _ in names]
    eps = [ds.get(k, {}).get("episodes", 1) for k, _, _ in names]
    y = np.arange(3)[::-1]
    aa.barh(y, eps, color=PAPER, height=0.6)
    aa.barh(y, vals, color=[PETROL, RED, RED], height=0.6)
    for yi, v, e in zip(y, vals, eps):
        aa.text(max(v, 4) + 8, yi, f"{v} distinct of {e}", va="center", fontsize=10.5, fontweight="bold",
                color=PETROL if v > 1 else RED)
    aa.set_yticks(y)
    aa.set_yticklabels([n for _, n, _ in names], fontsize=10.5)
    aa.set_xlim(0, 720)
    aa.set_xticks([])
    aa.spines["bottom"].set_visible(False)
    ov = overlay(fig)
    txt(ov, 48, 262, "Distinct motions (episodes after de-duplication of joint trajectories)", size=11.5,
        weight="bold")
    # panel B: idle + visual variety
    ab = chart(fig, 48 + 160, 500, 440, 130)
    idle = [ds.get(k, {}).get("idle_frac_mean", 0) * 100 for k, _, _ in names]
    ab.barh(y, idle, color=GREY2, height=0.6)
    for yi, v in zip(y, idle):
        ab.text(v + 0.8, yi, f"{v:.0f}% idle frames", va="center", fontsize=10.5, color=GREY)
    ab.set_yticks(y)
    ab.set_yticklabels([n for _, n, _ in names], fontsize=10.5)
    ab.set_xlim(0, 45)
    ab.set_xticks([])
    ab.spines["bottom"].set_visible(False)
    txt(ov, 48, 472, "Idle frames (robot standing still): pauses without a visual cue", size=11.5, weight="bold")
    # right: what is missing -> what we did
    rx = 720
    box(ov, rx, 122, W - 48 - rx, 576, fc=WHITE, ec=LINE, r=10)
    txt(ov, rx + 22, 140, "What the handover data cannot teach", size=15, weight="bold", color=RED)
    gaps = [("Box never moves", "box pose offsets are 0 in all 84; only the lighting changes"),
            ("No corrections", "one clean path: the model never sees how to recover from drift"),
            ("No failures", "every episode passes: no negative examples for a success judge"),
            ("Silent pause", "2.5 s wait before the right arm with nothing in the image to trigger it")]
    yy = 178
    for h, s in gaps:
        txt(ov, rx + 22, yy, "✕", size=14, color=RED, weight="bold", family="DejaVu Sans")
        txt(ov, rx + 48, yy, h, size=12, weight="bold")
        txt(ov, rx + 48, yy + 22, s, size=10.5, color=GREY, wrap=58)
        yy += 66
    ax2 = ov
    ax2.plot([rx + 22, W - 70], [yy + 2, yy + 2], color=LINE, lw=1)
    txt(ov, rx + 22, yy + 18, "What we did about it", size=15, weight="bold", color=PETROL)
    fixes = ["Generate demos in sim with the box moved (±15 mm, ±5°)",
             "Add 0.6° noise to executed motion, keep the clean label (DART)",
             "Cut idle stretches; keep failed episodes for analysis",
             "Fix the train / validation split before training"]
    yy += 54
    for s in fixes:
        txt(ov, rx + 22, yy, "✓", size=14, color=GREEN, weight="bold", family="DejaVu Sans")
        txt(ov, rx + 48, yy + 1, s, size=11, wrap=62)
        yy += 36
    k_ok = sum(r["success"] for r in d["expert"])
    box(ov, rx + 22, 620, W - 48 - rx - 44, 60, fc=PETROL_L, r=8)
    txt(ov, rx + 40, 650, f"v1 data: {k_ok} generated episodes, {k_ok} different box poses", size=12.5,
        weight="bold", color="#006B6B", va="center")
    save(fig, "F06_dataset_intelligence")


# ================================================================ F07 v0 failure analysis
def f07_v0(d):
    v0 = d["v0"]
    n = len(v0)
    fig, ax = canvas("Why the provided model fails in closed loop",
                     "Offline error is tiny (0.39°), yet the robot never completes the handover in simulation",
                     source=f"{n} v0 development runs, mixed settings (assist on/off, supervisor), sim/out/*/result.json")
    # funnel
    keys = [("left_grasp", "Left grasp"), ("right_grasp", "Right grasp"), ("handover", "Handover"),
            ("placed", "Placed")]
    af = chart(fig, 150, 160, 330, 250)
    ks = [sum(r[k] for r in v0) for k, _ in keys]
    y = np.arange(4)[::-1]
    af.barh(y, [n] * 4, color=PAPER, height=0.62)
    af.barh(y, ks, color=[ORANGE, ORANGE, ORANGE, RED], height=0.62)
    for yi, k in zip(y, ks):
        lo, hi = wilson(k, n)
        af.errorbar(k, yi, xerr=[[k - lo * n], [hi * n - k]], color=INK, capsize=4, lw=1.2)
        af.text(n + 0.8, yi, f"{k}/{n}", va="center", fontsize=12, fontweight="bold", color=INK)
        af.text(n + 5.0, yi, f"({lo * 100:.0f}-{hi * 100:.0f}%)", va="center", fontsize=9.5, color=GREY)
    af.set_yticks(y)
    af.set_yticklabels([l for _, l in keys], fontsize=11.5)
    af.set_xlim(0, n)
    af.set_xticks([])
    af.spines["bottom"].set_visible(False)
    ov = overlay(fig)
    txt(ov, 48, 124, "Milestones reached (95% Wilson interval)", size=12.5, weight="bold")
    # failure reasons
    from collections import Counter
    cnt = Counter(v0_failure_class(r) for r in v0)
    order = sorted(cnt.items(), key=lambda kv: -kv[1])
    ar = chart(fig, 220, 470, 330, 200)
    yy = np.arange(len(order))[::-1]
    colors = [GREY2 if "short" in k else (RED if "pause" in k or "over" in k else ORANGE) for k, _ in order]
    ar.barh(yy, [v for _, v in order], color=colors, height=0.62)
    for yi, (_, v) in zip(yy, order):
        ar.text(v + 0.25, yi, str(v), va="center", fontsize=11, fontweight="bold")
    ar.set_yticks(yy)
    ar.set_yticklabels([k for k, _ in order], fontsize=9.5)
    ar.set_xticks([])
    ar.spines["bottom"].set_visible(False)
    txt(ov, 48, 440, "How each run ended", size=12.5, weight="bold")
    # root causes
    rx = 640
    box(ov, rx, 122, W - 48 - rx, 576, fc=BLACK, r=10)
    txt(ov, rx + 24, 142, "Root causes, each checked on traces", size=15, weight="bold", color=ORANGE)
    causes = [("Covariate shift", "All demos follow one path. A 2-4 cm drift is a state the model never saw; "
               "the thin box tips over."),
              ("Narrow close window", "With recorded inputs it closes only for queries at frames 321-324; "
               "across plans the decision flickers."),
              ("Gripper value is a phase cue", "A held box read as 0.47 instead of 0.42 slows the carry and "
               "starts an early release."),
              ("Silent pause", "The demo waits 2.5 s with no visual cue: the policy waits forever."),
              ("Weak grip + jerky plans", "After the grasp the box slips or pivots in the hand.")]
    yy = 186
    for i, (h, s) in enumerate(causes):
        num_badge(ov, rx + 38, yy + 12, str(i + 1), fc=ORANGE, r=13, size=11)
        txt(ov, rx + 62, yy, h, size=12.5, weight="bold", color=WHITE)
        txt(ov, rx + 62, yy + 23, s, size=10.5, color="#C8C8C8", wrap=78)
        yy += 86
    box(ov, rx + 24, 640, W - 48 - rx - 48, 44, fc=ORANGE, r=6)
    txt(ov, rx + 40, 662, "Lesson: fix the data, not the controller. Measure over N runs, never one.", size=12,
        weight="bold", color=WHITE, va="center")
    save(fig, "F07_v0_failure_analysis")


# ================================================================ F08 v1 data generation
def f08_datagen(d):
    ex = d["expert"]
    n, k = len(ex), sum(r["success"] for r in ex)
    lo, hi = wilson(k, n)
    fig, ax = canvas("Fixing the data: an expert that generates demonstrations in simulation",
                     "Same task, box moved every episode, small noise on the motion, failures filtered out",
                     source="sim/expert_demos.py · sim/out/expert_demos*/**/meta.json · split sim/out/expert_split_v1.json")
    # scatter
    asc = chart(fig, 100, 150, 330, 330)
    asc.add_patch(patches.Rectangle((-15, -15), 30, 30, fc=PETROL_L, ec="none", zorder=0))
    for r in ex:
        if r["success"]:
            asc.scatter(r["dx_mm"], r["dz_mm"], s=90, color=PETROL, edgecolor=WHITE, lw=1.2, zorder=3)
        else:
            asc.scatter(r["dx_mm"], r["dz_mm"], s=120, color=RED, marker="X", edgecolor=WHITE, lw=1, zorder=4)
    asc.scatter([0], [0], s=160, marker="*", color=ORANGE, zorder=5, edgecolor=WHITE)
    asc.set_xlim(-20, 20)
    asc.set_ylim(-20, 20)
    asc.set_aspect("equal")
    asc.set_xlabel("box shift x (mm)")
    asc.set_ylabel("box shift along cart (mm)")
    asc.axhline(0, color=LINE, lw=0.8)
    asc.axvline(0, color=LINE, lw=0.8)
    asc.text(-14, -14, "sampled \u00b115 mm, \u00b15\u00b0", fontsize=9.5, color="#006B6B", va="bottom")
    ov = overlay(fig)
    txt(ov, 48, 120, "Where the box was placed (each dot = one episode)", size=12.5, weight="bold")
    txt(ov, 100, 540, "● success", size=10.5, color=PETROL, weight="bold")
    txt(ov, 190, 540, "\u2716 failure", size=10.5, color=RED, weight="bold", family="DejaVu Sans")
    txt(ov, 290, 540, "\u2605 original 84 demos", size=10.5, color=ORANGE, weight="bold", family="DejaVu Sans")
    txt(ov, 100, 562, "Failures: box at -14.8 mm knocked over on approach; box slipped from the left grip.",
        size=10, color=GREY, wrap=56)
    # grasp alignment
    ag = chart(fig, 560, 150, 300, 330)
    eL = [r["err_L"] for r in ex if r["success"] and r["err_L"] is not None]
    eR = [r["err_R"] for r in ex if r["success"] and r["err_R"] is not None]
    rng = np.random.default_rng(0)
    for xi, e, c in ((0, eL, ORANGE), (1, eR, PETROL)):
        ag.scatter(xi + rng.uniform(-0.15, 0.15, len(e)), e, s=55, color=c, alpha=0.85, edgecolor=WHITE)
        ag.plot([xi - 0.25, xi + 0.25], [np.median(e)] * 2, color=INK, lw=2)
    ag.axhline(4, color=RED, lw=1.2, ls="--")
    ag.text(1.45, 4.1, "close only below 4 mm", color=RED, fontsize=9.5, ha="right", va="bottom")
    ag.set_xticks([0, 1])
    ag.set_xticklabels(["left grasp", "right grasp"], fontsize=11)
    ag.set_ylim(0, 5)
    ag.set_xlim(-0.6, 1.6)
    ag.set_ylabel("gripper-to-box error at close (mm)")
    txt(ov, 520, 120, "Grasp alignment of the expert", size=12.5, weight="bold")
    # KPIs
    rx = 940
    kpis = [(f"{k} / {n}", f"episodes succeed ({lo * 100:.0f}-{hi * 100:.0f}% CI)", PETROL),
            ("34 → 6 min", "per episode: real-time RTX instead of path tracing", ORANGE),
            ("14 + 2", "train / held-out episodes, split fixed before training", BLACK),
            ("29,055", "new training frames for v1 (+3,000 LoRA steps)", BLACK)]
    yy = 122
    for big, small, c in kpis:
        box(ov, rx, yy, W - 48 - rx, 102, fc=PAPER, r=8)
        ov.add_patch(patches.Rectangle((rx, yy), 5, 102, fc=c, zorder=3))
        txt(ov, rx + 20, yy + 12, big, size=24, weight="bold", color=c)
        txt(ov, rx + 20, yy + 56, small, size=10.5, color=GREY, wrap=40)
        yy += 112
    box(ov, 48, 600, W - 96, 94, fc=BLACK, r=8)
    txt(ov, 66, 614, "How the expert works", size=12, weight="bold", color=ORANGE)
    txt(ov, 66, 640, "Plays the recorded motion in time, pauses the clock at each grasp until a Jacobian correction "
        "centres the gripper on the moved box (< 4 mm), then continues. Executed motion gets 0.6° noise; "
        "the label stays clean, so the model learns to correct.", size=11, color=WHITE, wrap=160)
    save(fig, "F08_v1_data_generation")


# ================================================================ F09 offline + latency
def f09_offline(d):
    fig, ax = canvas("Offline accuracy and speed",
                     "Fine-tuning matters, and the policy is fast enough for real-time control",
                     source="pi05/README.md (9 held-out episodes) · in-sim latency from v0 runs")
    aa = chart(fig, 110, 160, 330, 300)
    aa.bar([0, 1], [6.79, 0.39], color=[GREY2, ORANGE], width=0.55)
    for x, v in ((0, 6.79), (1, 0.39)):
        aa.text(x, v + 0.15, f"{v:.2f}°", ha="center", fontsize=15, fontweight="bold",
                color=ORANGE if x else GREY)
    aa.set_xticks([0, 1])
    aa.set_xticklabels(["π0.5 base\n(zero-shot)", "π0.5 fine-tuned\n(checkpoint v0)"], fontsize=11)
    aa.set_yticks([])
    aa.spines["left"].set_visible(False)
    ab = chart(fig, 520, 160, 330, 300)
    ab.bar([0, 1], [0.248, 0.006], color=[GREY2, PETROL], width=0.55)
    for x, v in ((0, 0.248), (1, 0.006)):
        ab.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=15, fontweight="bold", color=PETROL if x else GREY)
    ab.set_xticks([0, 1])
    ab.set_xticklabels(["π0.5 base\n(zero-shot)", "π0.5 fine-tuned\n(checkpoint v0)"], fontsize=11)
    ab.set_yticks([])
    ab.spines["left"].set_visible(False)
    ov = overlay(fig)
    txt(ov, 110, 124, "Arm error per joint (mean abs., deg)", size=12.5, weight="bold")
    txt(ov, 520, 124, "Gripper error (0 … 1 scale)", size=12.5, weight="bold")
    # latency
    lat = [r["lat"]["p50"] for r in d["v0"] if r.get("lat")]
    lat95 = [r["lat"]["p95"] for r in d["v0"] if r.get("lat")]
    al = chart(fig, 1010, 160, 270, 300)
    rng = np.random.default_rng(1)
    al.axhspan(0, 333, color=PETROL_L, zorder=0, lw=0)
    al.scatter(rng.uniform(-0.18, 0.18, len(lat)), lat, s=40, color=ORANGE, edgecolor=WHITE, zorder=3)
    al.scatter(1 + rng.uniform(-0.18, 0.18, len(lat95)), lat95, s=40, color=PETROL, edgecolor=WHITE, zorder=3)
    al.axhline(333, color=RED, lw=1.5)
    al.text(1.45, 340, "333 ms = motion per plan", color=RED, fontsize=9.5, ha="right", va="bottom")
    al.text(0.5, 250, "real-time budget", color="#006B6B", fontsize=10, ha="center", fontweight="bold")
    al.set_xticks([0, 1])
    al.set_xticklabels(["p50", "p95"], fontsize=11)
    al.set_xlim(-0.5, 1.5)
    al.set_ylim(0, 380)
    al.set_ylabel("ms per 10-step plan")
    txt(ov, 960, 124, f"Latency in {len(lat)} sim runs", size=12.5, weight="bold")
    txt(ov, 960, 486, f"median p50 {np.median(lat):.0f} ms, worst p95 {max(lat95):.0f} ms", size=10.5,
        color=GREY, wrap=44)
    box(ov, 48, 560, W - 96, 132, fc=ORANGE_L, r=10)
    txt(ov, 70, 580, "But: offline error is not task success", size=15, weight="bold", color=ORANGE)
    txt(ov, 70, 612, "0.39° average error on recorded frames still drifts 2-4 cm in closed loop, because "
        "each small error moves the robot into states the demonstrations never showed. That is why stage 04 "
        "evaluates in the simulator, over many runs, with milestones and confidence intervals.", size=12,
        wrap=150)
    save(fig, "F09_offline_accuracy_latency")


# ================================================================ F10 v0 vs v1
def f10_v0v1(d):
    fig, ax = canvas("v0 → v1: does generated data help?",
                     "Controlled evaluation: raw policy (no assists), new box positions, same render mode, N runs each",
                     source="Fill presentation/data/v0_v1_eval.json from sim/eval_summary.py, then re-run this script")
    keys = ["left_grasp", "right_grasp", "handover", "placed"]
    labs = ["Left grasp", "Right grasp", "Handover", "Placed"]
    ev = d["v0v1"]
    a = chart(fig, 110, 150, 820, 430)
    xs = np.arange(4)
    w = 0.36
    for j, (ver, c) in enumerate((("v0", GREY2), ("v1", ORANGE))):
        if ev and ver in ev:
            n = ev[ver]["n"]
            for i, k in enumerate(keys):
                kk = ev[ver][k]
                lo, hi = wilson(kk, n)
                p = kk / n
                a.bar(xs[i] + (j - 0.5) * w, p * 100, w * 0.92, color=c)
                a.errorbar(xs[i] + (j - 0.5) * w, p * 100, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK,
                           capsize=4, lw=1.2)
                a.text(xs[i] + (j - 0.5) * w, hi * 100 + 2, f"{kk}/{n}", ha="center", fontsize=10.5,
                       fontweight="bold")
        elif j == 0:
            a.text(1.5, 55, "Results pending: v1 is training (23 Oct, 3,000 steps)\n"
                   "Run sim/eval_batch.sh for v0 and v1, then fill presentation/data/v0_v1_eval.json",
                   ha="center", va="center", fontsize=12, color=GREY,
                   bbox=dict(boxstyle="round,pad=0.8", fc=PAPER, ec=LINE))
    a.set_xticks(xs)
    a.set_xticklabels(labs, fontsize=12)
    a.set_ylim(0, 110)
    a.set_xlim(-0.5, 3.5)
    a.set_ylabel("runs reaching milestone (%)")
    a.grid(axis="y", color=PAPER)
    ov = overlay(fig)
    box(ov, 110, 112, 20, 14, fc=GREY2, r=2)
    txt(ov, 136, 119, "v0  provided checkpoint", size=10.5, va="center")
    box(ov, 330, 112, 20, 14, fc=ORANGE, r=2)
    txt(ov, 356, 119, "v1  + 14 generated expert demos", size=10.5, va="center")
    rx = 980
    box(ov, rx, 150, W - 48 - rx, 430, fc=PAPER, r=10)
    txt(ov, rx + 20, 168, "Test protocol", size=14, weight="bold", color=ORANGE)
    proto = ["Raw policy only: no alignment assist, no latches, no retries",
             "New box seeds (20000+), normal and wide (±25 mm) sets",
             "Same RTX real-time render mode for both versions",
             "Per-milestone success with 95% Wilson interval",
             "v1 offline error on the 2 held-out episodes"]
    yy = 204
    for s in proto:
        txt(ov, rx + 20, yy, "•", size=13, color=ORANGE)
        txt(ov, rx + 36, yy + 1, s, size=10.8, wrap=30)
        yy += 70
    txt(ov, 110, 610, "Expected: fewer drift failures near the box. Not fixable by v1: poses the expert itself "
        "fails on (> 15 mm shift) → v2 adds DAgger corrections on v1's own rollouts.", size=11.5, color=GREY,
        wrap=150)
    save(fig, "F10_v0_vs_v1_results")


# ================================================================ F11 bigger picture
def f11_bigger(d):
    fig, ax = canvas("The bigger picture: from one handover to a cathlab companion",
                     "Same loop, more skills; autonomy grows only as fast as the evidence",
                     source="Roadmap is a proposal; items beyond 'Now' are not built yet")
    cols = [("NOW", "Hackathon", ORANGE,
             ["Box handover in Isaac Sim", "Loop 01-05 runs end to end", "v0 failure analysis, v1 data",
              "Supervisor + interlocks"]),
            ("NEXT", "Weeks", PETROL,
             ["DAgger: expert labels v1's own mistakes", "More objects in the scene: scissors, pill bottle, "
              "glass plate", "VLM success judge on run videos", "Domain randomisation for real cameras"]),
            ("THEN", "Months", NAVY,
             ["Real FR3 Duo, same policy server", "Nurse-in-the-loop labels from the tablet",
              "Skill library: fetch, hand over, restock storage", "Orchestrator picks skill per request"]),
            ("LATER", "Network", BLACK,
             ["Share validated skills across robots and sites", "Each site's failures improve all",
              "Versioned, signed-off model releases", "Audit trail for every run"])]
    x0, cw, g = 48, 294, 18
    for i, (k, when, c, items) in enumerate(cols):
        x = x0 + i * (cw + g)
        box(ax, x, 122, cw, 300, fc=WHITE, ec=LINE, r=10)
        box(ax, x, 122, cw, 52, fc=c, r=10, z=2)
        ax.add_patch(patches.Rectangle((x, 152), cw, 22, fc=c, zorder=2))
        txt(ax, x + 18, 148, k, size=17, weight="bold", color=WHITE, va="center", z=6)
        txt(ax, x + cw - 18, 148, when, size=11, color=WHITE, va="center", ha="right", z=6)
        y = 190
        for s in items:
            txt(ax, x + 18, y, "•", size=13, color=c if c != BLACK else ORANGE)
            txt(ax, x + 34, y + 1, s, size=11, wrap=36)
            y += 56
        if i < 3:
            arrow(ax, (x + cw + 2, 148), (x + cw + g - 2, 148), color=GREY2, lw=2, head=10)
    # cathlab skill map with hero image
    place(fig, os.path.join(ASSET, "sim_robot_gui.png"), 48, 444, 380, 250, crop=(190, 240, 1625, 900),
          border=LINE)
    ov = overlay(fig)
    box(ov, 58, 454, 168, 26, fc=BLACK, r=5, z=8, alpha=0.85)
    txt(ov, 68, 467, "today: one skill in sim", size=10, color=WHITE, va="center", weight="bold", z=9)
    sx = 450
    txt(ov, sx, 446, "Cathlab skills the same loop can teach", size=14, weight="bold")
    skills = [("Hand over supplies", "box, tray, packs", True), ("Fetch from dynamic storage", "drawers, shelves",
                                                                 False),
              ("Pass instruments", "scissors, bowls", False), ("Prepare the table", "lay out a procedure set",
                                                               False),
              ("Restock & tidy", "after the case", False), ("Bring on request", "voice or tablet", False)]
    for j, (h, s, done) in enumerate(skills):
        cx = sx + (j % 3) * 280
        cy = 482 + (j // 3) * 66
        box(ov, cx, cy, 266, 56, fc=ORANGE_L if done else PAPER, ec=ORANGE if done else "none", r=8, lw=1.5)
        txt(ov, cx + 14, cy + 10, h, size=11.5, weight="bold")
        txt(ov, cx + 14, cy + 32, s + ("  ·  in sim now" if done else ""), size=9.8,
            color=ORANGE if done else GREY, weight="bold" if done else "normal")
    box(ov, sx, 622, W - 48 - sx, 72, fc=BLACK, r=8)
    txt(ov, sx + 18, 658, "Each new skill = new demos + the same 5 stages. The nurse decides what the robot learns "
        "next.", size=12.5, color=WHITE, weight="bold", va="center", wrap=100)
    save(fig, "F11_bigger_picture_roadmap")


# ================================================================ F12 safety layers
def f12_safety(d):
    fig, ax = canvas("Safety by layers, human on top",
                     "The learned policy never acts alone: rules, a supervisor and the nurse sit between it and the "
                     "world", source="sim/policy_rollout.py, sim/recovery.py (runtime layers in our sim stack)")
    layers = [("Nurse", "starts, watches, stops, confirms; signs off new versions", PETROL, WHITE),
              ("Supervisor", "grasp miss · stall 5 s · tip > 35° · drop → rewind & retry "
               "(max 2) or ask", BLACK, WHITE),
              ("Interlocks", "left opens only after right holds; right opens only over the tray", NAVY, WHITE),
              ("Limits", "joint limits, smoothing after grasp, gripper force cap", GREY, WHITE),
              ("π0.5 policy", "images + instruction → joint targets", ORANGE, WHITE)]
    cx, cy = 340, 420
    for i, (h, s, c, tc) in enumerate(layers):
        r = 270 - i * 52
        ax.add_patch(patches.Circle((cx, cy), r, fc=c, ec=WHITE, lw=3, zorder=1 + i))
        if i < 4:
            txt(ax, cx, cy - r + 26, h, size=12.5 if i else 13.5, weight="bold", color=tc, ha="center",
                va="center", z=10)
    txt(ax, cx, cy, "π0.5\npolicy", size=15, weight="bold", color=WHITE, ha="center", va="center", z=10)
    rx = 680
    yy = 136
    for h, s, c, _ in layers:
        box(ax, rx, yy, W - 48 - rx, 98, fc=PAPER, r=8)
        ax.add_patch(patches.Rectangle((rx, yy), 6, 98, fc=c, zorder=3))
        txt(ax, rx + 24, yy + 18, h, size=14, weight="bold", color=c if c not in (GREY,) else INK)
        txt(ax, rx + 24, yy + 46, s, size=11.5, color=INK, wrap=72)
        yy += 112
    save(fig, "F12_safety_layers")


# ================================================================ F13 operator interface
def f13_interface(d):
    fig, ax = canvas("Operator interface: what the nurse sees",
                     "Concept for the bedside tablet, built on the live view and run logs we already produce",
                     source="Concept mock-up; camera images from a simulated episode")
    # tablet frame
    tx, ty, tw, th = 48, 116, 860, 580
    box(ax, tx, ty, tw, th, fc=BLACK, r=22)
    sx, sy, sw, sh = tx + 18, ty + 18, tw - 36, th - 36
    box(ax, sx, sy, sw, sh, fc="#F7F7F7", r=10, z=2)
    # top bar
    box(ax, sx, sy, sw, 44, fc=WHITE, r=10, z=3)
    txt(ax, sx + 18, sy + 22, "Cathlab Companion", size=13, weight="bold", color=ORANGE, va="center")
    pill(ax, sx + 210, sy + 10, "Box handover · policy v1", fc=PETROL_L, tc="#006B6B", size=9.5, z=8)
    pill(ax, sx + sw - 16, sy + 10, "● RUNNING", fc=GREEN, size=9.5, ha="right", z=8)
    # cameras
    cwid = (sw - 52) / 3
    for j, c in enumerate(("external", "left_wrist", "right_wrist")):
        place(fig, frame(c, 4), sx + 16 + j * (cwid + 10), sy + 58, cwid, 190 if j == 0 else 190)
    ov = overlay(fig)
    for j, lab in enumerate(("Room", "Left hand", "Right hand")):
        box(ov, sx + 24 + j * (cwid + 10), sy + 66, 80, 22, fc=BLACK, r=4, z=8, alpha=0.75)
        txt(ov, sx + 64 + j * (cwid + 10), sy + 77, lab, size=9, color=WHITE, ha="center", va="center", z=9)
    # progress steps
    py = sy + 266
    txt(ov, sx + 16, py, "Progress", size=11, weight="bold", color=GREY)
    steps = [("Left grasp", "done"), ("Carry", "now"), ("Handover", ""), ("Right carry", ""), ("Place", "")]
    sw5 = (sw - 32) / 5
    for j, (s, st) in enumerate(steps):
        x = sx + 16 + j * sw5
        fc = PETROL if st == "done" else (ORANGE if st == "now" else LINE)
        box(ov, x, py + 22, sw5 - 8, 10, fc=fc, r=5)
        txt(ov, x, py + 40, s + ("  ✓" if st == "done" else ""), size=10.5,
            weight="bold" if st else "normal", color=INK if st else GREY2)
    # status + controls
    by = py + 80
    box(ov, sx + 16, by, 420, 150, fc=WHITE, ec=LINE, r=8)
    txt(ov, sx + 32, by + 14, "Status", size=11, weight="bold", color=GREY)
    lines = [("Grip on box", "secure", GREEN), ("Box upright", "4° tilt", GREEN),
             ("Plan latency", "95 ms", GREEN), ("Retries used", "0 of 2", GREY)]
    for k, (a, b, c) in enumerate(lines):
        txt(ov, sx + 32, by + 42 + k * 25, a, size=10.5)
        txt(ov, sx + 420, by + 42 + k * 25, b, size=10.5, weight="bold", color=c, ha="right")
    bx = sx + 456
    box(ov, bx, by, sw - 472, 66, fc=RED, r=10)
    txt(ov, bx + (sw - 472) / 2, by + 33, "STOP", size=20, weight="bold", color=WHITE, ha="center", va="center")
    bw2 = (sw - 472 - 10) / 2
    box(ov, bx, by + 78, bw2, 72, fc=GREEN_L, ec=GREEN, r=10, lw=1.5)
    txt(ov, bx + bw2 / 2, by + 114, "✓  Done right", size=12, weight="bold", color=GREEN, ha="center",
        va="center")
    box(ov, bx + bw2 + 10, by + 78, bw2, 72, fc=RED_L, ec=RED, r=10, lw=1.5)
    txt(ov, bx + bw2 * 1.5 + 10, by + 114, "✕  Not right", size=12, weight="bold", color=RED, ha="center",
        va="center")
    # callouts right
    rx = 940
    callouts = [("Plain-language request", "Voice or tap; the orchestrator maps it to a skill and version."),
                ("Live cameras", "The same three images the policy sees: no hidden state."),
                ("Progress by milestone", "The evaluation milestones double as the nurse's progress bar."),
                ("One big STOP", "Always visible; pause and resume without losing the task."),
                ("One-tap label", "“Done right / not right” becomes a success label for the next "
                 "training cycle.")]
    yy = 122
    for i, (h, s) in enumerate(callouts):
        num_badge(ov, rx + 14, yy + 14, str(i + 1), fc=ORANGE, r=14, size=11)
        txt(ov, rx + 40, yy + 3, h, size=12.5, weight="bold")
        txt(ov, rx + 40, yy + 27, s, size=10.5, color=GREY, wrap=44)
        yy += 114
    save(fig, "F13_operator_interface")


FIGS = {"F00": f00_story, "F01": f01_problem, "F02": f02_interaction, "F03": f03_pipeline, "F04": f04_policy_io,
        "F05": f05_storyboard, "F06": f06_dataset, "F07": f07_v0, "F08": f08_datagen, "F09": f09_offline,
        "F10": f10_v0v1, "F11": f11_bigger, "F12": f12_safety, "F13": f13_interface}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    data = load_data()
    for k, f in FIGS.items():
        if not args.only or k in args.only:
            f(data)
