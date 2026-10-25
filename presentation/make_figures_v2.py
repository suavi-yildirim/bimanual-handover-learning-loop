"""v2 figures: minimal text, one idea per figure. Reuses helpers and data from make_figures.py.

Run: $ISAAC_PY presentation/make_figures_v2.py [--only V01 V07]
Writes presentation/figures_v2/<name>.png (200 dpi, 16:9) and .pdf.
"""
import argparse
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import make_figures as mf
from make_figures import (ASSET, BLACK, GREEN, GREEN_L, GREY, GREY2, H, INK, LINE, NAVY, NAVY_L, ORANGE, ORANGE_L,
                          ORANGE_M, PAPER, PETROL, PETROL_L, PETROL_M, RED, RED_L, SKY, W, WHITE, YELLOW_L, arrow, box,
                          chart, frame, num_badge, overlay, person, place, robot, txt, wilson)

OUT2 = os.path.join(mf.PRES, "figures_v2")
PETROL_D = "#006B6B"


def canvas(title, source=None):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.text(48, 70, title, fontsize=32, fontweight="bold", color=ORANGE, va="baseline")
    if source:
        ax.text(W - 48, 728, source, fontsize=8.5, color=GREY2, va="center", ha="right")
    return fig, ax


def save(fig, name):
    os.makedirs(OUT2, exist_ok=True)
    fig.savefig(os.path.join(OUT2, name + ".png"), dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(os.path.join(OUT2, name + ".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name)


def big_stat(ax, x, y, big, small, color, w=250):
    txt(ax, x, y, big, size=40, weight="bold", color=color)
    txt(ax, x, y + 66, small, size=14, color=GREY, wrap=int(w / 8))


def ring_nodes(ax, cx, cy, R, labels, colors, r=34, size=15, lab_size=15, start=-90, ring_c=PETROL_M, lw=10,
               icons=None):
    n = len(labels)
    ang = [start + i * 360 / n for i in range(n)]
    ax.add_patch(patches.Circle((cx, cy), R, fc="none", ec=ring_c, lw=lw, zorder=2))
    for i in range(n):
        a1 = ang[(i + 1) % n] + (360 if i == n - 1 else 0) - math.degrees(r / R) - 2
        p0 = (cx + R * math.cos(math.radians(a1 - 5)), cy + R * math.sin(math.radians(a1 - 5)))
        p1 = (cx + R * math.cos(math.radians(a1)), cy + R * math.sin(math.radians(a1)))
        arrow(ax, p0, p1, color=colors[i] if colors[i] != BLACK else INK, lw=0.1, head=22, z=3)
    pts = []
    for i, (lab, a) in enumerate(zip(labels, ang)):
        px, py = cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))
        pts.append((px, py))
        ax.add_patch(patches.Circle((px, py), r, fc=colors[i], ec=WHITE, lw=3, zorder=5))
        if icons and icons[i]:
            icons[i](ax, px, py)
        else:
            txt(ax, px, py + 1, f"{i + 1:02d}", size=size, weight="bold", color=WHITE, ha="center", va="center",
                z=7)
        ox, oy = math.cos(math.radians(a)), math.sin(math.radians(a))
        ha = "center" if abs(ox) < 0.3 else ("left" if ox > 0 else "right")
        va = "center" if abs(oy) < 0.5 else ("bottom" if oy < 0 else "top")
        txt(ax, px + ox * (r + 14), py + oy * (r + 14), lab, size=lab_size, weight="bold", ha=ha, va=va)
    return pts


# ================================================================ V01 big picture in one image
def v01_story(d):
    fig, ax = canvas("A robot that learns new cathlab tasks")
    place(fig, os.path.join(ASSET, "sim_robot_closeup.png"), 48, 110, 560, 470, crop=(0, 0, 578, 492))
    ov = overlay(fig)
    person(ov, 104, 560, s=1.3, color=PETROL)
    mf.bubble(ov, 150, 470, 330, 62, "“Hand me the box.”", tail=(132, 508), size=17)
    # loop
    labels = ["Ingest", "Check data", "Train", "Evaluate\nin sim", "Serve"]
    cols = [PETROL, PETROL, PETROL, ORANGE, PETROL]
    ring_nodes(ov, 960, 330, 165, labels, cols, r=36, size=16, lab_size=16)
    txt(ov, 960, 318, "learning", size=18, color=GREY, ha="center", va="center")
    txt(ov, 960, 344, "loop", size=18, color=GREY, ha="center", va="center")
    arrow(ov, (620, 330), (720, 330), color=ORANGE, lw=4, head=24)
    # numbers
    hd = d["datasets"].get("fr3_duo_handover_v9_night_84", {})
    v0, ex = d["v0"], d["expert"]
    att = [r for r in v0 if r["attempt"]]
    stats = [(f"{hd.get('episodes', 84)} → {hd.get('distinct_motions', 1)}", "demos → motions", BLACK),
             (f"{sum(r['left_grasp'] for r in att)}/{len(att)}", "v0 grasps and lifts", ORANGE),
             (f"{sum(r['success'] for r in ex)}/{len(ex)}", "new demos succeed", PETROL)]
    for i, (b, s, c) in enumerate(stats):
        x = 48 + i * 196
        txt(ov, x, 606, b, size=30, weight="bold", color=c)
        txt(ov, x, 654, s, size=13, color=GREY)
    txt(ov, 740, 640, "Failures in simulation become the next training data.", size=16, weight="bold",
        color=PETROL_D, wrap=48)
    save(fig, "V01_big_picture")


# ================================================================ V02 problem
def v02_problem(d):
    fig, ax = canvas("The problem")
    cols = [(48, "Nurses are overloaded", "staff shortage in the cathlab", PETROL, "person"),
            (474, "Robots are programmed", "one engineer per task", GREY, "robot_x"),
            (900, "Robots that learn", "from demonstrations", ORANGE, "robot")]
    for i, (x, h, s, c, ic) in enumerate(cols):
        box(ax, x, 130, 385, 420, fc=ORANGE_L if i == 2 else PAPER, r=14)
        cx = x + 192
        if ic == "person":
            person(ax, cx, 330, s=3.2, color=c)
        else:
            robot(ax, cx, 340, s=2.6, color=c)
        if ic == "robot_x":
            ax.plot([cx - 70, cx + 70], [190, 330], color=RED, lw=10, solid_capstyle="round", zorder=9)
            ax.plot([cx - 70, cx + 70], [330, 190], color=RED, lw=10, solid_capstyle="round", zorder=9)
        txt(ax, cx, 410, h, size=22, weight="bold", ha="center", color=INK)
        txt(ax, cx, 450, s, size=16, ha="center", color=GREY)
        if i < 2:
            arrow(ax, (x + 392, 340), (cols[i + 1][0] - 8, 340), color=ORANGE, lw=4, head=24)
    box(ax, 48, 586, W - 96, 96, fc=BLACK, r=12)
    txt(ax, W / 2, 634, "We build the loop that teaches, tests and ships robot skills.", size=21, weight="bold",
        color=WHITE, ha="center", va="center")
    save(fig, "V02_problem")


# ================================================================ V03 nurse-robot collaboration
def _ico_text(sym, size=24):
    def f(ax, x, y):
        txt(ax, x, y + 1, sym, size=size, weight="bold", color=WHITE, ha="center", va="center", z=8,
            family="DejaVu Sans")
    return f


def v03_team(d):
    fig, ax = canvas("Nurse and robot, one team")
    labels = ["Nurse asks", "Robot acts", "Supervisor\nchecks", "Nurse confirms", "Loop learns"]
    cols = [PETROL, ORANGE, BLACK, PETROL, NAVY]
    icons = [lambda a, x, y: person(a, x, y + 22, s=0.85, color=WHITE),
             lambda a, x, y: robot(a, x, y + 26, s=0.65, color=WHITE),
             _ico_text("⚠", 26), _ico_text("✓", 28), _ico_text("↻", 28)]
    cx, cy = 430, 405
    ring_nodes(ax, cx, cy, 225, labels, cols, r=46, lab_size=17, icons=icons, ring_c=LINE, lw=8)
    place(fig, os.path.join(ASSET, "sim_robot_closeup.png"), cx - 120, cy - 90, 240, 180, crop=(60, 0, 578, 492))
    # right: three rules
    rules = [("Robot", "task autonomy, no patient contact", ORANGE),
             ("Supervisor", "retries, then asks the nurse", BLACK),
             ("Nurse", "signs off every new version", PETROL)]
    y = 180
    for h, s, c in rules:
        box(ax, 820, y, 465, 120, fc=PAPER, r=12)
        ax.add_patch(patches.Circle((862, y + 60), 18, fc=c, zorder=4))
        txt(ax, 896, y + 30, h, size=20, weight="bold", color=c if c != BLACK else INK)
        txt(ax, 896, y + 66, s, size=15, color=GREY)
        y += 140
    save(fig, "V03_nurse_robot_team")


# ================================================================ V04 pipeline flowchart
def v04_pipeline(d):
    fig, ax = canvas("The learning loop we built")
    ds, v0, ex = d["datasets"], d["v0"], d["expert"]
    hd = ds.get("fr3_duo_handover_v9_night_84", {})
    stages = [("01", "Ingest", f"{sum(v['episodes'] for v in ds.values()) + len(ex)}", "episodes"),
              ("02", "Check data", f"{hd.get('distinct_motions', 1)}/{hd.get('episodes', 84)}", "distinct"),
              ("03", "Fine-tune", "0.39°", "offline error"),
              ("04", "Evaluate", f"{sum(r['left_grasp'] for r in v0 if r['attempt'])}/{sum(r['attempt'] for r in v0)}",
               "v0 grasps, 0 places"),
              ("05", "Serve", "95 ms", "per plan")]
    x0, cw, g, y0, ch = 60, 222, 25, 120, 230
    for i, (n, name, big, small) in enumerate(stages):
        x = x0 + i * (cw + g)
        hl = n == "04"
        box(ax, x, y0, cw, ch, fc=ORANGE if hl else BLACK, r=14)
        txt(ax, x + 20, y0 + 20, n, size=18, weight="bold", color=WHITE if hl else ORANGE)
        txt(ax, x + 20, y0 + 52, name, size=21, weight="bold", color=WHITE)
        txt(ax, x + 20, y0 + 118, big, size=34, weight="bold", color=WHITE if hl else PETROL_M)
        txt(ax, x + 20, y0 + 172, small, size=14, color=WHITE if hl else GREY2)
        if i < 4:
            arrow(ax, (x + cw + 3, y0 + ch / 2), (x + cw + g - 3, y0 + ch / 2), color=ORANGE, lw=3, head=14)
    yl = y0 + ch + 30
    xa, xb = x0 + 4 * (cw + g) + cw / 2, x0 + cw / 2
    ax.plot([xa, xa, xb], [y0 + ch, yl, yl], color=ORANGE, lw=4, zorder=2, solid_capstyle="round")
    arrow(ax, (xb, yl), (xb, y0 + ch + 2), color=ORANGE, lw=4, head=20)
    box(ax, W / 2 - 170, yl - 16, 340, 32, fc=WHITE, r=4, z=3)
    txt(ax, W / 2, yl, "failures → new demos", size=16, weight="bold", color=ORANGE, ha="center", va="center")
    ys = yl + 34
    hs = 712 - ys
    place(fig, os.path.join(ASSET, "sim_robot_gui.png"), 60, ys, 600, hs, crop=(190, 236, 1340, 900), border=LINE)
    fw = (W - 60 - 690) / 3 - 10
    for j, i in enumerate((3, 5, 9)):
        place(fig, frame("external", i), 690 + j * (fw + 15), ys, fw, hs, border=LINE, lw=1)
    ov = overlay(fig)
    box(ov, 72, ys + 12, 180, 30, fc=BLACK, r=6, z=8, alpha=0.85)
    txt(ov, 82, ys + 27, "Isaac Sim", size=13, color=WHITE, weight="bold", va="center", z=9)
    for j, s in enumerate(("grasp", "handover", "placed")):
        box(ov, 700 + j * (fw + 15), ys + 12, 104, 28, fc=BLACK, r=6, z=8, alpha=0.8)
        txt(ov, 752 + j * (fw + 15), ys + 26, s, size=12, color=WHITE, weight="bold", ha="center", va="center", z=9)
    save(fig, "V04_learning_loop")


# ================================================================ V05 policy in/out
def v05_io(d):
    fig, ax = canvas("What goes in, what comes out")
    for j, c in enumerate(("external", "left_wrist", "right_wrist")):
        place(fig, frame(c, 3), 48 + j * 130, 140, 120, 120, border=LINE, lw=1)
    txt(ax, 48, 272, "3 cameras", size=16, weight="bold")
    box(ax, 48, 330, 380, 64, fc=WHITE, ec=BLACK, r=12, lw=2)
    txt(ax, 238, 362, "“Hand the box over\nand place it on the tray.”", size=13, ha="center", va="center")
    txt(ax, 48, 404, "instruction", size=16, weight="bold")
    for row, c in enumerate((ORANGE, PETROL)):
        for i in range(8):
            box(ax, 48 + i * 47, 462 + row * 46, 42, 38, fc=BLACK if i == 7 else c, r=5,
                alpha=0.9 if i == 7 else 0.35)
    txt(ax, 48, 560, "16 joints", size=16, weight="bold")
    mx, my, mw, mh = 520, 200, 280, 300
    box(ax, mx, my, mw, mh, fc=BLACK, r=16)
    txt(ax, mx + mw / 2, my + 120, "π0.5", size=54, weight="bold", color=ORANGE, ha="center", va="center")
    txt(ax, mx + mw / 2, my + 200, "vision-language-action", size=15, color=WHITE, ha="center", va="center")
    txt(ax, mx + mw / 2, my + 232, "LoRA fine-tuned", size=13, color=GREY2, ha="center", va="center")
    for yy in (200, 362, 505):
        arrow(ax, (440, yy), (mx - 8, my + mh / 2 + (yy - 362) * 0.3), color=GREY2, lw=2.5, head=14)
    arrow(ax, (mx + mw + 8, my + mh / 2), (868, my + mh / 2), color=ORANGE, lw=4, head=22)
    a = d["ep_action"]
    t0, t1, c0 = 300, 360, 322
    t = np.arange(t0, t1) / 30.0
    ac = chart(fig, 900, 160, 385, 300)
    cols = [ORANGE, "#F08A3E", ORANGE_M, PETROL, PETROL_M, NAVY, SKY]
    for j in range(7):
        ac.plot(t, np.degrees(a[t0:t1, j] - a[t0, j]), color=cols[j], lw=2.2)
    ac.axvspan(c0 / 30, (c0 + 10) / 30, color=ORANGE_L, zorder=0)
    ac.set_xticks([])
    ac.set_yticks([])
    ac.spines["left"].set_visible(False)
    ov = overlay(fig)
    txt(ov, 900, 476, "10-step plan, both arms", size=16, weight="bold")
    txt(ov, 900, 504, "30 Hz · new plan every 333 ms", size=14, color=GREY)
    box(ov, 900, 560, 385, 120, fc=PAPER, r=12)
    txt(ov, 920, 580, "95 ms", size=40, weight="bold", color=ORANGE)
    txt(ov, 920, 642, "inference per plan (RTX 4090)", size=14, color=GREY)
    save(fig, "V05_policy_in_out")


# ================================================================ V06 dataset
def v06_dataset(d):
    fig, ax = canvas("84 demos, 1 motion", source="dataset_check/reports/*/report.json")
    ds = d["datasets"]
    rows = [("fr3_duo_rand500", "Cube stacking"), ("fr3_duo_handover_v9_night_84", "Box handover")]
    a = chart(fig, 260, 150, 760, 330)
    y = np.array([1, 0])
    eps = [ds[k]["episodes"] for k, _ in rows]
    dist = [ds[k]["distinct_motions"] for k, _ in rows]
    a.barh(y, eps, color=PAPER, height=0.55)
    a.barh(y, dist, color=[PETROL, RED], height=0.55)
    for yi, dd, e in zip(y, dist, eps):
        a.text(max(dd, e) + 12, yi, f"{dd} of {e}", va="center", fontsize=22, fontweight="bold",
               color=PETROL if dd > 1 else RED)
    a.set_yticks(y)
    a.set_yticklabels([n for _, n in rows], fontsize=18)
    a.set_xlim(0, 640)
    a.set_xticks([])
    a.spines["bottom"].set_visible(False)
    ov = overlay(fig)
    txt(ov, 260, 128, "distinct motions per dataset", size=15, color=GREY)
    gaps = ["box never moves", "no corrections", "no failures"]
    for i, s in enumerate(gaps):
        x = 48 + i * 420
        box(ov, x, 548, 400, 120, fc=RED_L, r=12)
        txt(ov, x + 30, 608, "✕", size=28, color=RED, weight="bold", va="center", family="DejaVu Sans")
        txt(ov, x + 80, 608, s, size=21, weight="bold", va="center")
    save(fig, "V06_dataset_one_motion")


# ================================================================ V07 v0 closed loop
def v07_v0(d):
    allr = d["v0"]
    v0 = [r for r in allr if r["attempt"]]
    n = len(v0)
    fig, ax = canvas("v0 in closed loop: grasps, carries, then stalls",
                     source=f"{n} of {len(allr)} v0 sim runs (excludes {len(allr) - n} debug runs: killed early or "
                            "old box-lift workaround) · mixed settings · sim/out/*/result.json")
    keys = [("left_grasp", "Left grasp\n+ lift"), ("right_grasp", "Right grasp"), ("handover", "Handover"),
            ("placed", "Placed")]
    af = chart(fig, 190, 150, 440, 420)
    ks = [sum(r[k] for r in v0) for k, _ in keys]
    y = np.arange(4)[::-1]
    af.barh(y, [n] * 4, color=PAPER, height=0.6)
    af.barh(y, ks, color=[ORANGE, ORANGE, ORANGE, RED], height=0.6)
    for yi, k in zip(y, ks):
        lo, hi = wilson(k, n)
        af.errorbar(k, yi, xerr=[[k - lo * n], [hi * n - k]], color=INK, capsize=5, lw=1.5)
        af.text(n + 0.4, yi, f"{k}/{n}", va="center", fontsize=20, fontweight="bold")
    af.set_yticks(y)
    af.set_yticklabels([l for _, l in keys], fontsize=17)
    af.set_xlim(0, n)
    af.set_xticks([])
    af.spines["bottom"].set_visible(False)
    short = {"pause": "Waits at pause", "carrying": "Stalls carrying", "Right": "Right grasp miss",
             "over": "Knocks box over", "short": "Other"}
    cnt = Counter()
    for r in v0:
        c = mf.v0_failure_class(r)
        cnt[next(v for k, v in short.items() if k in c)] += 1
    order = sorted(cnt.items(), key=lambda kv: -kv[1])
    ar = chart(fig, 950, 150, 300, 420)
    yy = np.arange(len(order))[::-1]
    ar.barh(yy, [v for _, v in order], color=[GREY2 if k == "Other" else RED for k, _ in order], height=0.6)
    for yi, (_, v) in zip(yy, order):
        ar.text(v + 0.2, yi, str(v), va="center", fontsize=18, fontweight="bold")
    ar.set_yticks(yy)
    ar.set_yticklabels([k for k, _ in order], fontsize=15)
    ar.set_xticks([])
    ar.spines["bottom"].set_visible(False)
    ov = overlay(fig)
    txt(ov, 48, 120, "milestones (95% CI)", size=15, color=GREY)
    txt(ov, 790, 120, "how runs ended", size=15, color=GREY)
    box(ov, 48, 610, W - 96, 76, fc=BLACK, r=12)
    txt(ov, W / 2, 648, "With alignment assist, v0 grasps and carries. The handover is where it breaks.", size=22, weight="bold", color=WHITE,
        ha="center", va="center")
    save(fig, "V07_v0_closed_loop")


# ================================================================ V08 expert data
def v08_expert(d):
    ex = d["expert"]
    n, k = len(ex), sum(r["success"] for r in ex)
    fig, ax = canvas("Fix the data: move the box", source="sim/out/expert_demos*/**/meta.json")
    asc = chart(fig, 120, 130, 470, 470)
    asc.add_patch(patches.Rectangle((-15, -15), 30, 30, fc=PETROL_L, ec="none", zorder=0))
    for r in ex:
        if r["success"]:
            asc.scatter(r["dx_mm"], r["dz_mm"], s=170, color=PETROL, edgecolor=WHITE, lw=1.5, zorder=3)
        else:
            asc.scatter(r["dx_mm"], r["dz_mm"], s=220, color=RED, marker="X", edgecolor=WHITE, zorder=4)
    asc.scatter([0], [0], s=380, marker="*", color=ORANGE, edgecolor=WHITE, zorder=5)
    asc.set_xlim(-20, 20)
    asc.set_ylim(-20, 20)
    asc.set_aspect("equal")
    asc.set_xticks([-15, 0, 15])
    asc.set_yticks([-15, 0, 15])
    asc.tick_params(labelsize=13)
    asc.set_xlabel("box shift x (mm)", fontsize=14)
    asc.set_ylabel("box shift y (mm)", fontsize=14)
    ov = overlay(fig)
    leg = [("★", "original demos", ORANGE), ("●", "new: success", PETROL), ("✖", "new: failed", RED)]
    for i, (m, s, c) in enumerate(leg):
        txt(ov, 700, 180 + i * 46, m, size=22, color=c, va="center", family="DejaVu Sans")
        txt(ov, 740, 180 + i * 46, s, size=17, va="center")
    lo, hi = wilson(k, n)
    txt(ov, 700, 360, f"{k} / {n}", size=64, weight="bold", color=PETROL)
    txt(ov, 700, 455, f"expert demos succeed ({lo * 100:.0f}-{hi * 100:.0f}% CI)", size=15, color=GREY)
    txt(ov, 700, 520, "34 \u2192 6 min", size=40, weight="bold", color=ORANGE)
    txt(ov, 700, 585, "per generated episode (real-time rendering)", size=15, color=GREY)
    save(fig, "V08_expert_data")


# ================================================================ V09 offline + latency
def v09_offline(d):
    fig, ax = canvas("Fine-tuning works, and it is fast",
                     source="pi05/README.md (9 held-out episodes) · latency from v0 sim runs")
    aa = chart(fig, 80, 150, 520, 430)
    aa.bar([0, 1], [6.79, 0.39], color=[GREY2, ORANGE], width=0.55)
    for x, v, c in ((0, 6.79, GREY), (1, 0.39, ORANGE)):
        aa.text(x, v + 0.15, f"{v:.2f}°", ha="center", fontsize=26, fontweight="bold", color=c)
    aa.set_xticks([0, 1])
    aa.set_xticklabels(["base π0.5", "fine-tuned"], fontsize=17)
    aa.set_ylim(0, 8.2)
    aa.set_yticks([])
    aa.spines["left"].set_visible(False)
    lat = [r["lat"]["p50"] for r in d["v0"]]
    lat95 = [r["lat"]["p95"] for r in d["v0"]]
    al = chart(fig, 820, 150, 430, 430)
    rng = np.random.default_rng(1)
    al.axhspan(0, 333, color=PETROL_L, zorder=0, lw=0)
    al.scatter(rng.uniform(-0.15, 0.15, len(lat)), lat, s=90, color=ORANGE, edgecolor=WHITE, zorder=3)
    al.scatter(1 + rng.uniform(-0.15, 0.15, len(lat95)), lat95, s=90, color=PETROL, edgecolor=WHITE, zorder=3)
    al.axhline(333, color=RED, lw=2)
    al.text(1.45, 343, "333 ms budget", color=RED, fontsize=15, ha="right", va="bottom", fontweight="bold")
    al.set_xticks([0, 1])
    al.set_xticklabels(["p50", "p95"], fontsize=17)
    al.set_xlim(-0.5, 1.5)
    al.set_ylim(0, 390)
    al.set_yticks([0, 100, 200, 300])
    al.tick_params(labelsize=13)
    ov = overlay(fig)
    txt(ov, 80, 120, "arm error offline", size=15, color=GREY)
    txt(ov, 760, 120, "ms per plan, 28 sim runs", size=15, color=GREY)
    save(fig, "V09_offline_and_latency")


# ================================================================ V10 task storyboard
def v10_task(d):
    fig, ax = canvas("The task", source="expert demo, Isaac Sim")
    idx = [1, 3, 4, 5, 6, 9]
    labels = ["start", "left grasp", "carry", "handover", "right carries", "placed"]
    fw, g = 190, 14.6
    for j, i in enumerate(idx):
        place(fig, frame("external", i), 48 + j * (fw + g), 110, fw, fw, border=LINE, lw=1)
    ov = overlay(fig)
    for j in range(6):
        x = 48 + j * (fw + g)
        num_badge(ov, x + 22, 132, str(j + 1), r=15, size=13)
        txt(ov, x + fw / 2, 316, labels[j], size=16, weight="bold", ha="center")
    s = d["ep_state"]
    ac = chart(fig, 100, 380, W - 160, 280)
    t = np.arange(len(s)) / 30
    ac.plot(t, s[:, 7], color=ORANGE, lw=3.5)
    ac.plot(t, s[:, 15], color=PETROL, lw=3.5)
    ac.set_ylim(0.35, 1.05)
    ac.set_xlim(0, t[-1])
    ac.set_yticks([0.43, 0.99])
    ac.set_yticklabels(["closed", "open"], fontsize=14)
    ac.tick_params(axis="x", labelsize=13)
    ac.set_xlabel("seconds", fontsize=14)
    ac.text(t[40], 0.64, "left gripper", color=ORANGE, fontsize=16, fontweight="bold")
    ac.text(t[1250], 0.92, "right gripper", color=PETROL, fontsize=16, fontweight="bold")
    txt(ov, 48, 362, "grippers", size=15, color=GREY)
    save(fig, "V10_task")


# ================================================================ V11 roadmap / bigger picture
def v11_roadmap(d):
    fig, ax = canvas("From one handover to a cathlab companion")
    cols = [("NOW", ORANGE, ["Box handover in sim", "Loop runs end to end"]),
            ("NEXT", PETROL, ["DAgger corrections", "More objects"]),
            ("THEN", NAVY, ["Real FR3 Duo", "Nurse labels on tablet"]),
            ("LATER", BLACK, ["Skills shared across sites"])]
    x0, cw, g = 48, 294, 18
    for i, (k, c, items) in enumerate(cols):
        x = x0 + i * (cw + g)
        box(ax, x, 110, cw, 220, fc=WHITE, ec=LINE, r=14, lw=1.5)
        box(ax, x, 110, cw, 60, fc=c, r=14, z=2)
        ax.add_patch(patches.Rectangle((x, 145), cw, 25, fc=c, zorder=2))
        txt(ax, x + 22, 141, k, size=22, weight="bold", color=WHITE, va="center", z=6)
        y = 196
        for s in items:
            ax.add_patch(patches.Circle((x + 30, y + 11), 6, fc=c if c != BLACK else ORANGE, zorder=4))
            txt(ax, x + 46, y, s, size=16, wrap=24)
            y += 58
        if i < 3:
            arrow(ax, (x + cw + 3, 140), (x + cw + g - 3, 140), color=GREY2, lw=2.5, head=12)
    txt(ax, 48, 364, "Skills the same loop can teach", size=18, weight="bold")
    skills = [("Hand over supplies", True), ("Fetch from storage", False), ("Pass instruments", False),
              ("Prepare the table", False), ("Restock & tidy", False), ("Bring on request", False)]
    for j, (s, done) in enumerate(skills):
        x = 48 + (j % 3) * 418
        y = 402 + (j // 3) * 96
        box(ax, x, y, 400, 80, fc=ORANGE if done else PAPER, r=12)
        txt(ax, x + 24, y + 40, s, size=19, weight="bold", color=WHITE if done else INK, va="center")
        if done:
            txt(ax, x + 376, y + 40, "in sim now", size=13, color=WHITE, va="center", ha="right")
    box(ax, 48, 608, W - 96, 84, fc=BLACK, r=12)
    txt(ax, W / 2, 650, "New skill = new demos + the same loop", size=22, weight="bold", color=WHITE,
        ha="center", va="center")
    save(fig, "V11_roadmap")


# ================================================================ V12 safety layers
def v12_safety(d):
    fig, ax = canvas("Safety in layers")
    layers = [("Nurse", PETROL, "stop, confirm, sign off"), ("Supervisor", BLACK, "retry, then ask the nurse"),
              ("Interlocks", NAVY, "release only when safe"), ("Limits", GREY, "joint and force limits"),
              ("π0.5 policy", ORANGE, "proposes the motion")]
    cx, cy = 400, 410
    for i, (h, c, _) in enumerate(layers):
        r = 290 - i * 56
        ax.add_patch(patches.Circle((cx, cy), r, fc=c, ec=WHITE, lw=4, zorder=1 + i))
        if i < 4:
            txt(ax, cx, cy - r + 29, h, size=18, weight="bold", color=WHITE, ha="center", va="center", z=10)
    txt(ax, cx, cy, "π0.5\npolicy", size=24, weight="bold", color=WHITE, ha="center", va="center", z=10)
    y = 150
    for h, c, s_ in layers:
        ax.add_patch(patches.Circle((800, y + 26), 16, fc=c, zorder=4))
        txt(ax, 832, y + 4, h, size=21, weight="bold", color=c if c not in (BLACK, GREY) else INK)
        txt(ax, 832, y + 36, s_, size=16, color=GREY)
        y += 108
    save(fig, "V12_safety_layers")


FIGS = {"V01": v01_story, "V02": v02_problem, "V03": v03_team, "V04": v04_pipeline, "V05": v05_io,
        "V06": v06_dataset, "V07": v07_v0, "V08": v08_expert, "V09": v09_offline, "V10": v10_task,
        "V11": v11_roadmap, "V12": v12_safety}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    data = mf.load_data()
    for k, f in FIGS.items():
        if not args.only or k in args.only:
            f(data)
