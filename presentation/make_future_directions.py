"""V17: future directions, simple 16:9 figure: voice commands, two arms multitasking, VR teleoperation.

Same visual language as V13/V15/V16. Run from the repo root:  python presentation/make_future_directions.py
Writes presentation/figures_v2/V17_future_directions.png and .pdf. Concept figure: no measured results.
"""
import os
import sys

from matplotlib import patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (GREY, GREY2, INK, ORANGE, ORANGE_L, PAPER, PETROL, PETROL_L, WHITE,  # noqa: E402
                          arrow, box, bubble, num_badge, person, txt)
from make_figures_v2 import canvas, save  # noqa: E402

L, R = 48, 1285
PW, GAP = 395, 26
PY, PH = 120, 470
XS = [L + i * (PW + GAP) for i in range(3)]
ARM = "#4A4A4A"


def arm(ax, pts, color=ARM, open_grip=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=9, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    for x, y in pts[:-1]:
        ax.add_patch(patches.Circle((x, y), 6, fc=WHITE, ec=color, lw=2, zorder=6))
    x, y = pts[-1]
    d = 11 if open_grip else 8
    ax.plot([x - d, x - d], [y, y + 16], color=color, lw=4, solid_capstyle="round", zorder=5)
    ax.plot([x + d, x + d], [y, y + 16], color=color, lw=4, solid_capstyle="round", zorder=5)
    ax.plot([x - d, x + d], [y, y], color=color, lw=4, zorder=5)


def dual_arm(ax, cx, base_y, left_tip, right_tip, s=1.0):
    """FR3 Duo style: pedestal with two arms; tips are gripper positions."""
    box(ax, cx - 18 * s, base_y - 70 * s, 36 * s, 70 * s, fc=ARM, r=4, z=4)
    box(ax, cx - 40 * s, base_y - 4, 80 * s, 10, fc=GREY2, r=3, z=4)
    sh_l, sh_r = (cx - 16 * s, base_y - 66 * s), (cx + 16 * s, base_y - 66 * s)
    el_l = (sh_l[0] - 40 * s, sh_l[1] - 44 * s)
    el_r = (sh_r[0] + 40 * s, sh_r[1] - 44 * s)
    arm(ax, [sh_l, el_l, left_tip])
    arm(ax, [sh_r, el_r, right_tip])


def panel(ax, i, n, head, body, tag, fc=PAPER):
    x = XS[i]
    box(ax, x, PY, PW, PH, fc=fc, r=16)
    num_badge(ax, x + 28, PY + 30, n, fc=ORANGE, r=15, size=12)
    txt(ax, x + 52, PY + 30, head, size=15, weight="bold", va="center")
    txt(ax, x + 22, PY + 330, body, size=10.5, color=GREY, wrap=47)
    box(ax, x + 22, PY + PH - 46, PW - 44, 30, fc=WHITE, ec=ORANGE, lw=1.4, r=8, z=3)
    txt(ax, x + PW / 2, PY + PH - 31, tag, size=10.5, weight="bold", color=ORANGE, ha="center", va="center", z=4)


def main():
    fig, ax = canvas("Where this goes next",
                     source="Concept · Challenge 2 · Cathlab Assistant Robot · 24 Oct 2026")
    txt(ax, L, 92, "From one scripted handover to a companion that listens, multitasks and keeps learning from people.",
        size=12, color=GREY)

    # 1. Voice
    panel(ax, 0, "1", "Talk to the robot",
          "The nurse asks in plain words. Speech becomes text, and the VLA model turns the instruction "
          "into arm motion. A new sentence is a new task, with no reprogramming.",
          "nurse starts and supervises")
    x0 = XS[0]
    person(ax, x0 + 70, PY + 250, s=1.3, color=PETROL)
    bubble(ax, x0 + 24, PY + 66, 230, 58, "“Please hand me\nthe scissors”", (x0 + 70, PY + 168),
           ec=PETROL, size=12)
    for k, (lab, fc, tc) in enumerate([("speech → text", PETROL_L, INK), ("VLA (pi0.5)", ORANGE_L, INK)]):
        bx, by = x0 + 132, PY + 196 + k * 44
        box(ax, bx, by, 130, 32, fc=fc, r=8, z=3)
        txt(ax, bx + 65, by + 16, lab, size=10, weight="bold", color=tc, ha="center", va="center", z=4)
    arrow(ax, (x0 + 104, PY + 212), (x0 + 129, PY + 212), color=GREY2, lw=2.2, head=10)
    arrow(ax, (x0 + 195, PY + 229), (x0 + 195, PY + 237), color=GREY2, lw=2.2, head=8)
    arrow(ax, (x0 + 264, PY + 256), (x0 + 286, PY + 256), color=GREY2, lw=2.2, head=10)
    dual_arm(ax, x0 + 330, PY + 300, (x0 + 270, PY + 182), (x0 + 372, PY + 160), s=0.8)
    # scissors in the right gripper
    sx, sy = x0 + 372, PY + 182
    ax.add_patch(patches.Circle((sx - 6, sy + 8), 5, fc="none", ec=ORANGE, lw=2.2, zorder=7))
    ax.add_patch(patches.Circle((sx + 6, sy + 8), 5, fc="none", ec=ORANGE, lw=2.2, zorder=7))
    ax.plot([sx - 3, sx + 6], [sy + 12, sy + 34], color=ORANGE, lw=2.4, zorder=7)
    ax.plot([sx + 3, sx - 6], [sy + 12, sy + 34], color=ORANGE, lw=2.4, zorder=7)

    # 2. Multitasking
    panel(ax, 1, "2", "Two arms, two jobs",
          "Each arm runs its own skill at the same time: one carries the box to the tray while the other "
          "fetches a bottle. A skill planner splits the work and keeps the arms apart.",
          "skill orchestration")
    x1 = XS[1]
    cx = x1 + PW / 2
    lt, rt = (cx - 118, PY + 150), (cx + 118, PY + 150)
    dual_arm(ax, cx, PY + 300, lt, rt, s=1.15)
    box(ax, lt[0] - 22, lt[1] + 16, 44, 30, fc=ORANGE, r=4, z=7)
    box(ax, rt[0] - 11, rt[1] + 12, 22, 52, fc=PETROL, r=6, z=7)
    box(ax, rt[0] - 6, rt[1] + 6, 12, 8, fc=PETROL, r=2, z=7)
    for (x, y), lab, c in (((lt[0], PY + 76), "box → tray", ORANGE), ((rt[0], PY + 76), "bottle → cart", PETROL)):
        box(ax, x - 60, y, 120, 28, fc=WHITE, ec=c, lw=1.6, r=8, z=3)
        txt(ax, x, y + 14, lab, size=10.5, weight="bold", color=c, ha="center", va="center", z=4)
    txt(ax, cx, PY + 90, "at the\nsame time", size=10, weight="bold", color=GREY2, ha="center", va="center")

    # 3. VR teleoperation
    panel(ax, 2, "3", "Teach it in VR",
          "An operator with a Meta Quest 3 moves both arms in the simulator. Every session becomes training "
          "data, and the operator steps in when the model drifts (DAgger).",
          "prototype runs in Isaac Sim")
    x2 = XS[2]
    hx, hy = x2 + 80, PY + 250
    person(ax, hx, hy, s=1.3, color=PETROL)
    box(ax, hx - 19, hy - 60, 38, 14, fc=INK, r=4, z=8)  # headset
    for dx in (-36, 36):
        box(ax, hx + dx - 5, hy - 26, 10, 22, fc=INK, r=4, z=8)  # controllers
    arrow(ax, (x2 + 128, PY + 200), (x2 + 236, PY + 200), color=ORANGE, lw=3, head=14, ls="--")
    txt(ax, x2 + 182, PY + 190, "moves", size=9.5, weight="bold", color=ORANGE, ha="center", va="bottom")
    dual_arm(ax, x2 + 310, PY + 300, (x2 + 252, PY + 176), (x2 + 368, PY + 176), s=0.8)
    # learning loop under the scene
    ly = PY + 100
    for k, lab in enumerate(["demo", "data", "retrain"]):
        bx = x2 + 40 + k * 110
        box(ax, bx, ly - 40, 86, 28, fc=WHITE, ec=PETROL, lw=1.4, r=8, z=3)
        txt(ax, bx + 43, ly - 26, lab, size=10, weight="bold", color=PETROL, ha="center", va="center", z=4)
        if k < 2:
            arrow(ax, (bx + 88, ly - 26), (bx + 108, ly - 26), color=PETROL, lw=2, head=9)

    # Road
    by = 618
    steps = [("Now", "one task in simulation"),
             ("Next", "VR demos and two-arm skills"),
             ("Later", "voice on the real FR3 Duo")]
    sw = (R - L - 2 * 40) / 3
    for k, (when, what) in enumerate(steps):
        x = L + k * (sw + 40)
        box(ax, x, by, sw, 46, fc=ORANGE_L if k else PAPER, r=10)
        txt(ax, x + 16, by + 23, when, size=12, weight="bold", color=ORANGE, va="center")
        txt(ax, x + 76, by + 23, what, size=10.5, color=INK, va="center")
        if k < 2:
            arrow(ax, (x + sw + 6, by + 23), (x + sw + 34, by + 23), color=ORANGE, lw=2.5, head=12)
    save(fig, "V17_future_directions")


if __name__ == "__main__":
    main()
