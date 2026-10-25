"""V13: the honest flowchart of what we did, what happened, and the next step (Quest 3 teleoperation).

Run (any python with matplotlib + pillow, from the repo root):  python presentation/make_flowchart_v3.py
Writes presentation/figures_v2/V13_our_path.png (2666x1500) and .pdf. Numbers: results/SUCCESS_RATES.md (24 Oct).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (BLACK, GREEN, GREEN_L, GREY, GREY2, INK, LINE, ORANGE, ORANGE_L, PAPER, PETROL,  # noqa: E402
                          PETROL_L, RED, RED_L, W, WHITE, arrow, box, num_badge, txt)
from make_figures_v2 import canvas, save  # noqa: E402

BW, BH = 232, 140      # step box
ROWS = (112, 340, 568)  # top of each row
XS = [48 + i * (BW + 26) for i in range(5)]


def step(ax, col, row, n, head, body, verdict=None, kind="ok", fc=PAPER, dashed=False):
    x, y = XS[col], ROWS[row]
    box(ax, x, y, BW, BH, fc=fc, ec=GREY2 if dashed else "none", ls="--" if dashed else "-", lw=1.6, r=12)
    num_badge(ax, x + 22, y + 24, n, fc={"ok": PETROL, "bad": RED, "open": GREY2, "next": ORANGE}[kind], r=14, size=11)
    txt(ax, x + 42, y + 24, head, size=12, weight="bold", va="center")
    txt(ax, x + 14, y + 46, body, size=9.3, color=GREY, wrap=31)
    if verdict:
        vc, vf = {"ok": (GREEN, GREEN_L), "bad": (RED, RED_L), "open": (GREY, PAPER), "next": (ORANGE, ORANGE_L)}[kind]
        box(ax, x + 10, y + BH - 36, BW - 20, 27, fc=vf if fc == PAPER else WHITE, ec=vc, lw=1.4, r=6, z=3)
        txt(ax, x + BW / 2, y + BH - 22.5, verdict, size=10.5, weight="bold", color=vc, ha="center", va="center", z=4)


def right(ax, col, row, color=GREY2):
    y = ROWS[row] + BH / 2
    arrow(ax, (XS[col] + BW + 3, y), (XS[col + 1] - 3, y), color=color, lw=2.5, head=12)


def row_label(ax, row, s, color):
    txt(ax, XS[4] + BW, ROWS[row] - 10, s, size=12, weight="bold", color=color, ha="right", va="bottom")


def main():
    fig, ax = canvas("What we did, and what we learned",
                     source="Isaac Sim closed loop, box handover · results/SUCCESS_RATES.md · 24 Oct 2026")

    # Row 1: the shortcut (supervisor's checkpoint)
    row_label(ax, 0, "Attempt 1: use the provided model (v0)", RED)
    step(ax, 0, 0, "1", "Organisers' demos", "84 + 10 handover episodes, 30 Hz, 3 cameras, 16 joints",
         "only lighting varies", "bad")
    step(ax, 1, 0, "2", "Dataset check", "Our Stage 02 tool: all 94 demos are one and the same motion",
         "verdict: NOT READY", "bad")
    step(ax, 2, 0, "3", "Given checkpoint", "pi0.5 + LoRA from the supervisor, trained on the handover demos "
         "(saved us training time)", "offline error 0.39°", "ok")
    step(ax, 3, 0, "4", "Test in Isaac Sim", "Raw model: no grasp at all. With our helpers "
         "(assist, supervisor): 10/29 grasps, 1 handover", "0 / 29 placed", "bad")
    step(ax, 4, 0, "5", "Why it fails", "Drifts 2–4 cm off the only path it knows, tips the box, waits at the "
         "recorded pause, box slips", "data, not model", "bad", fc=RED_L)
    for c in range(4):
        right(ax, c, 0)

    # Row 2: fix the data with our own expert
    row_label(ax, 1, "Attempt 2: make the missing data (v1)", PETROL)
    step(ax, 0, 1, "6", "Our expert policy", "Scripted in Isaac Sim: recorded path + grasp alignment on the "
         "true box pose", "task done", "ok", fc=PETROL_L)
    step(ax, 1, 1, "7", "Vary + add noise", "Box moved ±15 mm / ±5° each episode; DART noise so the data shows "
         "how to recover from drift", "16 / 18 succeed", "ok")
    step(ax, 2, 1, "8", "Fine-tune v1", "From the provided checkpoint on 14 expert demos (29k frames), "
         "2 held out", "checkpoint ready", "ok")
    step(ax, 3, 1, "9", "Evaluate v1", "Too little data (14 vs 94 demos) and no time left for a fair "
         "v0 vs v1 test", "not measured", "open")
    step(ax, 4, 1, "10", "Lesson", "A scripted expert is slow (6 min per demo) and only adds small variations "
         "around one path", "need richer demos", "open", fc=PAPER)
    for c in range(4):
        right(ax, c, 1)
    # failure feeds the expert: from step 5 down to step 6
    x5, x6 = XS[4] + BW / 2, XS[0] + BW / 2
    ym = ROWS[0] + BH + 30
    ax.plot([x5, x5, x6], [ROWS[0] + BH + 2, ym, ym], color=ORANGE, lw=3.5, zorder=2, solid_capstyle="round")
    arrow(ax, (x6, ym), (x6, ROWS[1] - 2), color=ORANGE, lw=3.5, head=16)
    box(ax, W / 2 - 190, ym - 13, 380, 26, fc=WHITE, r=4, z=3)
    txt(ax, W / 2, ym, "failures tell us which data is missing", size=11.5, weight="bold", color=ORANGE,
        ha="center", va="center", z=4)

    # Row 3: next step, Quest 3
    row_label(ax, 2, "Next: human demos in mixed reality (v2)", ORANGE)
    step(ax, 0, 2, "11", "Quest 3 in the sim", "Tried: headset streams the scene via CloudXR, "
         "walk around the robot with the sticks", "prototype runs", "next", fc=ORANGE_L, dashed=True)
    step(ax, 1, 2, "12", "Teleoperate arms", "Hands/controllers drive both arms; intuitive, any box pose, "
         "real recovery moves", "to build", "next", dashed=True)
    step(ax, 2, 2, "13", "Record + check", "Every session becomes a LeRobot episode, screened by our "
         "Stage 02 checker", "to build", "next", dashed=True)
    step(ax, 3, 2, "14", "Retrain + test", "Mix original, expert and human demos; seeded tests with "
         "95% confidence intervals", "to build", "next", dashed=True)
    step(ax, 4, 2, "15", "Correct the model", "Operator takes over when v2 drifts; the correction becomes "
         "new data (DAgger)", "loop closes", "next", dashed=True)
    for c in range(4):
        right(ax, c, 2, color=ORANGE)
    x10 = XS[4] + BW / 2
    ym2 = ROWS[1] + BH + 30
    ax.plot([x10, x10, x6], [ROWS[1] + BH + 2, ym2, ym2], color=ORANGE, lw=3.5, zorder=2, ls=(0, (4, 3)))
    arrow(ax, (x6, ym2), (x6, ROWS[2] - 2), color=ORANGE, lw=3.5, head=16)
    box(ax, W / 2 - 190, ym2 - 13, 380, 26, fc=WHITE, r=4, z=3)
    txt(ax, W / 2, ym2, "humans give the variety a script cannot", size=11.5, weight="bold", color=ORANGE,
        ha="center", va="center", z=4)
    save(fig, "V13_our_path")


if __name__ == "__main__":
    main()
