"""V16: simple one-page 16:9 summary of Stage 02 Dataset Intelligence (our dataset_check tool and what it found).

Same visual language as V13/V15. Run from the repo root:  python presentation/make_dataset_onepager.py
Writes presentation/figures_v2/V16_dataset_intelligence.png and .pdf.
Numbers: dataset_check/reports/DATA_PLAN_STATUS.md and dataset_check/README.md (24 Oct).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (GREY, INK, ORANGE, ORANGE_L, PAPER, PETROL, RED, RED_L, WHITE, YELLOW_L,  # noqa: E402
                          box, num_badge, txt)
from make_figures_v2 import canvas, save  # noqa: E402

L, R = 48, 1285
AMBER = "#B7791F"


def section(ax, x, y, s):
    txt(ax, x, y, s, size=13, weight="bold", color=ORANGE, va="bottom")


def verdict_card(ax, x, y, w, h, title, stats, verdict, ec, fc):
    box(ax, x, y, w, h, fc=fc, r=14)
    txt(ax, x + 20, y + 18, title, size=13, weight="bold")
    for i, (big, small) in enumerate(stats):
        sx = x + 20 + i * 120
        txt(ax, sx, y + 56, big, size=30, weight="bold", color=ec)
        txt(ax, sx, y + 106, small, size=10, color=GREY)
    box(ax, x + 20, y + h - 46, w - 40, 32, fc=WHITE, ec=ec, lw=1.6, r=8, z=3)
    txt(ax, x + w / 2, y + h - 30, verdict, size=13, weight="bold", color=ec, ha="center", va="center", z=4)


def main():
    fig, ax = canvas("Dataset Intelligence: is the data ready?",
                     source="Stage 02 · dataset_check/ · reports/DATA_PLAN_STATUS.md · 24 Oct 2026")
    txt(ax, L, 92, "Our tool reads any LeRobot dataset and returns a verdict, the episodes to drop and a data plan.",
        size=12, color=GREY)

    # What it checks
    section(ax, L, 160, "What it checks")
    checks = ["Format", "Grasp steps", "Idle time", "Action timing", "Diversity", "Outliers"]
    tw = (R - L - 5 * 14) / 6
    for i, head in enumerate(checks):
        x = L + i * (tw + 14)
        box(ax, x, 168, tw, 52, fc=PAPER, r=10)
        num_badge(ax, x + 22, 194, str(i + 1), fc=PETROL, r=12, size=10)
        txt(ax, x + 42, 194, head, size=12, weight="bold", va="center")

    # Verdicts
    y = 296
    section(ax, L, y - 10, "Verdict")
    cw, ch = 380, 220
    verdict_card(ax, L, y, cw, ch, "Siemens Healthineers dataset",
                 [("84", "demos"), ("1", "motion"), ("17%", "idle")], "NOT READY", RED, RED_L)
    verdict_card(ax, L + cw + 24, y, cw, ch, "Our sim expert data (v1)",
                 [("18", "demos"), ("18", "motions"), ("2%", "idle")], "CONDITIONALLY READY", AMBER, YELLOW_L)

    # What it told us
    fx = L + 2 * cw + 60
    section(ax, fx, y - 10, "What it told us")
    finds = [
        (RED, "Fix the data, not the model", "The provided demos are one motion: only the light changes."),
        (PETROL, "It finds bad demos by itself", "Flagged both failed expert demos automatically."),
        (ORANGE, "Scripted demos are not enough", "Box position explains only 12–43% of the motion."),
    ]
    for k, (c, head, body) in enumerate(finds):
        yy = y + k * 78
        box(ax, fx, yy, 6, 54, fc=c, r=2)
        txt(ax, fx + 18, yy + 4, head, size=12.5, weight="bold", color=INK)
        txt(ax, fx + 18, yy + 26, body, size=10.5, color=GREY, wrap=48)

    # Takeaway
    box(ax, L, 590, R - L, 70, fc=ORANGE_L, r=12)
    txt(ax, L + 22, 625, "Next demos should come from humans in VR: more variety than a script can give.",
        size=15, weight="bold", color=INK, va="center")
    save(fig, "V16_dataset_intelligence")


if __name__ == "__main__":
    main()
