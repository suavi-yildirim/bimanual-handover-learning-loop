"""V15: one page, what we did and learned in both simulators (Isaac Sim, then Genesis), small VLA trials, VR next.

Same visual language as V13. Run from the repo root:  python presentation/make_flowchart_both.py
Writes presentation/figures_v2/V15_what_we_learned.png and .pdf.
Numbers: results/SUCCESS_RATES.md, CLAUDE.md (replay, data loop v1), Genesis README (24 Oct);
~2 demos/min is the team's measurement.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (GREEN, GREEN_L, GREY, GREY2, ORANGE, ORANGE_L, PAPER, PETROL, PETROL_L, RED, RED_L,  # noqa: E402
                          W, WHITE, arrow, box, num_badge, txt)
from make_figures_v2 import canvas, save  # noqa: E402

BW, BH = 232, 146
XS = [48 + i * (BW + 26) for i in range(5)]
ROWS = (112, 336)
VLA_Y, VLA_H = 514, 88
KIND = {"ok": (PETROL, GREEN, GREEN_L), "bad": (RED, RED, RED_L), "open": (GREY2, GREY, PAPER),
        "next": (ORANGE, ORANGE, ORANGE_L)}


def step(ax, col, row, n, head, body, verdict, kind, fc=PAPER):
    x, y = XS[col], ROWS[row]
    badge, vc, vf = KIND[kind]
    box(ax, x, y, BW, BH, fc=fc, r=12)
    num_badge(ax, x + 22, y + 22, n, fc=badge, r=14, size=11)
    txt(ax, x + 42, y + 22, head, size=12, weight="bold", va="center")
    txt(ax, x + 14, y + 42, body, size=8.6, color=GREY, wrap=32)
    box(ax, x + 10, y + BH - 34, BW - 20, 26, fc=vf if fc == PAPER else WHITE, ec=vc, lw=1.4, r=6, z=3)
    txt(ax, x + BW / 2, y + BH - 21, verdict, size=10.5, weight="bold", color=vc, ha="center", va="center", z=4)


def right(ax, col, row, color=GREY2):
    y = ROWS[row] + BH / 2
    arrow(ax, (XS[col] + BW + 3, y), (XS[col + 1] - 3, y), color=color, lw=2.5, head=12)


def row_label(ax, y, s, color, sub=None):
    txt(ax, XS[4] + BW, y, s, size=12, weight="bold", color=color, ha="right", va="bottom")
    if sub:
        txt(ax, XS[0], y, sub, size=9.5, color=GREY, va="bottom")


def vla_card(ax, col, sim, head, body, verdict, kind):
    x, y = XS[col], VLA_Y
    _, vc, _ = KIND[kind]
    box(ax, x, y, BW, VLA_H, fc=WHITE, ec=GREY2, lw=1.2, r=10)
    txt(ax, x + 12, y + 10, sim, size=7.8, weight="bold", color=PETROL if sim.startswith("GENESIS") else RED)
    txt(ax, x + 12, y + 24, head, size=10.5, weight="bold")
    txt(ax, x + 12, y + 42, body, size=8.4, color=GREY, wrap=36)
    txt(ax, x + BW - 12, y + 9, verdict, size=8.6, weight="bold", color=vc, ha="right")


def main():
    fig, ax = canvas("What we did, and what we learned",
                     source="Isaac Sim 6.0 + Genesis World 1.4.5, FR3 Duo box handover · results/SUCCESS_RATES.md · "
                            "Genesis README · 24 Oct 2026")

    # Row 1: Isaac Sim
    row_label(ax, ROWS[0] - 8, "Approach 1: NVIDIA Isaac Sim 6.0 (Omniverse, Siemens Healthineers scene)", RED)
    step(ax, 0, 0, "1", "Provided dataset", "Siemens Healthineers dataset: 94 handover demos (LeRobot, 30 Hz). Our "
         "Stage 02 checker: one motion, only lighting varies", "not ready", "bad")
    step(ax, 1, 0, "2", "Provided checkpoint", "Siemens Healthineers pi0.5 + LoRA (v0). Replay works, but the "
         "model drifts 2–4 cm, tips the box, waits at the pause", "0 / 29 placed", "bad")
    step(ax, 2, 0, "3", "Our expert (v1)", "Scripted IK on the true box pose; box moved ±15 mm / ±5°; DART "
         "noise teaches corrections", "16 / 18 succeed", "ok")
    step(ax, 3, 0, "4", "Too slow", "RTX ray tracing ~57 ms per update (path tracing: 34 min). One Kit app per "
         "demo; more workers share one GPU", "6 min per demo", "bad")
    step(ax, 4, 0, "5", "Lesson", "v1 got 14 demos (vs 94), no fair test. Data is the problem, and making "
         "it here was too slow", "need a faster sim", "bad", fc=RED_L)
    for c in range(4):
        right(ax, c, 0)

    x5, x6 = XS[4] + BW / 2, XS[0] + BW / 2
    ym = ROWS[0] + BH + 24
    ax.plot([x5, x5, x6], [ROWS[0] + BH + 2, ym, ym], color=ORANGE, lw=3.5, zorder=2, solid_capstyle="round")
    arrow(ax, (x6, ym), (x6, ROWS[1] - 2), color=ORANGE, lw=3.5, head=16)
    box(ax, W / 2 - 230, ym - 13, 460, 26, fc=WHITE, r=4, z=3)
    txt(ax, W / 2, ym, "same task, rebuilt in a faster simulator", size=11.5, weight="bold", color=ORANGE,
        ha="center", va="center", z=4)

    # Row 2: Genesis
    row_label(ax, ROWS[1] - 8, "Approach 2: Genesis World 1.4.5 (open source, GPU physics)", PETROL)
    step(ax, 0, 1, "6", "Rebuilt scene", "Hospital room, carts, tray imported; official FR3v2 meshes and "
         "joint limits; contact physics, no grasp welds", "same handover", "ok", fc=PETROL_L)
    step(ax, 1, 1, "7", "Fast generator", "OpenGL rendering, compiled once, up to 12 runs on one 24 GB "
         "GPU; seeded, all attempts logged", "~2 demos / min", "ok", fc=PETROL_L)
    step(ax, 2, 1, "8", "Smarter expert", "Plans from the live box pose: reach check, 3 grasp tries, replans if "
         "the box moves > 2 cm, stops on collision", "10 / 10 smoke tests", "ok")
    step(ax, 3, 1, "9", "ACT + fast loader", "25 runs, box anywhere in 28 × 26 cm. Loader fix 2.9 s → 39 ms per "
         "sample. Loss 0.031: memorised", "fails at the pick", "open")
    step(ax, 4, 1, "10", "HG-DAgger + test", "Model drives, expert labels every tick and takes over on stall, "
         "drift or slip. Frozen seeds, 95% CI", "round 1 pending", "open")
    for c in range(4):
        right(ax, c, 1, color=PETROL)

    # Row 3: small VLA trials
    row_label(ax, VLA_Y - 8, "Small VLA trials", GREY, sub="Vision-language-action models we tried on the side")
    vla_card(ax, 0, "ISAAC SIM", "pi0.5 base, zero-shot", "Offline: arm error 6.79° vs 0.39° "
             "after fine-tuning", "needs fine-tune", "bad")
    vla_card(ax, 1, "ISAAC SIM", "pi0.5 v1 (LoRA)", "3,000 steps from v0 on 14 expert demos (29k frames)",
             "not yet tested", "open")
    vla_card(ax, 2, "GENESIS · PANDA", "SmolVLA, hybrid", "Old checkpoint drives one arm; "
             "handover stays scripted", "not validated", "open")
    vla_card(ax, 3, "GENESIS · FR3", "SmolVLA, both arms", "Dual-arm train + rollout path; "
             "gripper data to fix first", "data issue open", "next")
    vla_card(ax, 4, "GENESIS · FR3", "pi0.5 LoRA export", "Both arms, full handover, in openpi format",
             "ready to train", "ok")

    # Bottom: takeaway + VR next step
    by = VLA_Y + VLA_H + 20
    txt(ax, XS[0], by, "Key takeaway", size=12, weight="bold", color=ORANGE)
    txt(ax, XS[0], by + 22, "The model was not the problem, the data was; then the speed of making data was. "
        "A fast simulator plus an expert that corrects the model lets the loop improve itself.",
        size=11, color=GREY, wrap=74)
    vw, vh = 2 * BW + 26, 84
    vx = XS[3]
    box(ax, vx, by - 6, vw, vh, fc=ORANGE_L, ec=ORANGE, ls="--", lw=1.6, r=12)
    num_badge(ax, vx + 22, by + 16, "→", fc=ORANGE, r=14, size=12)
    txt(ax, vx + 42, by + 16, "Next: human demos in VR (Meta Quest 3)", size=12, weight="bold", va="center")
    txt(ax, vx + 14, by + 36, "Teleoperate both arms in Isaac Sim via CloudXR: any box pose, real "
        "recovery moves a script cannot give. Prototype runs.", size=9, color=GREY, wrap=68)
    save(fig, "V15_what_we_learned")


if __name__ == "__main__":
    main()
