"""V14: why the Genesis data generator is ~12x faster than our Isaac Sim v1 expert, and the loop it unlocks.

Same visual language as V13 (make_flowchart_v3.py). Run from the repo root:  python presentation/make_flowchart_genesis.py
Writes presentation/figures_v2/V14_genesis_engine.png and .pdf.
Numbers: Isaac v1 from CLAUDE.md "Data loop v1"; Genesis from the Genesis README (24 Oct) and the team's ~2 demos/min.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import ORANGE, PETROL, RED, RED_L, PETROL_L, ORANGE_L, PAPER, W, WHITE, arrow, box, txt  # noqa: E402
from make_figures_v2 import canvas, save  # noqa: E402
from make_flowchart_v3 import BH, ROWS, XS, right, row_label, step  # noqa: E402


def connector(ax, from_row, label, dashed=False):
    x5, x6 = XS[4] + 232 / 2, XS[0] + 232 / 2
    ym = ROWS[from_row] + BH + 30
    ax.plot([x5, x5, x6], [ROWS[from_row] + BH + 2, ym, ym], color=ORANGE, lw=3.5, zorder=2,
            ls=(0, (4, 3)) if dashed else "-", solid_capstyle="round")
    arrow(ax, (x6, ym), (x6, ROWS[from_row + 1] - 2), color=ORANGE, lw=3.5, head=16)
    box(ax, W / 2 - 280, ym - 13, 560, 26, fc=WHITE, r=4, z=3)
    txt(ax, W / 2, ym, label, size=11.5, weight="bold", color=ORANGE, ha="center", va="center", z=4)


def main():
    fig, ax = canvas("From 6 minutes to 30 seconds per demo",
                     source="Isaac v1: sim/expert_demos.py, gen_demos.sh · Genesis World 1.4.5 generator README · "
                            "24 Oct 2026")

    # Row 1: where Isaac v1 spent its time
    row_label(ax, 0, "Isaac Sim v1: why one demo took 6 minutes", RED)
    step(ax, 0, 0, "1", "Photoreal renderer", "RTX ray tracing, 3 cameras: ~57 ms per update (path tracing: 34 min "
         "per demo)", "render-bound", "bad")
    step(ax, 1, 0, "2", "One app per demo", "Every episode starts a full Omniverse Kit process and reloads the "
         "hospital USD stage", "fixed start-up cost", "bad")
    step(ax, 2, 0, "3", "Replay, not plan", "Expert follows the one recorded path; the clock stops at each "
         "grasp until aligned < 4 mm", "box only ±15 mm", "bad")
    step(ax, 3, 0, "4", "Parallel = no gain", "3 workers share one GPU renderer: more processes, same demos "
         "per hour", "GPU-bound", "bad")
    step(ax, 4, 0, "5", "Result", "16 / 18 demos succeed, 14 used to train v1: too few and too similar",
         "~10 demos / hour", "bad", fc=RED_L)
    for c in range(4):
        right(ax, c, 0)
    connector(ax, 0, "the bottleneck was the simulator, not the expert")

    # Row 2: what Genesis changes
    row_label(ax, 1, "Genesis: same handover, built for data volume", PETROL)
    step(ax, 0, 1, "6", "Light physics engine", "Genesis contact physics, GPU kernels compiled once; no grasp "
         "welds or teleporting", "no Kit overhead", "ok")
    step(ax, 1, 1, "7", "Cheap cameras", "OpenGL rasterizer, 256 px recordings, no ray tracing; previews "
         "at 16–64 px", "render not the limit", "ok")
    step(ax, 2, 1, "8", "Planning expert", "IK on the live box pose, reach check, up to 3 grasp tries, "
         "1.5× travel speed", "any layout in bounds", "ok")
    step(ax, 3, 1, "9", "Many sims per GPU", "Up to 12 generators on a 24 GB GPU; failed attempts kept, "
         "only successes admitted", "scales with processes", "ok")
    step(ax, 4, 1, "10", "Result", "Box anywhere in a 28 × 26 cm pickup region, seeded and reproducible",
         "~2 demos / min (~12×)", "ok", fc=PETROL_L)
    for c in range(4):
        right(ax, c, 1, color=PETROL)
    connector(ax, 1, "12× more data per hour buys time for training rounds", dashed=True)

    # Row 3: the loop it unlocks
    row_label(ax, 2, "The loop it unlocks (ACT now, pi0.5 next)", ORANGE)
    step(ax, 0, 2, "11", "Generate", "Seeded layouts → LeRobot export + provenance for every attempt",
         "25 workflows (gray v1)", "ok", fc=ORANGE_L, dashed=True)
    step(ax, 1, 2, "12", "Check", "Stage 02 checker per shard; whole-shard validation split, no mixing "
         "scenes or speeds", "to run", "next", dashed=True)
    step(ax, 2, 2, "13", "Train", "ACT with fast loader (2.9 s → 39 ms per sample); pi0.5 LoRA export ready",
         "trained, fails pick", "open", dashed=True)
    step(ax, 3, 2, "14", "HG-DAgger", "Student drives, expert labels every tick, takes over on stall, "
         "drift or slip", "round 1 pending", "next", dashed=True)
    step(ax, 4, 2, "15", "Benchmark", "Frozen seeds 900000+, student only, success with Wilson 95% CI",
         "loop closes", "next", dashed=True)
    for c in range(4):
        right(ax, c, 2, color=ORANGE)
    save(fig, "V14_genesis_engine")


if __name__ == "__main__":
    main()
