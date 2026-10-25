#!/usr/bin/env python3
"""Offline study of the demo-transform expert (kinematics only, no physics or collisions).

1. Reachability map over box displacement (dx, dz) x yaw.
2. Generate N varied-box demonstrations (seeded) in the dataset action format, for replay in Isaac Sim.
Outputs in expert/out/: reachability.csv, fig_reachability.{pdf,png}, fig_generated_paths.{pdf,png},
generated/<name>.npz (actions 16-D @ 30 Hz, box displacement, flags) and generated/manifest.json.

Usage: python run_expert_study.py [--n 50] [--seed 0] [--range-mm 30] [--yaw-deg 15] [--grid-step 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dataset_check"))
from demo_transform import BoxDelta, DemoTransformExpert, plan_with_symmetry  # noqa: E402
from kinematics import load_model  # noqa: E402
from dsc.checks import analyse  # noqa: E402
from dsc.io_lerobot import load  # noqa: E402
from dsc.plots import C, _panel  # noqa: E402

DATA = HERE.parent / "C2_Teach-the-robot_VLA/Dataset/fr3_duo_handover_84_ep.zip"
BOX = "WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"


def setup(episode=0):
    m = load_model()
    ds = load(DATA)
    r = analyse(ds)
    A = ds.action[ds.episode_slices()[episode]]
    ev = {k: int(r.episodes.loc[episode, f"t_{k.replace(' ', '_')}"]) for k in r.canonical_events}
    o = m["objects"][BOX]
    pivot = (np.array(o["bbox_min"]) + np.array(o["bbox_max"])) / 2
    ex = DemoTransformExpert(m["arms"]["left"], m["arms"]["right"], A, ds.fps, ev["L grasp"], ev["L release"])
    return m, ds, ex, pivot, ev


def reachability(ex, pivot, step=20, lim=100, yaws=(-90, -45, 0, 45, 90)):
    rows = []
    vals = np.arange(-lim, lim + 1, step)
    for yaw in yaws:
        for dx in vals:
            for dz in vals:
                p, flip = plan_with_symmetry(ex, pivot, dx, dz, yaw, stride=3)
                rows.append(dict(yaw=yaw, dx=int(dx), dz=int(dz), ok=bool(p.ok), flip=bool(flip),
                                 margin=p.min_margin_deg, dev=p.max_dev_from_demo_deg, speed=p.max_joint_speed))
    return rows, vals


def fig_reach(rows, vals, yaws, out):
    fig, axs = plt.subplots(1, len(yaws), figsize=(2.6 * len(yaws) + 0.8, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, yaw in zip(axs, yaws):
        Z = np.full((len(vals), len(vals)), np.nan)
        for rr in rows:
            if rr["yaw"] == yaw:
                i, j = list(vals).index(rr["dz"]), list(vals).index(rr["dx"])
                Z[i, j] = rr["dev"] if rr["ok"] else np.nan
        im = ax.imshow(Z, origin="lower", extent=[vals[0] - 10, vals[-1] + 10, vals[0] - 10, vals[-1] + 10],
                       cmap=cmap, vmin=0, vmax=90)
        for rr in rows:
            if rr["yaw"] == yaw and not rr["ok"]:
                ax.plot(rr["dx"], rr["dz"], "x", color=C["red"], ms=5, mew=1.2)
        ax.plot(0, 0, "+", color="white", ms=8)
        ax.set_title(f"yaw {yaw:+d} deg")
        ax.set_xlabel("box dx (mm)")
        ax.grid(False)
    axs[0].set_ylabel("box dz (mm)")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
    cb.set_label("max joint deviation from demo (deg)")
    n_ok = sum(r["ok"] for r in rows)
    fig.suptitle(f"Kinematic reachability of the demo-transform expert: {n_ok}/{len(rows)} feasible "
                 "(x = infeasible: IK or joint limit). No collision check.", x=0.01, ha="left", fontsize=9)
    fig.savefig(out / "fig_reachability.pdf")
    fig.savefig(out / "fig_reachability.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range-mm", type=float, default=30.0)
    ap.add_argument("--yaw-deg", type=float, default=15.0)
    ap.add_argument("--grid-step", type=int, default=20)
    ap.add_argument("--skip-grid", action="store_true")
    a = ap.parse_args(argv)
    out = HERE / "out"
    gen = out / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    m, ds, ex, pivot, ev = setup()
    L = m["arms"]["left"]

    if not a.skip_grid:
        t0 = time.time()
        yaws = (-90, -45, 0, 45, 90)
        rows, vals = reachability(ex, pivot, a.grid_step, 100, yaws)
        (out / "reachability.csv").write_text(
            "yaw,dx,dz,ok,flip,margin_deg,dev_deg,max_speed\n" + "\n".join(
                f"{r['yaw']},{r['dx']},{r['dz']},{int(r['ok'])},{int(r['flip'])},{r['margin']:.2f},{r['dev']:.2f},{r['speed']:.3f}"
                for r in rows))
        fig_reach(rows, vals, yaws, out)
        print(f"reachability: {sum(r['ok'] for r in rows)}/{len(rows)} feasible ({time.time() - t0:.0f}s)")

    # generate demonstrations
    rng = np.random.default_rng(a.seed)
    manifest = dict(source_dataset=str(DATA.name), source_episode=0, fps=ds.fps, events=ev, pivot=pivot.tolist(),
                    seed=a.seed, range_mm=a.range_mm, yaw_deg=a.yaw_deg,
                    action_layout="left q1..7 (rad), left gripper (1 open, 0.48 pre-shape, 0 closed), right q1..7, right gripper",
                    note="kinematically planned; physics, contacts and collisions must be verified in Isaac Sim",
                    episodes=[])
    paths = []
    for i in range(a.n):
        dx, dz = rng.uniform(-a.range_mm, a.range_mm, 2)
        yaw = rng.uniform(-a.yaw_deg, a.yaw_deg)
        p, flip = plan_with_symmetry(ex, pivot, dx, dz, yaw, stride=1)
        name = f"gen_{i:03d}"
        np.savez_compressed(gen / f"{name}.npz", actions=p.actions.astype(np.float32),
                            box_delta=np.array([dx, dz, yaw]), box_T=BoxDelta(pivot, dx, dz, yaw).at(1.0),
                            ok=p.ok, flip=flip)
        manifest["episodes"].append(dict(name=name, dx_mm=round(dx, 2), dz_mm=round(dz, 2), yaw_deg=round(yaw, 2),
                                         flip=flip, ok=p.ok, min_margin_deg=round(p.min_margin_deg, 2),
                                         max_dev_deg=round(p.max_dev_from_demo_deg, 2),
                                         max_joint_speed=round(p.max_joint_speed, 3)))
        if i < 12:
            paths.append((dx, dz, yaw, np.array([L.fk(q[:7])[:3, 3] for q in p.actions[: ev["L release"]: 3]])))
    (gen / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_ok = sum(e["ok"] for e in manifest["episodes"])
    print(f"generated {a.n} demos, {n_ok} kinematically feasible -> {gen}")

    # figure: left TCP paths of generated demos (top view) with box footprints
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    o = m["objects"][BOX]
    lo, hi = np.array(o["bbox_min"]), np.array(o["bbox_max"])
    demo = np.array([L.fk(q[:7])[:3, 3] for q in ex.A[: ev["L release"]: 3]])
    for dx, dz, yaw, P in paths:
        corners = np.array([[lo[0], lo[2]], [hi[0], lo[2]], [hi[0], hi[2]], [lo[0], hi[2]], [lo[0], lo[2]]])
        c = np.c_[corners[:, 0], np.full(5, pivot[1]), corners[:, 1]]
        T = BoxDelta(pivot, dx, dz, yaw).at(1.0)
        cw = (T[:3, :3] @ c.T).T + T[:3, 3]
        axs[0].plot(cw[:, 0], cw[:, 2], color=C["orange"], lw=0.6, alpha=0.8)
        axs[0].plot(P[:, 0], P[:, 2], color=C["blue"], lw=0.5, alpha=0.6)
    axs[0].plot(demo[:, 0], demo[:, 2], color=C["black"], lw=1.2, label="recorded demo")
    axs[0].plot([], [], color=C["blue"], lw=0.8, label="generated (left TCP)")
    axs[0].plot([], [], color=C["orange"], lw=0.8, label="displaced box")
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("z (m)")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Generated approach-and-handover paths (top view)")
    d = np.array([[e["dx_mm"], e["dz_mm"], e["yaw_deg"]] for e in manifest["episodes"]])
    ok = np.array([e["ok"] for e in manifest["episodes"]])
    sc = axs[1].scatter(d[:, 0], d[:, 1], c=d[:, 2], cmap="coolwarm", s=18, vmin=-a.yaw_deg, vmax=a.yaw_deg,
                        edgecolors=[C["black"] if k else C["red"] for k in ok], linewidths=0.4)
    fig.colorbar(sc, ax=axs[1], fraction=0.05).set_label("yaw (deg)")
    axs[1].set_xlabel("box dx (mm)")
    axs[1].set_ylabel("box dz (mm)")
    axs[1].set_title(f"Sampled box poses: {n_ok}/{a.n} feasible")
    axs[1].set_aspect("equal")
    for ax, l in zip(axs, "ab"):
        _panel(ax, l)
    fig.tight_layout()
    fig.savefig(out / "fig_generated_paths.pdf")
    fig.savefig(out / "fig_generated_paths.png")


if __name__ == "__main__":
    main()
