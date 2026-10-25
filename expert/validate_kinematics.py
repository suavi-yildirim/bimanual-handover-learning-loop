#!/usr/bin/env python3
"""Validate the USD-derived FR3 Duo kinematics against the recorded handover dataset (no simulator needed).

Checks
  1. Internal: FK(q_home) == authored TCP pose in the USD; analytic vs numeric Jacobian; IK round trips.
  2. Data: recorded start state == USD home pose; TCP at frame 0 == authored TCP.
  3. Data: kinematic carry. Box pose in the left TCP frame at the left grasp -> handed to the right TCP at the
     left release -> predicted box pose at the right release, compared with the generator's final_box_pose.
Outputs: validation.json and fig_kinematics_validation.{pdf,png} in expert/out/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dataset_check"))
from kinematics import inv_T, load_model, make_T, pose_error, quat_to_mat, rotvec  # noqa: E402
from dsc.checks import analyse  # noqa: E402
from dsc.io_lerobot import load  # noqa: E402
from dsc.plots import C, _panel  # noqa: E402  (shared figure style)

DATA = HERE.parent / "C2_Teach-the-robot_VLA/Dataset/fr3_duo_handover_84_ep.zip"
BOX = "WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"


def main():
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    m = load_model()
    L, R = m["arms"]["left"], m["arms"]["right"]
    res = dict(internal={}, data={})

    # 1. internal consistency
    rng = np.random.default_rng(0)
    for s, a in m["arms"].items():
        e = pose_error(a.fk(a.q_home), np.array(m["authored"][s]["tcp_authored"]))
        q = a.q_home + 0.1
        J, T0 = a.jacobian(q)
        Jn = np.stack([pose_error(T0, a.fk(q + 1e-6 * np.eye(7)[i])) / 1e-6 for i in range(7)], 1)
        ok, perr = [], []
        for _ in range(100):
            qt = np.clip(a.q_home + rng.normal(0, 0.4, 7), a.lower + 0.1, a.upper - 0.1)
            _, info = a.ik(a.fk(qt), a.q_home)
            ok.append(info["ok"])
            perr.append(info["pos_err"])
        res["internal"][s] = dict(fk_vs_authored_mm=float(np.linalg.norm(e[:3]) * 1e3),
                                  jacobian_max_abs_err=float(np.abs(J - Jn).max()),
                                  ik_success=float(np.mean(ok)), ik_median_pos_err_mm=float(np.median(perr) * 1e3))

    # 2./3. against data
    ds = load(DATA)
    r = analyse(ds)
    sl = ds.episode_slices()
    box0 = np.array(m["objects"][BOX]["T"])
    carry = []
    for e in sorted(sl):
        q = ds.state[sl[e]]
        t = {k: int(r.episodes.loc[e, f"t_{k.replace(' ', '_')}"]) for k in r.canonical_events}
        box_in_L = inv_T(L.fk(q[t["L grasp"], :7])) @ box0
        box_w = L.fk(q[t["L release"], :7]) @ box_in_L
        box_in_R = inv_T(R.fk(q[t["L release"], 8:15])) @ box_w
        pred = R.fk(q[t["R release"], 8:15]) @ box_in_R
        fb = np.array(ds.extra_raw[e]["final_box_pose"])
        Tf = make_T(quat_to_mat(fb[6], *fb[3:6]), fb[:3])   # final_box_pose = [x, y, z, qx, qy, qz, qw]
        carry.append(dict(episode=e, dpos_mm=((pred[:3, 3] - Tf[:3, 3]) * 1e3).tolist(),
                          rot_err_deg=float(np.degrees(np.linalg.norm(rotvec(Tf[:3, :3] @ pred[:3, :3].T))))))
    q0 = ds.state[sl[0]][0]
    res["data"] = dict(
        start_vs_home_deg_max=float(np.degrees(np.abs(np.r_[q0[:7] - L.q_home, q0[8:15] - R.q_home]).max())),
        tcp0_vs_authored_mm=float(np.linalg.norm(L.fk(q0[:7])[:3, 3] - np.array(m["authored"]["left"]["tcp_authored"])[:3, 3]) * 1e3),
        carry_pos_err_mm_median=float(np.median([np.linalg.norm(c["dpos_mm"]) for c in carry])),
        carry_rot_err_deg_median=float(np.median([c["rot_err_deg"] for c in carry])),
        carry_episodes=len(carry), events=list(r.canonical_events),
    )
    (out / "validation.json").write_text(json.dumps(res, indent=1))

    # ---- figure: TCP paths of episode 0 with objects and events
    q = ds.state[sl[0]]
    TL = np.array([L.fk(x[:7])[:3, 3] for x in q])
    TR = np.array([R.fk(x[8:15])[:3, 3] for x in q])
    t = {k: int(r.episodes.loc[0, f"t_{k.replace(' ', '_')}"]) for k in r.canonical_events}
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
    for ax, (i, j, lab) in zip(axs[:2], [(0, 2, ("x (m)", "z (m)")), (0, 1, ("x (m)", "y = height (m)"))]):
        for name, col in [(BOX, C["orange"]), ("ServingTray", C["grey"])]:
            o = m["objects"][name]
            lo, hi = np.array(o["bbox_min"]), np.array(o["bbox_max"])
            ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], color=col, alpha=0.35, lw=0))
            ax.text(hi[i], hi[j], " box" if name == BOX else " tray", fontsize=6.5, color=col)
        ax.plot(TL[:, i], TL[:, j], color=C["blue"], lw=1, label="left TCP")
        ax.plot(TR[:, i], TR[:, j], color=C["green"], lw=1, label="right TCP")
        for k, f in t.items():
            P = TL if k.startswith("L") else TR
            ax.plot(P[f, i], P[f, j], "o", ms=4, mfc="white", mec=C["red"])
            ax.annotate(k, (P[f, i], P[f, j]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
    axs[0].set_title("Top view: TCP paths (episode 0)")
    axs[1].set_title("Side view")
    axs[0].legend(loc="lower left")
    d = np.array([c["dpos_mm"] for c in carry])
    axs[2].bar(["dx", "dy", "dz", "|d|"], np.r_[np.median(d, 0), np.median(np.linalg.norm(d, axis=1))],
               color=[C["sky"]] * 3 + [C["blue"]])
    axs[2].axhline(0, color="k", lw=0.5)
    axs[2].set_ylabel("predicted - recorded final box position (mm)")
    axs[2].set_title(f"Kinematic carry check ({len(carry)} episodes)")
    axs[2].text(0.03, 0.95, f"rotation error {res['data']['carry_rot_err_deg_median']:.2f} deg\n"
                f"FK vs USD TCP {res['internal']['left']['fk_vs_authored_mm']:.1e} mm\n"
                f"IK success {res['internal']['left']['ik_success']:.0%}", transform=axs[2].transAxes, va="top",
                fontsize=6.5)
    for a, l in zip(axs, "abc"):
        _panel(a, l)
    fig.tight_layout()
    fig.savefig(out / "fig_kinematics_validation.pdf")
    fig.savefig(out / "fig_kinematics_validation.png")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
