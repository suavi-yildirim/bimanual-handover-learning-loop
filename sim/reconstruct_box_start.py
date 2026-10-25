"""Reconstruct the box start pose used in the recording from replayed gripper poses.

The shipped scene places the box elsewhere than the (unshipped) task layers did. The
arms replay accurately and the box does not slip in the grippers (slip_max ~0.14 mm in
fr3_episodes_extra.jsonl), so the box pose is rigidly tied to whichever gripper holds it:

    box-in-right-hand  X_R = inv(T_right @ release)  * B_final        (recorded final pose)
    box at handover    B_h = T_right @ right-close   * X_R
    box-in-left-hand   X_L = inv(T_left @ right-close) * B_h
    box at start       B_0 = T_left @ left-close     * X_L

Input: a replay summary JSON with `gripper_events` (tcp_pos_m, tcp_quat_wxyz), e.g.
    python sim/reconstruct_box_start.py sim/out/tcp_poses/replay_ep000.json
"""

import json
import sys

import numpy as np

EXTRA = ("C2_Teach-the-robot_VLA/Environment/Environment_Hackathon_2026/"
         "FR3Duo_Handover_Dataset_lerobot3/meta/fr3_episodes_extra.jsonl")


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q):
    return q * np.array([1, -1, -1, -1])


def qrot(q, v):
    return qmul(qmul(q, np.r_[0.0, v]), qconj(q))[1:]


def compose(a, b):
    """Pose a * pose b, poses as (position, quaternion wxyz)."""
    return a[0] + qrot(a[1], b[0]), qmul(a[1], b[1])


def inverse(a):
    qi = qconj(a[1])
    return -qrot(qi, a[0]), qi


def first_event(events, side, before, after):
    """First logged gripper change for `side` from `before` to `after` (command values, rounded to 3 dp)."""
    for e in events:
        if e["side"] == side and e["gripper_cmd"] == [before, after]:
            return e
    raise SystemExit(f"no {side} gripper event {before}->{after}")


def pose(e):
    return np.array(e["tcp_pos_m"], float), np.array(e["tcp_quat_wxyz"], float)


def main():
    summary = json.load(open(sys.argv[1]))
    episode = summary["episode"]
    with open(EXTRA) as fh:
        extra = [json.loads(line) for line in fh][episode]
    ev = summary["gripper_events"]

    left_close = first_event(ev, "left", 0.476, 0.0)
    right_close = first_event(ev, "right", 0.476, 0.0)
    right_release = first_event(ev, "right", 0.0, 0.005)  # opening command after the right-hand place

    fb = extra["final_box_pose"]
    b_final = np.array(fb[:3], float), np.array(fb[3:7], float)

    left_release = first_event(ev, "left", 0.0, 0.005)  # left lets go while the right still holds the box

    # Box-in-hand offsets are constant while held (slip is ~0.1 mm in the recording).
    x_r = compose(inverse(pose(right_release)), b_final)         # right holds 1083 .. 1901
    b_mid = compose(pose(right_close), x_r)                      # box pose at right grasp
    x_l = compose(inverse(pose(left_release)), b_mid)            # left holds 327 .. 1135; b_mid ~ box at 1135
    b_0 = compose(pose(left_close), x_l)                         # box pose at left grasp = start (box is static before)

    q0 = b_0[1] / np.linalg.norm(b_0[1])
    print(f"events used: left close f{left_close['frame']}, right close f{right_close['frame']}, "
          f"right release f{right_release['frame']}")
    print("reconstructed box START pose (prim origin, world):")
    print("  position m :", np.round(b_0[0], 4).tolist())
    print("  quat wxyz  :", np.round(q0, 5).tolist())
    print("box-in-left-hand offset (TCP frame) m:", np.round(x_l[0], 4).tolist())
    print("box-in-right-hand offset (TCP frame) m:", np.round(x_r[0], 4).tolist())
    print("box centre at right grasp (world) m:", np.round(b_mid[0], 4).tolist())
    json.dump({"episode": episode, "box_start_pos_m": b_0[0].tolist(), "box_start_quat_wxyz": q0.tolist()},
              open(sys.argv[1].replace(".json", "_box_start.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
