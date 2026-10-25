"""Demo-transform expert (MimicGen-style) for the FR3 Duo box handover.

Idea: the recorded demonstration solves the task for the nominal box pose. For a displaced box, the box-relative
part of the left-arm command trajectory (approach + grasp) is moved with the box; the displacement is blended in
before the grasp and blended out again during the lift, so the handover, the right arm and the placement replay
unchanged (the box sits identically in the left gripper). New joint commands come from damped-least-squares IK,
frame by frame, with a null-space pull towards the recorded posture.

Output has the dataset action layout (16 = left q1..7, left gripper, right q1..7, right gripper), so it can be
replayed in Isaac Sim exactly like a recorded episode. No collision checking here: verify in simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics import ArmChain, make_T, rot_axis


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


@dataclass
class BoxDelta:
    """Box displacement: yaw about the vertical (Y) axis through `pivot` (box centre), then a horizontal shift
    (dx, dz) in world axes. Blending is done on these parameters (a matrix lerp would rotate about the stage origin,
    which is ~23 m away). Convention of the organizers' box_pose {dx_mm, dz_mm, yaw_deg} fields: to confirm."""
    pivot: np.ndarray
    dx_mm: float = 0.0
    dz_mm: float = 0.0
    yaw_deg: float = 0.0

    def at(self, w: float = 1.0) -> np.ndarray:
        p = np.asarray(self.pivot, float)
        Ry = rot_axis("Y", np.radians(w * self.yaw_deg))
        return make_T(p=np.array([w * self.dx_mm, 0.0, w * self.dz_mm]) / 1000.0) @ make_T(p=p) @ Ry @ make_T(p=-p)


def box_delta(pivot, dx_mm=0.0, dz_mm=0.0, yaw_deg=0.0) -> np.ndarray:
    return BoxDelta(np.asarray(pivot, float), dx_mm, dz_mm, yaw_deg).at(1.0)


@dataclass
class Plan:
    actions: np.ndarray          # (T, 16)
    ok: bool
    ik_fail_frames: int
    max_pos_err_mm: float
    max_rot_err_deg: float
    min_margin_deg: float
    max_joint_speed: float       # rad/s
    max_dev_from_demo_deg: float


class DemoTransformExpert:
    def __init__(self, left: ArmChain, right: ArmChain, demo_actions: np.ndarray, fps: float,
                 t_grasp: int, t_release_L: int, blend_in_frac=0.8, blend_out_frac=0.5):
        """t_grasp: frame where the left gripper closes; t_release_L: frame where it opens at the handover."""
        self.L, self.R = left, right
        self.A = np.asarray(demo_actions, float)
        self.fps = fps
        self.tg, self.tr = int(t_grasp), int(t_release_L)
        n = len(self.A)
        # weight of the box displacement for each frame of the left arm
        w = np.zeros(n)
        t_in = int(self.tg * blend_in_frac)            # fully displaced from here on (final approach is rigid)
        w[:t_in] = min_jerk(np.arange(t_in) / max(t_in, 1))
        w[t_in:self.tg + 1] = 1.0
        t_out = self.tg + int((self.tr - self.tg) * blend_out_frac)
        k = np.arange(self.tg + 1, t_out)
        w[self.tg + 1:t_out] = 1.0 - min_jerk((k - self.tg) / max(t_out - self.tg, 1))
        self.weight = w
        self.TL_demo = [self.L.fk(a[:7]) for a in self.A]

    def plan(self, delta: BoxDelta, stride: int = 1) -> Plan:
        A = self.A.copy()
        q = A[0, :7].copy()
        n_fail, perr, rerr, speeds = 0, 0.0, 0.0, []
        lo, hi = self.L.lower, self.L.upper
        frames = list(range(0, len(A), stride))
        if frames[-1] != len(A) - 1:
            frames.append(len(A) - 1)
        q_prev, t_prev = q.copy(), 0
        offset = np.zeros(7)          # correction carried from frame to frame (0 for the nominal box)
        for t in frames:
            w = self.weight[t]
            if w < 1e-9:
                q = A[t, :7].copy()
                offset[:] = 0.0
            else:
                goal = delta.at(w) @ self.TL_demo[t]
                seed = np.clip(A[t, :7] + offset, lo, hi)
                # seed carries the previous correction (continuity); null space pulls back to the demo posture
                q, info = self.L.ik(goal, seed, iters=60, q_rest=A[t, :7], k_null=0.3, margin=np.radians(1))
                offset = q - A[t, :7]
                perr = max(perr, info["pos_err"])
                rerr = max(rerr, info["rot_err"])
                n_fail += int(not (info["pos_err"] < 1e-3 and info["rot_err"] < 5e-3))
            A[t, :7] = q
            if t > 0:
                speeds.append(np.abs(q - q_prev).max() * self.fps / max(t - t_prev, 1))
            q_prev, t_prev = q.copy(), t
        if stride > 1:   # fill skipped frames by linear interpolation of joint commands
            for j in range(7):
                A[:, j] = np.interp(np.arange(len(A)), frames, A[frames, j])
        margin = np.minimum(A[:, :7] - lo, hi - A[:, :7]).min()
        dev = np.abs(A[:, :7] - self.A[:, :7]).max()
        return Plan(A, n_fail == 0 and margin > np.radians(1), n_fail, perr * 1e3, np.degrees(rerr),
                    float(np.degrees(margin)), float(max(speeds) if speeds else 0.0), float(np.degrees(dev)))


def plan_with_symmetry(expert: "DemoTransformExpert", pivot, dx_mm=0.0, dz_mm=0.0, yaw_deg=0.0, stride=1):
    """The box is symmetric under a 180 deg turn about the vertical axis: try the nominal and the flipped grasp,
    return (plan, flipped) with the feasible one that deviates least from the demo."""
    cands = []
    for flip in (False, True):
        yaw = yaw_deg + (180.0 if flip else 0.0)
        yaw = (yaw + 180.0) % 360.0 - 180.0
        p = expert.plan(BoxDelta(np.asarray(pivot, float), dx_mm, dz_mm, yaw), stride=stride)
        cands.append((not p.ok, p.max_dev_from_demo_deg, flip, p))
    cands.sort(key=lambda c: (c[0], c[1]))
    return cands[0][3], cands[0][2]
