"""FR3 Duo kinematics (FK, Jacobian, damped-least-squares IK) built from the hackathon scene USD.

Model source: Environment_Hackathon_2026/Hackathon_FR3Duo_flattened.usd (PhysicsRevoluteJoint / PhysicsFixedJoint
prims, from <side>_fr3v2_link0 to <side>_tcp). `export_model()` writes robot_model.json, so pxr is only needed once;
`ArmChain.from_json()` works with numpy alone (Mac, Linux, inside Isaac Sim).

Frames: world = USD stage (metres, Y up). Joint convention as in USD physics:
    T_body1 = T_body0 @ T(localPos0, localRot0) @ Rx(q) @ inv(T(localPos1, localRot1))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_USD = HERE.parent / "C2_Teach-the-robot_VLA/Environment/Environment_Hackathon_2026/Hackathon_FR3Duo_flattened.usd"
ROBOT_PRIM = "/World/FR3Duo_withV_Mkit__1_"


# ---------------------------------------------------------------- small SE(3) helpers
def quat_to_mat(w, x, y, z) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z)"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q if q[0] >= 0 else -q


def make_T(R=np.eye(3), p=(0, 0, 0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def rot_axis(axis: str, q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    if axis == "X":
        R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "Y":
        R = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return make_T(np.array(R))


def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of rotation matrix R."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - a < 1e-6:  # 180 deg
        w = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(w))
        w[(i + 1) % 3] *= np.sign(R[i, (i + 1) % 3] + 1e-12)
        w[(i + 2) % 3] *= np.sign(R[i, (i + 2) % 3] + 1e-12)
        return a * w / np.linalg.norm(w)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def pose_error(T_cur: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    """6-vector [dp (m), drot (rad)] in world frame, pointing from current to goal."""
    return np.r_[T_goal[:3, 3] - T_cur[:3, 3], rotvec(T_goal[:3, :3] @ T_cur[:3, :3].T)]


# ---------------------------------------------------------------- arm chain
@dataclass
class Joint:
    name: str
    kind: str              # "revolute" | "fixed"
    A: np.ndarray          # T(localPos0, localRot0)
    B: np.ndarray          # inv(T(localPos1, localRot1))
    axis: str = "X"
    lower: float = -np.inf  # rad
    upper: float = np.inf


@dataclass
class ArmChain:
    side: str
    T_base: np.ndarray                 # world pose of <side>_fr3v2_link0
    joints: list = field(default_factory=list)   # link0 -> tcp, revolute + fixed
    q_home: np.ndarray | None = None   # authored joint angles in the USD (rad)

    @property
    def revolute(self):
        return [j for j in self.joints if j.kind == "revolute"]

    @property
    def lower(self):
        return np.array([j.lower for j in self.revolute])

    @property
    def upper(self):
        return np.array([j.upper for j in self.revolute])

    # ---- forward kinematics
    def fk_all(self, q):
        """World frames of every joint (before rotation) and the TCP."""
        T = self.T_base.copy()
        frames, k = [], 0
        for j in self.joints:
            Tj = T @ j.A
            if j.kind == "revolute":
                frames.append(Tj)
                T = Tj @ rot_axis(j.axis, q[k]) @ j.B
                k += 1
            else:
                T = Tj @ j.B
        return frames, T

    def fk(self, q) -> np.ndarray:
        return self.fk_all(q)[1]

    def jacobian(self, q):
        """Geometric Jacobian (6x7) of the TCP in world frame, plus the TCP pose."""
        frames, T = self.fk_all(q)
        p = T[:3, 3]
        J = np.zeros((6, len(frames)))
        for i, F in enumerate(frames):
            ax = F[:3, {"X": 0, "Y": 1, "Z": 2}[self.revolute[i].axis]]
            J[:3, i] = np.cross(ax, p - F[:3, 3])
            J[3:, i] = ax
        return J, T

    # ---- inverse kinematics
    def ik(self, T_goal, q0, iters=300, lam=0.02, pos_tol=1e-4, rot_tol=1e-3, max_step=0.2,
           w_rot=0.5, q_rest=None, k_null=0.02, margin=np.radians(3)):
        """Damped least squares with joint-limit clamping and a null-space pull towards q_rest.
        Returns (q, info)."""
        q = np.array(q0, float)
        lo, hi = self.lower + margin, self.upper - margin
        W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
        for it in range(iters):
            J, T = self.jacobian(q)
            e = pose_error(T, T_goal)
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
                return q, dict(ok=True, iters=it, pos_err=np.linalg.norm(e[:3]), rot_err=np.linalg.norm(e[3:]))
            Jw = W @ J
            JJt = Jw @ Jw.T + lam ** 2 * np.eye(6)
            J_pinv = Jw.T @ np.linalg.inv(JJt)
            dq = J_pinv @ (W @ e)
            if q_rest is not None:
                N = np.eye(len(q)) - J_pinv @ Jw
                dq += N @ (k_null * (q_rest - q))
            n = np.abs(dq).max()
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, lo, hi)
        J, T = self.jacobian(q)
        e = pose_error(T, T_goal)
        return q, dict(ok=False, iters=iters, pos_err=np.linalg.norm(e[:3]), rot_err=np.linalg.norm(e[3:]))

    # ---- serialisation
    def to_dict(self):
        return dict(side=self.side, T_base=self.T_base.tolist(),
                    q_home=None if self.q_home is None else self.q_home.tolist(),
                    joints=[dict(name=j.name, kind=j.kind, A=j.A.tolist(), B=j.B.tolist(), axis=j.axis,
                                 lower=None if not np.isfinite(j.lower) else j.lower,
                                 upper=None if not np.isfinite(j.upper) else j.upper) for j in self.joints])

    @classmethod
    def from_dict(cls, d):
        js = [Joint(j["name"], j["kind"], np.array(j["A"]), np.array(j["B"]), j["axis"],
                    -np.inf if j["lower"] is None else j["lower"], np.inf if j["upper"] is None else j["upper"])
              for j in d["joints"]]
        return cls(d["side"], np.array(d["T_base"]), js, None if d["q_home"] is None else np.array(d["q_home"]))


def load_model(path=HERE / "robot_model.json") -> dict:
    d = json.loads(Path(path).read_text())
    d["arms"] = {k: ArmChain.from_dict(v) for k, v in d["arms"].items()}
    return d


# ---------------------------------------------------------------- USD import (needs pxr = usd-core)
def export_model(usd_path=DEFAULT_USD, out=HERE / "robot_model.json") -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(usd_path))
    xc = UsdGeom.XformCache()

    def world(path):
        M = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(path))).T  # USD is row-vector convention
        return M

    def T_from(pos, rot):
        return make_T(quat_to_mat(rot.GetReal(), *rot.GetImaginary()), np.array(pos, float))

    joints_by_b0 = {}
    for p in Usd.PrimRange(st.GetPrimAtPath(ROBOT_PRIM)):
        if p.GetTypeName() in ("PhysicsRevoluteJoint", "PhysicsFixedJoint"):
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if b0 and b1:
                joints_by_b0.setdefault(str(b0[0]), []).append((p, j, str(b1[0])))

    arms, extra = {}, {}
    for side in ("left", "right"):
        link = f"{ROBOT_PRIM}/{side}_fr3v2_link0"
        target = f"{ROBOT_PRIM}/{side}_tcp"
        # depth-first search from link0 to tcp
        path = []

        def dfs(body):
            if body == target:
                return True
            for p, j, b1 in joints_by_b0.get(body, []):
                path.append((p, j, b1))
                if dfs(b1):
                    return True
                path.pop()
            return False

        assert dfs(link), f"no chain {link} -> {target}"
        chain, q_home = [], []
        for p, j, b1 in path:
            A = T_from(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get())
            B = inv_T(T_from(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))
            if p.GetTypeName() == "PhysicsRevoluteJoint":
                rj = UsdPhysics.RevoluteJoint(p)
                lo, hi = np.radians(rj.GetLowerLimitAttr().Get()), np.radians(rj.GetUpperLimitAttr().Get())
                axis = str(rj.GetAxisAttr().Get())
                chain.append(Joint(p.GetName(), "revolute", A, B, axis, float(lo), float(hi)))
                # authored angle: relative rotation between the two joint frames
                b0 = str(j.GetBody0Rel().GetTargets()[0])
                F0 = world(b0) @ A
                F1 = world(b1) @ inv_T(B)
                Rrel = F0[:3, :3].T @ F1[:3, :3]
                i1, i2 = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
                q_home.append(float(np.arctan2(Rrel[i2, i1], Rrel[i1, i1])))
            else:
                chain.append(Joint(p.GetName(), "fixed", A, B, "X"))
        arm = ArmChain(side, world(link), chain, np.array(q_home))
        arms[side] = arm
        extra[side] = dict(tcp_authored=world(target).tolist())

    # task objects: world pose and bounding box of rigid bodies near the robot
    objs = {}
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for name in ["WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01", "ServingTray", "SM_PillBottle_01v",
                 "NaturalBostonRoundBottle_A02_PR_NVD_01", "_37_scissors", "glass_plate", "Stackable"]:
        pr = st.GetPrimAtPath(f"/World/{name}")
        if not pr:
            continue
        box = bbc.ComputeWorldBound(pr).ComputeAlignedRange()
        M = world(pr.GetPath())
        scale = np.linalg.norm(M[:3, :3], axis=0)          # assets are resized with non-uniform scale
        T = make_T(M[:3, :3] / scale, M[:3, 3])            # rigid pose (prim origin, unscaled axes)
        objs[name] = dict(T=T.tolist(), scale=scale.tolist(), bbox_min=list(box.GetMin()), bbox_max=list(box.GetMax()))

    d = dict(source=str(usd_path), robot_prim=ROBOT_PRIM, units="m", up_axis="Y",
             arms={k: v.to_dict() for k, v in arms.items()}, authored=extra, objects=objs)
    Path(out).write_text(json.dumps(d, indent=1))
    return d


if __name__ == "__main__":
    d = export_model()
    for s, a in d["arms"].items():
        print(s, "q_home(deg) =", np.round(np.degrees(a["q_home"]), 2))
    print("objects:", list(d["objects"]))
