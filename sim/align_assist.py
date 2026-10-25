"""Grasp alignment assist: centre the grasping hand on the box in the last centimetres before the grasp.

The policy imitates one demonstrated motion but drifts 2-4 cm at the hand in closed loop (no corrective
examples in the data), which tips the tall, thin box. This layer adds a joint-space offset to the policy's
arm command so the TCP reaches box_centre + REF[side], the TCP-box offset at which the gripper closed in the
successful open-loop replay. The policy still decides when to close.

Per frame, for the arm whose grasp is pending (left first, then right while the left holds):
  - engage only within ENGAGE_M[side] of the grasp point (the policy does the gross motion)
  - lateral (x, z) first: no descent while the lateral error exceeds LATERAL_FIRST_M (rising is allowed)
  - Cartesian step capped at MAX_STEP_M, mapped to joints with damped least squares on the TCP Jacobian
  - total offset per joint capped at MAX_OFFSET_RAD
After that arm grasps, its offset is kept, plus a height floor: the assist usually brings the hand down before the
policy's own timing, and the policy then finishes its descent and dives the hand into the box. Until the box is
FLOOR_UNTIL_LIFT_M above its height at the grasp, the TCP is servoed back up to its grasp height whenever it sinks
below it (FLOOR_TOL_M). After the arm releases, the offset fades out over FADE_FRAMES. The box pose is sim ground truth; on a robot it would come from perception.
"""

import numpy as np

# TCP minus box centre (world, m) when each gripper closed in the open-loop replay with the box lying as shipped
# on a 30 mm support (sim/out/replay_support_30).
REF = {"L": np.array([0.0002, 0.0285, -0.0307]), "R": np.array([0.0801, -0.0339, 0.0002])}
ENGAGE_M = {"L": 0.12, "R": 0.20}  # the right hand meets a box carried by the policy, often well off its demo pose
LATERAL_FIRST_M = 0.005
MAX_STEP_M = 0.002
MAX_OFFSET_RAD = 0.25
DAMPING = 0.01
FADE_FRAMES = 60
FLOOR_TOL_M = 0.002
FLOOR_UNTIL_LIFT_M = 0.03


class AlignAssist:
    def __init__(self):
        self.offset = {"L": np.zeros(7), "R": np.zeros(7)}
        self.state = {"L": "idle", "R": "idle"}  # idle -> aligning -> holding -> fading -> done
        self.fade_left = {"L": 0, "R": 0}
        self.err_at_grasp = {"L": None, "R": None}
        self.status = "align assist idle"
        self.last_err = {"L": None, "R": None}  # latest TCP error to the grasp point (m, norm)
        self.floor = {"L": None, "R": None}  # (TCP height at the grasp, box height at the grasp) while active

    def pending_side(self, grasped):
        if not grasped["L"]:
            return "L"
        if not grasped["R"]:
            return "R"
        return None

    def update(self, k, grasped, released, tcp, box_centre, jac_pos):
        """grasped/released: per side bools. tcp: per side world position (3,). jac_pos: per side (3, 7).
        Returns the per-side 7-joint offsets to add to the policy's arm commands."""
        for side in ("L", "R"):
            st = self.state[side]
            if st in ("idle", "aligning") and grasped[side]:
                self.state[side] = "holding"
                err = box_centre + REF[side] - tcp[side]
                self.err_at_grasp[side] = np.round(1000 * err, 1).tolist()
                self.floor[side] = (float(tcp[side][1]), float(box_centre[1]))
                print(f"[assist] frame {k}: {side} grasped, TCP error to the replay grasp point "
                      f"{self.err_at_grasp[side]} mm, joint offset {np.round(np.degrees(self.offset[side]), 1).tolist()} deg",
                      flush=True)
            elif st == "holding" and self.floor[side] is not None:
                tcp_y0, box_y0 = self.floor[side]
                if box_centre[1] > box_y0 + FLOOR_UNTIL_LIFT_M:
                    # Box in the air: hand the arm back to the policy unmodified, so the carry follows its own
                    # path (a held offset drifts the arm 9-10 deg off the demonstration and it never reaches the
                    # pause before the right arm starts).
                    self.floor[side] = None
                    self.state[side], self.fade_left[side] = "fading", FADE_FRAMES
                    print(f"[assist] frame {k}: {side} box lifted, height floor off, correction fading out", flush=True)
                elif tcp[side][1] < tcp_y0 - FLOOR_TOL_M:
                    up = np.array([0.0, min(tcp_y0 - tcp[side][1], MAX_STEP_M), 0.0])
                    J = jac_pos[side]
                    dq = J.T @ np.linalg.solve(J @ J.T + DAMPING ** 2 * np.eye(3), up)
                    self.offset[side] = np.clip(self.offset[side] + dq, -MAX_OFFSET_RAD, MAX_OFFSET_RAD)
                    self.status = f"assist {side}: height floor, hand {1000 * (tcp_y0 - tcp[side][1]):.0f} mm low"
                if released[side]:
                    self.state[side], self.fade_left[side] = "fading", FADE_FRAMES
            elif st == "holding" and released[side]:
                self.state[side], self.fade_left[side] = "fading", FADE_FRAMES
            elif st == "fading":
                self.fade_left[side] -= 1
                self.offset[side] *= max(0, self.fade_left[side]) / (self.fade_left[side] + 1)
                if self.fade_left[side] <= 0:
                    self.state[side], self.offset[side] = "done", np.zeros(7)

        side = self.pending_side(grasped)
        if side is None or self.state[side] not in ("idle", "aligning"):
            return self.offset
        err = box_centre + REF[side] - tcp[side]
        if np.linalg.norm(err) > ENGAGE_M[side]:
            self.status = f"assist {side}: waiting ({1000 * np.linalg.norm(err):.0f} mm from grasp point)"
            return self.offset
        self.state[side] = "aligning"
        self.last_err[side] = float(np.linalg.norm(err))
        step = err.copy()
        lateral = np.hypot(err[0], err[2])
        if lateral > LATERAL_FIRST_M and step[1] < 0:
            step[1] = 0.0  # centre first, then descend
        n = np.linalg.norm(step)
        if n > MAX_STEP_M:
            step *= MAX_STEP_M / n
        J = jac_pos[side]
        dq = J.T @ np.linalg.solve(J @ J.T + DAMPING ** 2 * np.eye(3), step)
        self.offset[side] = np.clip(self.offset[side] + dq, -MAX_OFFSET_RAD, MAX_OFFSET_RAD)
        self.status = (f"assist {side}: aligning, error lateral {1000 * lateral:.0f} mm, "
                       f"vertical {1000 * err[1]:+.0f} mm")
        return self.offset
