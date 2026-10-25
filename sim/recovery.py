"""Failure detection and retry around the closed-loop policy (supervisor / skill layer).

The policy proposes an action every frame; `Supervisor.step` either passes it through or replaces it with a
scripted recovery motion, and can abort the episode with a reason. Detectors use signals a real robot also
has (gripper opening, joint state, progress along the demonstrated task) plus the box height, which is
sim-only and used only to stop when the box is on the floor.

Failures:
  grasp_miss  a gripper has been commanded closed for GRASP_CONFIRM frames but its opening says it holds
              nothing (state < EMPTY_CLOSED; closed on the box reads ~0.42, closed empty ~0.10).
  stall       no task progress for STALL_FRAMES: the nearest recorded frame has not advanced by STALL_MIN_PROGRESS,
              no gripper grasped or released, and the box has not moved BOX_PROGRESS_M (the recorded-frame measure
              is unreliable when the arm is far from the demonstrated path, e.g. after a grasp correction).
  pause       a stall while the left holds the box at the demonstrated pause before the right approach: the
              demonstration starts the right arm after ~2.5 s of standing still, with no visual cue, so the policy
              cannot tell when. The supervisor supplies the clock once: `pause_kick` asks the caller to commit the
              right gripper's pre-shape (the step that starts the right approach), and the stall timer restarts.
  box_dropped box centre below BOX_FLOOR_Y m: not recoverable here, abort.
  box_tipped  box tilted more than BOX_TIP_DEG from its start orientation while no gripper holds it (sim-only):
              the policy never saw a lying box, abort.
Recovery for one arm: hold the other arm at its last command (keeps a held box), open this gripper to pre-shape,
rewind this arm to the pose it was commanded REWIND frames ago (its own path, so the pose stays consistent with
wherever the other arm is), then hand back to the policy, which re-plans from there. A grasp or release by
either gripper counts as task progress for the stall detector. At most `max_retries` per arm, then abort.
"""

import numpy as np

GRIP = {"L": 7, "R": 15}
ARM = {"L": slice(0, 7), "R": slice(8, 15)}
PRE_SHAPE = 0.476
CLOSE_CMD = 0.25
EMPTY_CLOSED = 0.20  # closed on the box reads ~0.34-0.42, closed empty ~0.10
HOLDING = 0.3  # gripper state above this while commanded closed = something between the fingers
GRASP_CONFIRM = 30  # the recorded gripper drive is slow; give it time to finish closing
STALL_FRAMES = 150
PROGRESS_WINDOW = (-60, 150)
R_APPROACH_FRAMES = 300  # recorded frames before the right grasp that count as its approach  # progress search window around the best progress so far (recorded frames)
STALL_MIN_PROGRESS = 20
BOX_FLOOR_Y = 0.6
BOX_TIP_DEG = 35.0
BOX_PROGRESS_M = 0.02
REWIND = 90  # retry: the failing arm goes back to where it was commanded this many frames ago
OPEN_FRAMES, MOVE_FRAMES, SETTLE_FRAMES = 20, 45, 10


def progress_frame(state, rec_states, lo=0, hi=None):
    """Nearest recorded frame by all 14 arm joints, searched in rec_states[lo:hi]."""
    arm = np.r_[0:7, 8:15]
    lo, hi = max(0, lo), len(rec_states) if hi is None else min(len(rec_states), hi)
    return lo + int(np.abs(rec_states[lo:hi, arm] - state[arm]).max(axis=1).argmin())


def quat_matrix(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def holding(side, action, state):
    return action[GRIP[side]] < CLOSE_CMD and state[GRIP[side]] > HOLDING


class Supervisor:
    def __init__(self, rec_states, close_frames, max_retries=2):
        """close_frames: recorded grasp frame per side, e.g. {"L": 327, "R": 1083}."""
        self.rec = rec_states
        self.close_frames = close_frames
        self.max_retries = max_retries
        self.retries = {"L": 0, "R": 0}
        self.closed_for = {"L": 0, "R": 0}
        self.plan = []  # scripted 16-d targets still to execute
        self.history = []  # commanded 16-d actions, newest last
        self.held = (False, False)
        self.box_up0 = None  # box "up" axis at the start
        self.last_retry = None  # (frame, side) of the latest retry
        self.box_at_progress = None  # box centre at the last progress
        self.pause_kick = None  # frame at which the right approach was started by the supervisor
        self.best_progress, self.best_at = 0, 0
        self.events = []
        self.status = "policy"

    @property
    def recovering(self):
        return bool(self.plan)

    def _event(self, k, kind, text):
        self.events.append({"frame": k, "kind": kind, "detail": text})
        print(f"[supervisor] frame {k}: {kind}: {text}", flush=True)

    def _retry(self, k, side, last_cmd, reason):
        """last_cmd: the last commanded 16-d action. Holding it (not the measured state) keeps the other gripper
        closed on the box and the arms where the controller already is (command and state differ by up to 5 deg)."""
        if self.retries[side] >= self.max_retries:
            return f"{reason}; {side} arm already retried {self.retries[side]} times"
        self.retries[side] += 1
        self.last_retry = (k, side)
        hold = last_cmd.copy()
        start = last_cmd[ARM[side]].copy()
        goal = self.history[max(0, len(self.history) - REWIND)][ARM[side]].copy()
        plan = []
        for _ in range(OPEN_FRAMES):
            t = hold.copy()
            t[GRIP[side]] = PRE_SHAPE
            plan.append(t)
        for i in range(1, MOVE_FRAMES + 1):
            t = plan[-1].copy()
            a = 0.5 - 0.5 * np.cos(np.pi * i / MOVE_FRAMES)  # smooth start and stop
            t[ARM[side]] = start + a * (goal - start)
            plan.append(t)
        plan += [plan[-1].copy() for _ in range(SETTLE_FRAMES)]
        self.plan = plan
        self.closed_for = {"L": 0, "R": 0}
        self.best_at = k + len(plan)  # do not count the recovery time as a stall
        self._event(k, "retry", f"{reason} -> {side} arm: open, rewind {REWIND} frames along its own path, "
                                f"attempt {self.retries[side] + 1}/{self.max_retries + 1}")
        self.status = f"RECOVERY {side} (retry {self.retries[side]}/{self.max_retries}): {reason}"
        return None

    def step(self, k, state, action, box_centre, box_quat=None):
        """Returns (action to apply, abort reason or None, True if the policy must re-plan next frame)."""
        if self.plan:
            target = self.plan.pop(0)
            self.history.append(target)
            if not self.plan:
                self.status = "policy (resumed after recovery)"
                return target, None, True
            return target, None, False

        box_centre = np.asarray(box_centre, dtype=np.float64)
        box_y = float(box_centre[1])
        if self.box_at_progress is None or np.linalg.norm(box_centre - self.box_at_progress) >= BOX_PROGRESS_M:
            self.box_at_progress, self.best_at = box_centre, k  # the box is being moved: progress
        self.history.append(np.asarray(action, dtype=np.float64).copy())
        held = (holding("L", action, state), holding("R", action, state))
        if held != self.held:  # a grasp or release is progress
            self.held, self.best_at = held, k
        if box_quat is not None and not any(held):
            rot = quat_matrix(box_quat)
            if self.box_up0 is None:
                self.box_up0 = rot.T @ np.array([0.0, 1.0, 0.0])  # box-frame axis that points up at the start
            tilt = np.degrees(np.arccos(np.clip((rot @ self.box_up0)[1], -1.0, 1.0)))  # its angle to world up
            if tilt > BOX_TIP_DEG:
                self._event(k, "box_tipped", f"box tilted {tilt:.0f} deg with no gripper holding it")
                return action, f"box tipped over ({tilt:.0f} deg)", False
        if box_y < BOX_FLOOR_Y:
            self._event(k, "box_dropped", f"box centre at {box_y:.2f} m")
            return action, "box dropped to the floor", False

        for side in ("L", "R"):
            if action[GRIP[side]] < CLOSE_CMD:
                self.closed_for[side] += 1
            else:
                self.closed_for[side] = 0
            if self.closed_for[side] >= GRASP_CONFIRM and state[GRIP[side]] < EMPTY_CLOSED:
                self._event(k, "grasp_miss", f"{side} gripper closed on nothing (state {state[GRIP[side]]:.2f})")
                abort = self._retry(k, side, action, f"{side} grasp missed")
                if abort:
                    return action, abort, False
                return self.plan.pop(0), None, False

        # Progress is searched near the best so far, so a similar-looking later pose cannot make it jump ahead.
        prog = progress_frame(state, self.rec, self.best_progress + PROGRESS_WINDOW[0],
                              self.best_progress + PROGRESS_WINDOW[1])
        if prog >= self.best_progress + STALL_MIN_PROGRESS:
            self.best_progress, self.best_at = prog, k
        elif k - self.best_at >= STALL_FRAMES:
            # Which grasp is pending follows from what the grippers hold, not from the progress estimate.
            if not holding("L", action, state) and not holding("R", action, state):
                side = "L"
            elif holding("L", action, state) and not holding("R", action, state):
                # Before the right arm's approach phase the left arm is carrying: retrying the right arm
                # there only moves an uninvolved arm.
                side = "R" if self.best_progress >= self.close_frames["R"] - R_APPROACH_FRAMES else None
            else:
                side = None
            self._event(k, "stall", f"no task progress for {k - self.best_at} frames "
                                    f"(stuck near recorded frame {self.best_progress})")
            if (side is None and self.pause_kick is None and holding("L", action, state)
                    and self.best_progress >= self.close_frames["R"] - R_APPROACH_FRAMES - 30):
                self.pause_kick, self.best_at = k, k
                self._event(k, "pause_kick", f"waiting at the demonstrated pause (recorded frame {self.best_progress}): "
                                             "starting the right approach")
                self.status = "pause: right approach started by the supervisor"
                return action, None, False
            if side is None:
                return action, f"stalled near recorded frame {self.best_progress} while carrying the box", False
            abort = self._retry(k, side, action, f"stalled near recorded frame {self.best_progress}")
            if abort:
                return action, abort, False
            return self.plan.pop(0), None, False

        self.status = f"policy | progress: recorded frame {self.best_progress} | retries L {self.retries['L']} R {self.retries['R']}"
        return action, None, False
