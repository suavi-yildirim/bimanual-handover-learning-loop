"""Replay a recorded handover episode inside an already-open Isaac Sim window.

Open Hackathon_FR3Duo_flattened.usd in Isaac Sim, then in Window > Script Editor run:

    EPISODE = 0
    exec(open("<repo>/sim/replay_in_gui.py").read())

Same replay as replay_episode.py (recorded actions as joint targets, 2 physics
ticks per 30 Hz frame), but driven from a physics pre-step callback so it runs
in the open GUI. Progress and the final summary are printed to the Console tab.
Running it again restarts the replay.
"""

import asyncio
import builtins
import importlib
import math
import os
import sys
import traceback

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

SIM_DIR = os.path.join(os.environ.get("REPO_DIR", "."), "sim")  # exec() has no __file__; set REPO_DIR
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)
import replay_episode as rp  # noqa: E402  (shared loading / conversion helpers)

# The Script Editor keeps imported modules between runs; pick up edits to replay_episode.py.
rp = importlib.reload(rp)

_episode = globals().get("EPISODE", 0)

# Extra finger travel (rad) past the stall point while holding the box.
GRIP_SQUEEZE_RAD = 0.05
# Stall detection: a finger counts as closing above MOVING_SPEED (rad/s) and as blocked below STALL_SPEED.
GRIP_MOVING_SPEED = 0.05
GRIP_STALL_SPEED = 0.02
GRIP_STALL_MARGIN_RAD = 0.05

# Drop the callback from a previous run of this script.
def _drop_callback():
    cb = getattr(builtins, "_fr3_replay_cb", None)
    if cb is not None:
        SimulationManager.deregister_callback(cb)
        builtins._fr3_replay_cb = None


_drop_callback()


async def _run(episode):
    states, actions, raw_rows, extra = rp.load_episode(rp.DEFAULT_DATASET, episode)
    ticks = rp.ticks_per_frame(raw_rows, ignore_cut=False)
    n = len(actions)
    print(f"[replay] episode {episode}: {n} frames", flush=True)

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    # Restart from the authored scene so the box is back on the cart.
    timeline.stop()
    await app.next_update_async()
    timeline.play()
    for _ in range(3):
        await app.next_update_async()

    # Friction and gripper drive as in the recording (the shipped gripper drive is much stiffer).
    rp.apply_recorded_physics(omni.usd.get_context().get_stage())

    arms = []
    for side in ("left", "right"):
        art = Articulation(f"{rp.ROBOT}/{side}_arm_root_joint")
        names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_ctrl"]
        arms.append((art, art.get_dof_indices(names).numpy()))
    box = RigidPrim(rp.BOX)

    # Per side: held finger target (None = not holding) and whether the finger was closing last frame.
    grip_hold = [None, None]
    grip_moving = [False, False]

    def gripper_target(i, cmd, q_now, v_now):
        # A finger that was closing and has stopped while still commanded to close is blocked by the box.
        # Hold there so the box width sets the grip and friction holds it.
        if grip_hold[i] is not None and cmd >= grip_hold[i]:
            return grip_hold[i]
        grip_hold[i] = None
        closing = cmd > q_now + GRIP_STALL_MARGIN_RAD
        if closing and grip_moving[i] and abs(v_now) < GRIP_STALL_SPEED:
            # Aim slightly past the stall so the drive keeps a squeeze on the box.
            grip_hold[i] = min(q_now + GRIP_SQUEEZE_RAD, cmd)
            grip_moving[i] = False
            return grip_hold[i]
        grip_moving[i] = closing and v_now > GRIP_MOVING_SPEED
        return cmd

    def set_arms(vec16, teleport):
        for i, ((art, idx), q) in enumerate(zip(arms, rp.to_joint_targets(vec16))):
            if teleport:
                grip_hold[i] = None
                grip_moving[i] = False
                art.set_dof_positions(q.reshape(1, -1), dof_indices=idx)
                art.set_dof_velocities(np.zeros((1, len(idx))), dof_indices=idx)
            else:
                q_now = art.get_dof_positions(dof_indices=idx).numpy().reshape(-1)[7]
                v_now = art.get_dof_velocities(dof_indices=idx).numpy().reshape(-1)[7]
                q[7] = gripper_target(i, q[7], q_now, v_now)
            art.set_dof_position_targets(q.reshape(1, -1), dof_indices=idx)

    def read_arms():
        out = []
        for art, idx in arms:
            q = art.get_dof_positions(dof_indices=idx).numpy().reshape(-1).astype(np.float64)
            q[7] = 1.0 - q[7] / rp.GRIPPER_CLOSED_RAD
            out.append(q)
        return np.concatenate(out)

    set_arms(states[0], teleport=True)
    errors = np.zeros((n, 16))
    st = {"k": 0, "left": 0}

    def on_pre_step(dt, context):
        k = st["k"]
        if k >= n:
            return
        if st["left"] == 0:
            errors[k] = read_arms() - states[k]
            set_arms(actions[k], teleport=False)
            st["left"] = int(ticks[k])
            if k % 300 == 0:
                arm_err = np.degrees(np.abs(errors[k][[*range(7), *range(8, 15)]])).max()
                print(f"[replay] frame {k}/{n}  max arm joint error {arm_err:.2f} deg", flush=True)
        st["left"] -= 1
        if st["left"] == 0:
            st["k"] += 1

    builtins._fr3_replay_cb = SimulationManager.register_callback(on_pre_step, SimulationEvent.PHYSICS_PRE_STEP)

    while st["k"] < n:
        if not timeline.is_playing():
            print(f"[replay] timeline stopped at frame {st['k']}/{n}; replay aborted", flush=True)
            break
        await app.next_update_async()

    _drop_callback()
    if st["k"] < n:
        return

    arm_cols = [*range(7), *range(8, 15)]
    pos, quat = box.get_world_poses()
    box_pos = pos.numpy().reshape(-1).tolist()
    box_quat = quat.numpy().reshape(-1).tolist()
    print(f"[replay] finished {n} frames", flush=True)
    print(f"[replay] arm joint error: RMSE {np.degrees(np.sqrt((errors[:, arm_cols] ** 2).mean())):.3f} deg, "
          f"max {np.degrees(np.abs(errors[:, arm_cols]).max()):.3f} deg", flush=True)
    print(f"[replay] final box position (m): {[round(v, 4) for v in box_pos]}", flush=True)
    if extra and extra.get("final_box_pose"):
        ref = extra["final_box_pose"]
        print(f"[replay] recorded final box position (m): {[round(v, 4) for v in ref[:3]]}", flush=True)
        print(f"[replay] box position error {1000 * math.dist(box_pos, ref[:3]):.1f} mm, "
              f"rotation error {rp.quat_angle_deg(box_quat, ref[3:7]):.2f} deg", flush=True)
    print("[replay] simulation keeps playing; press Stop to reset the scene", flush=True)


async def _run_safe(episode):
    try:
        await _run(episode)
    except Exception:
        _drop_callback()
        print("[replay] FAILED:\n" + traceback.format_exc(), flush=True)


asyncio.ensure_future(_run_safe(_episode))
