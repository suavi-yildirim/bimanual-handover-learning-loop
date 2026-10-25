"""Thumbstick locomotion for the Quest 3 in the Isaac Sim VR session.

Kit's built-in XR navigation gets no controller data from the CloudXR runtime (hand poses and sticks
read zero in sim/xr_input_probe.py). This script reads the controllers the way Isaac Sim's teleop
extension does, through its own OpenXR session with isaacteleop DeviceIO trackers, and moves the VR
anchor (/World/XR_Anchor, created by sim/start_vr.py) itself.

Run after sim/start_vr.py, in the Script Editor:
    exec(open("<repo>/sim/vr_locomotion.py").read())
Set STOP_NAV = True before the exec line to stop it. Running it again restarts it.

Controls: left stick = walk (relative to where you look), right stick left/right = snap turn 30 deg,
right stick up/down = rise / sink.
"""

import builtins
import contextlib
import math
import os
import sys
import traceback

import omni.kit.app
import omni.usd
from omni.kit.xr.core import XRCore
from pxr import Gf, Usd, UsdGeom

TELEOP_PREBUNDLE = os.path.join(os.environ.get("ISAAC_ENV", os.path.expanduser("~/isaac-env-6")),
                                "lib/python3.12/site-packages/isaacsim/exts/isaacsim.replicator.teleop/pip_prebundle")
if TELEOP_PREBUNDLE not in sys.path:
    sys.path.append(TELEOP_PREBUNDLE)

ANCHOR = "/World/XR_Anchor"
WALK_SPEED = 1.5        # m/s at full stick
LIFT_SPEED = 0.75       # m/s at full stick
SNAP_DEG = 30.0
DEADZONE = 0.15
SNAP_ON, SNAP_OFF = 0.7, 0.3
UP = Gf.Vec3d(0, 1, 0)  # the scene is Y-up, metres


def _stop_previous():
    old = getattr(builtins, "_xr_nav", None)
    if old is None:
        return
    sub, stack = old
    try:
        sub.unsubscribe()
        stack.close()
    except Exception:
        traceback.print_exc()
    builtins._xr_nav = None
    print("[nav] stopped", flush=True)


def _dz(v):
    return 0.0 if abs(v) < DEADZONE else v


def _start():
    import isaacteleop.deviceio as deviceio
    import isaacteleop.oxr as oxr

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(ANCHOR)
    if not prim.IsValid():
        raise RuntimeError(f"{ANCHOR} not found; run sim/start_vr.py first")
    layer_id = stage.GetSessionLayer().identifier
    core = XRCore.get_singleton()

    # Start from the anchor's current pose (translation + yaw about Y).
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    fwd = m.TransformDir(Gf.Vec3d(0, 0, -1))
    state = {
        "pos": Gf.Vec3d(m.ExtractTranslation()),
        "yaw": math.degrees(math.atan2(-fwd[0], -fwd[2])),
        "snap_armed": True,
        "seen": False,
        "moved": False,
    }

    controllers = deviceio.ControllerTracker()
    head = deviceio.HeadTracker()
    trackers = [controllers, head]
    stack = contextlib.ExitStack()
    try:
        oxr_session = stack.enter_context(
            oxr.OpenXRSession("IsaacSimVRNav", deviceio.DeviceIOSession.get_required_extensions(trackers)))
        session = stack.enter_context(deviceio.DeviceIOSession.run(trackers, oxr_session.get_handles()))
    except Exception:
        stack.close()
        raise

    def write_anchor():
        mat = Gf.Matrix4d()
        mat.SetRotateOnly(Gf.Rotation(UP, state["yaw"]))
        mat.SetTranslateOnly(state["pos"])
        core.set_world_transform_matrix(ANCHOR, mat, layer_id)

    def on_update(event):
        try:
            if not session.update():
                return
            dt = min(event.payload.get("dt", 1 / 60), 0.1)
            left = controllers.get_left_controller(session)
            right = controllers.get_right_controller(session)
            left = left.data if left is not None else None
            right = right.data if right is not None else None
            if left is None and right is None:
                return
            if not state["seen"]:
                state["seen"] = True
                print(f"[nav] controllers seen (left={left is not None}, right={right is not None})", flush=True)

            lx = _dz(left.inputs.thumbstick_x) if left is not None else 0.0
            ly = _dz(left.inputs.thumbstick_y) if left is not None else 0.0
            rx = right.inputs.thumbstick_x if right is not None else 0.0
            ry = _dz(right.inputs.thumbstick_y) if right is not None else 0.0

            # Head pose in physical space: direction to walk in and pivot for turning.
            h = head.get_head(session)
            h = h.data if h is not None else None
            head_phys = Gf.Vec3d(0, 0, 0)
            look = Gf.Vec3d(0, 0, -1)
            if h is not None and h.is_valid:
                p, q = h.pose.position, h.pose.orientation
                head_phys = Gf.Vec3d(p.x, 0.0, p.z)
                look = Gf.Rotation(Gf.Quatd(q.w, Gf.Vec3d(q.x, q.y, q.z))).TransformDir(Gf.Vec3d(0, 0, -1))
            flat = Gf.Vec3d(look[0], 0.0, look[2])
            flat = flat.GetNormalized() if flat.GetLength() > 1e-3 else Gf.Vec3d(0, 0, -1)
            right_dir = Gf.Vec3d(-flat[2], 0.0, flat[0])

            changed = False
            rot = Gf.Rotation(UP, state["yaw"])
            if lx or ly:
                step = (flat * ly + right_dir * lx) * (WALK_SPEED * dt)
                state["pos"] += rot.TransformDir(step)
                changed = True
            if ry:
                state["pos"] += UP * (ry * LIFT_SPEED * dt)
                changed = True

            # Snap turn about the head so the view pivots in place.
            if state["snap_armed"] and abs(rx) > SNAP_ON:
                head_world = state["pos"] + rot.TransformDir(head_phys)
                state["yaw"] -= math.copysign(SNAP_DEG, rx)
                state["pos"] = head_world - Gf.Rotation(UP, state["yaw"]).TransformDir(head_phys)
                state["snap_armed"] = False
                changed = True
            elif abs(rx) < SNAP_OFF:
                state["snap_armed"] = True

            if changed:
                if not state["moved"]:
                    state["moved"] = True
                    print("[nav] moving", flush=True)
                write_anchor()
        except Exception:
            print("[nav] update FAILED:\n" + traceback.format_exc(), flush=True)
            _stop_previous()

    write_anchor()
    sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        on_update, name="xr_thumbstick_nav")
    builtins._xr_nav = (sub, stack)
    print(f"[nav] started: anchor {ANCHOR} at ({state['pos'][0]:.2f}, {state['pos'][1]:.2f}, "
          f"{state['pos'][2]:.2f}), yaw {state['yaw']:.0f} deg. Left stick = walk, right stick = turn / up-down.",
          flush=True)


_stop_previous()
if not globals().get("STOP_NAV", False):
    try:
        _start()
    except Exception:
        print("[nav] FAILED:\n" + traceback.format_exc(), flush=True)
