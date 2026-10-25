"""Start the OpenXR VR session in the open Isaac Sim XR VR window, with Quest 3 controller navigation.

Order matters: CloudXR configures the device only once the headset client is connected.
  1. Keep sim/start_cloudxr.sh (`python -m isaacteleop.cloudxr`) running.
  2. Quest 3 browser: open https://<workstation-ip>:48322 and accept the certificate, then open
     https://nvidia.github.io/IsaacTeleop/client/ , enter <workstation-ip> and connect.
  3. In the Isaac Sim viewport, fly the Perspective camera to where you want to stand, looking the
     way you want to face. Then Window > Script Editor:
         exec(open("<repo>/sim/start_vr.py").read())

Your VR origin is a fixed prim (/World/XR_Anchor, session layer only) on the floor under that camera.
Anchoring to the active camera instead attached the headset to the robot when the viewport looked
through the robot's head camera, and physics then re-pinned it every frame, so the sticks did nothing.

Controls (right hand dominant, Isaac Sim "vr" layout): left stick = fly, right stick forward = teleport
arc (release to jump), right stick left/right = turn, B = menu, A = laser, trigger = select, grip = grab.
Set STOP_VR = True before the exec line to end the session.
"""

import asyncio
import math
import traceback

import carb
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.xr.core import XRCore, XRSettings
from pxr import Gf, Usd, UsdGeom

ANCHOR = "/World/XR_Anchor"
PERSP = "/OmniverseKit_Persp"
FLOOR = "/World/Floor"
# Thumbstick fly speed in m/s; the room is small, so keep it moderate.
NAV_SPEED = 1.5


def _place_anchor(stage):
    """Put the anchor on the floor under the Perspective camera, facing the camera's yaw (Y-up stage)."""
    cam = UsdGeom.Xformable(stage.GetPrimAtPath(PERSP)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = cam.ExtractTranslation()
    fwd = -Gf.Vec3d(cam.GetRow3(2))  # cameras look down their local -Z
    yaw = math.degrees(math.atan2(-fwd[0], -fwd[2]))

    floor_y = 0.0
    floor = stage.GetPrimAtPath(FLOOR)
    if floor.IsValid():
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        floor_y = bbox.ComputeWorldBound(floor).ComputeAlignedRange().GetMax()[1]

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        xf = UsdGeom.Xform.Define(stage, ANCHOR)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(pos[0], floor_y, pos[2]))
        xf.AddRotateYOp().Set(yaw)
    print(f"[vr] anchor {ANCHOR} at ({pos[0]:.2f}, {floor_y:.2f}, {pos[2]:.2f}), yaw {yaw:.0f} deg", flush=True)


async def _start_vr():
    app = omni.kit.app.get_app()
    core = XRCore.get_singleton()
    xs = XRSettings.get_singleton()

    # Restart a running session so the new anchor and settings are picked up cleanly.
    if core.is_xr_viewport_enabled():
        core.request_disable_profile()
        for _ in range(30):
            await app.next_update_async()

    # Leave the robot's head camera, otherwise the XR view follows it.
    viewport = get_active_viewport()
    if viewport is not None and viewport.camera_path.pathString != PERSP:
        print(f"[vr] viewport camera {viewport.camera_path} -> {PERSP}", flush=True)
        viewport.camera_path = PERSP
        await app.next_update_async()

    _place_anchor(omni.usd.get_context().get_stage())

    xs.set_setting("profile/persistent/system/display", "OpenXR")
    xs.set_setting("profile/persistent/anchorMode", "custom anchor")
    xs.set_setting("profile/stage/customAnchor", ANCHOR)
    xs.set_setting("profile/persistent/controllers/visible", True)
    xs.set_setting("profile/persistent/enableNavigationControls", True)
    xs.set_setting("profile/persistent/floorAlignedNavigation", True)
    xs.set_setting("profile/persistent/navigation/speed", NAV_SPEED)
    xs.set_setting("profile/persistent/teleport/arc/maxHeight", 5.0)
    settings = carb.settings.get_settings()
    settings.set("/persistent/xr/profile/vr/tools/layout", "vr")
    # OpenXR runtime = the one named by XR_RUNTIME_JSON (CloudXR) in the environment Isaac Sim was launched from.
    settings.set("/persistent/xr/system/openxr/runtime", "system")

    XRCore.request_enable_profile("vr")
    print("[vr] VR started (OpenXR via CloudXR). Left stick = fly, right stick forward = teleport.", flush=True)


async def _start_vr_safe():
    try:
        await _start_vr()
    except Exception:
        print("[vr] FAILED:\n" + traceback.format_exc(), flush=True)


if globals().get("STOP_VR", False):
    XRCore.get_singleton().request_disable_profile()
    print("[vr] stop requested", flush=True)
else:
    asyncio.ensure_future(_start_vr_safe())
