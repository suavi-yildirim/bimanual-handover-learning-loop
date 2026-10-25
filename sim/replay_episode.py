"""Open-loop replay of a recorded FR3 Duo handover episode in Isaac Sim.

Loads the hackathon USD scene, puts both arms in the episode's first recorded
state, then plays the recorded `action` column as joint position targets at
30 Hz (2 physics ticks of 60 Hz per frame). Logs how far the simulated joints
drift from the recorded `observation.state` and where the box ends up compared
with the recorded final box pose.

Usage (from any directory):
    $ISAAC_PY sim/replay_episode.py --episode 0 --headless
    $ISAAC_PY sim/replay_episode.py --episode 0          # watch in GUI
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(
    HERE, "..", "C2_Teach-the-robot_VLA", "Environment", "Environment_Hackathon_2026"
)
# The scene the handover data was recorded on: its sha256 (587db360...) is the base layer in every
# episode's stack_lock (meta/fr3_episodes_extra.jsonl). Extracted complete from the Environment zip; the
# unzipped folder lacks its robot asset. The local Hackathon_FR3Duo_flattened.usd was re-saved after a sim
# session on 22 Oct (box 5.7 mm lower, physics values baked in) and is not used.
DEFAULT_USD = os.environ.get(
    "FR3_SCENE_USD",
    os.path.expanduser("~/c2_data/scene_v2/Environment_Hackathon_2026/Collected_Hackathon_FR3Duo_v2/Hackathon_FR3Duo.usd"),
)
DEFAULT_DATASET = os.path.join(ENV_DIR, "FR3Duo_Handover_Dataset_lerobot3")

ROBOT = "/World/FR3Duo_withV_Mkit__1_"
BOX = "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"
RECORDED_BOX_HEIGHT_M = 0.1028  # full asset height; see set_box_height()
PHYSICS_DT = 1.0 / 60.0
TICKS_PER_FRAME = 2
# Dataset gripper contract (meta/fr3_conversion.json): norm = 1 - joint_rad / 0.8, 1 = open.
GRIPPER_CLOSED_RAD = 0.8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--usd", default=DEFAULT_USD)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    p.add_argument("--ignore-hold-cut", action="store_true",
                   help="always step 2 ticks per frame, even where the converter cut out raw rows")
    p.add_argument("--out", default=os.path.join(HERE, "out"))
    p.add_argument("--keep-open", action="store_true", help="GUI only: keep the window open after the replay")
    p.add_argument("--render-every", type=int, default=None,
                   help="render every N frames (default: never when headless, every frame in the GUI)")
    p.add_argument("--box-lift-mm", type=float, default=0.0,
                   help="workaround for the missing task layers: raise the box this much at start and hold it "
                        "(gravity off) until the left gripper has closed")
    p.add_argument("--box-offset", type=float, nargs=4, default=None, metavar=("DX", "DY", "DZ", "YAW"),
                   help="move the box from its shipped pose (m, Y up; yaw in deg about +Y), e.g. the image fit "
                        "from sim/fit_box_pose.py")
    p.add_argument("--recorded-physics", action="store_true",
                   help="apply the box mass/friction and gripper drive values listed in the dataset metadata")
    p.add_argument("--grip-max-force", type=float, default=None,
                   help="with --recorded-physics: override the gripper drive max force (metadata value 1.0, units "
                        "unconfirmed)")
    p.add_argument("--box-support-mm", type=float, default=0.0,
                   help="raise the box onto an invisible static support (resting, gravity on), shipped shape and pose")
    p.add_argument("--box-standing", type=int, choices=(1, -1), default=None,
                   help="stand the box on its end (107 mm vertical) as in the recording; 1 or -1 picks which end is up")
    p.add_argument("--box-height-m", type=float, default=None,
                   help=f"rescale the box to this height, resting where it is (recording fit: {RECORDED_BOX_HEIGHT_M})")
    return p.parse_args()


def load_episode(dataset_dir, episode):
    import numpy as np
    import pyarrow.parquet as pq

    data_dir = os.path.join(dataset_dir, "data")
    files = sorted(
        os.path.join(root, f) for root, _, fs in os.walk(data_dir) for f in fs if f.endswith(".parquet")
    )
    cols = ["observation.state", "action", "frame_index", "episode_index", "raw_state_row"]
    rows = None
    for f in files:
        t = pq.read_table(f, columns=cols, filters=[("episode_index", "=", episode)])
        if t.num_rows:
            rows = t.sort_by("frame_index").to_pydict()
            break
    if rows is None:
        raise SystemExit(f"episode {episode} not found in {data_dir}")

    extra = None
    extra_path = os.path.join(dataset_dir, "meta", "fr3_episodes_extra.jsonl")
    if os.path.exists(extra_path):
        with open(extra_path) as fh:
            for i, line in enumerate(fh):
                rec = json.loads(line)
                if rec.get("episode_index", i) == episode:
                    extra = rec
                    break

    return (
        np.asarray(rows["observation.state"], dtype=np.float64),
        np.asarray(rows["action"], dtype=np.float64),
        np.asarray(rows["raw_state_row"], dtype=np.int64).reshape(-1),
        extra,
    )


def ticks_per_frame(raw_rows, ignore_cut):
    """Physics ticks to hold each action; >2 where the converter's hold cut removed raw rows."""
    import numpy as np

    n = len(raw_rows)
    if ignore_cut:
        return np.full(n, TICKS_PER_FRAME, dtype=np.int64)
    ticks = np.diff(raw_rows, append=raw_rows[-1] + TICKS_PER_FRAME)
    return np.clip(ticks, TICKS_PER_FRAME, None)


def to_joint_targets(vec16):
    """Dataset 16-vector -> (left 8, right 8) joint values in rad; gripper norm -> control joint rad."""
    import numpy as np

    out = []
    for half in (vec16[:8], vec16[8:]):
        q = np.array(half, dtype=np.float64)
        q[7] = (1.0 - q[7]) * GRIPPER_CLOSED_RAD
        out.append(q)
    return out


def quat_rotate(q_wxyz, v):
    import numpy as np

    w, x, y, z = q_wxyz
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def apply_recorded_physics(stage):
    """Values from the environment doc and fr3_episodes_extra.jsonl (`gripper_drive`).

    The recording applied these through override layers that were not shipped. The gripper drive
    numbers are used as-is in USD units; whether the recorder logged them in the same units is
    unconfirmed.
    """
    from pxr import UsdPhysics, UsdShade

    box = stage.GetPrimAtPath(BOX)
    UsdPhysics.MassAPI.Apply(box).CreateMassAttr(0.10)
    body = stage.GetPrimAtPath(f"{BOX}/SM_WhiteCorrugatedBox_A34_Body_01")
    mat_path = UsdShade.MaterialBindingAPI(body).GetDirectBinding("physics").GetMaterialPath()
    if not mat_path:
        # The recording scene (Collected_Hackathon_FR3Duo_v2) binds no physics material to the box.
        mat_path = "/World/BoxPhysicsMaterial"
        UsdShade.Material.Define(stage, mat_path)
        UsdShade.MaterialBindingAPI.Apply(body).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)), UsdShade.Tokens.weakerThanDescendants, "physics")
    mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    mat.CreateStaticFrictionAttr(0.9)
    mat.CreateDynamicFrictionAttr(0.8)
    for side in ("left", "right"):
        drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{ROBOT}/{side}_robotiq_85_base_link/{side}_gripper_ctrl"), "angular")
        drive.GetStiffnessAttr().Set(0.13333333)
        drive.GetDampingAttr().Set(0.00666667)
        drive.GetMaxForceAttr().Set(1.0)
    print(f"[replay] recorded physics applied (box mass 0.10 kg, friction 0.9/0.8 on {mat_path}, "
          "gripper drive k=0.133 d=0.0067 maxF=1.0)", flush=True)


def set_box_height(stage, height_m):
    """Rescale the box to `height_m` tall, bottom kept where it rests. Call before play (a scale edit on a
    live physics body invalidates the articulation views).

    The asset is 16 x 16 x 10.28 cm; the shipped prim squashes it to 3.1 x 10.7 x 7.2 cm (scale 0.2, 0.7, 0.7,
    prim scale y = vertical, origin at the box top). The recording fits the full 10.28 cm height: its final
    pose (top y 1.1229) then rests on the serving tray, and the start top (0.932) is where the grasp closes,
    which is why the 42-48 mm lift worked.
    """
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(BOX)
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    scale, translate = ops["xformOp:scale"], ops["xformOp:translate"]
    sx, sy, sz = scale.Get()
    old_h = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange().GetSize()[1]
    new_sy = sy * height_m / old_h
    scale.Set(Gf.Vec3f(sx, new_sy, sz) if isinstance(scale.Get(), Gf.Vec3f) else Gf.Vec3d(sx, new_sy, sz))
    t = translate.Get()
    translate.Set(type(t)(t[0], t[1] + (height_m - old_h), t[2]))
    print(f"[replay] box height {1000 * old_h:.1f} -> {1000 * height_m:.1f} mm (scale y {sy:.3f} -> {new_sy:.3f}), "
          "bottom unchanged", flush=True)


def stand_box_on_end(stage, sign=1):
    """Stand the box on its end (107 mm side vertical) where it lies, bottom on the same support.

    The environment doc (isaac_sim_hackathon_environment.html): box "approximately 31 x 72 x 107 mm", and the
    recorded goal pose "places the box standing on its end"; the recorded final orientation gives a 107 mm
    vertical extent. The shipped scene lays it on its side (72 mm vertical). Rotates 90 deg about world X
    (sign picks which end is up), keeps the footprint centre and the bottom height. Call before play.
    """
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(BOX)
    cache = UsdGeom.BBoxCache(0, ["default", "render"])
    before = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    orient, translate = ops["xformOp:orient"], ops["xformOp:translate"]
    q = orient.Get()
    rx = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0 * sign).GetQuat()
    new = rx * Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary()))
    orient.Set(type(q)(new.GetReal(), *new.GetImaginary()))
    cache.Clear()
    after = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    shift = Gf.Vec3d((before.GetMin()[0] + before.GetMax()[0] - after.GetMin()[0] - after.GetMax()[0]) / 2,
                     before.GetMin()[1] - after.GetMin()[1],
                     (before.GetMin()[2] + before.GetMax()[2] - after.GetMin()[2] - after.GetMax()[2]) / 2)
    t = translate.Get()
    translate.Set(type(t)(t[0] + shift[0], t[1] + shift[1], t[2] + shift[2]))
    cache.Clear()
    size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
    print(f"[replay] box stood on its end (sign {sign:+d}): size x/y/z {1000 * size[0]:.0f}/{1000 * size[1]:.0f}/"
          f"{1000 * size[2]:.0f} mm, bottom unchanged", flush=True)


def set_box_vertical_size(stage, height_m):
    """Scale the box along world Y to `height_m`, bottom kept on its support. Works in any orientation: the prim
    scale component whose axis is vertical is changed. Call before play."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(BOX)
    cache = UsdGeom.BBoxCache(0, ["default", "render"])
    before = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    old_h = before.GetSize()[1]
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    scale, translate = ops["xformOp:scale"], ops["xformOp:translate"]
    sc = list(scale.Get())
    # Try each prim scale axis; keep the one that changes the world height.
    for i in range(3):
        trial = list(sc)
        trial[i] = sc[i] * 1.01
        scale.Set(type(scale.Get())(*trial))
        cache.Clear()
        if abs(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()[1] - old_h) > 1e-4:
            sc[i] *= height_m / old_h
            break
    scale.Set(type(scale.Get())(*sc))
    cache.Clear()
    after = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    t = translate.Get()
    translate.Set(type(t)(t[0] + (before.GetMin()[0] + before.GetMax()[0] - after.GetMin()[0] - after.GetMax()[0]) / 2,
                          t[1] + before.GetMin()[1] - after.GetMin()[1],
                          t[2] + (before.GetMin()[2] + before.GetMax()[2] - after.GetMin()[2] - after.GetMax()[2]) / 2))
    cache.Clear()
    size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
    print(f"[replay] box height {1000 * old_h:.0f} -> {1000 * size[1]:.0f} mm (size x/y/z {1000 * size[0]:.0f}/"
          f"{1000 * size[1]:.0f}/{1000 * size[2]:.0f} mm), bottom unchanged", flush=True)


def add_box_support(stage, lift_m):
    """Raise the box by `lift_m` onto an invisible static support, so it rests (gravity on) at the height where the
    recorded grasp closes. The box keeps its shipped shape and orientation (31 x 72 x 107 mm, lying on its side, as
    in the shipped scene and the recorded images). The support is collision-only: it is not rendered and casts no
    shadow. Call before play."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(BOX)
    cache = UsdGeom.BBoxCache(0, ["default", "render"])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    pad = 0.01
    cube = UsdGeom.Cube.Define(stage, "/World/BoxSupport")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d((lo[0] + hi[0]) / 2, lo[1] + lift_m / 2, (lo[2] + hi[2]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3d(hi[0] - lo[0] + 2 * pad, lift_m, hi[2] - lo[2] + 2 * pad))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)  # not rendered by the cameras
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    t = ops["xformOp:translate"].Get()
    ops["xformOp:translate"].Set(type(t)(t[0], t[1] + lift_m, t[2]))
    print(f"[replay] box raised {1000 * lift_m:.0f} mm onto an invisible support (shape and orientation as shipped)",
          flush=True)


def offset_box(box, offset):
    """Shift a RigidPrim from its current pose by (dx, dy, dz) m and rotate it by yaw deg about world +Y."""
    import numpy as np

    dx, dy, dz, yaw = offset
    pos, quat = (a.numpy().reshape(-1).astype(np.float64) for a in box.get_world_poses())
    h = math.radians(yaw) / 2
    w1, x1, y1, z1 = math.cos(h), 0.0, math.sin(h), 0.0
    w2, x2, y2, z2 = quat
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    box.set_world_poses(positions=(pos + np.array([dx, dy, dz])).reshape(1, -1), orientations=q.reshape(1, -1))


def quat_angle_deg(q1, q2):
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def main():
    args = parse_args()
    import numpy as np

    states, actions, raw_rows, extra = load_episode(args.dataset, args.episode)
    n_frames = len(actions) if args.max_frames is None else min(args.max_frames, len(actions))
    ticks = ticks_per_frame(raw_rows, args.ignore_hold_cut)
    print(f"[replay] episode {args.episode}: {len(actions)} frames, replaying {n_frames}; "
          f"{int((ticks[:n_frames] > TICKS_PER_FRAME).sum())} frame(s) span a hold cut", flush=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    ok, _ = stage_utils.open_stage(args.usd)
    if not ok:
        raise SystemExit(f"could not open {args.usd}")
    while stage_utils.is_stage_loading():
        simulation_app.update()

    from pxr import UsdGeom

    stage = stage_utils.get_current_stage()
    if args.recorded_physics:
        apply_recorded_physics(stage)
        if args.grip_max_force:
            from pxr import UsdPhysics
            for side in ("left", "right"):
                UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{ROBOT}/{side}_robotiq_85_base_link/{side}_gripper_ctrl"),
                                        "angular").GetMaxForceAttr().Set(args.grip_max_force)
            print(f"[replay] gripper drive max force {args.grip_max_force}", flush=True)
    if args.box_height_m:
        set_box_height(stage, args.box_height_m)
    if args.box_standing:
        stand_box_on_end(stage, args.box_standing)
    if args.box_support_mm:
        add_box_support(stage, args.box_support_mm / 1000.0)
    # World-space centre of the box at load time; its prim origin is not the geometric centre.
    rng = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(BOX)).ComputeAlignedRange()
    box_centre_at_load = np.array((rng.GetMin() + rng.GetMax()) / 2.0, dtype=np.float64)
    render_every = args.render_every if args.render_every is not None else (0 if args.headless else 1)

    SimulationManager.setup_simulation(dt=PHYSICS_DT)
    # One app update == one physics tick, so the loop below controls timing exactly.
    RenderingManager.set_dt(PHYSICS_DT)
    app_utils.play()
    simulation_app.update()

    arms = {}
    for side in ("left", "right"):
        art = Articulation(f"{ROBOT}/{side}_arm_root_joint")
        names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_ctrl"]
        arms[side] = (art, art.get_dof_indices(names).numpy())
    box = RigidPrim(BOX)
    _p0, _q0 = (a.numpy().reshape(-1).astype(np.float64) for a in box.get_world_poses())
    # Centre offset expressed in the box's rotation frame (conjugate quaternion rotates world -> box).
    box_local_centre = quat_rotate(_q0 * np.array([1, -1, -1, -1]), box_centre_at_load - _p0)

    def box_centre():
        pos, quat = box.get_world_poses()
        return pos.numpy().reshape(-1) + quat_rotate(quat.numpy().reshape(-1), box_local_centre)

    tcps = {side: RigidPrim(f"{ROBOT}/{side}_tcp") for side in ("left", "right")}
    # Frames where a gripper command changes (grasp / release events).
    # Release is a ramp of small steps; only its first step is logged.
    grip_events = {}
    for k in range(1, n_frames):
        for side, c in (("left", 7), ("right", 15)):
            if actions[k, c] != actions[k - 1, c] and (k < 2 or actions[k - 1, c] == actions[k - 2, c]):
                grip_events.setdefault(k, []).append(side)
    left_close = next((k for k in range(1, len(actions)) if actions[k, 7] == 0.0 and actions[k - 1, 7] > 0.0), None)
    box_held_until = (left_close + 15) if (args.box_lift_mm and left_close is not None) else None
    event_log = []

    def set_arms(vec16, teleport):
        for (art, idx), q in zip(arms.values(), to_joint_targets(vec16)):
            if teleport:
                art.set_dof_positions(q.reshape(1, -1), dof_indices=idx)
                art.set_dof_velocities(np.zeros((1, len(idx))), dof_indices=idx)
            art.set_dof_position_targets(q.reshape(1, -1), dof_indices=idx)

    def read_arms():
        out = []
        for art, idx in arms.values():
            q = art.get_dof_positions(dof_indices=idx).numpy().reshape(-1).astype(np.float64)
            q[7] = 1.0 - q[7] / GRIPPER_CLOSED_RAD  # approximate: converter used its own joint->norm map
            out.append(q)
        return np.concatenate(out)

    set_arms(states[0], teleport=True)
    if args.box_offset:
        offset_box(box, args.box_offset)
        print(f"[replay] box offset from shipped pose {args.box_offset}", flush=True)
    if box_held_until is not None:
        pos, quat = box.get_world_poses()
        lifted = pos.numpy().reshape(1, -1) + np.array([[0.0, args.box_lift_mm / 1000.0, 0.0]])
        box.set_world_poses(positions=lifted, orientations=quat.numpy().reshape(1, -1))
        box.set_enabled_gravities(False)
        print(f"[replay] box raised {args.box_lift_mm:.0f} mm, gravity off until frame {box_held_until}", flush=True)
    simulation_app.update()

    steps_before = SimulationManager.get_num_physics_steps()
    errors = np.zeros((n_frames, 16))
    t0 = time.time()
    for k in range(n_frames):
        if not simulation_app.is_running():
            print("[replay] app closed, stopping early", flush=True)
            n_frames = k
            errors = errors[:k]
            break
        errors[k] = read_arms() - states[k]
        if k in grip_events:
            bpos = box_centre()
            for side in grip_events[k]:
                tpos, tquat = (v.numpy().reshape(-1) for v in tcps[side].get_world_poses())
                c = 7 if side == "left" else 15
                event_log.append({
                    "frame": k, "side": side,
                    "gripper_cmd": [round(float(actions[k - 1, c]), 3), round(float(actions[k, c]), 3)],
                    "tcp_pos_m": [round(float(v), 4) for v in tpos],
                    "tcp_quat_wxyz": [round(float(v), 6) for v in tquat],
                    "box_centre_m": [round(float(v), 4) for v in bpos],
                    "tcp_to_box_mm": round(float(1000 * np.linalg.norm(tpos - bpos)), 1),
                })
        set_arms(actions[k], teleport=False)
        if k == box_held_until:
            box.set_enabled_gravities(True)
        if render_every and k % render_every == 0:
            # In this mode one app update == one physics tick.
            for _ in range(int(ticks[k])):
                simulation_app.update()
        else:
            SimulationManager.step(steps=int(ticks[k]))
        if k % 300 == 0:
            arm_err = np.degrees(np.abs(errors[k][[*range(7), *range(8, 15)]])).max()
            print(f"[replay] frame {k}/{n_frames}  max arm joint error {arm_err:.2f} deg", flush=True)

    physics_steps = SimulationManager.get_num_physics_steps() - steps_before
    expected_steps = int(ticks[:n_frames].sum())
    pos, quat = box.get_world_poses()
    box_pos = pos.numpy().reshape(-1).tolist()
    box_quat = quat.numpy().reshape(-1).tolist()

    arm_cols = [*range(7), *range(8, 15)]
    names = [f"{s}_q{i}" for s in ("L", "R") for i in range(1, 8)]
    summary = {
        "episode": args.episode,
        "frames_replayed": int(n_frames),
        "physics_steps": int(physics_steps),
        "expected_physics_steps": expected_steps,
        "wall_time_s": round(time.time() - t0, 1),
        "joint_rmse_deg": dict(zip(names, np.degrees(np.sqrt((errors[:, arm_cols] ** 2).mean(0))).round(3).tolist())),
        "joint_max_abs_deg": dict(zip(names, np.degrees(np.abs(errors[:, arm_cols]).max(0)).round(3).tolist())),
        "gripper_max_abs_norm_err_approx": {
            "left": round(float(np.abs(errors[:, 7]).max()), 3),
            "right": round(float(np.abs(errors[:, 15]).max()), 3),
        },
        "final_box_pos_m": [round(v, 4) for v in box_pos],
        "final_box_centre_m": [round(float(v), 4) for v in box_centre()],
        "options": {"box_lift_mm": args.box_lift_mm, "box_offset": args.box_offset, "recorded_physics": args.recorded_physics},
        "final_box_quat_wxyz": [round(v, 4) for v in box_quat],
        "gripper_events": event_log,
    }
    if extra and extra.get("final_box_pose") and n_frames == len(actions):
        ref = extra["final_box_pose"]
        summary["recorded_final_box_pos_m"] = [round(v, 4) for v in ref[:3]]
        summary["final_box_pos_error_mm"] = round(1000 * math.dist(box_pos, ref[:3]), 1)
        summary["final_box_rot_error_deg"] = round(quat_angle_deg(box_quat, ref[3:7]), 2)
        summary["replay_matches_recording"] = (
            summary["final_box_pos_error_mm"] < 20.0 and summary["final_box_rot_error_deg"] < 5.0
        )
    if physics_steps != expected_steps:
        summary["warning"] = "physics step count differs from expected; app updates are not 1:1 with physics ticks"

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, f"replay_ep{args.episode:03d}.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savetxt(os.path.join(args.out, f"replay_ep{args.episode:03d}_state_error.csv"), errors,
               delimiter=",", header=",".join(f"err_{i}" for i in range(16)), comments="")
    print("[replay] summary:\n" + json.dumps(summary, indent=2), flush=True)
    print(f"[replay] wrote {out_json}", flush=True)

    if args.keep_open and not args.headless:
        print("[replay] done; close the window to exit", flush=True)
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
