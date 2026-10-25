"""Closed-loop rollout of a served pi0.5 policy on the FR3 Duo handover in Isaac Sim.

Same scene setup and timing as replay_episode.py (recorded physics, optional box lift, 2 physics
ticks per 30 Hz action), but the actions come from the openpi policy server
(pi05/serve_checkpoint.py serve): each query sends the three 224x224 camera images, the 16-D state
and the prompt, gets an action chunk back and executes its first `--replan-every` actions.

Usage (policy server must already be running on --port):
    PYTHONPATH=$ISAAC_PYDEPS $ISAAC_PY sim/policy_rollout.py --headless
    PYTHONPATH=$ISAAC_PYDEPS $ISAAC_PY sim/policy_rollout.py   # watch in GUI

Live view (also headless): http://localhost:8090/ shows the three policy cameras with a status bar.
Speed: one render per 30 Hz frame at most (1 physics-only tick + 1 tick with render), and none on frames
where neither the policy, the video nor a live viewer needs an image. Video encoding runs in a thread.
"""

import argparse
import json
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align_assist as aa  # noqa: E402
import recovery as rc  # noqa: E402
import replay_episode as rp  # noqa: E402

PROMPT = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
# Dataset camera -> prim (meta/fr3_conversion.json, recorder.camera_params); rendered natively at 224 x 224.
CAMERAS = {
    "external_image": (f"{rp.ROBOT}/mount_link/cover_link/head_link/zed_mini_camera_link/"
                       "zed_mini_left_camera_frame/head_cam", 2.4000000953674316),
    "left_wrist_image": (f"{rp.ROBOT}/left_d405_color_frame/left_wrist_cam", 2.5290722846984863),
    "right_wrist_image": (f"{rp.ROBOT}/right_d405_color_frame/right_wrist_cam", 2.5290722846984863),
}
APERTURE_MM = 4.800000190734863
IMAGE_SIZE = 224
# Gripper state calibration sim -> dataset, from a full replay of episode 0 with recorded physics
# (sim/out/lift48_recphys): steady-state medians per command level (closed on box, pre-shape, open),
# identical for both arms. The recorder's joint->norm map differs from 1 - q / 0.8.
GRIP_SIM = (0.337, 0.472, 0.994)
GRIP_DATA = (0.424, 0.553, 0.995)


def grip_to_dataset(x):
    """Piecewise-linear map with linear extrapolation beyond the end points."""
    import numpy as np

    xs, ys = np.asarray(GRIP_SIM), np.asarray(GRIP_DATA)
    if x < xs[0]:
        return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if x > xs[-1]:
        return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(x, xs, ys))


# Recorded episode-0 phases by frame (gripper events: L close 327, R close 1083, L open 1135, R open 1901).
PHASES = ((327, "L approach"), (834, "L grasp+lift"), (1083, "R approach"), (1135, "handover"),
          (1901, "R place"), (10**9, "retreat"))


PRE_SHAPE_STATE = 0.551  # recorded gripper state while pre-shaped (dataset scale)
HOLD_STATE = 0.424  # recorded gripper state while closed on the box
HOLD_RANGE = (0.30, 0.53)  # gripper state that counts as holding the box (on-box 0.42-0.48, pre-shape 0.55)
GRASP_CONFIRM_FRAMES = 10
POST_GRASP_FRAMES = 45
LATCHED_HOLD_RANGE = (0.30, 0.65)  # commanded closed and stopped here = blocked by the box (empty closes to ~0.10)
GRASP_ALIGN_TOL_M = 0.004  # with --align-assist: close once the TCP is this close to the grasp point
GRASP_ALIGN_MAX_FRAMES = 60  # ... or after this long
CLOSE_GATE_M = 0.025  # with --align-assist: a close command counts only this close to the grasp point
# TCP minus box centre (world, mm) at the moment each gripper closed in the open-loop replay with the box
# lying as shipped on a 30 mm support (sim/out/replay_support_30); the next grasping arm is compared with it live.
STANDING_HEIGHT_M = 0.107  # documented box height standing on its end
TRAY_CENTRE_XZ = (-3.4987, 22.716)  # serving tray (world x, z); right release is allowed within TRAY_RELEASE_M
TRAY_RELEASE_M = 0.12
ALIGN_REF = {"left": (0.2, 28.5, -30.7), "right": (80.1, -33.9, 0.2)}


def grip_label(x):
    return "CLOSE" if x < 0.25 else ("PRE-SHAPE/HOLD" if x < 0.75 else "OPEN")


def describe_decision(chunk, state, rec_states):
    """One-line summaries of what the policy just decided, from its action chunk and the current state.

    Gripper: label of the first action, and the first step in the chunk where the label changes.
    Arm: largest joint motion commanded over the chunk (deg). Phase: nearest recorded frame by arm joints.
    """
    import numpy as np

    grips = []
    for side, g in (("L", 7), ("R", 15)):
        labels = [grip_label(x) for x in chunk[:, g]]
        change = next((i for i, lab in enumerate(labels) if lab != labels[0]), None)
        txt = f"{side} grip {labels[0]} ({chunk[0, g]:.2f}, state {state[g]:.2f})"
        if change is not None:
            txt += f" -> {labels[change]} in {change} steps"
        grips.append(txt)
    move = [np.degrees(np.abs(chunk[-1, s] - state[s]).max()) for s in (slice(0, 7), slice(8, 15))]
    arm = np.r_[0:7, 8:15]
    d = np.abs(rec_states[:, arm] - state[arm]).max(axis=1)
    near = int(d.argmin())
    phase = next(name for end, name in PHASES if near < end)
    return (" | ".join(grips),
            f"arm motion over chunk L {move[0]:.1f} R {move[1]:.1f} deg | nearest recorded frame {near} "
            f"({phase}, off by {np.degrees(d[near]):.1f} deg)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--episode", type=int, default=0, help="recorded episode giving the start state and reference final pose")
    p.add_argument("--usd", default=rp.DEFAULT_USD)
    p.add_argument("--dataset", default=rp.DEFAULT_DATASET)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=2300, help="30 Hz frames; a recorded episode has 2,247")
    p.add_argument("--replan-every", type=int, default=10, help="actions executed from each chunk before re-querying")
    p.add_argument("--raw-grip-state", action="store_true",
                   help="send 1 - q/0.8 as gripper state instead of the calibrated dataset value")
    p.add_argument("--action-delay", type=int, default=0,
                   help="apply each action this many 30 Hz frames late (recorded state lags the command by ~4 frames, "
                        "the sim drives by ~1)")
    p.add_argument("--box-lift-mm", type=float, default=0.0,
                   help="old workaround (before --box-height-m): box raised and held (gravity off) until the left "
                        "gripper has closed; 0 = off")
    p.add_argument("--box-support-mm", type=float, default=0.0,
                   help="raise the box onto an invisible support this high (replay calibration: 30 mm puts its top where "
                        "the recorded grasp closes). Default 0: the box rests on the cart as shipped and "
                        "--align-assist brings the hand down to it (same TCP-box offset as the replay)")
    p.add_argument("--box-standing", type=int, choices=(1, -1, 0), default=0,
                   help="stand the box on its end (107 mm vertical) as in the recording (environment doc: 31 x 72 x "
                        "107 mm, goal pose standing on its end); -1 puts the taped end up as in the recorded wrist "
                        "images (+1 shows the plain end); 0 = shipped pose, lying on its side")
    p.add_argument("--box-height-m", type=float, default=0.0,
                   help="experiment: box height (m) after standing it up, bottom on the cart; 0 = documented 107 mm. "
                        "The grasp targets move with the box top.")
    p.add_argument("--no-recorded-physics", action="store_true")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--out", default=os.path.join(HERE, "out", "rollouts"))
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--video-every", type=int, default=3,
                   help="render and record every Nth 30 Hz frame (encoded at 30/N fps, so it plays in real time); "
                        "1 = every frame, ~2x slower")
    p.add_argument("--no-frames", action="store_true", help="do not also save each video frame as frames/NNNNNN.jpg")
    p.add_argument("--stream-port", type=int, default=8090, help="live MJPEG view on this port; 0 = off")
    p.add_argument("--stream-scale", type=int, default=2, help="upscale factor of the live view")
    p.add_argument("--render-mode", choices=("rt", "pt"), default="rt",
                   help="rt: RTX real-time ray tracing; pt: path tracing (scene default, 64 spp, ~7-10x slower). "
                        "Evaluate with the mode the policy's training images used.")
    p.add_argument("--settle", type=int, default=None,
                   help="render updates with physics paused before each policy query: RTX path tracing lags 2 "
                        "updates and converges over ~8; the recorder also rendered until converged (max 12)")
    p.add_argument("--keep-open", action="store_true", help="GUI only: keep the window open at the end")
    p.add_argument("--open-approach", action=argparse.BooleanOptionalAction, default=False,
                   help="open each gripper fully (instead of the learned pre-shape) until it has grasped, for more "
                        "finger clearance around the box; the policy is shown the pre-shape state it expects")
    p.add_argument("--align-assist", action=argparse.BooleanOptionalAction, default=False,
                   help="centre the grasping hand on the box (sim box pose) in the last 12 cm before each grasp; "
                        "the policy still decides when to close (sim/align_assist.py)")
    p.add_argument("--close-latch", action=argparse.BooleanOptionalAction, default=True,
                   help="once the policy commands a close for an approaching gripper, keep it closed until the grasp "
                        "is confirmed or the supervisor retries (the policy's close flickers across chunks; the "
                        "demonstrations never reopen before the handover)")
    p.add_argument("--release-interlock", action=argparse.BooleanOptionalAction, default=True,
                   help="a gripper holding the box opens only where the demonstrations open it: the left once the right "
                        "holds the box (handover), the right over the serving tray")
    p.add_argument("--post-grasp-smooth", type=float, default=0.5, metavar="DEG_PER_FRAME",
                   help="after a grasp, limit the holding arm's command change to this many degrees per frame for "
                        "POST_GRASP_FRAMES (the policy's chunk-to-chunk jumps lever the box in the weak grip); 0 = off")
    p.add_argument("--seed", type=int, default=None,
                   help="evaluation box pose: offset the box by a seeded random dx, dz (+-BOX_RANGE) and yaw (+-5 deg)")
    p.add_argument("--box-range-mm", type=float, default=15.0, help="with --seed: max |dx|, |dz| (expert data used 15)")
    p.add_argument("--raw", action="store_true",
                   help="evaluation mode: the policy alone. Real gripper readings, no latches, interlock, smoothing, "
                        "assist, open approach or supervisor recovery; failure detection still ends a run early")
    p.add_argument("--max-retries", type=int, default=2, help="supervisor retries per arm; 0 = detect and abort only")
    p.add_argument("--no-supervisor", action="store_true", help="raw policy, no failure detection or retries")
    p.add_argument("--dump-obs", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                   help="save every policy query (images, state, full action chunk) in this frame range to obs_dump.npz")
    return p.parse_args()


def main():
    args = parse_args()
    if args.raw:
        args.close_latch = args.release_interlock = args.open_approach = args.align_assist = False
        args.post_grasp_smooth = 0.0
        args.max_retries = 0
    if args.settle is None:
        args.settle = 2 if args.render_mode == "rt" else 8
    import numpy as np

    states, _actions, _raw_rows, extra = rp.load_episode(args.dataset, args.episode)

    from openpi_client import websocket_client_policy

    print(f"[rollout] connecting to policy server {args.host}:{args.port}", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[rollout] server metadata: {client.get_server_metadata()}", flush=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import UsdGeom

    ok, _ = stage_utils.open_stage(args.usd)
    if not ok:
        raise SystemExit(f"could not open {args.usd}")
    while stage_utils.is_stage_loading():
        simulation_app.update()

    stage = stage_utils.get_current_stage()
    import carb

    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting" if args.render_mode == "rt" else "PathTracing")
    print(f"[rollout] render mode {args.render_mode}, settle {args.settle}", flush=True)
    if not args.no_recorded_physics:
        rp.apply_recorded_physics(stage)
    if args.box_standing:
        rp.stand_box_on_end(stage, args.box_standing)
    if args.box_support_mm:
        rp.add_box_support(stage, args.box_support_mm / 1000.0)
    if args.box_height_m:
        if args.box_standing:
            rp.set_box_vertical_size(stage, args.box_height_m)
            # The fingers grip a fixed distance below the box top: shift the grasp targets with the top.
            dy = (args.box_height_m - STANDING_HEIGHT_M) / 2
            for side in aa.REF:
                aa.REF[side] = aa.REF[side] + np.array([0.0, dy, 0.0])
            for side in ALIGN_REF:
                ALIGN_REF[side] = (ALIGN_REF[side][0], ALIGN_REF[side][1] + 1000 * dy, ALIGN_REF[side][2])
        else:
            rp.set_box_height(stage, args.box_height_m)
    rng = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(rp.BOX)).ComputeAlignedRange()
    box_centre_at_load = np.array((rng.GetMin() + rng.GetMax()) / 2.0, dtype=np.float64)

    if args.headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            vp = get_active_viewport()
            if vp is not None:
                vp.updates_enabled = False
                print("[rollout] headless: default viewport rendering disabled", flush=True)
        except Exception as e:  # speed-up only
            print(f"[rollout] could not disable the viewport: {e!r}", flush=True)
    annotators = {}
    for key, (path, focal) in CAMERAS.items():
        cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        if not cam:
            raise SystemExit(f"camera prim missing: {path}")
        cam.GetFocalLengthAttr().Set(focal)
        cam.GetHorizontalApertureAttr().Set(APERTURE_MM)
        cam.GetVerticalApertureAttr().Set(APERTURE_MM)
        product = rep.create.render_product(path, (IMAGE_SIZE, IMAGE_SIZE))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([product])
        annotators[key] = annot

    SimulationManager.setup_simulation(dt=rp.PHYSICS_DT)
    RenderingManager.set_dt(rp.PHYSICS_DT)  # one app update == one physics tick
    app_utils.play()
    simulation_app.update()

    arms = {}
    for side in ("left", "right"):
        art = Articulation(f"{rp.ROBOT}/{side}_arm_root_joint")
        names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_ctrl"]
        arms[side] = (art, art.get_dof_indices(names).numpy())
    box = RigidPrim(rp.BOX)
    tcps = {side: RigidPrim(f"{rp.ROBOT}/{side}_tcp") for side in ("left", "right")}
    # Row of each TCP in its arm's (fixed-base) Jacobian, and the arm joint columns.
    jac_rows = {side: arms[side][0].link_names.index(f"{side}_tcp") - 1 for side in arms}

    def tcp_jacobians():
        out = {}
        for side, short in (("left", "L"), ("right", "R")):
            art, idx = arms[side]
            jac = art.get_jacobian_matrices().numpy()[0, jac_rows[side]]
            out[short] = jac[0:3, np.asarray(idx[:7])].astype(np.float64)
        return out
    _p0, _q0 = (a.numpy().reshape(-1).astype(np.float64) for a in box.get_world_poses())
    box_local_centre = rp.quat_rotate(_q0 * np.array([1, -1, -1, -1]), box_centre_at_load - _p0)

    def box_pose():
        pos, quat = box.get_world_poses()
        pos, quat = pos.numpy().reshape(-1).astype(np.float64), quat.numpy().reshape(-1).astype(np.float64)
        return pos, quat, pos + rp.quat_rotate(quat, box_local_centre)

    def set_arms(vec16, teleport):
        for (art, idx), q in zip(arms.values(), rp.to_joint_targets(vec16)):
            if teleport:
                art.set_dof_positions(q.reshape(1, -1), dof_indices=idx)
                art.set_dof_velocities(np.zeros((1, len(idx))), dof_indices=idx)
            art.set_dof_position_targets(q.reshape(1, -1), dof_indices=idx)

    def read_state():
        out = []
        for art, idx in arms.values():
            q = art.get_dof_positions(dof_indices=idx).numpy().reshape(-1).astype(np.float64)
            q[7] = 1.0 - q[7] / rp.GRIPPER_CLOSED_RAD
            if not args.raw_grip_state:
                q[7] = grip_to_dataset(q[7])
            out.append(q)
        return np.concatenate(out).astype(np.float32)

    def read_images(strict=True):
        imgs = {}
        for key, annot in annotators.items():
            data = np.asarray(annot.get_data())
            if data.ndim != 3 or data.shape[0] != IMAGE_SIZE:
                if not strict:
                    return None
                raise RuntimeError(f"camera {key}: no image yet (got shape {data.shape})")
            imgs[key] = np.ascontiguousarray(data[:, :, :3], dtype=np.uint8)
        return imgs

    set_arms(states[0], teleport=True)
    box_offset = [0.0, 0.0, 0.0, 0.0]
    if args.seed is not None:
        r = np.random.default_rng(args.seed)
        lim = args.box_range_mm / 1000.0
        box_offset = [float(r.uniform(-lim, lim)), 0.0, float(r.uniform(-lim, lim)), float(r.uniform(-5.0, 5.0))]
        rp.offset_box(box, box_offset)
        for _ in range(30):  # let it settle on the cart
            SimulationManager.step(steps=rp.TICKS_PER_FRAME)
        print(f"[rollout] seed {args.seed}: box offset dx {1000 * box_offset[0]:+.1f} mm, dz {1000 * box_offset[2]:+.1f} mm, "
              f"yaw {box_offset[3]:+.1f} deg", flush=True)
    box_held = args.box_lift_mm > 0
    if box_held:
        pos, quat = box.get_world_poses()
        lifted = pos.numpy().reshape(1, -1) + np.array([[0.0, args.box_lift_mm / 1000.0, 0.0]])
        box.set_world_poses(positions=lifted, orientations=quat.numpy().reshape(1, -1))
        box.set_enabled_gravities(False)
        print(f"[rollout] box raised {args.box_lift_mm:.0f} mm, gravity off until the left gripper closes", flush=True)
    def settled_images():
        """Render with physics paused until the path tracer has converged, then read the cameras."""
        app_utils.pause()
        for _ in range(args.settle):
            simulation_app.update()
        imgs = read_images()
        app_utils.play()
        return imgs

    # Physics paused while the render products start up (shader compilation can take many updates).
    app_utils.pause()
    for warm in range(1, 1001):
        simulation_app.update()
        if read_images(strict=False) is not None:
            break
    app_utils.play()
    print(f"[rollout] cameras ready after {warm} warm-up updates", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"rollout_{stamp}_ep{args.episode:03d}")
    os.makedirs(out_dir, exist_ok=True)
    import frame_stream

    recorder = None if args.no_video else frame_stream.VideoRecorder(
        os.path.join(out_dir, "video.mp4"), fps=30 / args.video_every,
        frames_dir=None if args.no_frames else os.path.join(out_dir, "frames"))
    live = frame_stream.LiveStream(args.stream_port) if args.stream_port else None

    # Recorded grasp frame per arm (first fully-closed command), used for the retry approach pose.
    close_frames = {side: int(np.argmax(_actions[:, g] < 0.1)) for side, g in rc.GRIP.items()}
    supervisor = None if args.no_supervisor else rc.Supervisor(states, close_frames, args.max_retries)
    abort_reason = None
    stop = {"signal": None}

    def request_stop(signum, _frame):
        stop["signal"] = signum  # finish the frame, then write video / trace / result

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    grasped = {"L": False, "R": False}  # per arm: holding since its last (re)approach
    closed_for = {"L": 0, "R": 0}  # frames the gripper has been commanded closed
    latched = {"L": False, "R": False}  # close committed, waiting for the grasp to be confirmed
    # Grasp step per arm, from the committed close until the grasp is confirmed: the arm holds the pose it had
    # when the policy decided to close (the assist finishes centring it) and the fingers close once aligned.
    grasp_exec = {"L": None, "R": None}
    policy_grip_cmd = {"L": 1.0, "R": 1.0}
    interlock_note = {"L": False, "R": False}
    grasped_at = {"L": None, "R": None}
    last_arm_cmd = {"L": None, "R": None}
    # Pre-shape latch: the demonstrations step a gripper open -> pre-shape -> closed and never back during its
    # approach; the policy's pre-shape flickers across chunks (seen at the pause before the right approach).
    preshaped = {"L": False, "R": False}  # latest gripper command from the policy itself (before any override)
    released = {"L": False, "R": False}  # per arm: opened again after holding
    assist = aa.AlignAssist() if args.align_assist else None

    def policy_state():
        """State sent to the policy, with each gripper shown at the level the demonstrations had in its phase.

        The policy uses the gripper value as a strong phase cue (offline: a held box read as 0.50 instead of 0.424
        halves the carry motion and starts opening the gripper). With --open-approach an approaching gripper is
        physically wide open but shown the pre-shape value, only while the policy itself commands pre-shape (the
        demonstrations keep a gripper fully open until its approach); a gripper holding the box is shown the
        recorded holding value, whatever width the box orientation leaves it at."""
        st = read_state()
        if args.raw:
            return st
        for side, g in rc.GRIP.items():
            if (args.open_approach and not grasped[side] and st[g] > PRE_SHAPE_STATE
                    and (preshaped[side] or rc.CLOSE_CMD <= policy_grip_cmd[side] < 0.75)):
                st[g] = PRE_SHAPE_STATE
            elif grasped[side] and not released[side] and LATCHED_HOLD_RANGE[0] < st[g] < LATCHED_HOLD_RANGE[1]:
                st[g] = HOLD_STATE
            elif latched[side] and not grasped[side] and st[g] < LATCHED_HOLD_RANGE[1]:
                st[g] = HOLD_STATE  # committed close, fingers closing: never show an in-between value
        return st

    def query_due(chunk, chunk_pos):
        if supervisor is not None and supervisor.recovering:
            return False
        return chunk is None or chunk_pos >= min(args.replan_every, len(chunk))

    def needs_images(k, chunk, chunk_pos):
        return (query_due(chunk, chunk_pos)
                or ((recorder is not None or (live is not None and live.watching)) and k % args.video_every == 0))

    chunk, chunk_pos = None, 0
    pending = []  # actions waiting out --action-delay
    latencies, gravity_on_frame, left_closed_frame = [], None, None
    log = {"gripper_cmd": [], "box_centre": [], "box_quat": [], "tcp_left": [], "tcp_right": [], "state": [],
           "action": [], "query_left_grip_min": []}
    dump = {key: [] for key in ("frame", "state", "chunk", *CAMERAS)}
    hud = "starting"
    centre = box_pose()[2]
    decision = ("", "")
    renders = 0
    steps_before = SimulationManager.get_num_physics_steps()
    t0 = time.time()
    n_done = 0
    for k in range(args.max_frames):
        if not simulation_app.is_running():
            print("[rollout] app closed, stopping early", flush=True)
            break
        if stop["signal"] is not None:
            print(f"[rollout] signal {stop['signal']}, stopping and writing outputs", flush=True)
            break
        # Images are only fresh if the previous frame rendered; needs_images() decided that one frame ahead.
        if query_due(chunk, chunk_pos) and args.settle:
            images = settled_images()
        else:
            images = read_images() if needs_images(k, chunk, chunk_pos) else None
        if images is not None:
            if recorder is not None and k % args.video_every == 0:
                recorder.push(frame_stream.compose(images, CAMERAS, hud))
            if live is not None and live.watching:
                live.push(frame_stream.compose(images, CAMERAS, hud, scale=args.stream_scale))
        if query_due(chunk, chunk_pos):
            obs = {"images": images, "state": policy_state(), "prompt": args.prompt}
            tq = time.perf_counter()
            chunk = np.asarray(client.infer(obs)["actions"], dtype=np.float64)
            latencies.append(time.perf_counter() - tq)
            chunk_pos = 0
            log["query_left_grip_min"].append([k, round(float(chunk[:, 7].min()), 3)])
            decision = describe_decision(chunk, obs["state"], states)
            prog = rc.progress_frame(obs["state"], states)
            side = "left" if prog < close_frames["L"] else ("right" if prog < close_frames["R"] + 52 else None)
            if side:
                off = 1000 * (tcps[side].get_world_poses()[0].numpy().reshape(-1) - centre)
                err = off - np.array(ALIGN_REF[side])
                decision = (decision[0], decision[1] + f" | {side[0].upper()} tcp-box {np.round(off).astype(int).tolist()} mm, "
                                                       f"vs replay {np.round(err).astype(int).tolist()}")
            print(f"[policy] frame {k}: {decision[0]} | {decision[1]}", flush=True)
            if args.dump_obs and args.dump_obs[0] <= k <= args.dump_obs[1]:
                dump["frame"].append(k)
                dump["state"].append(obs["state"])
                dump["chunk"].append(chunk.astype(np.float32))
                for key, img in images.items():
                    dump[key].append(img)
        action = None
        if supervisor is None or not supervisor.recovering:
            action = np.array(chunk[chunk_pos], dtype=np.float64)
            chunk_pos += 1
            for side, g in rc.GRIP.items():
                policy_grip_cmd[side] = float(action[g])
                if args.close_latch and not grasped[side]:
                    if rc.CLOSE_CMD <= action[g] < 0.75 and not preshaped[side]:
                        preshaped[side] = True
                        print(f"[latch] frame {k}: {side} pre-shape committed (approach started)", flush=True)
                    if preshaped[side] and action[g] >= 0.75:
                        action[g] = rc.PRE_SHAPE
            # Close latch before the supervisor, so its grasp checks see the command actually sent.
            for side, g in rc.GRIP.items():
                if args.close_latch and not grasped[side]:
                    far = (assist is not None and (assist.last_err[side] is None
                                                   or assist.last_err[side] > CLOSE_GATE_M))
                    if action[g] < rc.CLOSE_CMD and not latched[side] and far:
                        action[g] = rc.PRE_SHAPE  # too far from the box: a close here can only miss
                    if action[g] < rc.CLOSE_CMD and not latched[side]:
                        latched[side] = True
                        grasp_exec[side] = {"arm": action[rc.ARM[side]].copy(), "k0": k}
                        print(f"[latch] frame {k}: {side} close committed, arm held for the grasp", flush=True)
                    if latched[side]:
                        ex = grasp_exec[side]
                        action[rc.ARM[side]] = ex["arm"]
                        err = assist.last_err[side] if assist is not None else None
                        aligned = err is None or err < GRASP_ALIGN_TOL_M or k - ex["k0"] >= GRASP_ALIGN_MAX_FRAMES
                        if aligned and "close_at" not in ex:
                            ex["close_at"] = k
                            print(f"[latch] frame {k}: {side} closing (TCP error "
                                  f"{'n/a' if err is None else f'{1000 * err:.1f} mm'})", flush=True)
                        action[g] = 0.0 if aligned else (1.0 if args.open_approach else rc.PRE_SHAPE)
        if supervisor is not None:
            action, abort_reason, replan = supervisor.step(k, read_state(), action, centre, box_pose()[1])
            if replan:
                chunk = None
            if abort_reason:
                print(f"[supervisor] frame {k}: ABORT: {abort_reason}", flush=True)
                break
            if supervisor.pause_kick == k and not grasped["R"] and not args.raw:
                preshaped["R"] = True
                print(f"[latch] frame {k}: R pre-shape committed by the supervisor (end of the demonstrated pause)",
                      flush=True)
            if supervisor.last_retry and supervisor.last_retry[0] == k:
                side = supervisor.last_retry[1]
                grasped[side] = released[side] = latched[side] = False  # this arm approaches again
                grasp_exec[side] = None
                if assist is not None:
                    assist.state[side] = "idle"
        action = np.array(action, dtype=np.float64)  # never edit the policy's chunk in place
        real = read_state()
        if args.release_interlock:
            over_tray = math.hypot(centre[0] - TRAY_CENTRE_XZ[0], centre[2] - TRAY_CENTRE_XZ[1]) < TRAY_RELEASE_M
            for side, g in rc.GRIP.items():
                if grasped[side] and not released[side] and action[g] >= rc.CLOSE_CMD:
                    allowed = grasped["R"] if side == "L" else over_tray
                    if not allowed:
                        if not interlock_note[side]:
                            print(f"[interlock] frame {k}: {side} open command held (box not handed over / not over the tray)",
                                  flush=True)
                            interlock_note[side] = True
                        action[g] = 0.0
        for side, g in rc.GRIP.items():
            closed_for[side] = closed_for[side] + 1 if action[g] < rc.CLOSE_CMD else 0
            hold_range = LATCHED_HOLD_RANGE if latched[side] else HOLD_RANGE
            if (not grasped[side] and closed_for[side] >= GRASP_CONFIRM_FRAMES
                    and hold_range[0] < real[g] < hold_range[1]):
                grasped[side] = True
                grasped_at[side] = k
                grasp_exec[side] = None  # the policy drives this arm again
            elif grasped[side] and not released[side] and action[g] >= rc.CLOSE_CMD:
                released[side] = True
            if args.open_approach and not grasped[side] and action[g] >= rc.CLOSE_CMD:
                action[g] = 1.0  # fully open while approaching; a close command passes through
        if args.post_grasp_smooth:
            step = np.radians(args.post_grasp_smooth)
            for side in ("L", "R"):
                arm = rc.ARM[side]
                if (grasped_at[side] is not None and k - grasped_at[side] < POST_GRASP_FRAMES
                        and last_arm_cmd[side] is not None):
                    action[arm] = last_arm_cmd[side] + np.clip(action[arm] - last_arm_cmd[side], -step, step)
                last_arm_cmd[side] = action[arm].copy()
        if assist is not None:
            tcp_now = {"L": tcps["left"].get_world_poses()[0].numpy().reshape(-1).astype(np.float64),
                       "R": tcps["right"].get_world_poses()[0].numpy().reshape(-1).astype(np.float64)}
            pend = assist.pending_side(grasped)
            ex = grasp_exec.get(pend) if pend is not None else None
            if ex is not None and "close_at" in ex:
                off = assist.offset  # fingers closing: hold still
            else:
                off = assist.update(k, grasped, released, tcp_now, np.asarray(centre, dtype=np.float64),
                                    tcp_jacobians())
            action[0:7] += off["L"]
            action[8:15] += off["R"]
        log["state"].append(read_state().round(4).tolist())
        log["action"].append(action.round(4).tolist())

        if left_closed_frame is None and action[7] < 0.1:
            left_closed_frame = k
        if box_held and left_closed_frame is not None and k >= left_closed_frame + 15:
            box.set_enabled_gravities(True)
            box_held, gravity_on_frame = False, k

        pending.append(action)
        set_arms(pending.pop(0) if len(pending) > args.action_delay else states[0], teleport=False)
        if not args.headless or needs_images(k + 1, chunk, chunk_pos):
            # Physics-only ticks, then one app update (= one tick + render): one render per frame, not two.
            SimulationManager.step(steps=rp.TICKS_PER_FRAME - 1)
            simulation_app.update()
            renders += 1
        else:
            SimulationManager.step(steps=rp.TICKS_PER_FRAME)
        n_done = k + 1

        _, bquat, centre = box_pose()
        log["box_quat"].append([round(float(v), 5) for v in bquat])
        for side, prim in tcps.items():
            log[f"tcp_{side}"].append([round(float(v), 4) for v in prim.get_world_poses()[0].numpy().reshape(-1)])
        log["gripper_cmd"].append([round(float(action[7]), 3), round(float(action[15]), 3)])
        log["box_centre"].append([round(float(v), 4) for v in centre])
        hud = (f"t {k / 30:5.1f} s  frame {k}  grip L {action[7]:.2f} R {action[15]:.2f}  "
               f"box height {centre[1]:.3f} m  policy {1000 * latencies[-1]:.0f} ms\n" + "\n".join(decision)
               + "\n" + (supervisor.status if supervisor is not None else "supervisor off")
               + (f" | {assist.status}" if assist is not None else ""))
        if k % 150 == 0:
            print(f"[rollout] frame {k}  grip cmd L {action[7]:.2f} R {action[15]:.2f}  "
                  f"box centre {np.round(centre, 3).tolist()}  last latency {1000 * latencies[-1]:.0f} ms", flush=True)

    wall = time.time() - t0
    physics_steps = SimulationManager.get_num_physics_steps() - steps_before
    if recorder is not None:
        recorder.close()
    if live is not None:
        live.close()

    box_pos, box_quat, centre = box_pose()
    lat_ms = 1000 * np.array(latencies[1:] or latencies)
    summary = {
        "episode_start_state": args.episode,
        "frames": n_done,
        "sim_fps": round(n_done / wall, 1),
        "realtime_factor": round(n_done / 30 / wall, 2),
        "renders": renders,
        "physics_steps": int(physics_steps),
        "physics_steps_ok": int(physics_steps) == n_done * rp.TICKS_PER_FRAME,
        "video_frames": recorder.frames if recorder is not None else 0,
        "replan_every": args.replan_every,
        "queries": len(latencies),
        "latency_ms": {"p50": round(float(np.percentile(lat_ms, 50)), 1),
                       "p95": round(float(np.percentile(lat_ms, 95)), 1)},
        "wall_time_s": round(time.time() - t0, 1),
        "left_closed_frame": left_closed_frame,
        "box_gravity_on_frame": gravity_on_frame,
        "box_lifted_mm_at_start": args.box_lift_mm,
        "box_lift_released": gravity_on_frame is not None or args.box_lift_mm == 0,
        "final_box_pos_m": [round(float(v), 4) for v in box_pos],
        "final_box_centre_m": [round(float(v), 4) for v in centre],
        "final_box_quat_wxyz": [round(float(v), 4) for v in box_quat],
        "supervisor": supervisor is not None,
        "max_retries": args.max_retries,
        "retries": supervisor.retries if supervisor is not None else None,
        "supervisor_events": supervisor.events if supervisor is not None else [],
        "abort_reason": abort_reason,
        "align_assist": assist is not None,
        "assist_tcp_error_at_grasp_mm": assist.err_at_grasp if assist is not None else None,
        "open_approach": args.open_approach,
        "raw": args.raw,
        "seed": args.seed,
        "box_offset_dx_dy_dz_yaw": box_offset,
        "render_mode": args.render_mode,
        "stopped_by_signal": stop["signal"],
    }
    if extra and extra.get("final_box_pose"):
        ref = extra["final_box_pose"]
        summary["recorded_final_box_pos_m"] = [round(v, 4) for v in ref[:3]]
        summary["final_box_pos_error_mm"] = round(1000 * math.dist(box_pos.tolist(), ref[:3]), 1)
        summary["final_box_rot_error_deg"] = round(rp.quat_angle_deg(box_quat.tolist(), ref[3:7]), 2)
        # Same placement threshold as the replay check; rotation is reported but not gated (unresolved in replay).
        summary["placed_near_recorded_pose"] = summary["final_box_pos_error_mm"] < 30.0
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "trace.json"), "w") as fh:
        json.dump(log, fh)
    if dump["frame"]:
        np.savez_compressed(os.path.join(out_dir, "obs_dump.npz"), **{k: np.stack(v) for k, v in dump.items()})
    print("[rollout] summary:\n" + json.dumps(summary, indent=2), flush=True)
    print(f"[rollout] wrote {out_dir}", flush=True)

    if args.keep_open and not args.headless:
        print("[rollout] done; close the window to exit", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
