"""Generate expert demonstrations of the box handover in our Isaac Sim scene (data loop, Stage 01 for v1).

The expert follows the recorded episode's joint trajectory in time, with the box as shipped (resting on the cart):
  - grasps: at the recorded close frame of each arm the clock stops; the alignment assist (sim/align_assist.py)
    centres the TCP on box_centre + REF[side] (the replay's grasp offset), the fingers close once the error is
    below ALIGN_TOL_M, and the clock resumes when the grasp is confirmed (gripper stopped on the box)
  - pauses: recorded idle stretches are cut to IDLE_KEEP frames (the policy cannot time an ambiguous pause)
  - variation: box offset per episode (+-BOX_DXZ_M in x and z, +-BOX_YAW_DEG yaw)
  - DART: a slowly varying noise (Ornstein-Uhlenbeck, NOISE_DEG) is added to the executed arm targets outside
    the grasp windows; the saved action is the clean expert command, so the data teaches corrections
Images are rendered every 2nd frame (settle updates with physics paused, same as the rollout's --settle) and
reused on the frame in between. Per episode: <out>/ep_NNN/{external,left_wrist,right_wrist}.mp4 (224x224, 30 fps),
data.npz (state, action float32 [T,16], dataset scale), meta.json (success, box offset, events).

    PYTHONPATH=$ISAAC_PYDEPS $ISAAC_PY sim/expert_demos.py --episodes 1
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align_assist as aa  # noqa: E402
import policy_rollout as pr  # noqa: E402
import recovery as rc  # noqa: E402
import replay_episode as rp  # noqa: E402

ALIGN_TOL_M = 0.004
ALIGN_MAX_FRAMES = 120
GRASP_CONFIRM = 10
IDLE_DEG_S = 0.5
IDLE_MIN_RUN = 15
IDLE_KEEP = 5
BOX_DXZ_M = 0.015
BOX_YAW_DEG = 5.0
NOISE_DEG = 0.6
NOISE_THETA = 0.05  # OU mean reversion per frame
TRAY_CENTRE_XZ = pr.TRAY_CENTRE_XZ
PLACE_TOL_M = 0.10  # success: box at rest in the tray footprint, centre within this of the tray centre


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "out", "expert_demos"))
    p.add_argument("--no-variation", action="store_true", help="box at the shipped pose, no DART noise")
    p.add_argument("--render-mode", choices=("rt", "pt"), default="rt",
                   help="rt: RTX real-time ray tracing (~57 ms per update with 3 cameras); pt: path tracing at the scene's "
                        "64 spp (~400-600 ms). Train and evaluate with the same mode.")
    p.add_argument("--settle", type=int, default=None, help="updates per image with physics paused (default rt 2, pt 3)")
    p.add_argument("--stream-port", type=int, default=8090)
    return p.parse_args()


def keep_mask(actions, states):
    """Recorded frames to keep: idle stretches (no joint motion, no gripper change) cut to IDLE_KEEP frames."""
    import numpy as np

    arm = np.r_[0:7, 8:15]
    speed = np.degrees(np.abs(np.diff(states[:, arm], axis=0)).max(axis=1)) * 30
    speed = np.r_[speed, speed[-1]]
    grip_change = np.r_[False, np.any(np.abs(np.diff(actions[:, [7, 15]], axis=0)) > 1e-3, axis=1)]
    idle = (speed < IDLE_DEG_S) & ~grip_change
    keep = np.ones(len(actions), bool)
    i = 0
    while i < len(idle):
        if idle[i]:
            j = i
            while j < len(idle) and idle[j]:
                j += 1
            if j - i >= IDLE_MIN_RUN:
                keep[i + IDLE_KEEP:j] = False
            i = j
        else:
            i += 1
    return keep


def main():
    args = parse_args()
    if args.settle is None:
        args.settle = 2 if args.render_mode == "rt" else 3
    import numpy as np

    rng_np = np.random.default_rng(args.seed)
    states, actions, _raw, _extra = rp.load_episode(rp.DEFAULT_DATASET, 0)
    close_frames = {side: int(np.argmax(actions[:, g] < 0.1)) for side, g in rc.GRIP.items()}
    keep = keep_mask(actions, states)
    print(f"[expert] recorded {len(actions)} frames, keeping {int(keep.sum())} after cutting idle stretches; "
          f"grasp frames {close_frames}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    import imageio_ffmpeg
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import UsdGeom

    import frame_stream

    live = frame_stream.LiveStream(args.stream_port) if args.stream_port else None

    for ep in range(args.episodes):
        ok, _ = stage_utils.open_stage(rp.DEFAULT_USD)
        while stage_utils.is_stage_loading():
            app.update()
        stage = stage_utils.get_current_stage()
        import carb

        carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting" if args.render_mode == "rt" else "PathTracing")
        rp.apply_recorded_physics(stage)
        try:
            from omni.kit.viewport.utility import get_active_viewport

            vp = get_active_viewport()
            if vp is not None:
                vp.updates_enabled = False
        except Exception:
            pass
        annotators = {}
        for key, (path, focal) in pr.CAMERAS.items():
            cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
            cam.GetFocalLengthAttr().Set(focal)
            cam.GetHorizontalApertureAttr().Set(pr.APERTURE_MM)
            cam.GetVerticalApertureAttr().Set(pr.APERTURE_MM)
            prod = rep.create.render_product(path, (pr.IMAGE_SIZE, pr.IMAGE_SIZE))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([prod])
            annotators[key] = annot
        rngb = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(rp.BOX)).ComputeAlignedRange()
        box_centre_at_load = np.array((rngb.GetMin() + rngb.GetMax()) / 2.0, dtype=np.float64)

        SimulationManager.setup_simulation(dt=rp.PHYSICS_DT)
        RenderingManager.set_dt(rp.PHYSICS_DT)
        app_utils.play()
        app.update()
        arms = {}
        for side in ("left", "right"):
            art = Articulation(f"{rp.ROBOT}/{side}_arm_root_joint")
            names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_ctrl"]
            arms[side] = (art, art.get_dof_indices(names).numpy())
        box = RigidPrim(rp.BOX)
        tcps = {"L": RigidPrim(f"{rp.ROBOT}/left_tcp"), "R": RigidPrim(f"{rp.ROBOT}/right_tcp")}
        jac_rows = {side: arms[side][0].link_names.index(f"{side}_tcp") - 1 for side in arms}
        _p0, _q0 = (a.numpy().reshape(-1).astype(np.float64) for a in box.get_world_poses())
        box_local = rp.quat_rotate(_q0 * np.array([1, -1, -1, -1]), box_centre_at_load - _p0)

        def box_centre():
            pos, quat = (a.numpy().reshape(-1).astype(np.float64) for a in box.get_world_poses())
            return pos + rp.quat_rotate(quat, box_local), quat

        def jacobians():
            out = {}
            for side, short in (("left", "L"), ("right", "R")):
                art, idx = arms[side]
                jac = art.get_jacobian_matrices().numpy()[0, jac_rows[side]]
                out[short] = jac[0:3, np.asarray(idx[:7])].astype(np.float64)
            return out

        def set_arms(vec16, teleport=False):
            for (art, idx), q in zip(arms.values(), rp.to_joint_targets(vec16)):
                if teleport:
                    art.set_dof_positions(q.reshape(1, -1), dof_indices=idx)
                    art.set_dof_velocities(np.zeros((1, len(idx))), dof_indices=idx)
                art.set_dof_position_targets(q.reshape(1, -1), dof_indices=idx)

        def read_state():
            out = []
            for art, idx in arms.values():
                q = art.get_dof_positions(dof_indices=idx).numpy().reshape(-1).astype(np.float64)
                q[7] = pr.grip_to_dataset(1.0 - q[7] / rp.GRIPPER_CLOSED_RAD)
                out.append(q)
            return np.concatenate(out)

        def render(settle):
            app_utils.pause()
            for _ in range(settle):
                app.update()
            imgs = {}
            for key, annot in annotators.items():
                data = np.asarray(annot.get_data())
                if data.ndim != 3 or data.shape[0] != pr.IMAGE_SIZE:
                    imgs = None
                    break
                imgs[key] = np.ascontiguousarray(data[:, :, :3], dtype=np.uint8)
            app_utils.play()
            return imgs

        set_arms(states[0], teleport=True)
        offset = [0.0, 0.0, 0.0, 0.0]
        if not args.no_variation:
            offset = [float(rng_np.uniform(-BOX_DXZ_M, BOX_DXZ_M)), 0.0, float(rng_np.uniform(-BOX_DXZ_M, BOX_DXZ_M)),
                      float(rng_np.uniform(-BOX_YAW_DEG, BOX_YAW_DEG))]
            rp.offset_box(box, offset)
        for warm in range(1, 1001):
            if render(1) is not None:
                break
        for _ in range(30):  # let the box settle on the cart
            SimulationManager.step(steps=rp.TICKS_PER_FRAME)

        ep_dir = os.path.join(args.out, f"ep_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        writers = {}
        for key in pr.CAMERAS:
            w = imageio_ffmpeg.write_frames(os.path.join(ep_dir, f"{key.replace('_image', '')}.mp4"),
                                            (pr.IMAGE_SIZE, pr.IMAGE_SIZE), fps=30, codec="libx264",
                                            pix_fmt_out="yuv420p", quality=None, macro_block_size=1,
                                            output_params=["-preset", "fast", "-crf", "16"])
            w.send(None)
            writers[key] = w

        assist = aa.AlignAssist()
        grasped = {"L": False, "R": False}
        released = {"L": False, "R": False}
        closed_for = {"L": 0, "R": 0}
        noise = np.zeros(16)
        sigma = np.radians(NOISE_DEG) * (0.0 if args.no_variation else 1.0)
        rec_states, rec_actions, events = [], [], []
        t, k, hold_since, closing = 0, 0, None, None
        images = None
        t0 = time.time()
        while t < len(actions) and k < 4000:
            # Clock: skip cut idle frames; stop at a grasp until it is confirmed.
            while t < len(actions) - 1 and not keep[t]:
                t += 1
            pending = next((s for s in ("L", "R") if not grasped[s] and t >= close_frames[s]), None)
            target = actions[min(t, close_frames[pending] - 1) if pending else t].astype(np.float64).copy()
            if pending:
                if hold_since is None:
                    hold_since = k
                    events.append({"frame": k, "event": f"{pending} grasp: aligning"})
                g = rc.GRIP[pending]
                err = assist.last_err[pending]
                if closing is None and ((err is not None and err < ALIGN_TOL_M) or k - hold_since >= ALIGN_MAX_FRAMES):
                    closing = k
                    events.append({"frame": k, "event": f"{pending} closing, TCP error "
                                                        f"{'n/a' if err is None else round(1000 * err, 1)} mm"})
                target[g] = 0.0 if closing is not None else rc.PRE_SHAPE
            # Assist offsets (alignment before each grasp, height floor, fade after the lift).
            tcp_now = {s: tcps[s].get_world_poses()[0].numpy().reshape(-1).astype(np.float64) for s in tcps}
            centre, quat = box_centre()
            if closing is not None and pending:
                off = assist.offset  # hold still while the fingers close
            else:
                off = assist.update(k, grasped, released, tcp_now, centre, jacobians())
            label = target.copy()
            label[0:7] += off["L"]
            label[8:15] += off["R"]
            # DART: executed = label + slowly varying noise, except in the grasp windows.
            noise = noise * (1 - NOISE_THETA) + sigma * math.sqrt(2 * NOISE_THETA) * rng_np.normal(size=16)
            noise[[7, 15]] = 0.0
            executed = label + (0.0 if pending else noise)

            if k % 2 == 0 or images is None:
                images = render(args.settle) or images
            st = read_state()
            if images is not None:
                for key, w in writers.items():
                    w.send(images[key])
                rec_states.append(st.astype(np.float32))
                rec_actions.append(label.astype(np.float32))
                if live is not None and live.watching:
                    hud = (f"expert ep {ep} frame {k} rec {t} | {assist.status}\n"
                           f"grasped L {grasped['L']} R {grasped['R']} | box {np.round(centre, 3).tolist()}")
                    live.push(frame_stream.compose(images, pr.CAMERAS, hud, scale=2))

            set_arms(executed)
            SimulationManager.step(steps=rp.TICKS_PER_FRAME)
            k += 1
            real = read_state()
            for s, g in rc.GRIP.items():
                closed_for[s] = closed_for[s] + 1 if executed[g] < rc.CLOSE_CMD else 0
                if (not grasped[s] and closed_for[s] >= GRASP_CONFIRM
                        and pr.LATCHED_HOLD_RANGE[0] < real[g] < pr.LATCHED_HOLD_RANGE[1]):
                    grasped[s] = True
                    events.append({"frame": k, "event": f"{s} grasped (state {real[g]:.2f})"})
                    hold_since, closing = None, None
                elif not grasped[s] and closed_for[s] >= 3 * GRASP_CONFIRM and real[g] < rc.EMPTY_CLOSED:
                    events.append({"frame": k, "event": f"{s} grasp missed"})
                    t = len(actions)  # failed episode
                elif grasped[s] and not released[s] and executed[g] >= rc.CLOSE_CMD:
                    released[s] = True
            if not pending:
                t += 1
            if k % 300 == 0:
                print(f"[expert] ep {ep} frame {k} rec {t} box {np.round(centre, 3).tolist()} "
                      f"grasped {grasped} {k / (time.time() - t0):.1f} fps", flush=True)

        for _ in range(60):  # let the box come to rest
            SimulationManager.step(steps=rp.TICKS_PER_FRAME)
        centre, quat = box_centre()
        d_tray = math.hypot(centre[0] - TRAY_CENTRE_XZ[0], centre[2] - TRAY_CENTRE_XZ[1])
        success = bool(grasped["L"] and grasped["R"] and released["R"] and d_tray < PLACE_TOL_M and 0.95 < centre[1] < 1.1)
        for w in writers.values():
            w.close()
        np.savez_compressed(os.path.join(ep_dir, "data.npz"), state=np.stack(rec_states), action=np.stack(rec_actions))
        meta = {"episode": ep, "success": success, "frames": len(rec_states), "box_offset_dx_dy_dz_yaw": offset,
                "final_box_centre_m": np.round(centre, 4).tolist(), "box_to_tray_centre_m": round(d_tray, 3),
                "events": events, "wall_s": round(time.time() - t0, 1), "dart_noise_deg": 0 if args.no_variation else NOISE_DEG,
                "render_mode": args.render_mode, "settle": args.settle}
        json.dump(meta, open(os.path.join(ep_dir, "meta.json"), "w"), indent=2)
        print(f"[expert] ep {ep}: {'SUCCESS' if success else 'FAILED'} | {len(rec_states)} frames | box "
              f"{np.round(centre, 3).tolist()} ({d_tray * 1000:.0f} mm from tray centre) | {meta['wall_s']} s | "
              f"events {[e['event'] for e in events]}", flush=True)
        app_utils.stop()

    if live is not None:
        live.close()
    app.close()


if __name__ == "__main__":
    main()
