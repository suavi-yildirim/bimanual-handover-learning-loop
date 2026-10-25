"""Fit the box start pose to the recorded camera images (the task layers that placed it were not shipped).

For each candidate pose (offset from the shipped pose: dx, dy (up), dz in m, yaw about +Y in deg) the box is
set kinematic at that pose, both arms are teleported to the recorded state of a few frames before the left
grasp, the head and left-wrist cameras are rendered and compared with the recorded images (per-channel
normalised L1). Coordinate search, coarse to fine. Reference images come from
~/c2_data/boxfit/ref_ep0.npz (episode 0 of the 10-episode set, extracted with PyAV in the openpi venv).

Usage:
    PYTHONPATH=$ISAAC_PYDEPS $ISAAC_PY sim/fit_box_pose.py
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import policy_rollout as pr  # noqa: E402
import replay_episode as rp  # noqa: E402

SCORED = {  # frame -> cameras compared there (the box is large in the left wrist view just before the grasp)
    0: ("external_image",),
    207: ("external_image", "left_wrist_image"),
    267: ("external_image", "left_wrist_image"),
    297: ("left_wrist_image",),
    317: ("left_wrist_image",),
}
BOX_HEIGHT_M = 0.0719  # shipped box height (bbox), origin at its top; bottom rests on the cart shelf at y 0.823
REF_KEYS = {"external_image": "external_camera", "left_wrist_image": "left_wrist_camera"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default=os.path.expanduser("~/c2_data/boxfit/ref_ep0.npz"))
    p.add_argument("--usd", default=rp.DEFAULT_USD)
    p.add_argument("--start", type=float, nargs=5, default=[0.0, 0.0, 0.0, 0.0, 1.0],
                   help="dx dy dz (m) yaw (deg) height-scale")
    p.add_argument("--steps", type=float, nargs=5, default=[0.02, 0.02, 0.02, 10.0, 0.2])
    p.add_argument("--rest-on-shelf", action="store_true",
                   help="tie dy to the height scale so the box bottom stays on the cart shelf (not searched)")
    p.add_argument("--rounds", type=int, default=4, help="step halvings")
    p.add_argument("--settle", type=int, default=10,
                   help="app updates per scored frame: RTX path tracing lags 2 updates and converges over ~8")
    p.add_argument("--out", default=os.path.join(HERE, "out", "box_fit"))
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np

    ref = np.load(args.ref)
    states = dict(zip(ref["frames"].tolist(), ref["states"]))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    ok, _ = stage_utils.open_stage(args.usd)
    if not ok:
        raise SystemExit(f"could not open {args.usd}")
    while stage_utils.is_stage_loading():
        simulation_app.update()
    stage = stage_utils.get_current_stage()
    # Render-only box: without physics APIs its pose and scale can change while the timeline plays
    # (a scale edit on a physics body invalidates the articulation views).
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rp.BOX)):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    box_ops = {o.GetOpName(): o for o in UsdGeom.Xformable(stage.GetPrimAtPath(rp.BOX)).GetOrderedXformOps()}

    annotators = {}
    for key in REF_KEYS:
        path, focal = pr.CAMERAS[key]
        cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        cam.GetFocalLengthAttr().Set(focal)
        cam.GetHorizontalApertureAttr().Set(pr.APERTURE_MM)
        cam.GetVerticalApertureAttr().Set(pr.APERTURE_MM)
        product = rep.create.render_product(path, (pr.IMAGE_SIZE, pr.IMAGE_SIZE))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([product])
        annotators[key] = annot

    SimulationManager.setup_simulation(dt=rp.PHYSICS_DT)
    RenderingManager.set_dt(rp.PHYSICS_DT)
    app_utils.play()
    simulation_app.update()

    arms = {}
    for side in ("left", "right"):
        art = Articulation(f"{rp.ROBOT}/{side}_arm_root_joint")
        names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_ctrl"]
        arms[side] = (art, art.get_dof_indices(names).numpy())
    # The box root's parent is /World (identity), so its translate/orient ops are world values; yaw about +Y
    # is pre-multiplied onto the orient op (the later rotateX:unitsResolve op only converts Z-up to Y-up).
    p0 = np.array(box_ops["xformOp:translate"].Get(), dtype=np.float64)
    o0 = box_ops["xformOp:orient"].Get()
    q0 = np.array([o0.GetReal(), *o0.GetImaginary()], dtype=np.float64)
    orient_type = type(o0)
    print(f"[fit] shipped box origin {p0.round(4).tolist()} orient op {q0.round(4).tolist()}", flush=True)

    def norm(img):
        x = img.astype(np.float32)
        return (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-3)

    refs = {(f, k): norm(ref[f"{REF_KEYS[k]}_{f}"]) for f, keys in SCORED.items() for k in keys}

    scale_op = box_ops["xformOp:scale"]
    scale0 = scale_op.Get()

    def set_height_scale(h):
        # Scale component 1 is the box height; it grows downward from the box top (the prim origin).
        scale_op.Set(Gf.Vec3f(scale0[0], scale0[1] * h, scale0[2]))

    def full(x):
        x = np.array(x, dtype=np.float64)
        if args.rest_on_shelf:
            x[1] = BOX_HEIGHT_M * (x[4] - 1.0)
        return x

    def box_pose(x):
        dx, dy, dz, yaw = x[:4]
        h = math.radians(yaw) / 2
        qy = np.array([math.cos(h), 0.0, math.sin(h), 0.0])  # rotation about world +Y
        w1, x1, y1, z1 = qy
        w2, x2, y2, z2 = q0
        q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        return p0 + np.array([dx, dy, dz]), q

    renders = {}

    def score(x, keep=False):
        x = full(x)
        set_height_scale(x[4])
        pos, quat = box_pose(x)
        total, n = 0.0, 0
        for f, keys in SCORED.items():
            for (art, idx), q in zip(arms.values(), rp.to_joint_targets(states[f])):
                art.set_dof_positions(q.reshape(1, -1), dof_indices=idx)
                art.set_dof_velocities(np.zeros((1, len(idx))), dof_indices=idx)
                art.set_dof_position_targets(q.reshape(1, -1), dof_indices=idx)
            box_ops["xformOp:translate"].Set(Gf.Vec3d(*pos.tolist()))
            box_ops["xformOp:orient"].Set(orient_type(float(quat[0]), *quat[1:].tolist()))
            for _ in range(args.settle):  # arms hold the teleported pose (targets = positions), box is kinematic
                simulation_app.update()
            for k in keys:
                img = np.asarray(annotators[k].get_data())[:, :, :3]
                total += float(np.abs(norm(img) - refs[(f, k)]).mean())
                n += 1
                if keep:
                    renders[(f, k)] = img.copy()
        return total / n

    t0 = time.time()
    x = np.array(args.start, dtype=np.float64)
    shipped = np.array([0, 0, 0, 0, 1.0])
    baseline = {"shipped": score(shipped), "start": score(x), "shipped_repeat": score(shipped)}
    print(f"[fit] render noise: shipped scored {baseline['shipped']:.4f} and {baseline['shipped_repeat']:.4f}", flush=True)
    best = baseline["start"]
    print(f"[fit] score shipped {baseline['shipped']:.4f}, start {x.tolist()} {best:.4f}", flush=True)
    steps = np.array(args.steps, dtype=np.float64)
    evals = 3
    for rnd in range(args.rounds):
        improved = True
        while improved:
            improved = False
            for i in [i for i in range(5) if not (args.rest_on_shelf and i == 1)]:
                for sign in (1, -1):
                    cand = x.copy()
                    cand[i] += sign * steps[i]
                    s = score(cand)
                    evals += 1
                    if s < best - 1e-5:
                        x, best, improved = cand, s, True
                        print(f"[fit] round {rnd} {np.round(x, 4).tolist()} score {best:.4f}", flush=True)
                        break
        steps /= 2

    x = full(x)
    score(x, keep=True)
    pos, quat = box_pose(x)
    os.makedirs(args.out, exist_ok=True)
    result = {
        "offset_from_shipped": {"dx_m": x[0], "dy_up_m": x[1], "dz_m": x[2], "yaw_deg": x[3]},
        "height_scale": x[4], "box_height_m": BOX_HEIGHT_M * x[4],
        "box_bottom_m": float(pos[1]) - BOX_HEIGHT_M * x[4], "rest_on_shelf": args.rest_on_shelf,
        "box_origin_m": pos.round(5).tolist(), "box_quat_wxyz": quat.round(6).tolist(),
        "score": best, "baseline_scores": baseline,
        "evals": evals, "seconds": round(time.time() - t0, 1),
        "frames_scored": {str(f): list(k) for f, k in SCORED.items()},
    }
    with open(os.path.join(args.out, "box_fit.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    import cv2

    rows = [np.concatenate([ref[f"{REF_KEYS[k]}_{f}"], renders[(f, k)]], axis=1) for f, keys in SCORED.items() for k in keys]
    cv2.imwrite(os.path.join(args.out, "ref_vs_fit.png"), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    print("[fit] result:\n" + json.dumps(result, indent=2), flush=True)
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
