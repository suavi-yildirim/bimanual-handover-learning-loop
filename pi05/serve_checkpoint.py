"""Serve (or offline-check) an externally trained FR3 Duo pi0.5 checkpoint.

The checkpoint fr3-duo-checkpoint-6a5f7c87 was trained outside this repo: its norm stats sit under
`assets/fr3_duo` and its action stats are absolute joint targets (means equal the state means), so it
needs `pi05_fr3duo_handover_lora` with delta actions off and asset id `fr3_duo`. The action horizon is not
stored in the checkpoint; `check` compares horizons against recorded validation frames.

Run from ~/openpi:
    uv run <repo>/pi05/serve_checkpoint.py serve --ckpt checkpoints/fr3duo_6a5f7c87 --horizon 30
    uv run <repo>/pi05/serve_checkpoint.py check --ckpt checkpoints/fr3duo_6a5f7c87 --horizons 10 30 50
"""

import argparse
import dataclasses
import glob
import io
import json
import logging
import time

import numpy as np

from openpi.models import pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

BASE_CONFIG = "pi05_fr3duo_handover_lora"
VAL_DIR = "~/.cache/huggingface/lerobot/local/fr3duo_handover_v0_val"
CAMERA_KEYS = ("external_image", "left_wrist_image", "right_wrist_image")


def make_config(horizon: int, delta: bool, asset_id: str) -> _config.TrainConfig:
    cfg = _config.get_config(BASE_CONFIG)
    data = dataclasses.replace(
        cfg.data, use_delta_joint_actions=delta, assets=_config.AssetsConfig(asset_id=asset_id)
    )
    model = pi0_config.Pi0Config(
        pi05=True, action_horizon=horizon, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    )
    return dataclasses.replace(cfg, data=data, model=model)


def make_policy(args, horizon: int):
    return _policy_config.create_trained_policy(
        make_config(horizon, args.delta, args.asset_id), args.ckpt, sample_kwargs={"num_steps": args.denoise_steps}
    )


def load_val_frames(n_episodes: int, stride: int):
    """(observation dict, recorded absolute action chunk [T, 16]) pairs from the converted val split."""
    import os

    import pandas as pd
    from PIL import Image

    files = sorted(glob.glob(os.path.join(os.path.expanduser(VAL_DIR), "data", "*", "*.parquet")))[:n_episodes]
    for f in files:
        df = pd.read_parquet(f)
        actions = np.stack(df["actions"].values).astype(np.float32)
        for k in range(0, len(df), stride):
            row = df.iloc[k]
            obs = {
                "images": {c: np.asarray(Image.open(io.BytesIO(row[c]["bytes"])).convert("RGB")) for c in CAMERA_KEYS},
                "state": np.asarray(row["state"], dtype=np.float32),
                "prompt": "Hand the box from the left gripper to the right gripper and place it on the serving tray.",
            }
            yield f, k, obs, actions[k:]


def check(args):
    frames = list(load_val_frames(args.episodes, args.stride))
    report = {}
    for horizon in args.horizons:
        policy = make_policy(args, horizon)
        errs, grip_errs, lat = [], [], []
        for _, _, obs, gt in frames:
            t0 = time.perf_counter()
            pred = policy.infer(obs)["actions"]
            lat.append(time.perf_counter() - t0)
            n = min(len(gt), len(pred), args.compare_steps)
            arm = [*range(7), *range(8, 15)]
            errs.append(np.degrees(np.abs(pred[:n, arm] - gt[:n, arm])).mean())
            grip_errs.append(np.abs(pred[:n, [7, 15]] - gt[:n, [7, 15]]).mean())
        report[horizon] = {
            "frames": len(frames),
            "arm_mae_deg": round(float(np.mean(errs)), 3),
            "gripper_mae": round(float(np.mean(grip_errs)), 4),
            # First call includes JIT compilation.
            "latency_ms_p50": round(1000 * float(np.median(lat[1:])), 1),
        }
        logging.info("horizon %d: %s", horizon, report[horizon])
        del policy
    print(json.dumps(report, indent=2))


def serve(args):
    from openpi.serving import websocket_policy_server

    policy = make_policy(args, args.horizon)
    logging.info("serving %s (horizon %d, delta %s) on port %d", args.ckpt, args.horizon, args.delta, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["serve", "check"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--asset-id", default="fr3_duo")
    p.add_argument("--delta", action="store_true", help="checkpoint was trained with delta joint actions")
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 50])
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--stride", type=int, default=150)
    p.add_argument("--compare-steps", type=int, default=10, help="chunk steps compared in `check`")
    args = p.parse_args()
    (check if args.mode == "check" else serve)(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
