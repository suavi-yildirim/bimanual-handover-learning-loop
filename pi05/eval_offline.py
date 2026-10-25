"""Stage 04 (offline): run a checkpoint on held-out episodes and compare its action chunks with the recording.

No simulator: every query uses the recorded observation, so this checks the action space, normalisation,
camera mapping and fit, not closed-loop success. Writes metrics.json and one figure per episode
(predicted vs recorded, chunks tiled open-loop: each chunk's first `stride` steps).

Run from ~/openpi:
    uv run <repo>/pi05/eval_offline.py \
        --config pi05_fr3duo_6a5f7c87 --checkpoint checkpoints/fr3duo_6a5f7c87 --episodes 0 1
"""

import dataclasses
import json
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies import policy_config
from openpi.training import config as _config

NAMES = [f"L q{i}" for i in range(1, 8)] + ["L grip"] + [f"R q{i}" for i in range(1, 8)] + ["R grip"]
ARM = [*range(0, 7), *range(8, 15)]
GRIP = [7, 15]
CAMS = ("external_image", "left_wrist_image", "right_wrist_image")


@dataclasses.dataclass
class Args:
    config: str
    checkpoint: pathlib.Path
    repo_id: str = "local/fr3duo_handover_v0_val"
    episodes: tuple[int, ...] = ()
    """Episode indices within repo_id; empty = all."""
    action_horizon: int | None = None
    """Override the config's action horizon (the horizon is not stored in the weights)."""
    stride: int | None = None
    """Frames between queries; default = action horizon (chunks tile the episode)."""
    swap_wrists: bool = False
    """Feed the left wrist image as right and vice versa (camera-mapping check)."""
    out: pathlib.Path | None = None


def to_uint8_hwc(img) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = (255 * x).round().astype(np.uint8)
    return np.moveaxis(x, 0, -1) if x.shape[0] == 3 else x


def main(a: Args) -> None:
    cfg = _config.get_config(a.config)
    if a.action_horizon is not None:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=a.action_horizon))
    H = cfg.model.action_horizon
    stride = a.stride or H
    out = a.out or pathlib.Path("eval_offline") / f"{a.config}_h{H}{'_swap' if a.swap_wrists else ''}"
    out.mkdir(parents=True, exist_ok=True)

    policy = policy_config.create_trained_policy(cfg, a.checkpoint)
    ds = LeRobotDataset(a.repo_id)
    ds.hf_dataset = ds.hf_dataset.with_format(None)  # plain python/PIL rows, faster than torch tensors
    lo, hi = ds.episode_data_index["from"].tolist(), ds.episode_data_index["to"].tolist()
    episodes = a.episodes or tuple(range(len(lo)))
    cams = dict(zip(CAMS, CAMS))
    if a.swap_wrists:
        cams["left_wrist_image"], cams["right_wrist_image"] = "right_wrist_image", "left_wrist_image"

    step_err, chunk_err, latency, per_episode = [], [], [], {}
    for e in episodes:
        rows = ds.hf_dataset.select(range(lo[e], hi[e]))
        rec = np.stack([np.asarray(r, np.float32) for r in rows["actions"]])
        L = len(rec)
        stitched = np.full_like(rec, np.nan)
        for t in range(0, L, stride):
            r = rows[t]
            obs = {
                "images": {slot: to_uint8_hwc(r[src]) for slot, src in cams.items()},
                "state": np.asarray(r["state"], np.float32),
                "prompt": ds.meta.tasks[int(r["task_index"])],
            }
            t0 = time.monotonic()
            pred = policy.infer(obs)["actions"]
            latency.append(time.monotonic() - t0)
            n = min(H, L - t)
            err = np.abs(pred[:n] - rec[t : t + n])
            chunk_err.append(err.mean(0))
            step_err.extend((k, *err[k]) for k in range(n))
            stitched[t : t + min(stride, n)] = pred[: min(stride, n)]
        d = np.abs(stitched - rec)
        per_episode[e] = {
            "arm_mae_deg": float(np.degrees(np.nanmean(d[:, ARM]))),
            "grip_mae": float(np.nanmean(d[:, GRIP])),
            "frames": L,
            "queries": len(range(0, L, stride)),
        }
        print(f"[eval] ep {e}: arm MAE {per_episode[e]['arm_mae_deg']:.2f} deg, grip MAE {per_episode[e]['grip_mae']:.3f}", flush=True)

        fig, axs = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
        tt = np.arange(L) / ds.fps
        for j, ax in enumerate(axs.flat):
            ax.plot(tt, rec[:, j], color="0.3", lw=1.2, label="recorded")
            ax.plot(tt, stitched[:, j], color="tab:orange", lw=0.9, label="predicted")
            ax.set_title(NAMES[j], fontsize=9)
            ax.tick_params(labelsize=7)
        axs[0, 0].legend(fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel("time (s)", fontsize=8)
        fig.suptitle(f"{a.config} h={H}  ep {e}  arm MAE {per_episode[e]['arm_mae_deg']:.2f} deg  (rad / gripper cmd)")
        fig.tight_layout()
        fig.savefig(out / f"ep{e:02d}.png", dpi=110)
        plt.close(fig)

    se = np.array(step_err)
    by_step = [float(np.degrees(se[se[:, 0] == k][:, 1:][:, ARM].mean())) for k in range(H) if (se[:, 0] == k).any()]
    ce = np.array(chunk_err)
    lat = np.array(latency[1:] or latency)  # first call includes JIT compile
    metrics = {
        "config": a.config, "checkpoint": str(a.checkpoint), "repo_id": a.repo_id, "action_horizon": H,
        "stride": stride, "swap_wrists": a.swap_wrists, "episodes": per_episode,
        "arm_mae_deg": float(np.degrees(ce[:, ARM].mean())),
        "grip_mae": float(ce[:, GRIP].mean()),
        "arm_mae_deg_by_chunk_step": by_step,
        "latency_s": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "first_call": latency[0]},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(len(by_step)), by_step, marker=".")
    ax.set_xlabel("step within chunk (1/30 s)")
    ax.set_ylabel("arm MAE (deg)")
    ax.set_title(f"error vs prediction step, h={H}")
    fig.tight_layout()
    fig.savefig(out / "error_by_chunk_step.png", dpi=110)
    print(json.dumps({k: metrics[k] for k in ("action_horizon", "arm_mae_deg", "grip_mae", "latency_s")}, indent=1))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
