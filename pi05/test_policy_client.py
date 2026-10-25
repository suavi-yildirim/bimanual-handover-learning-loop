"""Stage 05 smoke test: query the running policy server over websocket with recorded val frames.

Start the server first (from ~/openpi):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_fr3duo_6a5f7c87 --policy.dir=checkpoints/fr3duo_6a5f7c87
Then (from ~/openpi):
    uv run <repo>/pi05/test_policy_client.py
"""

import dataclasses
import time

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy

ARM = [*range(0, 7), *range(8, 15)]


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    repo_id: str = "local/fr3duo_handover_v0_val"
    episode: int = 0
    frames: tuple[int, ...] = (0, 300, 600, 900, 1200, 1500, 1800)


def main(a: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    print("[client] server metadata:", client.get_server_metadata())
    ds = LeRobotDataset(a.repo_id)
    ds.hf_dataset = ds.hf_dataset.with_format(None)
    start = int(ds.episode_data_index["from"][a.episode])
    end = int(ds.episode_data_index["to"][a.episode])
    rows = ds.hf_dataset.select(range(start, end))
    rec = np.stack([np.asarray(r, np.float32) for r in rows["actions"]])
    for t in a.frames:
        r = rows[t]
        obs = {
            "images": {k: np.asarray(r[k]) for k in ("external_image", "left_wrist_image", "right_wrist_image")},
            "state": np.asarray(r["state"], np.float32),
            "prompt": ds.meta.tasks[int(r["task_index"])],
        }
        t0 = time.monotonic()
        out = client.infer(obs)
        dt = time.monotonic() - t0
        act = out["actions"]
        n = min(len(act), len(rec) - t)
        err = np.degrees(np.abs(act[:n, ARM] - rec[t : t + n, ARM]).mean())
        grip = act[0, [7, 15]].round(2)
        print(f"[client] frame {t:5d}: chunk {act.shape}, arm MAE {err:.2f} deg, grippers {grip}, round trip {1000 * dt:.0f} ms")


if __name__ == "__main__":
    main(tyro.cli(Args))
