"""Data loop, Stage 01 for v1: expert demos generated in our Isaac Sim scene -> LeRobot v2.1 for openpi.

Reads sim/expert_demos.py episodes (<demos>/ep_NNN/{external,left_wrist,right_wrist}.mp4, data.npz, meta.json),
keeps successful episodes only, and writes them with the same features as convert_fr3duo_to_openpi.py
(images 224x224, state/actions float32[16], task string). Run from ~/openpi:
    uv run <repo>/pi05/convert_expert_demos.py \
        --demos <repo>/sim/out/expert_demos
"""

import dataclasses
import json
import pathlib
import shutil

import av
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
CAMERAS = {"external": "external_image", "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"}
IMG_SHAPE = (224, 224, 3)


@dataclasses.dataclass
class Args:
    demos: tuple[pathlib.Path, ...]
    """Folders holding ep_NNN episode dirs (several generation batches can be merged)."""
    repo_id: str = "local/fr3duo_sim_expert_v1"
    include_failed: bool = False


def decode(path: pathlib.Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def main(a: Args) -> None:
    out = HF_LEROBOT_HOME / a.repo_id
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=a.repo_id, robot_type="fr3_duo", fps=30,
        features={
            **{name: {"dtype": "image", "shape": IMG_SHAPE, "names": ["height", "width", "channel"]}
               for name in CAMERAS.values()},
            "state": {"dtype": "float32", "shape": (16,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (16,), "names": ["actions"]},
        },
        image_writer_threads=16, image_writer_processes=0,
    )
    log = []
    for root in a.demos:
        for ep in sorted(root.glob("ep_*")):
            meta = json.loads((ep / "meta.json").read_text())
            if not meta["success"] and not a.include_failed:
                log.append({"episode": str(ep), "kept": False, "reason": "failed"})
                continue
            d = np.load(ep / "data.npz")
            imgs = {cam: decode(ep / f"{cam}.mp4") for cam in CAMERAS}
            n = min(len(d["state"]), *(len(v) for v in imgs.values()))
            for i in range(n):
                ds.add_frame({**{name: imgs[cam][i] for cam, name in CAMERAS.items()},
                              "state": d["state"][i], "actions": d["action"][i], "task": TASK})
            ds.save_episode()
            log.append({"episode": str(ep), "kept": True, "frames": n})
            print(f"[convert] {ep.name}: {n} frames", flush=True)
    (out / "meta" / "expert_conversion.json").write_text(json.dumps(log, indent=2))
    kept = [x for x in log if x["kept"]]
    print(f"[convert] {len(kept)} episodes, {sum(x['frames'] for x in kept)} frames -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
