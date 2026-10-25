"""Stage 01: merge + filter the FR3 Duo handover sets and rewrite them as LeRobot v2.1 for openpi.

Implements C2_pi05_Training_Process.md, sections 3 and 5:
- merge sources in the given order (84-set = 0..83, 10-set = 84..93),
- split train/val from the dataset_check manifests,
- trim idle runs > 0.5 s (keep the first and last 0.25 s), drop `success`,
- write `local/fr3duo_handover_v0` (train) and `local/fr3duo_handover_v0_val` (val).

Run from ~/openpi so the pinned LeRobot writes the files:
    uv run examples/fr3duo/convert_fr3duo_to_openpi.py \
        --sources ~/c2_data/fr3_duo_handover_84_ep.zip ~/c2_data/FR3Duo_Handover_Dataset_lerobot3.zip
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import time
import zipfile

import av
import numpy as np
import pyarrow.parquet as pq
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

PROJECT = pathlib.Path(os.environ.get("REPO_DIR", pathlib.Path(__file__).resolve().parents[1]))
DEFAULT_MANIFESTS = (
    PROJECT / "dataset_check/reports/fr3_duo_handover_v9_night_84/manifest.json",
    PROJECT / "dataset_check/reports/FR3Duo_Handover_Dataset_lerobot3/manifest.json",
)
CAMERAS = {  # source video key -> feature name in the converted set
    "observation.images.external_camera": "external_image",
    "observation.images.left_wrist_camera": "left_wrist_image",
    "observation.images.right_wrist_camera": "right_wrist_image",
}
ARM = [*range(0, 7), *range(8, 15)]
GRIP = [7, 15]
IMG_SHAPE = (224, 224, 3)


@dataclasses.dataclass
class Args:
    sources: tuple[pathlib.Path, ...]
    """Dataset zips or unzipped folders, in merge order."""
    manifests: tuple[pathlib.Path, ...] = DEFAULT_MANIFESTS
    """dataset_check manifest.json per source (same order); gives the train/val split."""
    repo_id: str = "local/fr3duo_handover_v0"
    unzip_dir: pathlib.Path = pathlib.Path("~/c2_data/raw").expanduser()
    idle_speed: float = 0.02
    """rad/s; max arm joint speed below this (and gripper command unchanged) = idle."""
    trim_min_s: float = 0.5
    """Only idle runs longer than this are trimmed."""
    trim_keep_s: float = 0.25
    """Kept at the start and at the end of each trimmed run."""
    max_episodes: int | None = None
    """Convert only the first N merged episodes (smoke test)."""
    no_trim: bool = False


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_root(src: pathlib.Path, unzip_dir: pathlib.Path) -> pathlib.Path:
    """Return the folder holding meta/info.json; zips are extracted once into unzip_dir/<stem>."""
    if src.suffix == ".zip":
        dest = unzip_dir / src.stem
        if not dest.exists():
            print(f"[convert] extracting {src} -> {dest}", flush=True)
            with zipfile.ZipFile(src) as z:
                z.extractall(dest)
        src = dest
    infos = sorted(src.glob("**/meta/info.json"), key=lambda p: len(p.parts))
    if not infos:
        raise FileNotFoundError(f"no meta/info.json under {src}")
    return infos[0].parent.parent


def keep_mask(state: np.ndarray, action: np.ndarray, fps: int, a: Args) -> np.ndarray:
    """Idle rule of dataset_check (dsc/checks.py), then trim runs > trim_min_s to their ends."""
    keep = np.ones(len(state), bool)
    if a.no_trim:
        return keep
    speed = np.r_[np.abs(np.diff(state[:, ARM], axis=0)).max(1) * fps, 0.0]
    dgrip = np.r_[np.abs(np.diff(action[:, GRIP], axis=0)).max(1), 0.0]
    idle = np.r_[False, (speed < a.idle_speed) & (dgrip < 1e-3), False].astype(int)
    starts, ends = np.flatnonzero(np.diff(idle) == 1), np.flatnonzero(np.diff(idle) == -1)
    k = int(round(a.trim_keep_s * fps))
    for s, t in zip(starts, ends):
        if (t - s) / fps > a.trim_min_s:
            keep[s + k : t - k] = False
    return keep


class VideoCache:
    """Decodes each shared v3.0 mp4 once; frame i of a file sits at pts i/fps."""

    def __init__(self, root: pathlib.Path, video_path: str, fps: int):
        self.root, self.video_path, self.fps = root, video_path, fps
        self.key = None
        self.frames: np.ndarray | None = None

    def get(self, cam: str, chunk: int, file: int) -> np.ndarray:
        if self.key != (cam, chunk, file):
            path = self.root / self.video_path.format(video_key=cam, chunk_index=chunk, file_index=file)
            with av.open(str(path)) as c:
                s = c.streams.video[0]
                frames = {round(float(f.pts * s.time_base) * self.fps): f.to_ndarray(format="rgb24") for f in c.decode(s)}
            n = max(frames) + 1
            if len(frames) != n:
                raise ValueError(f"{path}: {n - len(frames)} frames missing")
            self.frames = np.stack([frames[i] for i in range(n)])
            self.key = (cam, chunk, file)
        return self.frames


def new_dataset(repo_id: str, fps: int) -> LeRobotDataset:
    path = HF_LEROBOT_HOME / repo_id
    if path.exists():
        shutil.rmtree(path)
    image = {"dtype": "image", "shape": IMG_SHAPE, "names": ["height", "width", "channel"]}
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="fr3_duo",
        fps=fps,
        features={
            **{name: image for name in CAMERAS.values()},
            "state": {"dtype": "float32", "shape": (16,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (16,), "names": ["actions"]},
        },
        # Threads only: forked writer processes deadlocked after PyAV/pyarrow had started their thread pools.
        image_writer_threads=16,
        image_writer_processes=0,
    )


def main(a: Args) -> None:
    if len(a.sources) != len(a.manifests):
        raise ValueError("give one manifest per source")
    t0 = time.time()
    out = {"train": None, "val": None}
    log = {"repo_id": a.repo_id, "params": {k: str(v) for k, v in dataclasses.asdict(a).items()}, "sources": [], "episodes": []}
    merged = 0
    for src, man_path in zip(a.sources, a.manifests):
        root = dataset_root(src.expanduser(), a.unzip_dir)
        info = json.loads((root / "meta/info.json").read_text())
        manifest = json.loads(man_path.read_text())
        fps = info["fps"]
        if fps != 30 or info["robot_type"] != "fr3_duo":
            raise ValueError(f"{root}: expected fr3_duo at 30 fps, got {info['robot_type']} at {fps}")
        tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
        task_of = dict(zip(tasks["task_index"], tasks.index))  # v3.0 stores the task string as the index
        eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas().sort_values("episode_index")
        log["sources"].append({
            "source": str(src), "root": str(root), "sha256": sha256(src.expanduser()) if src.suffix == ".zip" else None,
            "manifest": str(man_path), "manifest_lineage": manifest["lineage"]["files"], "merged_offset": merged,
        })
        videos = VideoCache(root, info["video_path"], fps)
        data_tables = {}
        for _, ep in eps.iterrows():
            if a.max_episodes is not None and merged >= a.max_episodes:
                break
            e = int(ep["episode_index"])
            m = manifest["episodes"][str(e)]
            if m["verdict"] == "drop":
                print(f"[convert] skip source ep {e}: drop ({m['reason']})")
                continue
            split = "val" if m["split"] == "val" else "train"
            dkey = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
            if dkey not in data_tables:
                data_tables[dkey] = pq.read_table(root / info["data_path"].format(chunk_index=dkey[0], file_index=dkey[1])).to_pandas()
            d = data_tables[dkey]
            d = d[d["episode_index"] == e].sort_values("frame_index")
            state = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
            action = np.stack(d["action"].to_numpy()).astype(np.float32)
            task = task_of[int(d["task_index"].iloc[0])]
            keep = keep_mask(state, action, fps, a)
            images = {}
            for cam in CAMERAS:
                frames = videos.get(cam, int(ep[f"videos/{cam}/chunk_index"]), int(ep[f"videos/{cam}/file_index"]))
                first = int(round(float(ep[f"videos/{cam}/from_timestamp"]) * fps))
                images[cam] = frames[first + d["frame_index"].to_numpy()]
            if out[split] is None:
                out[split] = new_dataset(a.repo_id + ("_val" if split == "val" else ""), fps)
            ds = out[split]
            for i in np.flatnonzero(keep):
                ds.add_frame({
                    **{name: images[cam][i] for cam, name in CAMERAS.items()},
                    "state": state[i],
                    "actions": action[i],
                    "task": task,
                })
            ds.save_episode()
            log["episodes"].append({
                "merged_index": merged, "source_episode": e, "source": str(src), "split": split,
                "frames_in": len(keep), "frames_kept": int(keep.sum()),
                "out_episode": ds.meta.total_episodes - 1,
            })
            print(f"[convert] merged {merged:3d} (src ep {e:3d}) -> {split:5s} {int(keep.sum())}/{len(keep)} frames  [{time.time() - t0:.0f} s]", flush=True)
            merged += 1

    eps_log = log["episodes"]
    log["summary"] = {
        split: {
            "episodes": sum(r["split"] == split for r in eps_log),
            "frames_in": sum(r["frames_in"] for r in eps_log if r["split"] == split),
            "frames_kept": sum(r["frames_kept"] for r in eps_log if r["split"] == split),
            "merged_indices": [r["merged_index"] for r in eps_log if r["split"] == split],
        }
        for split in ("train", "val")
    }
    log["seconds"] = round(time.time() - t0, 1)
    for split, ds in out.items():
        if ds is not None:
            (ds.root / "meta/fr3duo_conversion.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log["summary"], indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))
