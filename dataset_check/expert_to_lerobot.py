"""Write the sim expert demos (sim/out/expert_demos*/.../ep_000/data.npz + meta.json) as a minimal LeRobot v3.0
folder (state/action only, no video) so check_dataset.py can run the same Stage 02 checks on them.

    python expert_to_lerobot.py OUT_DIR            (run from dataset_check/)
    python check_dataset.py OUT_DIR/fr3duo_sim_expert_v1 --no-video

Includes failed episodes (status FAIL) on purpose: the checker should flag them. The corrupted DISCARDED batch is
left out. Box offset goes to meta/fr3_episodes_extra.jsonl as episode_extra.box_pose, as in the organisers' data.
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
NAMES = [f"{s}_{j}" for s in ("left", "right") for j in [f"joint{i}" for i in range(1, 8)] + ["gripper"]]


def main(out):
    split = json.load(open(os.path.join(ROOT, "sim", "out", "expert_split_v1.json")))
    metas = sorted(glob.glob(os.path.join(ROOT, "sim/out/expert_demos*/*/*/ep_*/meta.json")))
    metas += glob.glob(os.path.join(ROOT, "sim/out/expert_test/ep_*/meta.json"))
    metas = [m for m in metas if "DISCARDED" not in m]
    root = os.path.join(out, "fr3duo_sim_expert_v1")
    os.makedirs(os.path.join(root, "meta", "episodes", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "chunk-000"), exist_ok=True)
    st, ac, cols, eps, extra = [], [], {k: [] for k in ("episode_index", "frame_index", "timestamp", "success",
                                                         "task_index", "index")}, [], []
    n = 0
    for e, m in enumerate(metas):
        meta = json.load(open(m))
        z = np.load(os.path.join(os.path.dirname(m), "data.npz"))
        s, a = z["state"].astype(np.float32), z["action"].astype(np.float32)
        T = len(s)
        st.append(s)
        ac.append(a)
        cols["episode_index"] += [e] * T
        cols["frame_index"] += list(range(T))
        cols["timestamp"] += list(np.arange(T, dtype=np.float32) / 30)
        cols["success"] += [bool(meta["success"])] * T
        cols["task_index"] += [0] * T
        cols["index"] += list(range(n, n + T))
        eps.append({"episode_index": e, "length": T, "tasks": [TASK], "data/chunk_index": 0, "data/file_index": 0})
        batch = os.path.relpath(os.path.dirname(os.path.dirname(m)), ROOT)
        batch = "sim/out/expert_test" if "expert_test" in m else batch
        off = meta.get("box_offset_dx_dy_dz_yaw", [0, 0, 0, 0])
        extra.append({"episode_index": e, "name": batch, "status": "PASS" if meta["success"] else "FAIL",
                      "split_v1": "train" if batch in split["train"] else "val" if batch in split["val"] else "excluded",
                      "dart_noise_deg": meta.get("dart_noise_deg"),
                      "episode_extra": {"box_pose": {"dx_mm": off[0] * 1000, "dz_mm": off[2] * 1000, "yaw_deg": off[3]}},
                      "frames": T})
        n += T
    S, A = np.concatenate(st), np.concatenate(ac)
    vec = lambda x: pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), 16)
    table = pa.table({"observation.state": vec(S), "action": vec(A), **{k: pa.array(v) for k, v in cols.items()}})
    pq.write_table(table, os.path.join(root, "data", "chunk-000", "file-000.parquet"))
    pq.write_table(pa.Table.from_pylist(eps), os.path.join(root, "meta", "episodes", "chunk-000", "file-000.parquet"))
    pq.write_table(pa.table({"task_index": [0], "task": [TASK]}), os.path.join(root, "meta", "tasks.parquet"))
    with open(os.path.join(root, "meta", "fr3_episodes_extra.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in extra) + "\n")
    feat = {"dtype": "float32", "shape": [16], "names": NAMES}
    info = {"codebase_version": "v3.0", "robot_type": "fr3_duo", "fps": 30, "total_episodes": len(metas),
            "total_frames": int(n), "splits": {"train": f"0:{len(metas)}"},
            "features": {"observation.state": feat, "action": feat}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}
    json.dump(info, open(os.path.join(root, "meta", "info.json"), "w"), indent=2)
    print(f"wrote {root}: {len(metas)} episodes, {n} frames")


if __name__ == "__main__":
    main(sys.argv[1])
