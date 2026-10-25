"""Read a LeRobot v3.0 dataset from a folder or directly from a .zip (no unzip needed)."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class Source:
    """Uniform file access for a dataset root inside a zip or a directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.is_zip = self.path.suffix == ".zip"
        if self.is_zip:
            self.zf = zipfile.ZipFile(self.path)
            names = self.zf.namelist()
            infos = [n for n in names if n.endswith("meta/info.json")]
            if not infos:
                raise FileNotFoundError(f"no meta/info.json in {path}")
            self.prefix = sorted(infos, key=len)[0][: -len("meta/info.json")]
            self._names = [n[len(self.prefix):] for n in names if n.startswith(self.prefix)]
        else:
            hits = sorted(self.path.rglob("meta/info.json"), key=lambda p: len(p.parts))
            if not hits:
                raise FileNotFoundError(f"no meta/info.json under {path}")
            self.root = hits[0].parent.parent
            self._names = [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()]

    @property
    def name(self) -> str:
        if self.is_zip:
            return self.prefix.strip("/") or self.path.stem
        return self.root.name

    def list(self, prefix: str = "", suffix: str = "") -> list[str]:
        return sorted(n for n in self._names if n.startswith(prefix) and n.endswith(suffix) and not n.endswith("/"))

    def exists(self, rel: str) -> bool:
        return rel in self._names

    def read(self, rel: str) -> bytes:
        if self.is_zip:
            return self.zf.read(self.prefix + rel)
        return (self.root / rel).read_bytes()

    def json(self, rel: str):
        return json.loads(self.read(rel))

    def parquet(self, rel: str, columns=None):
        return pq.read_table(io.BytesIO(self.read(rel)), columns=columns)

    def extract(self, rel: str, dest_dir: Path) -> Path:
        """Copy one member to dest_dir (needed for video decoding)."""
        dest = Path(dest_dir) / rel.replace("/", "__")
        if self.is_zip:
            with self.zf.open(self.prefix + rel) as fsrc, open(dest, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 1 << 20)
        else:
            shutil.copy(self.root / rel, dest)
        return dest

    def sha256(self, rel: str) -> str:
        return hashlib.sha256(self.read(rel)).hexdigest()


def _vec_column(table, name: str) -> np.ndarray:
    col = table.column(name).combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float64).reshape(len(col), -1)


@dataclass
class Dataset:
    name: str
    source: Source
    info: dict
    tasks: pd.DataFrame
    episodes_meta: pd.DataFrame
    state: np.ndarray            # (N, D)
    action: np.ndarray           # (N, D)
    frames: pd.DataFrame         # episode_index, frame_index, timestamp, success, task_index, index
    extra: pd.DataFrame | None = None   # flattened meta/fr3_episodes_extra.jsonl if present
    extra_raw: list = field(default_factory=list)
    conversion: dict | None = None
    norm_stats: dict = field(default_factory=dict)
    stats: dict | None = None
    data_files: list = field(default_factory=list)

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def channel_names(self) -> list[str]:
        names = self.info["features"].get("observation.state", {}).get("names")
        if isinstance(names, list) and len(names) == self.state.shape[1]:
            return names
        # documented order for fr3_duo (see CLAUDE.md)
        side = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
        return [f"left_{s}" for s in side] + [f"right_{s}" for s in side]

    def episode_slices(self) -> dict[int, slice]:
        ep = self.frames["episode_index"].to_numpy()
        out = {}
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
        ends = np.r_[starts[1:], len(ep)]
        for s, e in zip(starts, ends):
            out[int(ep[s])] = slice(int(s), int(e))
        return out


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list) and len(v) <= 4 and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                out[f"{key}[{i}]"] = x
        elif isinstance(v, list):
            out[key] = len(v) if len(v) > 4 else str(v)
        else:
            out[key] = v
    return out


def load(path: str | Path) -> Dataset:
    src = Source(path)
    info = src.json("meta/info.json")
    tasks = src.parquet("meta/tasks.parquet").to_pandas()
    ep_files = src.list("meta/episodes/", ".parquet")
    epm = pd.concat([src.parquet(f).to_pandas() for f in ep_files], ignore_index=True)
    epm = epm[[c for c in epm.columns if not c.startswith("stats/")]]

    data_files = src.list("data/", ".parquet")
    states, actions, frames = [], [], []
    scalar_cols = ["episode_index", "frame_index", "timestamp", "success", "task_index", "index"]
    for f in data_files:
        t = src.parquet(f)
        states.append(_vec_column(t, "observation.state"))
        actions.append(_vec_column(t, "action"))
        cols = [c for c in scalar_cols + ["raw_state_row", "raw_action_row"] if c in t.schema.names]
        frames.append(t.select(cols).to_pandas())
    frames = pd.concat(frames, ignore_index=True)
    state = np.concatenate(states)
    action = np.concatenate(actions)
    order = np.lexsort((frames["frame_index"].to_numpy(), frames["episode_index"].to_numpy()))
    if not np.all(order == np.arange(len(order))):
        frames = frames.iloc[order].reset_index(drop=True)
        state, action = state[order], action[order]
    if frames["success"].dtype == object:
        frames["success"] = frames["success"].map(lambda x: bool(np.asarray(x).ravel()[0]))

    ds = Dataset(src.name, src, info, tasks, epm, state, action, frames, data_files=data_files)

    if src.exists("meta/fr3_episodes_extra.jsonl"):
        rows = [json.loads(l) for l in src.read("meta/fr3_episodes_extra.jsonl").decode().splitlines() if l.strip()]
        ds.extra_raw = rows
        ds.extra = pd.DataFrame([_flatten(r) for r in rows])
    if src.exists("meta/fr3_conversion.json"):
        ds.conversion = src.json("meta/fr3_conversion.json")
    if src.exists("meta/stats.json"):
        ds.stats = src.json("meta/stats.json")
    for rel in src.list("", "norm_stats.json"):
        ds.norm_stats[rel] = src.json(rel)
    return ds
