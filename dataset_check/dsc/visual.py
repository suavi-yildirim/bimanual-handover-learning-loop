"""Visual diversity: sample frames per episode from the (AV1) videos and compute simple image statistics."""
from __future__ import annotations

import tempfile
from pathlib import Path

import av
import numpy as np
import pandas as pd

from .io_lerobot import Dataset


def sample_frames(ds: Dataset, camera: str = "observation.images.external_camera",
                  rel_times=(0.0, 0.5), max_episodes: int | None = None, tmp_dir: str | None = None):
    """Return {episode: [HxWx3 uint8 per rel_time]}. Decodes each video file once, sequentially."""
    em = ds.episodes_meta.set_index("episode_index").sort_index()
    if max_episodes:
        em = em.iloc[np.linspace(0, len(em) - 1, max_episodes).astype(int)]
    ci, fi = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
    t0c, t1c = f"videos/{camera}/from_timestamp", f"videos/{camera}/to_timestamp"
    out: dict[int, list] = {}
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        for (c, f), grp in em.groupby([ci, fi]):
            rel = ds.info["video_path"].format(video_key=camera, chunk_index=int(c), file_index=int(f))
            path = ds.source.extract(rel, Path(td))
            # wanted timestamps -> (episode, slot)
            want = []
            for e, row in grp.iterrows():
                dur = row[t1c] - row[t0c] - 1 / ds.fps
                for k, rt in enumerate(rel_times):
                    want.append((row[t0c] + rt * max(dur, 0), int(e), k))
            want.sort()
            got: dict[tuple, np.ndarray] = {}
            with av.open(str(path)) as cont:
                st = cont.streams.video[0]
                st.thread_type = "AUTO"
                wi = 0
                for frame in cont.decode(st):
                    t = float(frame.pts * st.time_base)
                    while wi < len(want) and t + 0.5 / ds.fps >= want[wi][0]:
                        got[(want[wi][1], want[wi][2])] = frame.to_ndarray(format="rgb24")
                        wi += 1
                    if wi >= len(want):
                        break
            path.unlink()
            for e in grp.index:
                out[int(e)] = [got.get((int(e), k)) for k in range(len(rel_times))]
    return out


def image_stats(frames: dict[int, list]) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-episode statistics of the first sampled frame, plus pairwise visual distance (16x16 grey, 0..1)."""
    rows, vecs, ids = [], [], []
    for e, fl in sorted(frames.items()):
        im = fl[0]
        if im is None:
            continue
        x = im.astype(np.float64) / 255.0
        lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        r, g, b = (x[..., i].mean() for i in range(3))
        rows.append(dict(episode_index=e, brightness=lum.mean(), contrast=lum.std(),
                         mean_r=r, mean_g=g, mean_b=b, rb_ratio=r / max(b, 1e-6)))
        h, w = lum.shape
        small = lum[: h - h % 16, : w - w % 16].reshape(16, h // 16, 16, w // 16).mean((1, 3))
        vecs.append(small.ravel())
        ids.append(e)
    V = np.stack(vecs)
    sq = (V ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * V @ V.T, 0) / V.shape[1])
    return pd.DataFrame(rows).set_index("episode_index"), dist
