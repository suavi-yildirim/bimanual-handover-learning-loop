#!/usr/bin/env python3
"""Stage 02 Dataset Intelligence for LeRobot v3.0 datasets.

Usage:
  python check_dataset.py DATASET [DATASET ...] [--out reports] [--no-video] [--tmp DIR]

DATASET may be a .zip or a folder containing meta/info.json. Outputs per dataset:
  reports/<name>/report.html, report.json, episodes.csv, manifest.json, figures/*.pdf|png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsc import plots, report  # noqa: E402
from dsc.checks import Params, analyse  # noqa: E402
from dsc.io_lerobot import load  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "reports"))
    ap.add_argument("--no-video", action="store_true", help="skip frame sampling (faster)")
    ap.add_argument("--tmp", default=None, help="temp dir for video extraction")
    ap.add_argument("--idle-speed", type=float, default=Params.idle_speed)
    ap.add_argument("--dup-threshold", type=float, default=Params.dup_threshold)
    ap.add_argument("--robust-z", type=float, default=Params.robust_z)
    ap.add_argument("--seed", type=int, default=Params.seed)
    a = ap.parse_args(argv)
    p = Params(idle_speed=a.idle_speed, dup_threshold=a.dup_threshold, robust_z=a.robust_z, seed=a.seed)

    out_root = Path(a.out)
    done = []
    for path in a.datasets:
        t0 = time.time()
        ds = load(path)
        r = analyse(ds, p)
        if not a.no_video:
            from dsc.visual import image_stats, sample_frames
            r.thumbs = sample_frames(ds, tmp_dir=a.tmp)
            r.visual, r.visual_dist = image_stats(r.thumbs)
            vd = r.visual_dist + np.diag(np.full(len(r.visual_dist), np.inf))
            r.summary["visual_nn_median"] = float(np.median(vd.min(1)))
        out = out_root / ds.name
        figs = plots.make_all(ds, r, out / "figures")
        files = report.write(ds, r, out, figs)
        done.append((ds, r))
        print(f"[{r.verdict}] {ds.name}: {len(r.episodes)} episodes, {len(figs)} figures, "
              f"{time.time() - t0:.1f}s -> {files['html']}")
    if len(done) > 1:
        f = plots.fig_compare(done, out_root / "_comparison")
        print("comparison ->", out_root / "_comparison" / f)


if __name__ == "__main__":
    main()
