"""Publication-style figures (vector PDF + 300 dpi PNG) for the dataset check."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from .checks import ARM, FR3_Q_LIMITS, Result
from .io_lerobot import Dataset

# Okabe-Ito colour-blind safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
         sky="#56B4E9", yellow="#F0E442", grey="#666666", black="#000000")
VERDICT_C = {"keep": C["blue"], "review": C["orange"], "drop": C["red"]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlelocation": "left",
})


def _panel(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    return f"{name}.png"


def _hist(ax, x, bins=30, color=C["blue"], xlabel="", unit_fmt="{:.3g}"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if np.ptp(x) < 1e-12:
        w = max(abs(x[0]) * 0.01, 1e-3)
        ax.bar([x[0]], [len(x)], width=w, color=color)
        ax.set_xlim(x[0] - 30 * w, x[0] + 30 * w)
        ax.set_ylim(0, len(x) * 1.35)
        ax.text(0.5, 0.97, f"no variation: all {len(x)} = {unit_fmt.format(x[0])}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.5, color=C["red"])
    else:
        ax.hist(x, bins=bins, color=color, edgecolor="white", linewidth=0.3)
        m = np.median(x)
        ax.axvline(m, color=C["black"], lw=0.8, ls="--")
        ax.text(m, ax.get_ylim()[1] * 0.95, f" median {unit_fmt.format(m)}", fontsize=6.5, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("episodes")


def _aligned_axis(r: Result):
    n = r.aligned.shape[1]
    sp = r.params.seg_points
    ticks = [sp * (i + 1) for i in range(len(r.canonical_events))]
    return np.arange(n), ticks


def _mark_events(ax, r: Result, labels=True):
    _, ticks = _aligned_axis(r)
    for t, name in zip(ticks, r.canonical_events):
        ax.axvline(t, color=C["grey"], lw=0.5, ls=":")
        if labels:
            ax.text(t, 1.0, name, transform=ax.get_xaxis_transform(), rotation=90, fontsize=5.5,
                    va="top", ha="right", color=C["grey"])


# ------------------------------------------------------------------ figures
def fig_overview(ds: Dataset, r: Result, out: Path) -> str:
    ep = r.episodes
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.4))
    _hist(axs[0], ep.duration_s, xlabel="episode duration (s)")
    axs[0].set_title("Episode length")
    _hist(axs[1], 100 * ep.idle_frac, color=C["orange"], xlabel="idle frames (%)")
    axs[1].set_title(f"Idle share (speed < {r.params.idle_speed} rad/s)")
    _hist(axs[2], ep.longest_idle_s, color=C["orange"], xlabel="longest pause (s)")
    axs[2].set_title("Longest pause per episode")
    # phase durations
    tcols = [f"t_{n.replace(' ', '_')}" for n in r.canonical_events]
    T = ep[tcols].to_numpy(float)
    knots = np.c_[np.zeros(len(T)), T, ep.length.to_numpy() - 1] / ds.fps
    dur = np.diff(knots, axis=1)
    names = ["start"] + list(r.canonical_events)
    labels = [f"{a} → {b}" for a, b in zip(names, list(r.canonical_events) + ["end"])]
    med = np.nanmedian(dur, 0)
    lo, hi = np.nanpercentile(dur, 5, 0), np.nanpercentile(dur, 95, 0)
    y = np.arange(len(med))
    axs[3].barh(y, med, xerr=[med - lo, hi - med], color=C["sky"], ecolor=C["black"],
                error_kw=dict(lw=0.6, capsize=2))
    axs[3].set_yticks(y, labels, fontsize=6.5)
    axs[3].invert_yaxis()
    axs[3].set_xlabel("duration (s), median, whiskers 5-95%")
    axs[3].set_title("Sub-task phases (from gripper events)")
    for a, l in zip(axs, "abcd"):
        _panel(a, l)
    fig.suptitle(f"{ds.name}: {len(ep)} episodes, {len(ds.frames):,} frames @ {ds.fps:g} Hz", x=0.01, ha="left",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    return _save(fig, out, "fig01_overview")


def fig_trajectories(ds: Dataset, r: Result, out: Path) -> str:
    A = r.aligned
    x, _ = _aligned_axis(r)
    names = [ds.channel_names[j] for j in ARM]
    fig, axs = plt.subplots(2, 7, figsize=(13, 3.8), sharex=True)
    rng = np.random.default_rng(0)
    show = rng.choice(len(A), size=min(25, len(A)), replace=False)
    for k, ax in enumerate(axs.ravel()):
        y = np.degrees(A[:, :, k])
        for i in show:
            ax.plot(x, y[i], color=C["grey"], lw=0.25, alpha=0.5)
        med = np.median(y, 0)
        p5, p95 = np.percentile(y, [5, 95], 0)
        col = C["blue"] if k < 7 else C["green"]
        ax.fill_between(x, p5, p95, color=col, alpha=0.25, lw=0)
        ax.plot(x, med, color=col, lw=1.1)
        _mark_events(ax, r, labels=(k in (0, 7)))
        ax.set_title(names[k].replace("_", " "), fontsize=7.5)
        if k % 7 == 0:
            ax.set_ylabel("joint angle (deg)")
        if k >= 7:
            ax.set_xlabel("phase-aligned time")
        ax.set_xticks([])
    fig.suptitle("Arm joint trajectories, time-warped to gripper events (median, 5-95% band, 25 random episodes)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    return _save(fig, out, "fig02_joint_trajectories")


def fig_timing(ds: Dataset, r: Result, out: Path) -> str:
    ep = r.episodes
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 2.6))
    tcols = [f"t_{n.replace(' ', '_')}" for n in r.canonical_events]
    data = [ep[c].dropna().to_numpy() / ds.fps for c in tcols]
    bp = axs[0].boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                        flierprops=dict(markersize=2, markerfacecolor=C["grey"], markeredgewidth=0))
    for b in bp["boxes"]:
        b.set(facecolor=C["sky"], linewidth=0.6)
    axs[0].set_yticks(range(1, len(tcols) + 1), r.canonical_events)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("time in episode (s)")
    axs[0].set_title("Gripper event timing")
    sd = [np.std(d) for d in data]
    axs[0].text(0.98, 0.02, "sd: " + ", ".join(f"{s:.2f}s" for s in sd), transform=axs[0].transAxes,
                ha="right", fontsize=6)
    x, _ = _aligned_axis(r)
    axs[1].fill_between(x, 0, 100 * r.idle_profile.mean(0), color=C["orange"], alpha=0.6, lw=0)
    _mark_events(axs[1], r)
    axs[1].set_ylim(0, 105)
    axs[1].set_ylabel("episodes idle (%)")
    axs[1].set_xlabel("phase-aligned time")
    axs[1].set_xticks([])
    axs[1].set_title("Where the robot pauses")
    sp = r.speed_profile
    med = np.median(sp, 0)
    q1, q3 = np.percentile(sp, [25, 75], 0)
    axs[2].fill_between(x, q1, q3, color=C["blue"], alpha=0.25, lw=0)
    axs[2].plot(x, med, color=C["blue"])
    axs[2].axhline(r.params.idle_speed, color=C["red"], lw=0.6, ls="--")
    axs[2].set_yscale("symlog", linthresh=r.params.idle_speed)
    axs[2].set_ylim(bottom=0)
    _mark_events(axs[2], r, labels=False)
    axs[2].set_ylabel("max joint speed (rad/s)")
    axs[2].set_xlabel("phase-aligned time")
    axs[2].set_xticks([])
    axs[2].set_title("Speed profile (median, IQR)")
    k = np.arange(len(r.lag_curve))
    axs[3].plot(k, np.degrees(r.lag_curve), "o-", color=C["purple"], ms=3)
    b = int(np.nanargmin(r.lag_curve))
    axs[3].plot([b], [np.degrees(r.lag_curve[b])], "o", color=C["red"], ms=6, mfc="none")
    axs[3].set_xlabel("lag k (frames)")
    axs[3].set_ylabel("RMS(action[t] - state[t+k]) (deg)")
    axs[3].set_title(f"Action-state lag: best k = {b}")
    for a, l in zip(axs, "abcd"):
        _panel(a, l)
    fig.tight_layout()
    return _save(fig, out, "fig03_timing_and_pauses")


def fig_diversity(ds: Dataset, r: Result, out: Path) -> str:
    ep = r.episodes
    D = np.degrees(r.dist)
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1]))
    if np.ptp(D) > 1e-9:
        order = leaves_list(linkage(squareform(D, checks=False), "average"))
    else:
        order = np.arange(len(D))
    im = axs[0].imshow(D[np.ix_(order, order)], cmap="viridis", interpolation="nearest", aspect="auto",
                       vmin=0, vmax=max(D.max(), 1e-3))
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
    cb.set_label("RMS distance (deg)")
    axs[0].set_xlabel("episode (clustered)")
    axs[0].set_ylabel("episode (clustered)")
    axs[0].set_title("Pairwise trajectory distance")
    axs[0].grid(False)
    nn = np.degrees(ep.nn_dist_rad.to_numpy())
    thr = np.degrees(r.params.dup_threshold)
    if np.ptp(nn) < 1e-12:
        axs[1].bar([0], [len(nn)], color=C["red"], width=0.5)
        axs[1].set_xticks([0], [f"{nn[0]:.2g}"])
        axs[1].text(0.5, 0.8, f"every episode has an\nidentical twin\n({r.summary['distinct_motions']} distinct motion)",
                    transform=axs[1].transAxes, ha="center", fontsize=7, color=C["red"])
        axs[1].set_xlabel("NN distance (deg)")
    else:
        bins = np.logspace(np.log10(max(nn[nn > 0].min(), 1e-4) * 0.8), np.log10(nn.max() * 1.2), 30)
        axs[1].hist(np.maximum(nn, bins[0]), bins=bins, color=C["blue"], edgecolor="white", lw=0.3)
        axs[1].set_xscale("log")
        axs[1].axvline(thr, color=C["red"], ls="--", lw=0.8)
        axs[1].text(thr, axs[1].get_ylim()[1] * 0.9, " duplicate\n threshold", color=C["red"], fontsize=6)
        axs[1].set_xlabel("nearest-neighbour distance (deg, log)")
    axs[1].set_ylabel("episodes")
    axs[1].set_title("Redundancy")
    flat = r.aligned.reshape(len(r.aligned), -1)
    Xc = flat - flat.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) if np.abs(Xc).max() > 1e-12 else np.zeros(1)
    if s.sum() > 0:
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        for v, col in VERDICT_C.items():
            m = (ep.verdict == v).to_numpy()
            if m.any():
                axs[2].scatter(np.degrees(Z[m, 0]), np.degrees(Z[m, 1]), s=6, color=col, label=f"{v} ({m.sum()})",
                               lw=0)
        axs[2].legend(loc="best", markerscale=2)
        ev = S ** 2 / (S ** 2).sum()
        axs[2].set_xlabel(f"PC1 ({ev[0]:.0%} var., deg)")
        axs[2].set_ylabel(f"PC2 ({ev[1]:.0%} var., deg)")
        cum = np.cumsum(ev)
        kk = np.arange(1, min(30, len(cum)) + 1)
        axs[3].plot(kk, 100 * cum[: len(kk)], "o-", ms=2.5, color=C["green"])
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        axs[3].axhline(90, color=C["grey"], ls=":", lw=0.6)
        axs[3].axvline(n90, color=C["red"], ls="--", lw=0.6)
        axs[3].text(n90, 50, f" {n90} PCs\n explain 90%", color=C["red"], fontsize=6.5)
        axs[3].set_ylim(0, 102)
    else:
        for a in axs[2:]:
            a.text(0.5, 0.5, "zero variance:\nall trajectories identical", ha="center", va="center",
                   transform=a.transAxes, color=C["red"])
            a.set_xticks([])
            a.set_yticks([])
    axs[2].set_title("PCA of trajectories")
    axs[3].set_xlabel("principal components")
    axs[3].set_ylabel("cumulative variance (%)")
    axs[3].set_title("Intrinsic dimensionality")
    for a, l in zip(axs, "abcd"):
        _panel(a, l)
    fig.tight_layout()
    return _save(fig, out, "fig04_motion_diversity")


def fig_quality(ds: Dataset, r: Result, out: Path) -> str:
    ep = r.episodes
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1, 1.2, 1]))
    srt = ep.sort_values("outlier_score", ascending=False)
    cols = [VERDICT_C[v] for v in srt.verdict]
    axs[0].bar(np.arange(len(srt)), srt.outlier_score, color=cols, width=1.0, lw=0)
    axs[0].axhline(r.params.robust_z, color=C["red"], ls="--", lw=0.7)
    axs[0].set_xlabel("episodes (sorted)")
    axs[0].set_ylabel("max |robust z| over features")
    axs[0].set_title("Outlier score per episode")
    axs[0].legend(handles=[Patch(color=c, label=f"{v} ({(ep.verdict == v).sum()})") for v, c in VERDICT_C.items()],
                  loc="upper right")
    for v, c in VERDICT_C.items():
        m = ep.verdict == v
        if m.any():
            axs[1].scatter(100 * ep.idle_frac[m], ep.rms_jerk[m], s=6, color=c, lw=0, label=v)
    axs[1].set_xlabel("idle frames (%)")
    axs[1].set_ylabel("RMS jerk (rad/s³)")
    axs[1].set_title("Smoothness vs idling")
    # joint range usage vs FR3 limits
    lim = np.r_[FR3_Q_LIMITS, FR3_Q_LIMITS]
    span = lim[:, 1] - lim[:, 0]
    lo = (r.joint_range["min"] - lim[:, 0]) / span
    hi = (r.joint_range["max"] - lim[:, 0]) / span
    p1 = (r.joint_range["p01"] - lim[:, 0]) / span
    p99 = (r.joint_range["p99"] - lim[:, 0]) / span
    y = np.arange(14)
    axs[2].barh(y, hi - lo, left=lo, color=C["sky"], height=0.6, label="min-max")
    axs[2].barh(y, p99 - p1, left=p1, color=C["blue"], height=0.3, label="1-99%")
    axs[2].axvline(0, color=C["red"], lw=0.8)
    axs[2].axvline(1, color=C["red"], lw=0.8)
    axs[2].set_yticks(y, [ds.channel_names[j].replace("_joint", " q") for j in ARM], fontsize=6.5)
    axs[2].invert_yaxis()
    axs[2].set_xlim(-0.05, 1.05)
    axs[2].set_xlabel("position within joint range (0 = lower, 1 = upper limit)")
    axs[2].set_title("Joint range used")
    axs[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    allm = np.r_[ep.min_joint_margin_deg_left, ep.min_joint_margin_deg_right]
    bins = np.linspace(allm.min() - 1, allm.max() + 1, 40)
    axs[3].hist(ep.min_joint_margin_deg_left, bins=bins, color=C["blue"], alpha=0.7, label="left arm")
    axs[3].hist(ep.min_joint_margin_deg_right, bins=bins, color=C["green"], alpha=0.7, label="right arm")
    axs[3].set_xlabel("min distance to a joint limit (deg)")
    axs[3].set_ylabel("episodes")
    axs[3].set_title("Safety margin to joint limits")
    axs[3].legend()
    for a, l in zip(axs, "abcd"):
        _panel(a, l)
    fig.tight_layout()
    return _save(fig, out, "fig05_quality_and_safety")


def fig_visual(ds: Dataset, r: Result, out: Path) -> str | None:
    if r.visual is None or not r.thumbs:
        return None
    vis, ep = r.visual, r.episodes
    fig = plt.figure(figsize=(12, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    ids = sorted(r.thumbs)
    pick = [ids[i] for i in np.linspace(0, len(ids) - 1, min(24, len(ids))).astype(int)]
    ncol = 6
    nrow = int(np.ceil(len(pick) / ncol))
    h, w = 224, 224
    canvas = np.ones((nrow * (h + 4), ncol * (w + 4), 3), np.uint8) * 255
    for i, e in enumerate(pick):
        im = r.thumbs[e][0]
        if im is None:
            continue
        rr, cc = divmod(i, ncol)
        canvas[rr * (h + 4): rr * (h + 4) + h, cc * (w + 4): cc * (w + 4) + w] = im[:h, :w]
    ax0.imshow(canvas)
    ax0.axis("off")
    ax0.set_title(f"First frame, head camera ({len(pick)} episodes, evenly spaced)")
    ax1 = fig.add_subplot(gs[1])
    sc = ax1.scatter(vis.brightness, vis.rb_ratio, s=8, c=vis.index, cmap="viridis", lw=0)
    ax1.set_xlabel("mean luminance (0-1)")
    ax1.set_ylabel("R/B ratio (warm > 1 > cool)")
    ax1.set_title("Lighting coverage")
    fig.colorbar(sc, ax=ax1, fraction=0.05, pad=0.02).set_label("episode")
    ax2 = fig.add_subplot(gs[2])
    vd = r.visual_dist + np.diag(np.full(len(r.visual_dist), np.inf))
    vnn = vd.min(1)
    mnn = np.degrees(ep.nn_dist_rad.reindex(vis.index).to_numpy())
    ax2.scatter(vnn, mnn, s=8, color=C["purple"], lw=0)
    ax2.set_xlabel("visual NN distance (grey level RMS)")
    ax2.set_ylabel("motion NN distance (deg)")
    ax2.set_title("What actually varies")
    ax2.text(0.97, 0.95, f"visual median {np.median(vnn):.3f}\nmotion median {np.median(mnn):.3g} deg",
             transform=ax2.transAxes, ha="right", va="top", fontsize=6.5)
    ax3 = fig.add_subplot(gs[3])
    x = ds.extra.set_index("episode_index") if ds.extra is not None else None
    ck = "randomization.requested.light.color_temperature_K"
    if x is not None and ck in x:
        kt = x[ck].reindex(vis.index)
        ax3.scatter(kt, vis.rb_ratio, s=8, color=C["red"], lw=0)
        rho, pv = spearmanr(kt, vis.rb_ratio)
        ax3.set_xlabel("requested colour temperature (K)")
        ax3.set_ylabel("measured R/B ratio")
        ax3.set_title("Randomisation check")
        ax3.text(0.97, 0.95, f"Spearman ρ = {rho:.2f}\np = {pv:.1e}", transform=ax3.transAxes, ha="right",
                 va="top", fontsize=6.5)
    else:
        ax3.hist(vis.brightness, bins=25, color=C["orange"], edgecolor="white", lw=0.3)
        ax3.set_xlabel("mean luminance (0-1)")
        ax3.set_ylabel("episodes")
        ax3.set_title("Brightness distribution")
    for a, l in zip([ax0, ax1, ax2, ax3], "abcd"):
        _panel(a, l)
    fig.tight_layout()
    return _save(fig, out, "fig06_visual_diversity")


def fig_generator_metrics(ds: Dataset, r: Result, out: Path) -> str | None:
    if ds.extra is None:
        return None
    x = ds.extra
    spec = [
        ("final_metrics.placement_offset_mm.horizontal", "placement offset (mm)", "Placement accuracy"),
        ("final_metrics.tilt_deg", "box tilt (deg)", "Final tilt"),
        ("carry_metrics.slip_max_mm", "max slip in gripper (mm)", "Carry slip"),
        ("carry_metrics.pad_f_right_awake.mean[0]", "right pad force, mean (N)", "Grip force"),
    ]
    spec = [s for s in spec if s[0] in x]
    ncol = int(np.ceil((len(spec) + 2) / 2))
    fig, axs = plt.subplots(2, ncol, figsize=(3.8 * ncol, 5.4))
    axs = axs.ravel()
    for extra_ax in axs[len(spec) + 2:]:
        extra_ax.axis("off")
    axs = axs[: len(spec) + 2]
    for ax, (col, lab, title) in zip(axs, spec):
        _hist(ax, x[col].astype(float), bins=20, color=C["green"], xlabel=lab)
        ax.set_title(title)
    # placement offset vs lighting: should be independent
    ax = axs[len(spec)]
    li = "randomization.requested.light.rect_intensity_scale"
    po = "final_metrics.placement_offset_mm.horizontal"
    if li in x and po in x:
        ax.scatter(x[li], x[po], s=8, color=C["blue"], lw=0)
        if np.ptp(x[po]) > 0:
            rr, pv = pearsonr(x[li], x[po])
            ax.text(0.97, 0.95, f"r = {rr:.2f}, p = {pv:.2f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5)
        ax.set_xlabel("light intensity scale")
        ax.set_ylabel("placement offset (mm)")
        ax.set_title("Outcome vs lighting")
    # randomisation factor ranges
    ax = axs[-1]
    cov = [c for c in r.coverage]
    y = np.arange(len(cov))
    ax.barh(y, [1 if c["varies"] else 0.04 for c in cov], color=[C["green"] if c["varies"] else C["red"] for c in cov])
    for i, c in enumerate(cov):
        ax.text(0.03 if not c["varies"] else 0.5, i, f"{c['min']:.3g} .. {c['max']:.3g} {c['unit']}",
                va="center", fontsize=5.5, ha="left" if not c["varies"] else "center",
                color="white" if c["varies"] else C["black"])
    ax.set_yticks(y, [c["factor"] for c in cov], fontsize=6)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_title("Variation factors (green = varied)")
    ax.grid(False)
    for a, l in zip(axs, "abcdefgh"):
        _panel(a, l)
    fig.suptitle("Organizer generator metadata (meta/fr3_episodes_extra.jsonl)", x=0.01, ha="left", fontsize=9,
                 y=1.0)
    fig.tight_layout()
    return _save(fig, out, "fig07_generator_metrics")


def fig_coverage(ds: Dataset, r: Result, out: Path) -> str | None:
    if ds.extra is not None:
        return None  # included in fig07
    cov = r.coverage
    fig, ax = plt.subplots(figsize=(4.5, 0.35 * len(cov) + 0.8))
    y = np.arange(len(cov))
    ax.barh(y, [1 if c["varies"] else 0.04 for c in cov], color=[C["green"] if c["varies"] else C["red"] for c in cov])
    for i, c in enumerate(cov):
        ax.text(0.5 if c["varies"] else 0.06, i, f"{c['min']:.3g} .. {c['max']:.3g} {c['unit']}", va="center",
                fontsize=6.5, ha="center" if c["varies"] else "left", color="white" if c["varies"] else C["black"])
    ax.set_yticks(y, [c["factor"] for c in cov])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title("Variation factors (green = varied, red = constant)")
    fig.tight_layout()
    return _save(fig, out, "fig07_coverage")


def fig_compare(results: list[tuple[Dataset, Result]], out: Path) -> str:
    names = [d.name.replace("fr3_duo_", "").replace("_lerobot3", "") for d, _ in results]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.6))
    vals = [
        ([r.summary["episodes"] for _, r in results], [r.summary["distinct_motions"] for _, r in results],
         "episodes", "Episodes vs distinct motions"),
    ]
    x = np.arange(len(names))
    b1 = axs[0].bar(x - 0.2, vals[0][0], 0.4, color=C["sky"], label="episodes")
    b2 = axs[0].bar(x + 0.2, vals[0][1], 0.4, color=C["blue"], label="distinct motions")
    axs[0].bar_label(b1, fontsize=6)
    axs[0].bar_label(b2, fontsize=6)
    axs[0].set_yscale("log")
    axs[0].set_ylim(0.7, max(vals[0][0]) * 4)
    axs[0].set_title(vals[0][3])
    axs[0].legend(loc="upper left", ncol=2)
    axs[1].bar(x, [100 * r.summary["idle_frac_mean"] for _, r in results], color=C["orange"])
    axs[1].set_title("Idle frames (%)")
    nnv = [np.degrees(r.summary["nn_dist_median"]) for _, r in results]
    bb = axs[2].bar(x, [max(v, 1e-4) for v in nnv], color=C["purple"])
    axs[2].bar_label(bb, labels=[f"{v:.2g}" for v in nnv], fontsize=6)
    axs[2].set_yscale("log")
    axs[2].set_title("Median NN distance (deg, log)")
    axs[3].bar(x, [r.summary["hours"] * 60 for _, r in results], color=C["green"])
    axs[3].set_title("Total demonstration time (min)")
    for a, l in zip(axs, "abcd"):
        a.set_xticks(x, names, rotation=20, ha="right", fontsize=6.5)
        _panel(a, l)
    fig.tight_layout()
    return _save(fig, out, "fig00_dataset_comparison")


def make_all(ds: Dataset, r: Result, out: Path) -> list[str]:
    figs = [fig_overview(ds, r, out), fig_trajectories(ds, r, out), fig_timing(ds, r, out),
            fig_diversity(ds, r, out), fig_quality(ds, r, out), fig_visual(ds, r, out),
            fig_generator_metrics(ds, r, out), fig_coverage(ds, r, out)]
    return [f for f in figs if f]
