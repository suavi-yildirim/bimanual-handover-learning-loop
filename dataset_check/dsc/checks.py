"""Stage 02 Dataset Intelligence: schema checks, per-episode quality, diversity, coverage, data plan."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io_lerobot import Dataset

# Joint limits of the sim robot (fr3v2 joints in Hackathon_FR3Duo_flattened.usd, deg -> rad). They reproduce the
# generator's joint_margin_deg exactly. Velocity limits: FR3 datasheet (not authored in the USD).
FR3_Q_LIMITS = np.radians([[-166.2, 166.2], [-105.2, 105.2], [-166.2, 166.2], [-176.3, -6.7],
                           [-164.8, 164.8], [25.2, 264.8], [-174.8, 174.8]])
FR3_DQ_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

ARM = [*range(0, 7), *range(8, 15)]   # 14 arm joints
GRIP = [7, 15]                          # left, right gripper
GRIP_NAMES = {7: "L", 15: "R"}


@dataclass
class Params:
    idle_speed: float = 0.02        # rad/s: max arm joint speed below this = idle
    min_idle_run_s: float = 1.0     # idle runs shorter than this are not reported as pauses
    dup_threshold: float = 1e-3     # rad RMS: phase-aligned trajectories closer than this are motion duplicates
    robust_z: float = 3.5           # outlier threshold on robust z-score
    seg_points: int = 25            # resample points per phase segment
    max_lag: int = 8                # frames searched for action -> state lag
    val_fraction: float = 0.1
    seed: int = 0


@dataclass
class Finding:
    id: str
    severity: str        # critical | warning | info | ok
    title: str
    detail: str
    fix: str = ""
    auto_fixable: bool = False


@dataclass
class Result:
    name: str
    params: Params
    findings: list = field(default_factory=list)
    episodes: pd.DataFrame | None = None
    canonical_events: tuple = ()
    aligned: np.ndarray | None = None       # (E, P, 14) phase-aligned arm trajectories
    aligned_knots: list = field(default_factory=list)
    dist: np.ndarray | None = None          # (E, E) RMS distance between aligned trajectories
    lag_curve: np.ndarray | None = None     # (max_lag+1,) RMS(action[t]-state[t+k])
    idle_profile: np.ndarray | None = None  # (E, P) idle flag on aligned time axis
    speed_profile: np.ndarray | None = None
    joint_range: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    coverage: list = field(default_factory=list)
    data_plan: list = field(default_factory=list)
    checklist: list = field(default_factory=list)
    verdict: str = ""
    verdict_reason: str = ""
    lineage: dict = field(default_factory=dict)
    visual: pd.DataFrame | None = None
    visual_dist: np.ndarray | None = None
    thumbs: dict = field(default_factory=dict)

    def add(self, *a, **k):
        self.findings.append(Finding(*a, **k))


# ---------------------------------------------------------------- helpers
def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-12:
        sd = np.nanstd(x)
        mad = sd if sd > 1e-12 else 1.0
    return (x - med) / mad


def gripper_events(g: np.ndarray, lo: float, hi: float) -> list[tuple[int, str]]:
    """Hysteresis threshold crossings. Convention: high = open."""
    events, is_open = [], g[0] > (lo + hi) / 2
    for t, v in enumerate(g):
        if is_open and v < lo:
            is_open = False
            events.append((t, "grasp"))
        elif not is_open and v > hi:
            is_open = True
            events.append((t, "release"))
    return events


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def resample(x: np.ndarray, n: int) -> np.ndarray:
    t0 = np.linspace(0, 1, len(x))
    t1 = np.linspace(0, 1, n)
    return np.stack([np.interp(t1, t0, x[:, j]) for j in range(x.shape[1])], 1)


def phase_align(x: np.ndarray, knots: list[int], seg_points: int) -> np.ndarray:
    parts = []
    for a, b in zip(knots[:-1], knots[1:]):
        seg = x[a:b + 1] if b > a else x[a:a + 2]
        parts.append(resample(seg, seg_points))
    return np.concatenate(parts)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def n_clusters(dist: np.ndarray, thr: float) -> np.ndarray:
    """Connected components of the graph dist < thr (single linkage)."""
    n = len(dist)
    lab = -np.ones(n, int)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = c
        while stack:
            j = stack.pop()
            for k in np.flatnonzero((dist[j] < thr) & (lab < 0)):
                lab[k] = c
                stack.append(k)
        c += 1
    return lab


# ---------------------------------------------------------------- main
def analyse(ds: Dataset, p: Params | None = None) -> Result:
    p = p or Params()
    r = Result(ds.name, p)
    fps = ds.fps
    sl = ds.episode_slices()
    ep_ids = sorted(sl)
    D = ds.state.shape[1]

    # ---------- schema / metadata
    info = ds.info
    n_ep, n_fr = len(ep_ids), len(ds.frames)
    if info.get("total_episodes") != n_ep:
        r.add("meta.total_episodes", "warning", "Episode count in info.json does not match data",
              f"info.json says {info.get('total_episodes')}, data has {n_ep}.", "Update info.json.", True)
    if info.get("total_frames") != n_fr:
        r.add("meta.total_frames", "warning", "Frame count in info.json does not match data",
              f"info.json says {info.get('total_frames')}, data has {n_fr}.", "Update info.json.", True)
    missing_ids = sorted(set(range(max(ep_ids) + 1)) - set(ep_ids))
    if missing_ids:
        r.add("meta.gaps", "info", "Episode indices are not contiguous", f"Missing: {missing_ids[:20]}")
    splits = info.get("splits", {})
    covered = set()
    for name, rng in splits.items():
        a, b = (int(x) for x in str(rng).split(":"))
        covered |= set(range(a, b))
    not_cov = [e for e in ep_ids if e not in covered]
    if not_cov:
        r.add("meta.splits", "critical", "Split definition drops episodes",
              f"splits = {splits} covers {len(set(ep_ids) & covered)} of {n_ep} episodes; "
              f"{len(not_cov)} episodes would be ignored by loaders that honour splits.",
              f'Set "splits": {{"train": "0:{max(ep_ids) + 1}"}} in a working copy.', True)
    if list(splits) == ["train"]:
        r.add("meta.no_val", "warning", "No held-out split",
              "Only a train split exists, so offline evaluation would test on training data.",
              f"Use the seeded {int(p.val_fraction * 100)}% validation split written to manifest.json.", True)
    if not isinstance(info["features"].get("observation.state", {}).get("names"), list):
        r.add("meta.names", "warning", "State/action channels have no names",
              "info.json has names = null; channel order must come from documentation.",
              "Add names (left_joint1..7, left_gripper, right_joint1..7, right_gripper).", True)
    tasks = [str(t) for t in (ds.tasks["task"] if "task" in ds.tasks.columns else ds.tasks.index)]
    if len(tasks) == 1:
        r.add("lang.single", "warning", "Only one language instruction",
              f'Every episode uses: "{tasks[0]}". A VLA cannot learn to follow different instructions from this.',
              "Add paraphrases and, for multi-task, more tasks.")
    vid_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    missing_vid = []
    for k in vid_keys:
        fi_col = f"videos/{k}/file_index"
        ci_col = f"videos/{k}/chunk_index"
        if fi_col not in ds.episodes_meta:
            missing_vid.append(k)
            continue
        for ci, fi in set(zip(ds.episodes_meta[ci_col], ds.episodes_meta[fi_col])):
            rel = info["video_path"].format(video_key=k, chunk_index=int(ci), file_index=int(fi))
            if not ds.source.exists(rel):
                missing_vid.append(rel)
    if missing_vid:
        r.add("video.missing", "critical", "Referenced video files are missing", f"{missing_vid[:5]}")

    # timestamps / frame indices
    ts_dev, bad_index = [], []
    for e in ep_ids:
        f = ds.frames.iloc[sl[e]]
        if not np.array_equal(f["frame_index"].to_numpy(), np.arange(len(f))):
            bad_index.append(e)
        ts_dev.append(np.max(np.abs(np.diff(f["timestamp"].to_numpy()) - 1 / fps)) if len(f) > 1 else 0)
    if bad_index:
        r.add("frames.index", "critical", "frame_index not contiguous", f"episodes {bad_index[:10]}")
    if max(ts_dev) > 0.5 / fps:
        r.add("frames.ts", "warning", "Timestamps not uniform", f"max deviation {max(ts_dev):.4f} s")

    finite = np.isfinite(ds.state).all(1) & np.isfinite(ds.action).all(1)
    if not finite.all():
        r.add("values.nonfinite", "critical", "NaN/Inf in state or action", f"{(~finite).sum()} frames")

    # success label
    succ = ds.frames["success"].astype(bool).to_numpy()
    if succ.all():
        r.add("label.success", "warning", "Success label carries no information",
              "success is True on every frame, including the first. It cannot be used to learn or score progress.",
              "Derive progress from sub-task events (this report) and judge success in simulation.")

    # constant channels
    const = [ds.channel_names[j] for j in range(D) if np.ptp(ds.state[:, j]) < 1e-6]
    if const:
        r.add("values.const", "warning", "Constant state channels", ", ".join(const))

    # gripper channel range (for event thresholds)
    g_lo, g_hi = {}, {}
    for j in GRIP:
        lo, hi = np.percentile(ds.action[:, j], [1, 99])
        # thresholds near the closed level: some generators release to a partly open "hold" level
        g_lo[j], g_hi[j] = lo + 0.2 * (hi - lo), lo + 0.4 * (hi - lo)
    binary_grip = all(len(np.unique(ds.action[:, j])) <= 2 for j in GRIP)
    levels = sorted({round(float(v), 2) for j in GRIP for v, c in zip(*np.unique(ds.action[:, j].round(2), return_counts=True))
                     if c > 0.02 * len(ds.action)})

    # ---------- per-episode features
    rows, events_all, aligned_src = [], {}, {}
    arm_lo = np.r_[FR3_Q_LIMITS[:, 0], FR3_Q_LIMITS[:, 0]]
    arm_hi = np.r_[FR3_Q_LIMITS[:, 1], FR3_Q_LIMITS[:, 1]]
    dq_lim = np.r_[FR3_DQ_LIMITS, FR3_DQ_LIMITS]
    lag_acc = np.zeros(p.max_lag + 1)
    for e in ep_ids:
        q = ds.state[sl[e]]
        a = ds.action[sl[e]]
        L = len(q)
        qa = q[:, ARM]
        v = np.diff(qa, axis=0) * fps
        speed = np.r_[np.abs(v).max(1), 0.0]
        dgrip = np.r_[np.abs(np.diff(a[:, GRIP], axis=0)).max(1), 0.0]
        idle = (speed < p.idle_speed) & (dgrip < 1e-3)
        idle_runs = [(s, t) for s, t in runs(idle) if (t - s) / fps >= p.min_idle_run_s]
        longest = max([(t - s) for s, t in runs(idle)] or [0])
        acc = np.diff(v, axis=0) * fps
        jerk = np.diff(acc, axis=0) * fps
        ev = []
        for j in GRIP:
            ev += [(t, f"{GRIP_NAMES[j]} {k}") for t, k in gripper_events(a[:, j], g_lo[j], g_hi[j])]
        ev.sort()
        events_all[e] = ev
        for k in range(p.max_lag + 1):
            lag_acc[k] += np.mean((a[: L - k, ARM] - q[k:, ARM]) ** 2) if k < L else np.nan
        margin = np.minimum(qa - arm_lo, arm_hi - qa)  # (L,14) rad
        da = np.abs(np.diff(a[:, ARM], axis=0)).max(1) * fps
        rows.append(dict(
            episode_index=e, length=L, duration_s=L / fps,
            idle_frac=float(idle.mean()), idle_s=float(idle.sum() / fps),
            longest_idle_s=longest / fps, n_pauses=len(idle_runs),
            pause_starts=";".join(str(s) for s, _ in idle_runs),
            path_len_rad=float(np.abs(np.diff(qa, axis=0)).sum()),
            peak_speed=float(speed.max()),
            speed_ratio_to_limit=float((np.abs(v) / dq_lim).max()) if len(v) else 0.0,
            rms_jerk=float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else 0.0,
            peak_action_rate=float(da.max()) if len(da) else 0.0,
            min_joint_margin_deg_left=float(np.degrees(margin[:, :7].min())),
            min_joint_margin_deg_right=float(np.degrees(margin[:, 7:].min())),
            joint_limit_violation=bool((margin < 0).any()),
            n_events=len(ev), event_seq=" | ".join(k for _, k in ev),
            event_frames=";".join(str(t) for t, _ in ev),
            success_frac=float(succ[sl[e]].mean()),
            success_first=bool(succ[sl[e]][0]),
            nonfinite=bool(~finite[sl[e]].all()),
            grip_out_of_range=bool(((a[:, GRIP] < -0.01) | (a[:, GRIP] > 1.01)).any()),
        ))
        aligned_src[e] = (qa, idle, speed)
    ep = pd.DataFrame(rows).set_index("episode_index")
    lag_curve = np.sqrt(lag_acc / len(ep_ids))
    r.lag_curve = lag_curve
    best_lag = int(np.nanargmin(lag_curve))

    # canonical event sequence
    seq_counts = Counter(tuple(k for _, k in events_all[e]) for e in ep_ids)
    canon, canon_n = seq_counts.most_common(1)[0]
    r.canonical_events = canon
    ep["canonical_seq"] = [tuple(k for _, k in events_all[e]) == canon for e in ep_ids]
    for i, name in enumerate(canon):
        ep[f"t_{name.replace(' ', '_')}"] = [events_all[e][i][0] if ep.loc[e, "canonical_seq"] else np.nan
                                            for e in ep_ids]

    # ---------- phase-aligned trajectories and distances
    Pn = p.seg_points * (len(canon) + 1)
    al, idle_al, speed_al = [], [], []
    for e in ep_ids:
        qa, idle, speed = aligned_src[e]
        L = len(qa)
        if ep.loc[e, "canonical_seq"]:
            knots = [0] + [t for t, _ in events_all[e]] + [L - 1]
            knots = list(np.maximum.accumulate(knots))
            al.append(phase_align(qa, knots, p.seg_points))
            idle_al.append(phase_align(idle[:, None].astype(float), knots, p.seg_points)[:, 0])
            speed_al.append(phase_align(speed[:, None], knots, p.seg_points)[:, 0])
        else:
            al.append(resample(qa, Pn))
            idle_al.append(resample(idle[:, None].astype(float), Pn)[:, 0])
            speed_al.append(resample(speed[:, None], Pn)[:, 0])
    A = np.stack(al)
    r.aligned, r.idle_profile, r.speed_profile = A, np.stack(idle_al), np.stack(speed_al)
    flat = A.reshape(len(A), -1)
    sq = (flat ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * flat @ flat.T, 0) / flat.shape[1])
    np.fill_diagonal(dist, 0)
    r.dist = dist
    d_off = dist + np.diag(np.full(len(dist), np.inf))
    ep["nn_dist_rad"] = d_off.min(1) if len(dist) > 1 else np.nan
    ep["nn_episode"] = [ep_ids[i] for i in d_off.argmin(1)]
    medoid = int(np.argmin(dist.sum(1)))
    ep["dist_to_medoid_rad"] = dist[medoid]
    labels = n_clusters(dist, p.dup_threshold)
    ep["motion_cluster"] = labels
    ep["motion_duplicate"] = ep["nn_dist_rad"] < p.dup_threshold
    q0 = np.stack([ds.state[sl[e]][0] for e in ep_ids])
    qT = np.stack([ds.state[sl[e]][-1] for e in ep_ids])

    # ---------- outliers and episode verdicts
    zcols = ["length", "idle_frac", "rms_jerk", "dist_to_medoid_rad", "peak_action_rate"]
    for c in zcols:
        ep[f"z_{c}"] = robust_z(ep[c].to_numpy())
    zmax = ep[[f"z_{c}" for c in zcols]].abs().max(1)
    ep["outlier_score"] = zmax
    verdicts, reasons = [], []
    for e in ep_ids:
        row = ep.loc[e]
        why = []
        if row.nonfinite:
            why.append("NaN/Inf")
        if row.joint_limit_violation:
            why.append("outside joint limits")
        if row.speed_ratio_to_limit > 1:
            why.append("exceeds FR3 velocity limit")
        if row.grip_out_of_range:
            why.append("gripper out of [0,1]")
        if not row.canonical_seq:
            why.append(f"unusual gripper sequence ({row.n_events} events)")
        hard = len(why) > 0
        soft = [f"{c} z={row[f'z_{c}']:+.1f}" for c in zcols if abs(row[f"z_{c}"]) > p.robust_z]
        if hard:
            verdicts.append("drop" if (row.nonfinite or row.joint_limit_violation) else "review")
        elif soft:
            verdicts.append("review")
        else:
            verdicts.append("keep")
        reasons.append("; ".join(why + soft))
    ep["verdict"] = verdicts
    ep["reason"] = reasons
    r.episodes = ep

    # ---------- dataset-level findings
    idle_mean = ep.idle_frac.mean()
    if idle_mean > 0.1:
        pause = ep.longest_idle_s.median()
        r.add("quality.idle", "warning", f"{idle_mean:.0%} of frames are idle",
              f"Arm speed < {p.idle_speed} rad/s and gripper unchanged. Median longest pause {pause:.1f} s. "
              "Imitation policies trained on this learn to stall.",
              "Trim idle runs longer than 0.5 s, or down-weight idle frames in the loss.", True)
    n_unique = len(set(labels))
    dup_frac = ep.motion_duplicate.mean()
    if dup_frac > 0.5:
        r.add("diversity.motion", "critical" if n_unique <= 2 else "warning",
              f"Motion diversity is very low: {n_unique} distinct motion(s) in {n_ep} episodes",
              f"{dup_frac:.0%} of episodes have a near-identical twin (phase-aligned RMS < {p.dup_threshold} rad). "
              f"Median NN distance {ep.nn_dist_rad.median():.2e} rad. The policy will see one trajectory and will "
              "not recover if the object or robot starts elsewhere.",
              "Generate demonstrations with varied object pose (and start pose) in Isaac Sim.")
    else:
        r.add("diversity.motion", "info", f"{n_unique} distinct motions in {n_ep} episodes",
              f"Median NN distance {ep.nn_dist_rad.median():.3f} rad, "
              f"median distance to medoid {ep.dist_to_medoid_rad.median():.3f} rad.")
    q0_spread = float(np.abs(q0[:, ARM] - np.median(q0[:, ARM], 0)).max())
    if q0_spread < 1e-3:
        r.add("diversity.start", "warning", "Robot start pose identical in every episode",
              f"Max deviation from median start pose {q0_spread:.1e} rad.",
              "Randomise the start pose (a few degrees) in generated demos.")
    ok_seq = ep.canonical_seq.mean()
    r.add("structure.events", "ok" if ok_seq == 1 else "warning",
          f"Gripper event sequence: {' -> '.join(canon)}",
          f"{canon_n}/{n_ep} episodes follow it ({ok_seq:.0%}). Used as sub-task boundaries for progress metrics.")
    r.add("timing.lag", "info", f"Action leads state by {best_lag} frame(s)",
          f"RMS(action[t]-state[t+k]) is minimal at k={best_lag} ({lag_curve[best_lag]:.2e} rad). "
          + ("Action = next state: an absolute joint-target action space." if best_lag == 1 and lag_curve[1] < 1e-4
             else "Action is a commanded target ahead of the measured state."))
    if not binary_grip and 2 < len(levels) <= 5:
        r.add("action.gripper_levels", "info", f"Gripper command uses {len(levels)} plateau levels: {levels}",
              "Open, pre-shape/hold and closed. Events are detected near the closed level.")
    if binary_grip:
        r.add("action.gripper", "info", "Gripper action is binary (0/1)",
              "Train with a classification-style or thresholded gripper head; do not average chunks across 0.5.")
    viol = int(ep.joint_limit_violation.sum())
    r.add("safety.limits", "critical" if viol else "ok",
          f"Joint limits: {viol} episodes outside the sim robot's joint limits",
          f"Min margin left {ep.min_joint_margin_deg_left.min():.1f} deg, right {ep.min_joint_margin_deg_right.min():.1f} deg. "
          "Limits from the scene USD (fr3v2 joints).")
    n_review = int((ep.verdict == "review").sum())
    n_drop = int((ep.verdict == "drop").sum())
    r.add("episodes.verdicts", "warning" if n_review + n_drop else "ok",
          f"Episodes: {n_ep - n_review - n_drop} keep, {n_review} review, {n_drop} drop",
          f"Robust z threshold {p.robust_z} on {', '.join(zcols)}.")

    # norm stats consistency
    for rel, ns in ds.norm_stats.items():
        try:
            nst = ns.get("norm_stats", ns)
            m = np.asarray(nst["state"]["mean"])[:D]
            dev = float(np.abs(m - ds.state.mean(0)).max())
            r.add(f"norm.{rel}", "ok" if dev < 1e-2 else "warning", f"{rel}: state mean matches data" if dev < 1e-2
                  else f"{rel}: state mean differs from data", f"max |diff| = {dev:.2e}",
                  "" if dev < 1e-2 else "Recompute norm stats.")
        except Exception as exc:  # unknown layout
            r.add(f"norm.{rel}", "info", f"{rel}: could not parse", str(exc))

    # ---------- metadata (organizer generator, if present)
    if ds.extra is not None:
        x = ds.extra
        if "frames" in x and np.any(x["frames"].to_numpy() != ep.length.reindex(x["episode_index"]).to_numpy()):
            r.add("extra.frames", "warning", "Frame counts in fr3_episodes_extra.jsonl disagree with data", "")
        if "status" in x:
            k = int((x["status"] == "PASS").sum())
            lo, hi = wilson(k, len(x))
            r.add("extra.status", "info", f"Generator status: {k}/{len(x)} PASS",
                  f"95% Wilson CI of generator success {lo:.2f}-{hi:.2f}. "
                  + ("No failed episodes are included, so the data cannot teach recovery or train a success classifier."
                     if k == len(x) else f"{len(x) - k} failed episode(s) included: exclude them from training, "
                     "keep them for evaluation or a success classifier."))
        lm = "joint_margin_deg.measured_min.left"
        if lm in x:
            ours = ep.min_joint_margin_deg_left.reindex(x["episode_index"]).to_numpy()
            d = x[lm].to_numpy() - ours
            off, sd = float(np.nanmedian(d)), float(np.nanstd(d))
            ok = abs(off) < 0.05 and sd < 0.05
            r.add("extra.margin_xcheck", "ok" if ok else "warning", "Cross-check: our joint-limit margin vs generator",
                  f"Generator margin - ours = {off:+.2f} deg (sd {sd:.2f}, left arm). "
                  + ("Identical: our limits match the generator's." if ok else "Differs: check the limits used."))

    # ---------- coverage of variation factors
    cov = []
    cov.append(dict(factor="robot start pose", unit="rad (max dev.)", min=0.0, max=q0_spread, varies=q0_spread > 1e-3))
    goal_spread = float(np.abs(qT[:, ARM] - np.median(qT[:, ARM], 0)).max())
    cov.append(dict(factor="robot end pose", unit="rad (max dev.)", min=0.0, max=goal_spread, varies=goal_spread > 1e-3))
    cov.append(dict(factor="motion (NN dist.)", unit="rad RMS", min=float(ep.nn_dist_rad.min()),
                    max=float(ep.nn_dist_rad.max()), varies=n_unique > 1))
    cov.append(dict(factor="language instruction", unit="count", min=1, max=len(tasks), varies=len(tasks) > 1))
    if ds.extra is not None:
        for col, name, unit in [
            ("randomization.requested.light.rect_intensity_scale", "light intensity (rect)", "x"),
            ("randomization.requested.light.dome_intensity_scale", "light intensity (dome)", "x"),
            ("randomization.requested.light.color_temperature_K", "colour temperature", "K"),
            ("randomization.requested.light.rect_tilt_deg[0]", "light tilt (axis 1)", "deg"),
            ("episode_extra.box_pose.dx_mm", "box position dx", "mm"),
            ("episode_extra.box_pose.dz_mm", "box position dz", "mm"),
            ("episode_extra.box_pose.yaw_deg", "box yaw", "deg"),
        ]:
            if col in ds.extra:
                v = ds.extra[col].astype(float)
                cov.append(dict(factor=name, unit=unit, min=float(v.min()), max=float(v.max()), varies=bool(v.std() > 1e-9)))
    r.coverage = cov
    gaps = [c["factor"] for c in cov if not c["varies"]]
    if gaps:
        r.add("coverage.gaps", "warning", f"Not varied in this dataset: {', '.join(gaps)}",
              "A policy has no evidence to generalise along these axes.")

    # ---------- VLA readiness checklist
    n_val = max(1, int(round(n_ep * p.val_fraction)))
    r.checklist = [
        ("Language instruction per episode", len(tasks) >= 1, f"{len(tasks)} unique"),
        ("Camera views", len(vid_keys) >= 1, f"{len(vid_keys)}: " + ", ".join(k.split('.')[-1] for k in vid_keys)),
        ("Image size", True, "x".join(str(s) for s in info["features"][vid_keys[0]]["shape"][:2]) if vid_keys else "-"),
        ("Control rate", fps >= 5, f"{fps:g} Hz" + (" (low: a 50-step chunk spans %.1f s)" % (50 / fps) if fps < 10 else "")),
        ("State/action dimension", True, f"{D} / {ds.action.shape[1]}"),
        ("Enough episodes (>= 50)", n_ep >= 50, str(n_ep)),
        ("Held-out split", list(splits) != ["train"], "manifest.json adds one" if list(splits) == ["train"] else "yes"),
        ("Split covers all episodes", not not_cov, f"{len(not_cov)} episodes outside splits"),
        ("Motion diversity", n_unique > max(3, 0.2 * n_ep), f"{n_unique} distinct"),
        ("Failure / recovery examples", not succ.all(), "none" if succ.all() else "present"),
        ("Normalisation stats", bool(ds.norm_stats) or ds.stats is not None,
         "openpi norm_stats.json" if ds.norm_stats else ("meta/stats.json" if ds.stats else "missing")),
        ("No NaN/Inf", bool(finite.all()), ""),
    ]

    # ---------- data plan (rule based)
    plan = []
    if not_cov:
        plan.append(("P0", "Fix split", f'Set splits to "0:{max(ep_ids) + 1}" before training (loaders would use only {len(set(ep_ids) & covered)} episodes).'))
    if n_unique <= max(3, 0.2 * n_ep):
        plan.append(("P0", "Add motion diversity",
                     "Generate demos with box pose dx/dz in +-1..3 cm and yaw +-10 deg (and start pose +-3 deg). "
                     "Target >= 50 new episodes for a first v1; hold out a seeded test grid."))
    if idle_mean > 0.1:
        plan.append(("P1", "Trim idle frames", f"Remove idle runs > 0.5 s (currently {idle_mean:.0%} of frames)."))
    if succ.all():
        plan.append(("P1", "Get failure labels", "Run the policy in sim, keep failed rollouts with stage labels; use them for "
                     "evaluation and as hard negatives for a success classifier (module A)."))
    if len(tasks) == 1:
        plan.append(("P2", "Language augmentation", "Add 5-10 paraphrases of the instruction; test with unseen phrasing."))
    if list(splits) == ["train"]:
        plan.append(("P1", "Held-out set", f"Use the {n_val}-episode validation split in manifest.json for offline metrics."))
    if fps < 10:
        plan.append(("P2", "Check control rate", f"At {fps:g} Hz, set action horizon accordingly (e.g. 10-16 steps)."))
    r.data_plan = sorted(plan, key=lambda x: x[0])

    # ---------- verdict
    crit = [f for f in r.findings if f.severity == "critical"]
    crit_hard = [f for f in crit if not f.auto_fixable]
    if crit_hard:
        r.verdict = "NOT READY"
        r.verdict_reason = ("For a policy that must generalise: " + "; ".join(f.title for f in crit_hard)
                            + ". Usable for a nominal-scene baseline and replay tests.")
    elif crit or any(f.severity == "warning" for f in r.findings):
        r.verdict = "CONDITIONALLY READY"
        r.verdict_reason = "Usable after the auto-fixes in the manifest; see warnings."
    else:
        r.verdict = "READY"
        r.verdict_reason = "No blocking issues."

    r.joint_range = dict(min=ds.state[:, ARM].min(0), max=ds.state[:, ARM].max(0),
                         p01=np.percentile(ds.state[:, ARM], 1, 0), p99=np.percentile(ds.state[:, ARM], 99, 0))
    r.summary = dict(
        dataset=ds.name, source=str(ds.source.path.name), codebase_version=info.get("codebase_version"),
        robot=info.get("robot_type"), fps=fps, episodes=n_ep, frames=n_fr,
        hours=round(n_fr / fps / 3600, 3), tasks=tasks, cameras=vid_keys, state_dim=D,
        length_median=float(ep.length.median()), length_min=int(ep.length.min()), length_max=int(ep.length.max()),
        idle_frac_mean=float(idle_mean), distinct_motions=int(n_unique),
        effective_episodes_ratio=round(n_unique / n_ep, 3),
        nn_dist_median=float(ep.nn_dist_rad.median()), action_state_lag=best_lag,
        canonical_events=list(canon), canonical_share=float(ok_seq),
        verdict=r.verdict, keep=int((ep.verdict == "keep").sum()), review=n_review, drop=n_drop,
    )

    # ---------- lineage + split
    rng = np.random.default_rng(p.seed)
    val = sorted(rng.choice(ep_ids, size=n_val, replace=False).tolist())
    r.lineage = dict(
        source_file=str(ds.source.path), files={f: ds.source.sha256(f) for f in ds.data_files + ["meta/info.json"]},
        params=p.__dict__, split=dict(seed=p.seed, val=val,
                                      train=[e for e in ep_ids if e not in val and ep.loc[e, "verdict"] != "drop"]),
    )
    return r
