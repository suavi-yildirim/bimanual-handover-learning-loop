# Genesis data engine: why it is faster, and why we use it

Challenge 2, box handover with FR3 Duo · 24 Oct 2026 · figure: `presentation/figures_v2/V14_genesis_engine.png`
(build: `python presentation/make_flowchart_genesis.py` from the repo root)

## 1. The problem it solves

Our v1 lesson (V13, step 10) was that we needed **more demos and more varied demos**. The Isaac Sim expert gave
16/18 successful demos at about **6 minutes each (~10 per hour)**, with the box moved only ±15 mm / ±5°. v1 had
14 training demos against 94 in the original set. That was too little for a fair v0 vs v1 test, and it left no time
for a second training round.

The Genesis generator produces about **2 demos per minute (~120 per hour, ~12× faster)** on the same handover task.

## 2. Where the time went, and what changed

| Cost driver | Isaac Sim v1 (`sim/expert_demos.py`) | Genesis generator | Effect |
|---|---|---|---|
| Rendering | RTX ray tracing, 3 cameras at 224 px, ~57 ms per update, 2 settle updates per image with physics paused. The scene default (path tracing, 64 spp) took 400–600 ms: 34 min per demo | OpenGL rasterizer (or software raycast headless), 256 px recordings, no ray tracing | Rendering was the main cost in Isaac. It is no longer the limit |
| Start-up | One full Omniverse Kit process plus hospital USD stage load **per episode** (`gen_demos.sh`), starts staggered by 20 s | Simulation kernels compiled once at first launch, then reused | Fixed cost per demo goes away |
| Expert design | Replays the one recorded 2,247-frame path in time. The clock stops at each grasp until the alignment assist reaches < 4 mm (up to 120 frames) | Plans from the **live box pose**: IK reachability check, approach above clutter, up to 3 grasp attempts, travel at 1.5× nominal speed (max 0.30 m/s), slow only near contact | Shorter episodes (about 45–49 s simulated) and no waiting on alignment |
| Parallelism | 3 workers shared one GPU renderer. More workers did **not** add throughput | Up to 12 simulations at once on a 24 GB GPU | Throughput grows with process count until GPU memory is full |
| Failures | Failed episodes found after the full run; 2/18 lost | Startup rejects bad layouts (overlap, no support, no clearance); collisions stop the task early; failed attempts kept for diagnosis, only successes admitted | Less wall time spent on attempts that cannot succeed |

**Main reason:** Isaac Sim is built for photoreal rendering, and our pipeline was GPU-bound by the renderer.
Genesis spends little on rendering and starts each episode cheaply, so the same GPU runs many generators at once.
The ~12× comes mostly from **parallel processes plus cheaper rendering**, not from a faster single episode.
To confirm: per-process wall time and the number of parallel generators behind the "2 per minute" figure.
This number should go on the slide next to the 6 min.

## 3. Variety, not only speed

| | Isaac v1 | Genesis |
|---|---|---|
| Box position | ±15 mm, ±5° around the recorded pose | Continuous XY over a 28 × 26 cm pickup region (`randomization_bounds`), seeded |
| Motion | One path plus DART noise (0.6°) | A new plan per layout, with regrasps when needed |
| Reproducibility | Seed per episode | Seed → identical layout; episode JSON stores the seed and initial object poses |
| Provenance | `meta.json` per episode | Asset hashes, camera calibration per frame, both-arm telemetry, all attempts in `genesis_provenance/attempt_XXXX` |

This addresses V13's "need richer demos": a policy only learns to handle box positions it has seen. Genesis covers a
region about 9× wider per axis than v1 (28 cm vs 3 cm), at 12× the rate.

## 4. Justification of the methodology

1. **Data-centric loop.** Our v0 analysis showed the failure was in the data (one motion, no corrections), not the
   model. When data is the bottleneck, demos per hour sets how fast the loop improves.
2. **Iteration speed.** At ~120 demos per hour, an overnight run (10 h) yields ~1,200 demos instead of ~100.
   That leaves time for several train → evaluate → DAgger rounds within the hackathon, which v1 could not do.
3. **Fit for purpose.** Photoreal rendering matters for sim-to-real transfer of images. It is not needed to prove the
   learning loop. We use the cheap simulator for the loop and keep Isaac Sim for the final visual check.
4. **Honest evaluation built in.** Frozen benchmark seeds (900000+) never used for collection, student-only runs,
   Wilson 95% intervals. This matches our Isaac evaluation rules.
5. **The expert can label any state.** A planning expert reads the live pose, so it can label states the student
   reaches by mistake. HG-DAgger needs exactly this, and the Isaac replay expert could not provide it.

## 5. Limits (state these in the pitch)

- **Different simulator, different images.** Genesis uses a rasterizer, Franka grippers instead of Robotiq, and its own
  wrist-camera mounts (`genesis_fr3_asymmetric_wrist_v3`). Models trained here do not transfer directly to the Isaac
  scene or to checkpoint 6a5f7c87. Never mix Genesis and Isaac data without labelling the source.
- **Physics shortcuts.** Self-collision is disabled; the room is only partly collision-enabled; box/bottle collision
  uses proxies. Friction and actuators are not calibrated against hardware.
- **Privileged state.** The expert uses simulator poses, not camera detection.
- **Validation is small.** 10/10 bounded smoke tests (up to 4 cm relocations), not a broad robustness study.
- **No learned success yet.** ACT on `fr3_duo_gray_v1` (25 workflows) knocks the box over during the pick (seed 900050).
  The cause is under diagnosis. No HG-DAgger round has finished.

## 6. The framework (same stages as our 01→05 loop)

| Stage | Genesis step | Tool / output | Status |
|---|---|---|---|
| 01 Ingest & generate | Seeded layouts → scripted planning expert → LeRobot export | `generate_genesis_dataset.sh`, `--robot fr3_duo`, `--training-format pi05` | working |
| 02 Dataset intelligence | Our Stage 02 checker per shard; split by whole workflow; refuse mixed scenes or speeds | `dataset_check/`, `run_act_training.sh prepare` | checker to run on Genesis shards |
| 03 Fine-tune | ACT (fast loader: 2.9 s → 39 ms per sample); pi0.5 LoRA via openpi | `run_act_training.sh train`, `docs/pi05_genesis_training.md` | ACT trained, pi0.5 export ready |
| 04 Evaluate | Frozen seeds, student only, gate logs where it would intervene, Wilson CI | `run_dagger_benchmark.sh` | ready |
| 05 Improve | HG-DAgger: student drives, teacher labels every tick, gate hands over on stall, phase budget, divergence or slip; merge rounds with `prepare --sources` | `control/hg_dagger.py` | round 1 pending |

**Next actions, in order:**
1. Find out why ACT fails the pick (benchmark seeds plus training seeds) before collecting more data.
2. Generate a large gray-box set (several hundred workflows) in parallel and run the Stage 02 checker on it.
3. Run HG-DAgger round 1 and benchmark v0 vs v1 on the frozen seeds with CIs.
4. Optional: replay a few Genesis-trained rollouts in Isaac Sim as a visual sim-to-sim check.
