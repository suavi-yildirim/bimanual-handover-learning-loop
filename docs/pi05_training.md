# C2 "Teach the Robot": Fine-tuning pi0.5 (openpi) on the FR3 Duo handover data

Process flow, v2, 23 Oct 2026

**Legend**
- **[VERIFIED]** checked against the openpi repo (README, `config.py`, examples) on 23 Oct
- **[OUR DATA]** measured by our dataset_check tool (`dataset_check/reports/`)
- **[TO CONFIRM]** our inference, check on the workstation before relying on it

---

## 0. Summary

- **Train data for v0:** merge the two handover sets → 94 episodes, 211,218 frames, ~1.96 h, 30 Hz. Do **not** use the cube set (`fr3_duo_rand500`) for this model.
- **Filtering:** no episode needs to be dropped. We trim long idle pauses, drop the useless `success` column, and make a seeded train/val split.
- **Format:** openpi reads an older LeRobot format than ours (v3.0), so one conversion script merges, filters and rewrites the data in the format openpi expects.
- **Model:** pi0.5 base + LoRA (fits one 24 GB GPU). Template: openpi's `pi05_aloha` (bimanual, 3 cameras, joint space), with our own 16-dim policy transforms.
- **Expected v0 result:** works in the nominal scene, fails when the box moves (the data contains one single motion). That failure is the baseline; v1 adds generated demos with varied box pose.

---

## 1. Where this fits in our loop

| Stage | What we do |
|---|---|
| 01 Ingest & Transform | Conversion script (section 5) |
| 02 Dataset Intelligence | dataset_check reports (section 2), re-run on the converted set |
| 03 Fine-Tune | openpi LoRA training (sections 6–8) |
| 04 Evaluate | Policy server + Isaac Sim closed loop (section 8) |
| 05 Version & Serve | Checkpoint + norm stats + config + dataset hash in the registry |

> **Before training:** run the **replay test** (recorded actions played open-loop in Isaac Sim). It validates action space and timing. If replay does not complete the task, training will not either.

---

## 2. Which data to use [OUR DATA]

| Set | Episodes | Frames | Rate | Use for v0? |
|---|---:|---:|---|---|
| `fr3_duo_handover_v9_night_84` | 84 | 188,748 | 30 Hz | **Yes** |
| `FR3Duo_Handover_Dataset_lerobot3` | 10 | 22,470 | 30 Hz | **Yes** |
| `fr3_duo_rand500` (cube stacking) | 490 | 172,526 | 6 Hz | **No** (other task, other rate) |

**Why the two handover sets can be merged**
- Same task string, same robot, same 3 cameras (224×224), same 16-dim state/action.
- No duplicates: seeds 75000–75083 (night lighting) vs 71100–71109 (lighting "L1").
- Joint trajectories are bit-identical in all 94 episodes; only lighting differs.
- All 94 episodes: generator status PASS, dataset_check verdict "keep", no joint-limit issues (min margin 13.1° left, 37.7° right).

**Why the cube set is excluded (for now)**
- Different task, and 6 Hz vs 30 Hz; mixing would need resampling and a multi-task setup.
- 61 of 490 episodes are flagged "review" (jerk / action-rate outliers, robust z > 3.5).
- Its `info.json` split is `"0:41"` and silently drops 449 episodes; fix to `"0:490"` if ever used.
- Use it for offline experiments and the dataset-intelligence demo, not for the handover model.

**Honest limitation:** dataset_check's verdict is **NOT READY** for a policy that must generalise. There is 1 distinct motion, an identical start pose, a box pose that never varies, and one instruction. This is fine for a nominal-scene baseline and replay tests, but not enough for robustness.

---

## 3. Filtering rules (applied in the conversion script)

1. **Episodes:** keep all 94 (0 review, 0 drop).
2. **Idle trimming:** 17% of frames are idle (max arm joint speed < 0.02 rad/s **and** gripper command unchanged). For every idle run longer than 0.5 s, keep the first 0.25 s and the last 0.25 s and drop the middle.
   - The robot is static during idle, so this creates no jump in state.
   - Keeping the last 0.25 s matters: actions lead state by 4 frames, so motion onset is already visible in the actions at the end of a pause.
   - Note: `manifest.json` → `trim_idle` stores only the **start** frame of each run (≥ 1 s), so the script recomputes the runs with the rule above.
3. **Columns:** drop `success` (True on every frame, no information). Keep state, action, 3 images, task.
4. **Action:** use the dataset's `action` column as-is (the command; leads state by 4 frames). Channel order: left q1–q7, left gripper, right q1–q7, right gripper. The gripper command has 3 levels (0.0 / 0.48 / 1.0); keep it absolute, do not binarise.
5. **Split** (seeded, from our manifests): val = 9 episodes, train = 85. Merged numbering (84-set = 0..83, 10-set = 84..93): **val = 1, 3, 6, 21, 24, 40, 49, 65, 92**. Val covers both lighting regimes.
   - Caveat: val has the same motion as train, so offline val loss is only a sanity check. The real test is closed loop in sim.
6. **After conversion:** re-run dataset_check on the new set and keep its manifest (sha256) as the dataset version for the registry.

---

## 4. Set up openpi (Ubuntu workstation, NVIDIA GPU)

Repo: <https://github.com/Physical-Intelligence/openpi>

**Folder layout on the workstation** (used by every command below):

| Folder | Content |
|---|---|
| `~/openpi` | openpi repo clone. Its `.venv` is created here by `uv sync`. **All `uv run` commands run from here.** |
| `~/c2_data` | Our dataset zips (`fr3_duo_handover_84_ep.zip`, `FR3Duo_Handover_Dataset_lerobot3.zip`) |
| `~/.cache/openpi` | Downloaded base checkpoints (automatic) |
| `~/.cache/huggingface/lerobot/local/fr3duo_handover_v0` | Converted dataset, written by our script (LeRobot default location) [TO CONFIRM] |

**Run from: `~` (home folder)**, clone the repo:

```bash
cd ~
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

**Run from: `~/openpi`**, install dependencies:

```bash
cd ~/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

- **[VERIFIED] Requirements:** Python ≥ 3.11. GPU > 22.5 GB for LoRA (e.g. RTX 4090), > 70 GB for full fine-tuning (A100 80 GB / H100), > 8 GB for inference.
- **Pins:** `jax[cuda12]==0.5.3`, `torch==2.7.1`, LeRobot at commit `0cf8648` (PR #1127).
- **[VERIFIED]** The README says openpi is tested on Ubuntu 22.04 only. Our workstation is 24.04: if `uv sync` or JAX/CUDA fails, use the Docker setup in `docs/docker.md`.
- **Checkpoint [VERIFIED]:** downloaded automatically on first use from `gs://openpi-assets/checkpoints/pi05_base` into `~/.cache/openpi` (override with `OPENPI_DATA_HOME`). No manual download is needed, but the machine needs internet on the first run.

---

## 5. Step A: conversion script (merge + filter + rewrite)

**Why:** openpi's pinned LeRobot uses the module path `lerobot.common.datasets` (seen in openpi's own example), i.e. the older LeRobot dataset layout. Our zips are LeRobot v3.0.

**[TO CONFIRM]** the exact version. **Run from: `~/openpi`** (after `uv sync`, so `.venv` exists):

```bash
cd ~/openpi
grep -rn "CODEBASE_VERSION =" .venv/lib/python3.*/site-packages/lerobot/
```

It prints a line like `CODEBASE_VERSION = "v2.1"`. If it prints nothing, LeRobot was installed elsewhere; use this instead (same folder):

```bash
cd ~/openpi
uv run python -c "from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION; print(CODEBASE_VERSION)"
```

Expected: v2.x. Either way, rewriting with the pinned library is the safe path.

**Pattern:** copy `examples/libero/convert_libero_data_to_lerobot.py` [VERIFIED]:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset.create(repo_id=..., robot_type=..., fps=..., features={...},
                           image_writer_threads=10, image_writer_processes=5)
# for each frame:
ds.add_frame({..., "task": instruction})
# after each episode:
ds.save_episode()
```

**Our script** (to write, ~150 lines): `convert_fr3duo_to_openpi.py`, saved as `~/openpi/examples/fr3duo/convert_fr3duo_to_openpi.py`
1. Read both zips: parquet with pyarrow, AV1 video with PyAV (it has the dav1d decoder).
2. Merge the 94 episodes in the numbering of section 3.5.
3. Apply the filtering rules of section 3 (idle trim, drop `success`).
4. Write with `LeRobotDataset.create(...)`:
   - `repo_id = "local/fr3duo_handover_v0"`, `robot_type = "fr3_duo"`, `fps = 30`
   - features: `external_image`, `left_wrist_image`, `right_wrist_image` → dtype `"image"`, shape `(224, 224, 3)`; `state` → float32 `(16,)`; `actions` → float32 `(16,)`
5. Write the val episode list next to it. openpi trains on everything it loads, so keep the val episodes out of the training dataset or put them under a second `repo_id`.

**Run from: `~/openpi`**, so the openpi venv (and its pinned LeRobot) writes the files:

```bash
cd ~/openpi
uv run examples/fr3duo/convert_fr3duo_to_openpi.py --zips ~/c2_data/fr3_duo_handover_84_ep.zip ~/c2_data/FR3Duo_Handover_Dataset_lerobot3.zip
```

**Disk:** images are stored as PNG frames (3 cameras × ~180k frames); expect tens of GB. **[TO CONFIRM]** check free space first.

---

## 6. Step B: policy transforms for our robot

**Do NOT reuse `aloha_policy.py` as-is [VERIFIED].** It hard-codes 14 dims (`actions[:, :14]`, grippers at indices 6 and 13) and applies ALOHA joint flips and gripper conversions (`adapt_to_pi=True`). On our 16-dim FR3 data this would silently corrupt state and actions.

New file `~/openpi/src/openpi/policies/fr3duo_policy.py`, modelled on `~/openpi/src/openpi/policies/aloha_policy.py`:

| Class | Mapping |
|---|---|
| `FR3DuoInputs` | `external_image` → `base_0_rgb`; `left_wrist_image` → `left_wrist_0_rgb`; `right_wrist_image` → `right_wrist_0_rgb` (all masks True). Images as uint8 HWC; state = 16 floats; no joint flips, no gripper conversion. Pass `actions` through during training, `prompt` from the task. |
| `FR3DuoOutputs` | Return `actions[:, :16]` |

---

## 7. Step C: training config (add to `~/openpi/src/openpi/training/config.py`)

**Data config:** copy `LeRobotAlohaDataConfig` → `LeRobotFR3DuoDataConfig` and change:
- Repack keys to our feature names (`external_image`, `left_wrist_image`, `right_wrist_image`, `state`, `actions`).
- Inputs/outputs → `FR3DuoInputs` / `FR3DuoOutputs`.
- Delta actions for the joints, grippers absolute: `DeltaActions` / `AbsoluteActions` with `make_bool_mask(7, -1, 7, -1)`. ALOHA uses `make_bool_mask(6, -1, 6, -1)` for its 6-joint arms [VERIFIED].
- Default prompt: *"Hand the box from the left gripper to the right gripper and place it on the serving tray."*

**Train config:** openpi ships no pi0.5 LoRA config [VERIFIED]. This one combines the pi0 LoRA pattern (`pi0_libero_low_mem_finetune`) with pi05 settings. **[TO CONFIRM]** that it runs.

```python
TrainConfig(
    name="pi05_fr3duo_handover_lora",
    model=pi0_config.Pi0Config(
        pi05=True, action_horizon=30,                      # 30 steps = 1 s at 30 Hz
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"),
    data=LeRobotFR3DuoDataConfig(
        repo_id="local/fr3duo_handover_v0",
        base_config=DataConfig(prompt_from_task=True)),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora").get_freeze_filter(),
    ema_decay=None,
    batch_size=32,
    num_train_steps=30_000,
)
```

- **Reference values [VERIFIED]:** `pi05_libero` = batch 256, 30k steps, action_horizon 10 (LIBERO runs at 10 Hz, so 1 s). `pi05_aloha_pen_uncap` = batch 64, 20k steps.
- **Rate choice:** we keep 30 Hz with action_horizon 30 (also ~1 s). Alternative: subsample to 15 Hz with horizon 15; this halves data and compute but needs care with the 4-frame lag.
- **Scale:** 30k steps × batch 32 ≈ 5–6 passes over the trimmed train set.
- **If memory runs out:** lower `batch_size` to 16.

---

## 8. Step D: norm stats, train, serve, evaluate

[VERIFIED] commands. **All three run from: `~/openpi`.**

1. Normalisation stats:

```bash
cd ~/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_fr3duo_handover_lora
```

2. Training (long; start it inside `tmux` so it survives a closed terminal):

```bash
cd ~/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_fr3duo_handover_lora --exp-name=v0 --overwrite
```

3. Policy server (replace `<step>` with a saved step number, e.g. `29999`; list them with `ls ~/openpi/checkpoints/pi05_fr3duo_handover_lora/v0/`):

```bash
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_fr3duo_handover_lora --policy.dir=checkpoints/pi05_fr3duo_handover_lora/v0/<step>
```

- Recompute norm stats whenever data or config changes. Never reuse the cube set's `norm_stats.json`.
- Checkpoints go to `~/openpi/checkpoints/<config>/<exp-name>/<step>`. The policy server listens on port 8000.
- **[TO CONFIRM]** training logs to Weights & Biases by default; disable it in the config if we have no account.
- Start the run overnight; check the loss curve and a few val frames in the morning.

**Evaluation in Isaac Sim.** Our client calls the server with 3 images + state + prompt and gets back a chunk of 30 actions; it executes them, observes, and repeats.
- **Suites (fixed seeds):** nominal; new lighting; box moved (dx/dz ±1–3 cm, yaw ±10°).
- **Report:** success rate with 95% Wilson CI, progress per sub-task (L grasp → R grasp → L release → R release), placement offset and tilt, latency p50/p95.
- **Offline sanity check:** action error and gripper timing on the 9 val episodes.

---

## 9. After v0: the v1 step

- Generate ≥ 50 demos with varied box pose (dx/dz ±1–3 cm, yaw ±10°) and start pose (±3°), using the scripted IK expert in Isaac Sim, the organizers' generator (it has `box_pose` fields), or GR00T-Mimic.
- Run them through the same conversion + dataset_check, add them to the v0 data, and train v1 with the same config. Compare v0 vs v1 on the same seeded suites.
- Later: 5–10 instruction paraphrases; keep failed rollouts for the success judge (module A).

---

## 10. Risks and open items

- [ ] Workstation: NVIDIA GPU ≥ 24 GB, Ubuntu (22.04 tested; 24.04 → Docker fallback).
- [ ] Pinned LeRobot dataset version (expected v2.x; the conversion handles it either way).
- [ ] The custom pi0.5 LoRA config runs and fits in memory.
- [ ] Replay test passes in Isaac Sim (validates action space, timing, 4-frame lag).
- [ ] Disk space for the converted dataset (tens of GB).
- [ ] W&B logging on/off.
- [ ] pi0.5 base was pre-trained on other robots, so our FR3 Duo joint space is new to it. With 1 motion and ~2 h of data, expect a strong fit to the nominal scene only.
- [ ] Ask mentors: is the box-handover scene the judged task? Is the episode generator available for v1 data?

---

**Sources:** github.com/Physical-Intelligence/openpi (README; `src/openpi/training/config.py`; `src/openpi/policies/aloha_policy.py`; `examples/libero/convert_libero_data_to_lerobot.py`; `pyproject.toml`), checked 23 Oct 2026. Data facts: `dataset_check/reports/*`.
