# dataset_check: Stage 02 "Dataset Intelligence"

Answers the slide questions for any LeRobot v3.0 dataset: *Is it ready for training? Which episodes are
invalid, redundant or unrepresentative? What is missing? Is it VLA-ready?* Reads `.zip` files directly, so
there is no need to unzip.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python check_dataset.py ../C2_Teach-the-robot_VLA/Dataset/fr3_duo_handover_84_ep.zip \
    ../C2_Teach-the-robot_VLA/Dataset/fr3_duo_rand500.zip
```
Options: `--no-video` (skip frame sampling), `--out DIR`, `--tmp DIR` (video extraction), `--idle-speed`,
`--dup-threshold`, `--robust-z`, `--seed`. Runtime is about 40 s per dataset with video and about 1 s without.

## Outputs, per dataset under `reports/<name>/`
| File | Content |
|---|---|
| `report.html` | Verdict (READY / CONDITIONALLY READY / NOT READY), findings, VLA checklist, data plan, figures |
| `report.json` | The same, machine-readable (input for the loop agent) |
| `episodes.csv` | ~40 metrics per episode (idle share, pauses, gripper events, jerk, joint margins, NN distance, outlier z-scores, verdict) |
| `manifest.json` | keep/review/drop per episode, seeded train/val split, sha256 of the data files, auto-fixes, idle runs to trim |
| `figures/*.pdf, *.png` | Vector PDF and 300 dpi PNG figures |

`reports/_comparison/` compares datasets when more than one is given.

## Method
- **Schema:** info.json counts vs data, split coverage, channel names, timestamps, NaN/Inf, video files present, norm stats vs data.
- **Sub-tasks:** gripper grasp/release events (hysteresis near the closed level), canonical event sequence, phase durations.
- **Idle:** max arm joint speed < 0.02 rad/s and gripper command unchanged; pauses >= 1 s.
- **Action semantics:** lag k minimising RMS(action[t] - state[t+k]). k = 1 with error 0 means action = next state.
- **Diversity:** trajectories time-warped piecewise-linearly between gripper events, then pairwise RMS distance, nearest neighbour, duplicate clusters (< 1e-3 rad), PCA dimensionality.
- **Outliers:** robust z (median/MAD) on length, idle share, jerk, distance to medoid and action rate, with threshold 3.5. Hard rules: NaN, outside joint limits (scene USD, fr3v2) or FR3 datasheet velocity limits, unusual event sequence.
- **Visual:** head-camera frames at 0% and 50% of each episode. Luminance, R/B colour balance and a 16x16 grey-level distance. Requested colour temperature is checked against the measured R/B ratio.
- **Coverage:** which factors vary (start/end pose, motion, instruction, light, box pose from generator metadata).

## Limits
Joint limits come from the scene USD and reproduce the generator's margins exactly. Motion diversity uses the robot trajectory only: object poses are not stored in the cube set.
