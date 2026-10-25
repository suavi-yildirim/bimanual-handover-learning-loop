# Stage 02 follow-up: what the dataset reports asked for, and what we did (24 Oct)

The 22 Oct reports (`reports/<dataset>/report.html`) gave the handover data the verdict **NOT READY**
("1 distinct motion in 84 episodes"). On 24 Oct we ran the same checker on our own sim expert data
(`reports/fr3duo_sim_expert_v1/`, converted with `expert_to_lerobot.py`, state/action only, no video).
Side-by-side figure: `reports/_comparison_original_vs_expert/fig00_dataset_comparison.png`.

## Verdicts
| Dataset | Episodes | Distinct motions | Idle frames | Demo time | Verdict |
|---|---|---|---|---|---|
| Organisers' handover (84) | 84 | **1** | 17% | 105 min | **NOT READY** |
| Organisers' handover (10) | 10 | 1 | 17% | 12 min | NOT READY |
| Our sim expert (v1 data) | 18 (16 PASS, 2 FAIL) | 18 | 2% | 19 min | **CONDITIONALLY READY** |
| Cube stacking (rand500, other task) | 490 | 490 | 27% | 8 h | CONDITIONALLY READY (split fix) |

## The report's data plan for the handover set, item by item
| Prio | Recommendation | Status | Evidence |
|---|---|---|---|
| P0 | Add motion diversity: box dx/dz ±1–3 cm, yaw ±10°, start pose ±3°, ≥ 50 new episodes | **Partly done** | 16 good episodes with box ±15 mm and ±5°. Start pose not varied; 16 instead of ≥ 50 (6 min per demo in sim) |
| P1 | Trim idle frames (17%) | **Done** | Expert data 2% idle; the v0 conversion trims idle runs > 0.5 s |
| P1 | Get failure labels from sim rollouts | **Partly done** | 29 scored v0 rollouts with stage milestones (`results/runs_index.csv`) and 2 failed expert episodes; not yet used for training or a success classifier |
| P1 | Held-out set | **Done** | 9 val episodes for the original set; 2 held out for v1 (`sim/out/expert_split_v1.json`), fixed before training |
| P2 | Language augmentation | Not done | Still one instruction in every dataset |
| – | Success label carries no information | **Done for our data** | Expert data has a real per-episode success label (16 PASS, 2 FAIL) |

## What the check of our own data shows (new findings)
1. **The checker found the bad demos itself.** It marks 4 of 18 episodes for review: the 2 failed demos (an
   unusual gripper sequence and a short length), and two more, below. These are the same episodes we left out of training by hand.
2. **"18 distinct motions" looks better than it is.** Part of the difference between episodes is the training
   noise, not the box position:
   - executed motion: nearest neighbour 0.84°; clean labels: 0.50°
   - label distance vs box-offset distance: correlation **−0.08** (none) over the whole trajectory
   - at the grasp, the box pose explains only **12–43%** of the joint variation (R², 16 episodes)
   So v1 sees only a weak link between "where the box is" and "how to move". This is the most likely reason
   16 demos are not enough. The checker should add a "variation explained by the varied factor" test.
3. **The noise makes the executed motion about 100× jerkier**: jerk 33 vs 0.3 rad/s³ on the clean expert run,
   while the labels stay smooth (1.3). This is the shaking you see in the videos. Because most episodes are
   noisy, the checker marks the one **clean** episode (`expert_test`) as the outlier. That is a limit of
   outlier rules on small, uniform sets.
4. **One slow alignment**: `expert_demos_rt/w2/batch_01` is 60 frames longer (the right grasp took longer to align); it is fine to keep.
5. **Volume**: 19 min of expert demos vs 105 min of the organisers' demos. Diversity went up, quantity went down.
6. **Not yet checked**: start pose, instructions, and the images. Our videos are H.264 MP4 files outside the LeRobot layout, so the visual check was skipped.

## Fixed in the tool
`dsc/checks.py`: the generator-status finding used to say "No failed episodes are included" even when some
failed. It now reports the number of failed episodes and what to do with them.

## One-line message for the slide
"Our checker said the organisers' data was not ready (84 demos, 1 motion). Our expert data is conditionally ready
(18 motions, the bad demos found automatically). But the check also shows that the box position drives only a small part of
the motion. That is why v2 needs human demos (Quest 3), not more scripted ones."

Reproduce (from `dataset_check/`, venv with `requirements.txt`):
```bash
python expert_to_lerobot.py /tmp/expert_ds
python check_dataset.py /tmp/expert_ds/fr3duo_sim_expert_v1 --no-video
```
