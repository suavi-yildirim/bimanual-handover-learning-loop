# pi05: pi0.5 + LoRA fine-tune on the FR3 Duo handover (Stages 01 and 03)

Implements `../C2_pi05_Training_Process.md`. The sources live here and are linked into `~/openpi`.

| File | Role | Linked into openpi as |
|---|---|---|
| `setup_openpi.sh` | Clone, `uv sync`, link files, apply patch (idempotent) | |
| `convert_fr3duo_to_openpi.py` | Merge 84+10 handover episodes, idle trim, split, write LeRobot v2.1 | `examples/fr3duo/` |
| `fr3duo_policy.py` | `FR3DuoInputs` / `FR3DuoOutputs` (16 dims, 3 cameras, no flips) | `src/openpi/policies/` |
| `openpi_fr3duo_config.patch` | `LeRobotFR3DuoDataConfig` + `pi05_fr3duo_handover_lora` in `config.py` | applied with `git apply` |
| `train_v0.sh` | Norm stats, then training detached with a log | |

## Run (workstation, from anywhere)
```bash
pi05/setup_openpi.sh
cd ~/openpi && ~/.local/bin/uv run examples/fr3duo/convert_fr3duo_to_openpi.py \
    --sources <Dataset>/fr3_duo_handover_84_ep.zip <Dataset>/FR3Duo_Handover_Dataset_lerobot3.zip
pi05/train_v0.sh smoke --num-train-steps=20     # memory / pipeline check
pi05/train_v0.sh v0                             # full run, log in ~/openpi/checkpoints/pi05_fr3duo_handover_lora/v0.log
```
Outputs: `~/.cache/huggingface/lerobot/local/fr3duo_handover_v0{,_val}` (conversion log in `meta/fr3duo_conversion.json`),
`~/openpi/assets/pi05_fr3duo_handover_lora/` (norm stats), `~/openpi/checkpoints/pi05_fr3duo_handover_lora/<exp>/<step>`.

## Verified on the workstation (23 Oct, Ubuntu 24.04, RTX 4090 24 GB)
- `uv sync` works natively (no Docker). Pinned LeRobot is v2.1, JAX 0.5.3 sees the GPU, PyAV 17 decodes AV1 (libdav1d).
- Converted frames match the source state exactly; images and task string come through.
- Transforms: joints become deltas, grippers stay absolute; output round trip error 2e-7.
- Idle trim keeps 2,077 of 2,247 frames per episode (7.6% removed; short pauses and the 0.25 s ends stay).
- W&B is off (`wandb_enabled=False`).

## Provided checkpoint `fr3-duo-checkpoint-6a5f7c87` (config `pi05_fr3duo_6a5f7c87`)
Extracted at `~/openpi/checkpoints/fr3duo_6a5f7c87` (params + norm stats under asset id `fr3_duo`; tar also has train_state).
Who trained it and with which config is to confirm; inferred from weights and stats:
- pi0.5 + LoRA (rank 16 on gemma_2b, 32 on gemma_300m), 16-dim, **absolute** joint actions (no delta transform).
- **action_horizon 10**: offline arm MAE over the first 10 steps 0.38 deg at h=10, 0.45 at h=30, 0.55 at h=50.
- State stats match our handover data (max diff 0.07), so it was likely trained on the same set (all 94 episodes?
  then our val episodes are not held out for it; to confirm).

Offline result on our 9 val episodes (`pi05/eval_offline.py`, outputs in `~/openpi/eval_offline/`):
arm MAE 0.39 deg, gripper MAE 0.006, latency p50 95 ms / p95 117 ms per 10-step chunk (RTX 4090).
Swapping the wrist cameras raises arm MAE to 0.60 deg, so the camera mapping (external -> base, left/right wrist) is right.

## Policy server (checkpoint 6a5f7c87)
```bash
cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_fr3duo_6a5f7c87 --policy.dir=checkpoints/fr3duo_6a5f7c87
```
Cap the memory fraction: JAX takes 75% of the GPU by default (22.8 GB) and leaves nothing for Isaac Sim; 0.4 uses ~10 GB.
`pi05/test_policy_client.py` checks it over websocket: chunk (10, 16), arm MAE 0.27-0.53 deg on val frames,
round trip ~110 ms. The first request takes ~25 s (JIT compile), so send one warm-up request before a sim rollout.

## Inference contract (sim client -> policy server, port 8000)
`{"images": {"external_image", "left_wrist_image", "right_wrist_image"}: uint8 224x224x3, "state": float32[16], "prompt": str}`
-> `{"actions": float32[30, 16]}` absolute joint targets (rad) + gripper command (1 open, 0 closed), 30 Hz.
