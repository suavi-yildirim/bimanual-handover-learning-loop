"""Policy transforms for the FR3 Duo (2 x 7-DoF FR3 + Robotiq 2F-85), modelled on aloha_policy.py.

State/action layout (16): left q1-q7, left gripper, right q1-q7, right gripper. Joints in rad, gripper
command normalised (1 open, 0.48 pre-shape/hold, 0 closed). No joint flips or gripper conversion: the
values go to the model as recorded, so the action dims must not be remapped (unlike AlohaInputs).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms

ACTION_DIM = 16
CAMERAS = {  # model image slot -> key sent by the dataset / sim client
    "base_0_rgb": "external_image",
    "left_wrist_0_rgb": "left_wrist_image",
    "right_wrist_0_rgb": "right_wrist_image",
}


def make_fr3duo_example() -> dict:
    """Random input in the format the sim client sends to the policy server."""
    return {
        "state": np.zeros((ACTION_DIM,), dtype=np.float32),
        "images": {key: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8) for key in CAMERAS.values()},
        "prompt": "Hand the box from the left gripper to the right gripper and place it on the serving tray.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FR3DuoInputs(transforms.DataTransformFn):
    """Expected inputs: images {external_image, left_wrist_image, right_wrist_image}, state [16],
    actions [horizon, 16] (training only), prompt."""

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        missing = set(CAMERAS.values()) - set(in_images)
        if missing:
            raise ValueError(f"FR3DuoInputs: missing cameras {sorted(missing)}, got {sorted(in_images)}")

        inputs = {
            "image": {slot: _parse_image(in_images[key]) for slot, key in CAMERAS.items()},
            "image_mask": {slot: np.True_ for slot in CAMERAS},
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FR3DuoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # The model pads to action_dim 32; only the first 16 dims are ours.
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
