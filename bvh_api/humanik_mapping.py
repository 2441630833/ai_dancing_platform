"""Map MediaPipe 33 landmarks to HumanIK-style joint positions (meters, world space)."""

from __future__ import annotations

import numpy as np

# Joint order used by BVH hierarchy (must match bvh_export.JOINT_PARENTS / JOINT_NAMES)
JOINT_ORDER = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
)


def _p(lm, i: int) -> np.ndarray:
    return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float64)


def landmarks_to_joint_positions(lm) -> dict[str, np.ndarray]:
    """Build joint centers from one pose's world landmarks (length 33)."""
    lh, rh = _p(lm, 23), _p(lm, 24)
    hips = (lh + rh) * 0.5
    ls, rs = _p(lm, 11), _p(lm, 12)
    chest = (ls + rs) * 0.5
    spine = hips + (chest - hips) * (1.0 / 3.0)
    spine1 = hips + (chest - hips) * (2.0 / 3.0)
    nose = _p(lm, 0)
    neck = chest + (nose - chest) * 0.38
    head = nose
    left_shoulder, right_shoulder = ls.copy(), rs.copy()
    left_arm, right_arm = _p(lm, 13), _p(lm, 14)
    left_fore, right_fore = _p(lm, 15), _p(lm, 16)
    left_hand, right_hand = _p(lm, 19), _p(lm, 20)
    left_up_leg, right_up_leg = lh.copy(), rh.copy()
    left_leg, right_leg = _p(lm, 25), _p(lm, 26)
    left_foot, right_foot = _p(lm, 27), _p(lm, 28)
    left_toe, right_toe = _p(lm, 31), _p(lm, 32)
    return {
        "Hips": hips,
        "Spine": spine,
        "Spine1": spine1,
        "Spine2": chest,
        "Neck": neck,
        "Head": head,
        "LeftShoulder": left_shoulder,
        "LeftArm": left_arm,
        "LeftForeArm": left_fore,
        "LeftHand": left_hand,
        "RightShoulder": right_shoulder,
        "RightArm": right_arm,
        "RightForeArm": right_fore,
        "RightHand": right_hand,
        "LeftUpLeg": left_up_leg,
        "LeftLeg": left_leg,
        "LeftFoot": left_foot,
        "LeftToe": left_toe,
        "RightUpLeg": right_up_leg,
        "RightLeg": right_leg,
        "RightFoot": right_foot,
        "RightToe": right_toe,
    }


def joint_positions_to_stacked(jp: dict[str, np.ndarray]) -> np.ndarray:
    """Shape (22, 3) in JOINT_ORDER."""
    return np.stack([jp[name] for name in JOINT_ORDER], axis=0)
