"""Video decode + MediaPipe Pose Landmarker (VIDEO) -> joint trajectories -> BVH."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from bvh_export import positions_to_bvh
from humanik_mapping import JOINT_ORDER, joint_positions_to_stacked, landmarks_to_joint_positions
from smooth import smooth_positions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def ensure_pose_model(models_dir: Path | None = None) -> Path:
    models_dir = models_dir or Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "pose_landmarker_lite.task"
    if not path.is_file():
        urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 — fixed Google URL
    return path


def _stack_to_frames(stack: np.ndarray) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    for i in range(stack.shape[0]):
        d = {JOINT_ORDER[j]: stack[i, j].copy() for j in range(len(JOINT_ORDER))}
        frames.append(d)
    return frames


def video_to_bvh(
    video_path: str | Path,
    *,
    max_frames: int = 30_000,
    smooth: bool = True,
) -> tuple[str, dict]:
    """
    Returns (bvh_text, meta) where meta includes fps, frame count, warnings.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    model_path = ensure_pose_model()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    raw: list[np.ndarray] = []
    ts = 0
    n_read = 0
    missed = 0
    last_lm = None

    while n_read < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        n_read += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts += int(round(1000.0 / max(fps, 1e-6)))
        res = landmarker.detect_for_video(mp_image, ts)

        if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
            lm = res.pose_world_landmarks[0]
            last_lm = lm
            jp = landmarks_to_joint_positions(lm)
            raw.append(joint_positions_to_stacked(jp))
        else:
            missed += 1
            if last_lm is not None:
                jp = landmarks_to_joint_positions(last_lm)
                raw.append(joint_positions_to_stacked(jp))
            else:
                raw.append(np.full((len(JOINT_ORDER), 3), np.nan, dtype=np.float64))

    cap.release()
    landmarker.close()

    if not raw:
        raise RuntimeError("no frames read from video")

    arr = np.stack(raw, axis=0)
    # Fill NaN (no pose at start) with first valid
    valid = ~np.isnan(arr).any(axis=(1, 2))
    if not valid.any():
        raise RuntimeError("pose never detected; try fixed full-body framing or a brighter clip")
    first = int(np.argmax(valid))
    for i in range(first):
        arr[i] = arr[first]
    for i in range(first, len(arr)):
        if np.isnan(arr[i]).any():
            arr[i] = arr[i - 1]

    if smooth:
        arr = smooth_positions(arr, fps)

    frames = _stack_to_frames(arr)
    rest = frames[0]
    bvh = positions_to_bvh(frames, fps, rest=rest)

    meta = {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": len(frames),
        "frames_read": n_read,
        "frames_without_detection": missed,
        "model": "pose_landmarker_lite",
    }
    return bvh, meta
