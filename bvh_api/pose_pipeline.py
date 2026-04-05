"""Video decode + MediaPipe Pose Landmarker (VIDEO) -> joint trajectories -> BVH."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.tasks.python.vision.drawing_styles import get_default_pose_landmarks_style
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from PIL import Image

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


def _resize_bgr_max_width(bgr: np.ndarray, max_w: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    scale = max_w / float(w)
    new_w = max_w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _draw_pose_skeleton(bgr: np.ndarray, normalized_landmarks: list) -> None:
    """Draw MediaPipe pose on BGR image in place."""
    draw_landmarks(
        bgr,
        normalized_landmarks,
        PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=get_default_pose_landmarks_style(),
        connection_drawing_spec=DrawingSpec(color=(245, 250, 250), thickness=3),
    )


def _bgr_frames_to_gif_bytes(frames: list[np.ndarray], fps: float) -> bytes:
    if not frames:
        return b""
    rgb_seq = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    pil_frames = [Image.fromarray(arr) for arr in rgb_seq]
    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


def _subsample_frames(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    n = len(frames)
    if n <= max_frames:
        return frames
    idx = np.linspace(0, n - 1, num=max_frames, dtype=np.int64)
    return [frames[int(i)] for i in idx]


def video_to_bvh(
    video_path: str | Path,
    *,
    max_frames: int = 30_000,
    smooth: bool = True,
    preview_gif: bool = False,
    preview_max_width: int = 640,
    gif_max_frames: int = 480,
) -> tuple[str, dict]:
    """
    Returns (bvh_text, meta) where meta includes fps, frame count, warnings.

    If preview_gif is True, meta also contains preview_gif_bytes (MP4 frames with
    skeleton overlay, GIF-encoded; may be subsampled to gif_max_frames).
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
    preview_frames: list[np.ndarray] | None = [] if preview_gif else None
    ts = 0
    n_read = 0
    missed = 0
    last_lm = None
    last_img_lm: list | None = None

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
            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                last_img_lm = list(res.pose_landmarks[0])
        else:
            missed += 1
            if last_lm is not None:
                jp = landmarks_to_joint_positions(last_lm)
                raw.append(joint_positions_to_stacked(jp))
            else:
                raw.append(np.full((len(JOINT_ORDER), 3), np.nan, dtype=np.float64))

        if preview_frames is not None:
            vis = _resize_bgr_max_width(bgr, preview_max_width)
            vis = vis.copy()
            if last_img_lm is not None:
                _draw_pose_skeleton(vis, last_img_lm)
            preview_frames.append(vis)

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

    meta: dict = {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": len(frames),
        "frames_read": n_read,
        "frames_without_detection": missed,
        "model": "pose_landmarker_lite",
    }
    if preview_frames is not None:
        gif_src = _subsample_frames(preview_frames, gif_max_frames)
        gif_fps = fps * (len(gif_src) / max(len(preview_frames), 1))
        meta["preview_gif_bytes"] = _bgr_frames_to_gif_bytes(gif_src, gif_fps)
        meta["preview_gif_frames"] = len(gif_src)
        meta["preview_gif_subsampled"] = len(preview_frames) > len(gif_src)
        meta["preview_gif_max_width"] = preview_max_width
    return bvh, meta
