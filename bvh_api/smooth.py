"""Temporal smoothing for joint trajectories (v1 — light; room for One-Euro / foot lock later)."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth_positions(positions: np.ndarray, fps: float) -> np.ndarray:
    """
    positions: (num_frames, num_joints, 3)
    """
    n = positions.shape[0]
    if n < 5:
        return _ema(positions, alpha=min(0.35, max(0.15, 8.0 / max(fps, 1e-6))))
    win = min(21, n)
    if win % 2 == 0:
        win -= 1
    win = max(5, win)
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 5:
        return _ema(positions, alpha=min(0.35, max(0.15, 8.0 / max(fps, 1e-6))))
    poly = 3 if win >= 7 else 2
    out = np.zeros_like(positions)
    for j in range(positions.shape[1]):
        for c in range(3):
            out[:, j, c] = savgol_filter(
                positions[:, j, c], window_length=win, polyorder=poly, mode="nearest"
            )
    return out


def _ema(positions: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(positions)
    out[0] = positions[0]
    for i in range(1, len(positions)):
        out[i] = alpha * positions[i] + (1.0 - alpha) * out[i - 1]
    return out
