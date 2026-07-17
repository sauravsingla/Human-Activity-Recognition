from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

SIGNALS: tuple[str, ...] = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)

ACTIVITIES: tuple[str, ...] = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)

Split = Literal["train", "test"]


@dataclass(frozen=True)
class HARSplit:
    signals: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray


def _load_matrix(path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required UCI HAR file not found: {path}")
    return np.loadtxt(path, dtype=dtype)


def load_split(dataset_dir: str | Path, split: Split) -> HARSplit:
    """Load one official subject-independent UCI HAR split.

    Returns signals with shape ``(samples, 128, 9)`` and zero-based labels.
    """
    root = Path(dataset_dir).expanduser().resolve()
    split_dir = root / split
    inertial_dir = split_dir / "Inertial Signals"

    channels = [
        _load_matrix(inertial_dir / f"{signal}_{split}.txt")
        for signal in SIGNALS
    ]
    shapes = {channel.shape for channel in channels}
    if len(shapes) != 1:
        raise ValueError(f"Signal files have inconsistent shapes: {sorted(shapes)}")

    signals = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    labels = _load_matrix(split_dir / f"y_{split}.txt", np.int64).reshape(-1) - 1
    subjects = _load_matrix(split_dir / f"subject_{split}.txt", np.int64).reshape(-1)

    if signals.shape[0] != labels.size or labels.size != subjects.size:
        raise ValueError("Signals, labels, and subjects contain different sample counts")
    if signals.shape[1:] != (128, len(SIGNALS)):
        raise ValueError(f"Expected signal shape (*, 128, 9), received {signals.shape}")
    if labels.size and (labels.min() < 0 or labels.max() >= len(ACTIVITIES)):
        raise ValueError("Labels must map to the six UCI HAR activities")

    return HARSplit(signals=signals, labels=labels, subjects=subjects)


def standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each sensor channel using training data only."""
    if train.ndim != 3 or test.ndim != 3 or train.shape[2] != test.shape[2]:
        raise ValueError("Expected train and test arrays shaped (samples, steps, channels)")

    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < np.finfo(np.float32).eps, 1.0, std)
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        mean.reshape(-1).astype(np.float32),
        std.reshape(-1).astype(np.float32),
    )
