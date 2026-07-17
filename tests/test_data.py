from pathlib import Path

import numpy as np
import pytest

from har.data import SIGNALS, load_split, standardize


def _write_split(root: Path, split: str, samples: int = 3) -> None:
    split_dir = root / split
    inertial = split_dir / "Inertial Signals"
    inertial.mkdir(parents=True)
    for index, signal in enumerate(SIGNALS):
        values = np.full((samples, 128), index + 1, dtype=np.float32)
        np.savetxt(inertial / f"{signal}_{split}.txt", values)
    np.savetxt(split_dir / f"y_{split}.txt", np.arange(1, samples + 1), fmt="%d")
    np.savetxt(split_dir / f"subject_{split}.txt", np.ones(samples), fmt="%d")


def test_load_split_returns_expected_shapes(tmp_path: Path) -> None:
    _write_split(tmp_path, "train")
    split = load_split(tmp_path, "train")
    assert split.signals.shape == (3, 128, 9)
    assert split.labels.tolist() == [0, 1, 2]
    assert split.subjects.tolist() == [1, 1, 1]


def test_standardize_uses_training_statistics() -> None:
    train = np.array([[[1.0], [3.0]]], dtype=np.float32)
    test = np.array([[[5.0], [7.0]]], dtype=np.float32)
    train_scaled, test_scaled, mean, std = standardize(train, test)
    np.testing.assert_allclose(train_scaled.reshape(-1), [-1.0, 1.0])
    np.testing.assert_allclose(test_scaled.reshape(-1), [3.0, 5.0])
    np.testing.assert_allclose(mean, [2.0])
    np.testing.assert_allclose(std, [1.0])


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required UCI HAR file"):
        load_split(tmp_path, "test")
