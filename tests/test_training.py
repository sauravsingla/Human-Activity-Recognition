from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from har.experiment import ExperimentConfig, subject_validation_masks  # noqa: E402
from har.models import build_model  # noqa: E402


def test_paper_gru_trains_one_batch() -> None:
    tf.keras.utils.set_random_seed(7)
    x = np.random.default_rng(7).normal(size=(8, 128, 9)).astype(np.float32)
    y = np.arange(8, dtype=np.int64) % 6
    model = build_model("gru", configuration="paper")
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    assert len(history.history["loss"]) == 1
    predictions = model.predict(x[:2], verbose=0)
    assert predictions.shape == (2, 6)
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(2), atol=1e-5)


def test_experiment_config_validation() -> None:
    ExperimentConfig(dataset_dir=Path("dataset"), model="cnn").validate()

    with pytest.raises(ValueError, match="epochs"):
        ExperimentConfig(
            dataset_dir=Path("dataset"), model="cnn", epochs=0
        ).validate()
    with pytest.raises(ValueError, match="validation_fraction"):
        ExperimentConfig(
            dataset_dir=Path("dataset"), model="cnn", validation_fraction=1.0
        ).validate()
    with pytest.raises(ValueError, match="validation_mode"):
        ExperimentConfig(
            dataset_dir=Path("dataset"),
            model="cnn",
            validation_mode="invalid",  # type: ignore[arg-type]
        ).validate()


def test_subject_validation_keeps_subjects_disjoint() -> None:
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    training_mask, validation_mask, held_out = subject_validation_masks(subjects, 0.25)
    assert held_out.tolist() == [4]
    assert not np.any(training_mask & validation_mask)
    assert set(subjects[training_mask]).isdisjoint(set(subjects[validation_mask]))


def test_subject_validation_requires_multiple_subjects() -> None:
    with pytest.raises(ValueError, match="at least two"):
        subject_validation_masks(np.ones(4, dtype=np.int64), 0.2)
