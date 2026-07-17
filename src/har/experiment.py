from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .data import ACTIVITIES, load_split, standardize
from .models import ModelConfiguration, ModelName, build_model

ValidationMode = Literal["window", "subject"]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_dir: Path
    model: ModelName
    configuration: ModelConfiguration = "paper"
    epochs: int = 45
    batch_size: int = 16
    seed: int = 42
    validation_fraction: float = 0.15
    validation_mode: ValidationMode = "window"
    patience: int = 5
    output_dir: Path = Path("artifacts")

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.validation_mode not in {"window", "subject"}:
            raise ValueError("validation_mode must be 'window' or 'subject'")
        if self.patience < 0:
            raise ValueError("patience cannot be negative")


def set_reproducible_seed(seed: int):
    """Configure repeatable Python, NumPy, and TensorFlow execution."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return tf


def subject_validation_masks(
    subjects: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/validation masks using complete subjects."""
    unique_subjects = np.unique(subjects)
    if unique_subjects.size < 2:
        raise ValueError("Subject validation requires at least two training subjects")
    validation_count = max(1, int(np.ceil(unique_subjects.size * fraction)))
    validation_count = min(validation_count, unique_subjects.size - 1)
    validation_subjects = unique_subjects[-validation_count:]
    validation_mask = np.isin(subjects, validation_subjects)
    training_mask = ~validation_mask
    return training_mask, validation_mask, validation_subjects


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train and evaluate one HAR model and persist reproducibility artifacts."""
    config.validate()
    tf = set_reproducible_seed(config.seed)
    from sklearn.metrics import classification_report, confusion_matrix

    train = load_split(config.dataset_dir, "train")
    test = load_split(config.dataset_dir, "test")
    x_train, x_test, mean, std = standardize(train.signals, test.signals)

    model = build_model(
        config.model,
        input_shape=x_train.shape[1:],
        configuration=config.configuration,
    )
    callbacks = []
    if config.patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        )

    fit_kwargs: dict[str, Any] = {}
    validation_subjects: list[int] = []
    fit_x = x_train
    fit_y = train.labels
    if config.validation_mode == "subject":
        train_mask, validation_mask, held_out = subject_validation_masks(
            train.subjects, config.validation_fraction
        )
        fit_x = x_train[train_mask]
        fit_y = train.labels[train_mask]
        fit_kwargs["validation_data"] = (
            x_train[validation_mask],
            train.labels[validation_mask],
        )
        validation_subjects = [int(value) for value in held_out]
    else:
        fit_kwargs["validation_split"] = config.validation_fraction

    started = time.perf_counter()
    history = model.fit(
        fit_x,
        fit_y,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
        **fit_kwargs,
    )
    training_seconds = time.perf_counter() - started

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    loss, accuracy = model.evaluate(x_test, test.labels, verbose=0)
    report = classification_report(
        test.labels,
        predictions,
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )

    run_dir = config.output_dir / f"{config.configuration}-{config.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.keras")
    np.savez(run_dir / "normalization.npz", mean=mean, std=std)
    np.save(run_dir / "predictions.npy", predictions)
    np.save(run_dir / "probabilities.npy", probabilities)

    history_payload = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (run_dir / "history.json").write_text(
        json.dumps(history_payload, indent=2), encoding="utf-8"
    )

    epochs_completed = len(history_payload.get("loss", []))
    val_losses = history_payload.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else epochs_completed

    metrics: dict[str, Any] = {
        "model": config.model,
        "configuration": config.configuration,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(test.labels, predictions).tolist(),
        "classification_report": report,
        "parameters": int(model.count_params()),
        "training_seconds": float(training_seconds),
        "epochs_requested": config.epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "validation_mode": config.validation_mode,
        "validation_subjects": validation_subjects,
        "fit_samples": int(fit_y.size),
        "train_samples": int(train.labels.size),
        "test_samples": int(test.labels.size),
        "train_subjects": int(np.unique(train.subjects).size),
        "test_subjects": int(np.unique(test.subjects).size),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "command": " ".join(sys.argv),
        "config": _json_safe(asdict(config)),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    return metrics
