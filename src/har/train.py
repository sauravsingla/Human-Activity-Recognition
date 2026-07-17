from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .data import ACTIVITIES, load_split, standardize
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UCI HAR deep-learning baseline")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("cnn", "lstm", "gru"), default="cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    train = load_split(args.dataset_dir, "train")
    test = load_split(args.dataset_dir, "test")
    x_train, x_test, mean, std = standardize(train.signals, test.signals)

    model = build_model(args.model, input_shape=x_train.shape[1:])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    model.fit(
        x_train,
        train.labels,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    loss, accuracy = model.evaluate(x_test, test.labels, verbose=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / f"{args.model}.keras")
    np.savez(args.output_dir / "normalization.npz", mean=mean, std=std)

    report = classification_report(
        test.labels,
        predictions,
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": args.model,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(test.labels, predictions).tolist(),
        "classification_report": report,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test_loss": loss, "test_accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
