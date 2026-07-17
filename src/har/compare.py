from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment

MODELS = ("cnn", "lstm", "gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CNN, LSTM, and GRU UCI HAR comparison"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--configuration", choices=("paper", "modern"), default="paper"
    )
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def _write_summary(output_dir: Path, results: list[dict]) -> None:
    columns = (
        "model",
        "configuration",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "parameters",
        "training_seconds",
        "epochs_completed",
        "best_epoch",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, separator]
    for result in results:
        values = []
        for column in columns:
            value = result[column]
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    results = []
    for model in MODELS:
        metrics = run_experiment(
            ExperimentConfig(
                dataset_dir=args.dataset_dir,
                model=model,
                configuration=args.configuration,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                patience=args.patience,
                output_dir=args.output_dir,
            )
        )
        results.append(metrics)
    _write_summary(args.output_dir, results)
    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
