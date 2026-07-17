from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one UCI HAR deep-learning model"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("cnn", "lstm", "gru"), default="cnn")
    parser.add_argument(
        "--configuration",
        choices=("paper", "modern"),
        default="paper",
        help="Use the historical paper architecture or a stronger modern baseline.",
    )
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-mode",
        choices=("window", "subject"),
        default="window",
        help="Use historical window validation or stricter subject-disjoint validation.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        ExperimentConfig(
            dataset_dir=args.dataset_dir,
            model=args.model,
            configuration=args.configuration,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            validation_mode=args.validation_mode,
            patience=args.patience,
            output_dir=args.output_dir,
        )
    )
    print(
        json.dumps(
            {
                "model": metrics["model"],
                "configuration": metrics["configuration"],
                "validation_mode": metrics["validation_mode"],
                "test_loss": metrics["test_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "artifacts": str(
                    args.output_dir / f"{args.configuration}-{args.model}"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
