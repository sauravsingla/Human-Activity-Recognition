# Human Activity Recognition

[![Tests](https://github.com/sauravsingla/Human-Activity-Recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/sauravsingla/Human-Activity-Recognition/actions/workflows/tests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26438%2Fijcse%2Fv8i11.2732-blue)](https://doi.org/10.26438/ijcse/v8i11.2732)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

Reproducible CNN, LSTM-RNN, and GRU-RNN experiments associated with:

> **Saurav Singla and Anjali Patel, “Comparative Study of the Deep Learning Neural Networks on the basis of the Human Activity Recognition,” International Journal of Computer Sciences and Engineering, vol. 8, no. 11, pp. 27–32, 2020.**  
> DOI: [10.26438/ijcse/v8i11.2732](https://doi.org/10.26438/ijcse/v8i11.2732)

## What this repository provides

- Explicit **paper** architectures derived from the original 2020 notebooks.
- Separate **modern** baselines that do not claim exact paper fidelity.
- Validated UCI HAR loading with the official train/test split.
- Training-only normalization, deterministic seeds, early stopping, and environment capture.
- One-command comparison of CNN, LSTM-RNN, and GRU-RNN.
- Saved models, predictions, probabilities, history, classification metrics, and comparison tables.
- Unit tests plus a TensorFlow training smoke test in GitHub Actions.

## Dataset and task

The experiments use the [UCI Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones):

- 30 volunteers with a waist-mounted smartphone
- accelerometer and gyroscope data sampled at 50 Hz
- windows of 128 readings with 50% overlap
- nine inertial channels
- six classes: walking, walking upstairs, walking downstairs, sitting, standing, and laying
- an official subject-independent training/test split

## Architecture configurations

| Configuration | CNN | LSTM-RNN / GRU-RNN | Intended use |
|---|---|---|---|
| `paper` | Two Conv1D layers with 32 filters and kernel size 3, dropout 0.6, pooling, flatten, Dense(50) | One 32-unit recurrent layer, dropout 0.5 | Reproduce the historical study setup |
| `modern` | Larger CNN with batch normalization and global average pooling | 64-unit recurrent layer plus Dense(64) | Contemporary reference baselines |

The original Colab notebooks remain as historical artifacts:

- [`CNN.ipynb`](CNN.ipynb)
- [`LSTM_RNN.ipynb`](LSTM_RNN.ipynb)
- [`GRU_RNN.ipynb`](GRU_RNN.ipynb)

See [`docs/paper-reproduction.md`](docs/paper-reproduction.md) for the paper-to-code mapping and methodological boundaries.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[train,test]"
```

For the reference pinned CPU environment:

```bash
pip install -r requirements-reproduction.txt
pip install -e .
```

Download and extract the UCI HAR archive. The supplied path must contain the dataset's `train` and `test` directories.

## Run one model

```bash
python -m har.train \
  --dataset-dir "/path/to/UCI HAR Dataset" \
  --model cnn \
  --configuration paper \
  --epochs 45 \
  --batch-size 16 \
  --seed 42
```

Installed command equivalent:

```bash
har-train --dataset-dir "/path/to/UCI HAR Dataset" --model cnn
```

## Run the complete paper comparison

```bash
python -m har.compare \
  --dataset-dir "/path/to/UCI HAR Dataset" \
  --configuration paper \
  --epochs 45 \
  --batch-size 16 \
  --seed 42
```

Installed command equivalent:

```bash
har-compare --dataset-dir "/path/to/UCI HAR Dataset"
```

Each run creates a directory such as `artifacts/paper-cnn/` containing:

- `model.keras`
- `normalization.npz`
- `predictions.npy`
- `probabilities.npy`
- `history.json`
- `metrics.json`

The comparison command additionally creates:

- `comparison.csv`
- `comparison.json`
- `comparison.md`

The summary contains test accuracy, macro precision, macro recall, macro F1, weighted F1, parameter count, training time, completed epochs, and best epoch.

## Test

Data and configuration tests without TensorFlow:

```bash
pip install -e ".[test]"
pytest tests/test_data.py
```

Complete suite including model construction and a one-batch training smoke test:

```bash
pip install -e ".[train,test]"
pytest
```

## Reproducibility and research integrity

- The official UCI training and test split is preserved.
- Normalization statistics are fitted only on training signals.
- Labels are converted from `1..6` to `0..5` and trained with sparse categorical cross-entropy.
- Python, NumPy, and TensorFlow seeds are recorded.
- Deterministic TensorFlow operations are enabled where supported.
- Runtime versions, platform, parameter count, sample counts, and training settings are stored in `metrics.json`.
- Historical paper/notebook values must be labelled separately from newly generated reproduction results.
- Exact numerical equality across historical and current environments is not guaranteed.

No new reproduction score is claimed in this README until the complete comparison has been run and its generated result files have been reviewed. See [`results/README.md`](results/README.md).

## Project structure

```text
src/har/data.py        validated UCI HAR loading and normalization
src/har/models.py      paper and modern CNN/LSTM/GRU architectures
src/har/experiment.py  reproducible training, evaluation, and artifact capture
src/har/train.py       single-model command
src/har/compare.py     three-model comparison command
tests/                 data, architecture, and training smoke tests
docs/                  reproduction methodology
results/               policy and reviewable result summaries
```

## Citation

```bibtex
@article{singla2020comparative,
  title={Comparative Study of the Deep Learning Neural Networks on the basis of the Human Activity Recognition},
  author={Singla, Saurav and Patel, Anjali},
  journal={International Journal of Computer Sciences and Engineering},
  volume={8},
  number={11},
  pages={27--32},
  year={2020},
  doi={10.26438/ijcse/v8i11.2732}
}
```

GitHub's **Cite this repository** function is supported through [`CITATION.cff`](CITATION.cff).

## License

The source code is available under the [MIT License](LICENSE). The UCI HAR dataset is distributed separately under its own terms.
