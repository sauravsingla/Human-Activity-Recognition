# Human Activity Recognition

Reproducible deep-learning baselines for **Human Activity Recognition (HAR)** using the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The repository follows the experimental setup introduced by Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones* (ESANN 2013):

- 30 volunteers carrying a waist-mounted smartphone
- accelerometer and gyroscope signals sampled at 50 Hz
- fixed windows of 128 readings with 50% overlap
- nine inertial channels
- six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying
- official subject-independent train/test split

## Implementations

| Model | Purpose |
|---|---|
| 1D CNN | Learns local temporal patterns across sensor channels |
| LSTM | Models longer temporal dependencies |
| GRU | Provides a lighter recurrent alternative |

The original exploratory Colab notebooks remain available:

- [`CNN.ipynb`](CNN.ipynb)
- [`LSTM_RNN.ipynb`](LSTM_RNN.ipynb)
- [`GRU_RNN.ipynb`](GRU_RNN.ipynb)

A reusable Python implementation is provided under `src/har` with validated data loading, train-only normalization, reproducible seeds, early stopping, model export, and classification metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[train,test]"
```

Download and extract the UCI HAR archive. The path supplied to the training command must contain the `train` and `test` directories from `UCI HAR Dataset`.

## Train

```bash
python -m har.train \
  --dataset-dir "/path/to/UCI HAR Dataset" \
  --model cnn \
  --epochs 30 \
  --batch-size 64
```

Valid model names are `cnn`, `lstm`, and `gru`. Outputs are written to `artifacts/` by default:

- trained Keras model
- normalization statistics
- test accuracy and loss
- confusion matrix
- per-class precision, recall, and F1 score

## Test

```bash
pytest
```

The automated tests validate file handling, tensor shapes, zero-based labels, and leakage-safe normalization without downloading the full dataset.

## Reproducibility notes

- The official UCI train/test split is preserved; samples are not randomly mixed across subjects.
- Normalization statistics are fitted only on training signals.
- Random seeds are set for Python, NumPy, and TensorFlow.
- Exact scores can vary across TensorFlow versions and hardware.

## Citation

```bibtex
@inproceedings{anguita2013public,
  title={A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis},
  booktitle={21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
  year={2013}
}
```

## License

Code in this repository is provided for research and educational use. The UCI HAR Dataset is distributed separately under its own terms.