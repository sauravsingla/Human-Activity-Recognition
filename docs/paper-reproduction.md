# Paper reproduction guide

This repository supports two explicit experiment configurations:

- `paper`: preserves the architectures and main training settings represented in the original 2020 notebooks.
- `modern`: provides stronger contemporary baselines and must not be reported as an exact reproduction of the paper.

## Paper-to-code mapping

| Published-study component | Repository implementation |
|---|---|
| UCI HAR inertial-signal loading | `src/har/data.py::load_split` |
| Training-only channel normalization | `src/har/data.py::standardize` |
| CNN architecture | `src/har/models.py::build_paper_model` with `name="cnn"` |
| LSTM-RNN architecture | `src/har/models.py::build_paper_model` with `name="lstm"` |
| GRU-RNN architecture | `src/har/models.py::build_paper_model` with `name="gru"` |
| Single-model experiment | `src/har/train.py` |
| Three-model comparison | `src/har/compare.py` |
| Metrics and environment capture | `src/har/experiment.py` |
| Historical executable records | `CNN.ipynb`, `LSTM_RNN.ipynb`, `GRU_RNN.ipynb` |

## Dataset protocol

The UCI HAR dataset contains 128-step windows from nine inertial channels and six activity classes. The repository preserves the official training and test directories and does not merge or reshuffle subjects across that boundary.

The reusable implementation converts labels from the dataset's `1..6` representation to `0..5` and uses sparse categorical cross-entropy. This is mathematically equivalent to one-hot labels with categorical cross-entropy while avoiding unnecessary label expansion.

## Reproduction command

```bash
python -m har.compare \
  --dataset-dir "/path/to/UCI HAR Dataset" \
  --configuration paper \
  --epochs 45 \
  --batch-size 16 \
  --seed 42
```

The command trains CNN, LSTM-RNN, and GRU-RNN under one shared evaluation pipeline and writes per-model artifacts plus `comparison.csv`, `comparison.json`, and `comparison.md`.

## Architecture fidelity

The `paper` CNN uses two 32-filter Conv1D layers with kernel size 3, dropout 0.6, max pooling, flattening, a 50-unit dense layer, and a six-class softmax output.

The `paper` recurrent models use one 32-unit LSTM or GRU layer, dropout 0.5, and a six-class softmax output.

The `modern` configuration intentionally differs by using larger layers, batch normalization or additional dense layers. Results from that configuration should be labelled as modern baselines rather than paper reproduction results.

## Reproducibility boundaries

Random seeds and deterministic TensorFlow operations are enabled where supported. Exact values can still vary because of TensorFlow version, CPU/GPU kernels, hardware, and historical notebook-environment differences.

The repository does not claim that a score has been reproduced until a complete run has generated the corresponding metrics file. Historical notebook outputs and newly generated results must be reported separately.
