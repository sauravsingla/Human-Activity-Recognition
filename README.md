# Human Activity Recognition

Reproducible implementation of the experiments reported in:

> **Saurav Singla and Anjali Patel, “Comparative Study of the Deep Learning Neural Networks on the basis of the Human Activity Recognition,” International Journal of Computer Sciences and Engineering, vol. 8, no. 11, pp. 27–32, 2020.**  
> DOI: [10.26438/ijcse/v8i11.2732](https://doi.org/10.26438/ijcse/v8i11.2732)

The paper compares three deep-learning architectures for smartphone-sensor Human Activity Recognition:

- Long Short-Term Memory Recurrent Neural Network (**LSTM-RNN**)
- Gated Recurrent Unit Recurrent Neural Network (**GRU-RNN**)
- One-dimensional Convolutional Neural Network (**CNN**)

## Experimental basis

The experiments use the [UCI Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), as used in the published study:

- 30 volunteers carrying a waist-mounted smartphone
- accelerometer and gyroscope signals sampled at 50 Hz
- fixed windows of 128 readings with 50% overlap
- nine inertial signal channels
- six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying
- official subject-independent training and test split

The UCI dataset publication describes the data-generation protocol; the Singla–Patel paper is the primary publication associated with this repository and its comparison of CNN, LSTM-RNN, and GRU-RNN.

## Implementations

| Model | Role in the comparative study |
|---|---|
| LSTM-RNN | Learns long-term temporal dependencies in sensor sequences |
| GRU-RNN | Recurrent alternative with fewer parameters than LSTM |
| 1D CNN | Learns local temporal patterns using convolutional filters |

The original exploratory Colab notebooks remain available:

- [`CNN.ipynb`](CNN.ipynb)
- [`LSTM_RNN.ipynb`](LSTM_RNN.ipynb)
- [`GRU_RNN.ipynb`](GRU_RNN.ipynb)

A reusable implementation is provided under `src/har` with validated data loading, training-only normalization, deterministic seeds, early stopping, model export, and consistent evaluation metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[train,test]"
```

Download and extract the UCI HAR archive. The dataset path supplied to the training command must contain the `train` and `test` directories from `UCI HAR Dataset`.

## Train one model

```bash
python -m har.train \
  --dataset-dir "/path/to/UCI HAR Dataset" \
  --model cnn \
  --epochs 45 \
  --batch-size 16
```

Valid model names are `cnn`, `lstm`, and `gru`. The default paper-reproduction settings use 45 epochs and a batch size of 16, matching the original notebooks.

Outputs are written to `artifacts/` by default:

- trained Keras model
- normalization statistics
- test accuracy and loss
- confusion matrix
- per-class precision, recall, and F1 score

Run the command separately for all three models to reproduce the comparison.

## Test

```bash
pytest
```

The tests validate file handling, tensor dimensions, zero-based labels, and leakage-safe normalization without requiring the complete dataset.

## Reproducibility notes

- The official UCI training/test split is preserved; subjects are not randomly mixed.
- Normalization statistics are fitted only on training data.
- Random seeds are set for Python, NumPy, and TensorFlow.
- Exact numerical scores may vary across TensorFlow versions, hardware, and random-number implementations.
- Results should be reported for all three architectures using the same split and evaluation procedure.

## Citation

Please cite the paper associated with this repository:

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

The dataset methodology can additionally be cited using the original UCI HAR dataset publication.

## License

Code in this repository is provided for research and educational use. The UCI HAR Dataset is distributed separately under its own terms.
