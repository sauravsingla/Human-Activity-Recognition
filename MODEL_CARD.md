# Model Card

## Model overview

This repository contains CNN, LSTM-RNN, and GRU-RNN models for Human Activity Recognition using the UCI HAR smartphone-sensor dataset.

Two configurations are provided:

- `paper`: compact architectures aligned with the historical notebooks associated with the 2020 Singla–Patel study.
- `modern`: stronger contemporary baselines that are intentionally not presented as exact paper reproductions.

## Intended use

The models are intended for:

- research and education in time-series classification;
- reproducibility studies of the associated paper;
- benchmarking CNN, LSTM, and GRU architectures on the official UCI HAR split.

They are not intended for safety-critical monitoring, medical diagnosis, workplace surveillance, or decisions about individuals.

## Data

The UCI HAR dataset contains inertial signals from 30 volunteers performing six activities. Inputs are windows of 128 time steps across nine accelerometer and gyroscope channels.

The dataset is distributed separately and is not included in this repository.

## Evaluation

The repository reports accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrices, parameter counts, training duration, and reproducibility metadata.

Numerical results should only be published after a complete run using the pinned environment and official dataset split.

## Limitations

- Performance may vary across TensorFlow versions, hardware, and deterministic-operation support.
- The dataset is controlled and relatively small compared with real-world sensor deployments.
- Results may not generalize to different devices, body placements, sampling rates, demographics, or unseen activities.
- Random window validation may be optimistic; subject-disjoint validation is available and recommended for stronger evaluation.

## Ethical considerations

Human activity inference can create privacy and surveillance risks. Users should obtain appropriate consent, minimize retained sensor data, and avoid using the models for covert or high-stakes monitoring.

## Citation

Please cite the associated paper and this repository using `CITATION.cff`.
