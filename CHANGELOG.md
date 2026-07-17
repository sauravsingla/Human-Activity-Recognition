# Changelog

All notable changes to this project are documented here.

## 1.0.0 - 2026-07-17

### Added

- Exact paper-style CNN, LSTM-RNN, and GRU-RNN model configurations.
- Separate modern baseline architectures that do not claim paper fidelity.
- Validated UCI HAR loading with training-only standardization.
- Reproducible training and three-model comparison commands.
- Window-based and subject-disjoint validation strategies.
- Model, history, prediction, metric, environment, and comparison artifacts.
- Unit tests, TensorFlow smoke tests, linting, coverage, and GitHub Actions CI.
- Pinned reproduction environment, MIT licence, citation metadata, and research documentation.
- Contributor, security, conduct, and dependency-maintenance policies.

### Historical material

- The original 2020 CNN, LSTM-RNN, and GRU-RNN Colab notebooks remain available as historical experiment artifacts.

## Reproduction status

The repository contains the complete reproduction workflow. Numerical reproduction results must only be added after all three models have been run from a clean pinned environment on the official UCI HAR split and the generated artifacts have been reviewed.
