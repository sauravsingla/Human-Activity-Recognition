# Contributing

Contributions that improve reproducibility, correctness, documentation, or test coverage are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[train,test]"
pytest
```

## Contribution rules

1. Keep the `paper` model configuration faithful to the historical notebooks. Architecture changes belong in the `modern` configuration unless supported by the publication or original experiment records.
2. Do not claim a reproduced metric without a complete run and its generated environment metadata.
3. Preserve the official UCI HAR train/test split.
4. Fit preprocessing parameters using training data only.
5. Add or update tests for behavioral changes.
6. Use focused commits and clear commit messages.
7. Do not commit the UCI HAR dataset, trained models, probability arrays, or large generated artifacts.

## Validation checklist

Before submitting a change, run:

```bash
python -m compileall -q src tests
pytest tests/test_data.py
pytest
```

The complete suite requires TensorFlow. Documentation-only changes should still verify that links, commands, citation metadata, and terminology remain accurate.
