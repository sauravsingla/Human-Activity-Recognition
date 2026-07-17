## Summary

Describe the change and why it is needed.

## Change type

- [ ] Paper-fidelity correction
- [ ] Modern-model improvement
- [ ] Data or preprocessing change
- [ ] Test or CI improvement
- [ ] Documentation change

## Scientific integrity checklist

- [ ] Paper and modern configurations remain clearly separated.
- [ ] No unverified numerical result is presented as reproduced.
- [ ] Dataset files, credentials, and generated model artifacts are not committed.
- [ ] Any architecture or preprocessing change is documented.

## Validation

List the commands executed and their outcomes.

```text
pytest
ruff check src tests
```

## Reproducibility impact

Explain whether this changes architecture, preprocessing, validation, dependencies, random seeds, or expected results.
