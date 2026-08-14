# CREDO count-SDE v4 contributor rules

- Never modify or write into the sibling `../CREDO/` checkout.
- Keep biological data and cohort-named adapters outside this repository.
- All persisted scientific objects are strict, hash-bound, and versioned.
- New channels require positive, null, adversarial, and ablation tests.
- Do not change a checkpoint schema in the same change that adds a model channel.
- Run `python -m pytest -q` and `python -m ruff check .` before handoff.
