# Reproducibility

Development reproduction uses the committed dependency locks and exact frozen
CREDO receipt:

```bash
(cd vendor && sha256sum --check SHA256SUMS)
python -m pip install -e '.[test]'
ruff format --check src tests scripts
ruff check src tests scripts
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider --cov
python scripts/generate_schemas.py
git diff --exit-code -- schemas
python -m build
```

The synthetic fixture is deterministic, contains no biology, and exercises
all three intents. A release receipt records Python, Torch, CUDA where relevant,
the wheel and sdist hashes, schemas, locks, source-tree hash, compatibility
receipt, tests, and sealed synthetic run.

Exact CPU resume is a supported envelope. CUDA/BF16, memory ceilings,
throughput, and numerical resume tolerances require the scheduled GPU workflow
and are not inferred from successful CPU execution.
