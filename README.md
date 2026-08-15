# CREDO count-SDE v4

Independent count-native finite-measure SDE recipe for longitudinal
perturbation screens. This repository implements three gated intents:

```text
count_state → count_measure → count_context
```

It is a sibling of, not a modification to, the frozen CREDO checkout. Version
`4.0.0.dev15` is engineering software; it is not a biological result and cannot
be relabeled as stable `4.0`.

Implemented surfaces include strict hash-bound contracts, sparse count storage,
source-only representation preparation, exact complete-denominator count
likelihood, separate state/fitness/context channels, immutable checkpoint and
resume, streaming inference, four-branch contrasts, one-shot evaluation, and a
sealed aggregate. The repository contains no cohort adapter or biological data.

Dev15 adds an explicitly pilot-only, low-rank source × target state interaction.
It persists update 0 and selects among the global null, a sister-guide
target-terminal model, and the target-plus-interaction model. Each non-null
family must clear a frozen improvement margin; the selected family and update
budget are then refit on the complete outer-training information set. This
contract does not turn pooled guide centroids into finite-measure or
biological-replicate evidence.

## Quick start

```bash
python -m pip install -e '.[test]'
credo-v4 synthetic --output /tmp/credo-v4-demo --intent count_context
credo-v4 prepare /tmp/credo-v4-demo/config.yaml
credo-v4 compile /tmp/credo-v4-demo/config.yaml
credo-v4 train /tmp/credo-v4-demo/config.yaml
credo-v4 finalize /tmp/credo-v4-demo/config.yaml
credo-v4 evaluate /tmp/credo-v4-demo/config.yaml
credo-v4 seal /tmp/credo-v4-demo/config.yaml
credo-v4 verify /tmp/credo-v4-demo/work/sealed --level full
```

See [architecture](docs/architecture.md), [contracts](docs/contracts.md),
[lifecycle](docs/lifecycle.md), and [release policy](docs/release.md).

Current verification status is recorded in
[implementation-status.md](docs/implementation-status.md) and
`receipts/local-validation.json`.
