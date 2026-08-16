# CREDO count-SDE v4

Independent count-native finite-measure SDE recipe for longitudinal
perturbation screens. This repository implements three gated intents:

```text
count_state → count_measure → count_context
```

It is a sibling of, not a modification to, the frozen CREDO checkout. Version
`4.0.0.dev20` is engineering software; it is not a biological result and cannot
be relabeled as stable `4.0`.

Implemented surfaces include strict hash-bound contracts, sparse count storage,
source-only representation preparation, exact complete-denominator count
likelihood, separate state/fitness/context channels, immutable checkpoint and
resume, streaming inference, four-branch contrasts, one-shot evaluation, and a
sealed aggregate. The repository contains no cohort adapter or biological data.

Dev20 retains the two immutable, failed T01 Hellinger candidates and adds the
independent T04 fixed-truth particle-engine qualification. T01 blocks the
pooled real-data state-dynamics path; it does not block T02A raw-count/mass
noise qualification or the synthetic T04–T07 numerical path. T00 has an
explicit, immutable pooled finite-measure API and receipt. The T03 source ×
target hardening is engineering code, not a qualified real-cohort result.

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

# Independent component test; no cohort or GPU is used.
credo-v4 qualify-particle-engine --output /tmp/T04_PARTICLE_ENGINE
```

See [architecture](docs/architecture.md), [contracts](docs/contracts.md),
[lifecycle](docs/lifecycle.md),
[component qualification](docs/component-qualification.md),
[the detailed T01 record](docs/t01-representation-qualification.md),
[the detailed T04 record](docs/t04-particle-engine-qualification.md), and
[release policy](docs/release.md).

Current verification status is recorded in
[implementation-status.md](docs/implementation-status.md) and
`receipts/local-validation.json`.
