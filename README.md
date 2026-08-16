# CREDO count-SDE v4

Independent count-native finite-measure SDE recipe for longitudinal
perturbation screens. This repository implements three gated intents:

```text
count_state → count_measure → count_context
```

It is a sibling of, not a modification to, the frozen CREDO checkout. Version
`4.0.0.dev25` is engineering software; it is not a biological result and cannot
be relabeled as stable `4.0`.

Implemented surfaces include strict hash-bound contracts, sparse count storage,
source-only representation preparation, exact complete-denominator count
likelihood, separate state/fitness/context channels, immutable checkpoint and
resume, streaming inference, four-branch contrasts, one-shot evaluation, and a
sealed aggregate. The repository contains no cohort adapter or biological data.

Dev25 retains the two immutable, failed T01 Hellinger candidates, passed T04
fixed-pool particle-engine qualification, and completed T02A calibration. It
adds a no-retraining amendment that places both dev23 T07S null and nonzero
tests in duration-integrated endpoint units, separates nonzero checkpoint
selection from false promotion, and retains the tightened 0.05 false-promotion
limit. T01 blocks the
pooled real-data state-dynamics path; it does not block T02A raw-count/mass
noise qualification or the passed synthetic T04/T07S numerical path. T00 has an
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

# Learned constant-reaction recovery; synthetic CPU catalogs only.
credo-v4 qualify-reaction --output /tmp/T07S_REACTION_RECOVERY

# Correct a manifest-verified dev23 T07S bundle without rerunning its optimizer.
credo-v4 amend-reaction \
  --t07s-bundle /path/to/T07S_REACTION_RECOVERY \
  --output /tmp/T07S_REACTION_METRIC_AMENDMENT

# Independent T02A calibration against a passed T00 bundle and raw CountStore.
credo-v4 qualify-raw-noise \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --count-store /path/to/counts.h5 \
  --output /tmp/T02A_RAW_COUNT_MASS_NOISE
```

See [architecture](docs/architecture.md), [contracts](docs/contracts.md),
[lifecycle](docs/lifecycle.md),
[component qualification](docs/component-qualification.md),
[the detailed T01 record](docs/t01-representation-qualification.md),
[the detailed T02A record](docs/t02a-raw-count-mass-noise.md),
[the detailed T04 record](docs/t04-particle-engine-qualification.md),
[the dev23 T07S record](docs/t07s-reaction-recovery.md),
[the dev24 R1 amendment](docs/t07s-reaction-metric-amendment.md),
[the authoritative dev25 unified R0/R1 amendment](docs/t07s-null-interval-amendment.md), and
[release policy](docs/release.md).

Current verification status is recorded in
[implementation-status.md](docs/implementation-status.md) and
`receipts/local-validation.json`.
