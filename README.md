# CREDO count-SDE v4

Independent count-native finite-measure SDE recipe for longitudinal
perturbation screens. This repository implements three gated intents:

```text
count_state → count_measure → count_context
```

It is a sibling of, not a modification to, the frozen CREDO checkout. Version
`4.0.0.dev29` is engineering software; it is not a biological result and cannot
be relabeled as stable `4.0`.

Implemented surfaces include strict hash-bound contracts, sparse count storage,
source-only representation preparation, exact complete-denominator count
likelihood, separate state/fitness/context channels, immutable checkpoint and
resume, streaming inference, four-branch contrasts, one-shot evaluation, and a
sealed aggregate. The repository contains no cohort adapter or biological data.

Dev27 retains the two immutable, failed T01 Hellinger candidates, passed T04
fixed-pool particle-engine qualification, and completed T02A calibration. It
adds a no-retraining amendment that places both dev23 T07S null and nonzero
tests in duration-integrated endpoint units, separates nonzero checkpoint
selection from false promotion, and retains the tightened 0.05 false-promotion
limit. It adds the frozen T07R-A0 pooled relative-guide likelihood pilot. The
production likelihood matches an independent reference on training-only
synthetic counts, but its fixed subset concentration is now explicitly labeled
as a fold-subcomposition v1 method. Dev27 adds the physical-pool conditional-DM
v2 forensic correction. It again selects update 0 and the sister-guide target
reference is superior under both conditional multinomial and conditional-DM
uncertainty surfaces. The selector/protocol is retired; the numerical estimator
and constant-reaction family are not broadly retired. T01 blocks the
pooled real-data state-dynamics path; it does not block T02A raw-count/mass
noise qualification or the passed synthetic T04/T07S numerical path. T00 has an
explicit, immutable pooled finite-measure API and receipt. The T03 source ×
target hardening is engineering code, not a qualified real-cohort result.

Dev28 added a content-addressed sharded CSR store, persistent process-local
readers, and a bounded resumable shard writer whose typed checkpoint truncates
uncommitted payload after interruption. It also freezes the checkpoint-only
dimension-zero multinomial decoder required by G04 and evidence-only G14 claim,
robustness, multiplicity, and seal contracts. External GSE314342 G00 probes
verify exact reordered/duplicated reads and zero planned guide fragmentation,
but fail the frozen H100 loader-throughput gate. The full approximately 765 GB
CountStore was therefore not built, and no biological or outer-donor result is
promoted.

Dev29 replaces the impossible direct-raw-store H100 rate requirement with a
two-tier data plane: immutable source authority and virtual canonical access
feed fold-native compact views, which are qualified only by integrated wait,
utilization, parity, and bounded-memory gates. It also closes the shard
finalization crash window, binds appends to exact frozen chunk identities,
bounds reader handles, restores physical checkpoint chronology, and hardens the
G14 evidence graph. See the
[G00 two-tier data-plane contract](docs/g00-two-tier-data-plane.md).

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

# One-fold, CPU-only pooled relative-guide likelihood qualification.
credo-v4 qualify-pooled-reaction \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --t02a-amendment /path/to/T02A_INTERPRETATION_AMENDMENT \
  --t07s-amendment /path/to/T07S_NULL_INTERVAL_AMENDMENT \
  --fold-assignment /path/to/guide_fold_assignment.parquet \
  --output /tmp/T07R_A0_POOLED_LIKELIHOOD

# One permitted, historically exposed physical-pool denominator correction.
credo-v4 correct-pooled-reaction \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --t02a-bundle /path/to/T02A_raw_count_mass_noise_v2 \
  --t02a-amendment /path/to/T02A_INTERPRETATION_AMENDMENT \
  --t07s-amendment /path/to/T07S_NULL_INTERVAL_AMENDMENT \
  --fold-assignment /path/to/guide_fold_assignment.parquet \
  --output /tmp/T07R_A0_PHYSICAL_POOL_CONDITIONAL_DM_V2_R2

# Independent T02A calibration against a passed T00 bundle and raw CountStore.
credo-v4 qualify-raw-noise \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --count-store /path/to/counts.h5 \
  --output /tmp/T02A_RAW_COUNT_MASS_NOISE
```

See [architecture](docs/architecture.md), [contracts](docs/contracts.md),
[lifecycle](docs/lifecycle.md),
[the G00 two-tier data-plane contract](docs/g00-two-tier-data-plane.md),
[component qualification](docs/component-qualification.md),
[the detailed T01 record](docs/t01-representation-qualification.md),
[the detailed T02A record](docs/t02a-raw-count-mass-noise.md),
[the detailed T04 record](docs/t04-particle-engine-qualification.md),
[the dev23 T07S record](docs/t07s-reaction-recovery.md),
[the dev24 R1 amendment](docs/t07s-reaction-metric-amendment.md),
[the authoritative dev25 unified R0/R1 amendment](docs/t07s-null-interval-amendment.md),
[the dev26/v1 and dev27/v2 T07R-A0 record](docs/t07r-a0-pooled-likelihood.md), and
[release policy](docs/release.md).

Current verification status is recorded in
[implementation-status.md](docs/implementation-status.md) and
`receipts/local-validation.json`.
