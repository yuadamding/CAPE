# T02A raw-count and relative-mass noise qualification

Last verified: 2026-08-15. Status: authoritative dev21 component contract and
completed Renz development calibration. This is a measurement-noise result,
not model performance or biology.

## Decision

T02A passes. The cohort-neutral implementation completed 100 cell split-half
repeats at each checkpoint and 100 relative-mass catalog bootstraps against
the exact passed T00 population. Every output is content-addressed,
environment-bound, published manifest-last, and fully reverified with the
complete CountStore parent.

The Renz noise ID is
`ca3b8c5c6575a399c6125047530d92a8d337cc0a54cc525dfb5daf9d4e0e5310`.
The complete external report is
[`T02A_RUN_REPORT.md`](../../credo_v4_renz_t02a_noise_20260815/T02A_RUN_REPORT.md).

## Dependency and channel boundary

```text
T00 passed pooled population ──→ T02A raw-count/mass noise ──→ frozen tolerances

T01 representation ──╳── not read
learned model ────────╳── not read
biological outcomes ─╳── not read for threshold selection
```

T02A uses no representation, state field, diffusion, reaction, ecology,
decoder, optimizer, checkpoint, or particle. Its `pass` status means the
calibration is complete and internally valid. It makes no assertion that the
measured noise is small.

## Raw-count estimand

For each retained guide and checkpoint, cells are randomly split into two
balanced halves. Raw counts are summed independently and compared by:

- Hellinger distance;
- Jensen–Shannon divergence;
- symmetric multinomial deviance per count;
- pseudobulk gene Spearman correlation; and
- top-variable-gene overlap.

The default protocol requires at least 100 deterministic repeats. Guide
metrics are averaged within perturbation target and then equally across
targets. Controls are reported separately, preventing the pooled control
category from receiving the weight of one ordinary perturbation target.

Pseudobulk Spearman and top-gene overlap use a checkpoint-specific frozen
2,000-gene universe selected by guide-pseudobulk `log1p(CPM)` variance. Top-50
overlap uses absolute smoothed log-enrichment relative to the checkpoint-wide
pool. The feature indices and selection scores are exported rather than
silently recomputed later.

## Relative-mass estimand

For checkpoint total `N_t` and Jeffreys-smoothed observed guide probabilities

\[
\widehat p_{g,t}=\frac{n_{g,t}+0.5}{\sum_h(n_{h,t}+0.5)},
\]

T02A samples independent catalogs

\[
n^{(b)}_{\cdot,t}\sim\operatorname{Multinomial}(N_t,\widehat p_{\cdot,t})
\]

and recomputes

\[
y_g^{(b)}=\log\widehat p^{(b)}_{g,60}-\log\widehat p^{(b)}_{g,4}.
\]

The exported floors cover target-balanced interval RMSE, sign stability,
guide and target ranks, and top/bottom-`k` stability.

## Renz calibration result

The exact parents are T00 ID
`7313a9d990900148618edc0fe6a72d5f41e2594641b10d48c278370240964f58`
and CountStore SHA-256
`cb7bd724b0adf3b74ec7141d6aa5605566c2b8f3f9b2f3d846482bc7b7efafd4`.
The population contains 495 guides, 150 perturbation targets, 277,200 cells,
and 34,699 assay-common features.

| Frozen quantity | Value |
| --- | ---: |
| Target-balanced Hellinger q95 | 0.154232 |
| Control-guide Hellinger q95 | 0.231343 |
| Target-balanced Jensen–Shannon q95 | 0.0231605 |
| Target-balanced deviance/count q95 | 0.0459571 |
| Target-balanced pseudobulk Spearman q05 | 0.881957 |
| Target-balanced top-50 overlap q05 | 0.330921 |
| Interval log-mass RMSE q95 | 0.132578 |
| Minimum detectable absolute interval log-mass effect | 0.289973 |
| Sign accuracy q05 | 0.930112 |
| Guide-rank Spearman q05 | 0.978563 |
| Target-rank Spearman q05 | 0.980054 |
| Top-20 / bottom-20 overlap q05 | 0.90 / 0.80 |

These thresholds were frozen before learned-model inspection. T01B may use the
control Hellinger tolerance; pooled T07R may use the mass RMSE margin; T12 may
use the protected raw-count metrics. Changing the population, feature scope,
pseudocount, target weighting, or quantile definition creates a new T02A
protocol.

## Public API and CLI

```python
from credo_count_sde_v4 import api

api.qualify_raw_noise(
    output_directory,
    pooled_bundle=t00_directory,
    count_store=count_store_path,
)

api.verify_raw_noise(
    output_directory,
    pooled_bundle=t00_directory,
    count_store=count_store_path,
)
```

```bash
credo-v4 qualify-raw-noise \
  --pooled-bundle T00_pooled_data_contract \
  --count-store counts.h5 \
  --split-repeats 100 \
  --mass-bootstrap-repeats 100 \
  --output T02A_raw_count_mass_noise
```

Implementation:
[`noise/qualification.py`](../src/credo_count_sde_v4/noise/qualification.py).

## Artifact surface

| Artifact | Purpose |
| --- | --- |
| `raw-count-mass-noise.json` | typed content-addressed bundle |
| `RAW_SPLIT_HALF_METRICS.parquet` | guide/checkpoint/repeat raw metrics |
| `RAW_REPEAT_SUMMARY.parquet` | target-balanced and control summaries |
| `RAW_TARGET_SUMMARY.parquet` | per-target raw stability |
| `VARIABLE_GENES.parquet` | frozen checkpoint-specific gene universes |
| `MASS_BOOTSTRAP_METRICS.parquet` | catalog-level mass/rank stability |
| `MASS_GUIDE_NOISE.parquet` | per-guide mass effect error |
| `MASS_TARGET_NOISE.parquet` | per-target mass effect error |
| `FROZEN_THRESHOLDS.json` | downstream calibration values |
| `TEST_RECEIPT.json` | typed detailed decision |
| `COMPONENT_RECEIPT.json` | common component surface |
| `INPUTS.sha256`, `IMPLEMENTATION.sha256` | parent and code identities |
| `artifacts.json`, `COMMITTED`, `SHA256SUMS` | transactional integrity |

## Limitations

Cell halves are technical sampling replicates, not mice. Multinomial catalogs
condition on observed pooled frequencies and omit systematic WTA-library,
batch, capture, and biological-replicate variation. Consequently these floors
are necessary protected tolerances but not a complete biological noise model.
T02A does not unblock T02B, T03R, or pooled dynamics while T01 remains open.
