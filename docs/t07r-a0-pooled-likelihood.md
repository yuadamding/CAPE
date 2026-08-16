# T07R-A0 pooled relative-guide likelihood qualification

Last verified: 2026-08-16. Status: authoritative dev26 component record;
`fail_retired`. Scope: one historically exposed Renz/GSE235325 guide-within-
target development fold, CPU only.

## Decision

The production complete-denominator likelihood passes its estimator-parity
gate but fails real pooled noninferiority. T07R-A0 does not advance and must not
be retried with more updates, GPU memory, or additional folds under this
contract.

This is a relative guide-representation test. It is not absolute population
growth, cell proliferation, state transport, ecology, or a biological claim.

## Frozen contract

- parents: passed T00 pooled data, dev22 T02A interpretation authority, and
  dev25 T07S-A;
- exact 495-guide P4-source-eligible population and immutable guide-to-target
  catalog;
- outer fold 0, inner-validation fold 1, fitting folds 2 and 3;
- updates `0, 25, 50, 100, 200`, with update 0 selectable;
- fresh zero-initialized refit on all non-outer guides;
- constant target-level relative reaction only;
- controls exact zero; pool reference fixed zero; smoothing 0.5;
- fixed concentration 1,000 and ridge penalty 0.05 in both the independent
  reference and production estimator;
- no representation, CountStore, drift, diffusion, selection, ecology,
  decoder, particles, or GPU;
- paired 4,000-draw terminal-catalog bootstrap for the one-shot outer result.

The direct reference uses an independent SciPy penalized Dirichlet–multinomial
objective and analytic gradient. The production route uses `CountSDEModel`,
the exact count objective, the same target hierarchy/penalty, and
`count_probabilities`.

## Gate A: estimator parity

On a training-only synthetic catalog generated from the fitted reference:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Maximum guide-probability error | `7.8147993e-08` | `<1e-6` |
| Maximum target-effect error | `9.5233413e-07` | `<5e-6` |

Gate A passes. This qualifies implementation parity, not cohort prediction.

## Gate B: real pooled noninferiority

Update 0 had the smallest inner-validation loss (`0.0273122`); nonzero updates
were all approximately `0.0274282`. The fresh refit therefore remains source-
frequency persistence.

| Metric | Result |
| --- | ---: |
| Production M2 NLL/count | `0.0230634238` |
| Sister-guide target M1 NLL/count | `0.0224216577` |
| M2 − M1 | `+0.0006417661` |
| Paired 95% interval | `[+0.0006210237,+0.0006623849]` |
| One-sided 95% upper | `+0.0006592353` |
| Training-only noninferiority margin | `1e-08` |

Gate B fails decisively. The target-level signal learned from sister guides is
useful as a comparator, but the selected production adapter collapses to
persistence and is materially worse. Under the preregistered stop rule, the
adapter is retired and no four-fold run is justified.

## Evidence and reproduction

The external authority is
[`credo_v4_renz_t07r_a0_20260816/`](../../credo_v4_renz_t07r_a0_20260816/),
especially its `RUN_REPORT.md`, typed bundle, receipt, per-guide probabilities,
selection curve, bootstrap deltas, selected safe tensors, and checksums.

```bash
credo-v4 qualify-pooled-reaction \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --t02a-amendment /path/to/T02A_INTERPRETATION_AMENDMENT \
  --t07s-amendment /path/to/T07S_NULL_INTERVAL_AMENDMENT \
  --fold-assignment /path/to/guide_fold_assignment.parquet \
  --output /new/path/T07R_A0_POOLED_LIKELIHOOD
```

The verifier rechecks every parent and artifact, recomputes estimator parity,
reruns selection/refit and the exact bootstrap, reloads every model tensor, and
rejects any byte or statistic mismatch.
