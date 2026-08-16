# T07S learned constant-reaction recovery

Last verified: 2026-08-16. Status: immutable dev23 training/parameter-recovery
record. Its primary endpoint RMSE table is superseded by the
[dev24 duration-correct amendment](t07s-reaction-metric-amendment.md).

The committed external evidence directory is
`credo_v4_t07s_reaction_recovery_20260815`. Its qualification ID is
`f6d264d4059f0116d2f1e7f5db2b3c5ce1c8a45eb25bb924057ffaf58aafa4b9`
and its detailed receipt ID is
`a4d204c9580ce3ddc386816f40578ad61562bf0c6528df43e5c6b704061fef8d`.

## Decision

The learned constant target-reaction parameters, R0 false-promotion audit, and
protected invariants pass. The dev23 R1 endpoint delta below mixed per-time
rates with cumulative interval changes and is retained only as historical
evidence; it must not be cited as the numerical authority.

T07S passes both required regimes:

- R0: a zero-reaction falsification and null-margin calibration;
- R1: recovery of nonzero target reaction contrasts on independent catalogs.

This opens design of the pooled scalar mass-only T07R-A test. It does not
qualify pooled state dynamics, diffusion, ecology, a decoder, an SDE mechanism,
absolute abundance, or any biological claim.

## Isolated model

The exact component ID is `T07S_REACTION_RECOVERY`. The channel contract is:

| Channel | Status |
|---|---|
| drift | fixed at zero |
| diffusion | fixed at zero |
| reaction | trainable target contrast |
| ecology | off |
| decoder | off |
| selection | off |
| pool-reference fitness | fixed at zero; non-identifiable under normalization |
| concentration | fixed |

Only `CountSDEModel.target_fitness` enters the optimizer. Controls receive an
exact-zero target residual. The production `raw_fitness`, source-exposure-
weighted `relative_fitness`, complete-denominator Dirichlet–multinomial loss,
`count_probabilities`, and streaming particle rollout are used directly.

The pool-reference scalar is intentionally not trained: adding one constant to
every category in a complete denominator does not change its probabilities.
Within each pool, the reported reaction obeys

\[
\sum_g w_{g,0}\,r_g=0,
\]

where `w` is the Jeffreys-smoothed source exposure. This is the identifiable
relative-fitness gauge; it is not absolute growth.

## Frozen synthetic design

The design has one control category plus 12 nonzero target effects from -1.10
to +1.10. Each complete pool contains three control guides and two guides per
target. Source counts vary by guide, pool durations are 0.75, 1.0, or 1.25,
and each terminal pool has 60,000 multinomial counts.

The candidate grid is `0, 25, 50, 100, 200` Adam updates at learning rate
0.05 in deterministic CPU float64. Update 0 remains selectable. Selection uses
separate training and validation catalogs. The selected budget is then rerun
from a fresh zero initialization on their union before the test catalog is
opened.

R0 uses 59 independent calibration refits and 60 separate audit refits. Its
margin is

\[
\epsilon_R=\max\{0,-Q_{0.05}(\Delta_{R0})\}.
\]

The audit false-promotion rate is evaluated against that already-frozen
margin. R1 uncertainty resamples the 12 target units 4,000 times and applies
the same target draw to the learned reaction and zero-reaction baseline.

## Result

| Quantity | Result |
|---|---:|
| R0 calibration refits | 59 |
| R0 independent audit refits | 60 |
| frozen null margin | 0.000000 |
| R0 audit false promotions | 0/60 |
| one-sided 95% false-promotion upper bound | 0.048703 |
| selected R1 update | 100 |
| R1 target-balanced RMSE delta | -0.567863 |
| paired target-bootstrap 95% interval | [-0.695670, -0.410235] |
| recovered raw-reaction RMSE | 0.017233 |
| nonzero-effect sign accuracy | 1.000000 |
| recovered channel RMS | 0.675928 |
| maximum weighted-gauge error | 3.82e-17 |
| maximum rollout mass relative error | 1.33e-15 |

The zero margin is the observed result of this exact R0 protocol. Update 0 did
not prevent every nonzero internal checkpoint selection: 9/59 calibration and
4/60 audit refits selected a nonzero update. It did prevent false promotion in
the audit (0/60). The margin is not a biological effect size and must not be
replaced by the T02A value `0.132578`.

## Promotion boundary

T07S demonstrates that this implementation can learn identifiable constant
target reaction contrasts from complete synthetic counts and propagate them
as finite-measure relative mass. It does not show that target reaction is
predictable in a real screen.

A future T07R-A comparison must preregister nested zero/global/leave-one-guide-
out target baselines and use a paired shared-catalog bootstrap of
`loss(new) - loss(strongest baseline)`. The global T02A tables are descriptive
sampling references, not fold-local selectors. The pooled state path remains
blocked by T01.

## Reproduction and evidence

```bash
credo-v4 qualify-reaction --output T07S_REACTION_RECOVERY
```

Publication is no-clobber, manifest-last, and checksum-bound. The bundle
contains:

- `NULL_REFITS.parquet` and `NULL_MODEL_EFFECTS.parquet`;
- `RECOVERY_CURVE.parquet`, `RECOVERY_SERIES.parquet`, and
  `TARGET_METRICS.parquet`;
- every target-bootstrap selection in `BOOTSTRAP_TARGET_DRAWS.parquet`;
- the post-selection `SELECTED_MODEL.safetensors`;
- typed detailed and role-aware component receipts;
- configuration, implementation, model-card, activity, and checksum records.

The verifier reloads the selected safe tensors, reconstructs row and target
metrics, replays the persisted bootstrap draws, recomputes the null margin and
audit bound, and reruns gauge/probability/rollout invariants.
