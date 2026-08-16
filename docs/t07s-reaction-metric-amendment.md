# T07S-A duration-correct reaction metric amendment

Last verified: 2026-08-16. Status: immutable dev24 R1 correction, superseded as
the unified decision authority by the
[dev25 R0/R1 amendment](t07s-null-interval-amendment.md); no cohort or
biological claim.

Dev24 fixed the primary nonzero R1 endpoint but retained the dev23
rate-versus-interval `NULL_REFITS.parquet` for its R0 margin. The numerical
decision was unchanged, but the mixed-unit evidence wiring means this page and
bundle remain provenance records rather than the final unified metric authority.

## Why an amendment was required

Dev23 correctly learned a per-unit-time target reaction rate, but its primary
endpoint table compared that rate directly with a duration-integrated centered
log-frequency change. Because the synthetic pools use durations 0.75, 1.0,
and 1.25, the two columns had inconsistent time units.

Dev24 evaluates the cumulative endpoint estimand:

\[
\widehat y_g(T)=T\widehat r_g^{\mathrm{rel}}
\]

against the observed source-exposure-centered interval log-frequency change.
Every row and receipt labels this quantity explicitly. A regression test over
durations 0.5, 1.0, and 2.0 prevents a future rate-versus-interval comparison.

## No-retraining contract

The amendment consumes the immutable dev23 qualification, verifies its
manifest and typed identities, and copies these bytes unchanged:

- `SELECTED_MODEL.safetensors`;
- `RECOVERY_CURVE.parquet`;
- `NULL_REFITS.parquet` and `NULL_MODEL_EFFECTS.parquet`;
- `BOOTSTRAP_TARGET_DRAWS.parquet`.

`PARENT_LINK.json` binds the parent qualification, parent bundle and artifact-
manifest hashes, the selected-model hash, and `optimizer_rerun=false`. Only the
duration-correct recovery/target metrics and their dependent receipts are
regenerated.

## Corrected result

| Quantity | Dev24 result |
|---|---:|
| selected update | 100 |
| interval-effect target-balanced RMSE | 0.033191 |
| zero-reaction RMSE | 0.706508 |
| paired RMSE delta | -0.673317 |
| target-bootstrap 95% interval | [-0.829965, -0.482187] |
| raw-reaction RMSE | 0.017233 |
| nonzero-effect sign accuracy | 1.000000 |

The correction strengthens the already-passing result. It does not change the
learned coefficients, selected checkpoint, gauge, probability normalization,
fixed-channel checks, or streaming rollout result.

## R0 semantics

Update 0 was selectable, but nonzero checkpoints were internally selected in
9/59 calibration refits and 4/60 audit refits. No audit refit falsely promoted.
The typed receipt therefore reports checkpoint-selection counts separately and
uses `r0_false_promotion_guard_pass`; it does not claim a false-selection guard.

The frozen maximum one-sided false-promotion bound is now 0.05. The observed
zero-failure upper 95% bound is 0.048703 and passes.

## Qualified and unqualified reaction channels

This component is now named **T07S-A**. It qualifies constant target-level
average relative reaction in complete synthetic count catalogs. It does not
qualify state-dependent centered reaction or within-guide selective
reweighting; that future component is **T07S-B**.

Consequently:

- T07R-A0 pooled scalar mass-likelihood design was executed in dev26 and
  retired after failing real-fold noninferiority;
- T07R-A1 predictive advancement requires an additional permitted source-side
  predictor beyond the target-shared scalar;
- T07R-B and joint T08 remain blocked by T01/T02B.

Duration-generalization, weak-signal/depth, and guide-heterogeneity synthetic
extensions remain robustness work, not prerequisites for the present software
identity correction.

## Reproduction

```bash
credo-v4 amend-reaction \
  --t07s-bundle /path/to/immutable/dev23/T07S_REACTION_RECOVERY \
  --output /path/to/new/dev24/T07S_REACTION_METRIC_AMENDMENT
```

Publication is no-clobber and manifest-last. Verification reloads the preserved
safe tensors, checks every parent/copied hash, recomputes row and target metrics,
replays all 4,000 persisted target-bootstrap selections, and reruns protected
gauge, probability, control, fixed-channel, and rollout invariants.
