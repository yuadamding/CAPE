# T07S-A unified R0/R1 interval-metric amendment

Last verified: 2026-08-16. Status: authoritative dev25 evidence correction;
synthetic software qualification only, with no optimizer rerun, cohort result,
or biological claim.

## Decision

Dev25 closes the final dev24 evidence-consistency gap. Both the zero-reaction
R0 calibration/audit and the learned-reaction R1 test now use the same primary
estimand:

\[
y_g(T)=\text{centered interval log-frequency change},\qquad
\widehat y_g(T)=T\widehat r_g^{\mathrm{rel}}.
\]

The T07S-A conclusion remains passed. Constant target-average relative
reaction is recovered in complete synthetic catalogs. State-dependent
reaction, within-guide selection, ecology, absolute growth, and biological
effects remain unqualified.

## Evidence-only derivation

The immutable dev23 `NULL_REFITS.parquet` is preserved as
`legacy_null_refits`. Dev25 adds `NULL_INTERVAL_REFITS.parquet`. For each of
the 59 calibration and 60 audit repeats it:

1. reads the persisted seed, selected update, and fitted target coefficients;
2. reconstructs the deterministic three-pool null test catalog from
   `seed + 2`;
3. assigns the persisted coefficients directly to the production model;
4. computes the source-exposure-centered relative-fitness rate;
5. multiplies each rate by its pool duration;
6. recomputes the target-balanced interval-effect RMSE and zero-reaction RMSE.

No Adam step, checkpoint selection, or post-selection refit is invoked.
`optimizer_rerun=false` is present in the typed receipt, amendment, parent
link, and null-calibration record. The selected model and all training-derived
parent artifacts remain byte-identical.

## Unified result

The authoritative external evidence directory is
`credo_v4_t07s_null_interval_amendment_20260816`. Its amendment ID is
`0330881eb6703f4cdb78e59d862fd6c2d539f1cb5b482ec891a32e9f00b609d9`
and its detailed receipt ID is
`6e10bc15f49b58d2f6664310da418378d349de74c3bab6c47aff4ae98671436e`.

| Quantity | Dev25 result |
|---|---:|
| R0 calibration repeats | 59 |
| R0 audit repeats | 60 |
| R0 required margin | 0 |
| R0 false promotions | 0/60 |
| one-sided upper 95% bound | 0.048703 |
| selected R1 update | 100 |
| R1 interval-effect target-balanced RMSE | 0.033191 |
| zero-reaction RMSE | 0.706508 |
| paired RMSE delta | -0.673317 |
| target-bootstrap 95% interval | [-0.829965, -0.482187] |
| raw-reaction RMSE | 0.017233 |
| nonzero-effect sign accuracy | 1.000000 |

The R0 margin and false-promotion decisions are derived only from
`interval_effect_rmse_delta` in the new corrected table. Tests faithfully
recreate the legacy metric and require at least one nonzero-selected repeat to
differ between legacy and corrected deltas.

## Reproduction

```bash
credo-v4 amend-reaction \
  --t07s-bundle /path/to/immutable/dev23/T07S_REACTION_RECOVERY \
  --output /path/to/new/dev25/T07S_REACTION_NULL_INTERVAL_AMENDMENT
```

Publication is no-clobber and manifest-last. Verification recomputes the
corrected R0 table from persisted coefficients, recomputes R1 from the
preserved model, replays all 4,000 target-bootstrap selections, checks copied
parent bytes, and reruns all protected numerical invariants.

T07R-A0 proceeded from this unified dev25 authority in dev26. Estimator parity
passed, but the predeclared real fold selected update 0 and failed
noninferiority; the adapter is retired. See the
[authoritative T07R-A0 record](t07r-a0-pooled-likelihood.md). Its scope remains
a pooled relative guide-representation likelihood qualification, not absolute
clone growth or proliferation.
