# G00 Dev30-A-r2 contract finalization

> **Superseded contract draft.** External review found incomplete source-record
> projection and artifact wiring. Dev31 closes those blockers; use
> [G00 Dev31 provenance finalization](g00-dev31-provenance-finalization.md) as
> the active contract authority. This r2 document does not authorize B0.

Last verified: 2026-08-16
Status: contract-only engineering amendment
Authority: source and execution contracts in `4.0.0.dev30`

## Decision

Dev30-A-r2 closes the remaining contract blockers identified after commit
`346cd36b495649c3867ab160fd1f995af8570dcf`. It does not execute G00C or G00D,
fit an expression model, inspect a feature-selection outcome, open a scale curve,
or use held-out stimulated expression for scientific inference.

The scientific status is unchanged:

| Component | Status |
|---|---|
| accepted Dev29 G00A v1 | pass, immutable |
| accepted Dev29 G00B v1 | pass, immutable |
| v1-to-v2 source-plane upgrade | contract ready; not executed |
| G00C miniature fold | contract and verifier ready; not executed |
| G00D H100 qualification | contract and verifier ready; not executed |
| G04/G07/G08 | blocked |
| biological evidence | none |

## V1-to-v2 authority amendment

`G00SourcePlaneV2Amendment` binds the accepted v1 G00A and G00B identifiers and
files, exactly 12 immutable GSE314342 source hashes, the v2 crosswalk, numerical
audit, construction derivation receipt, row locator, builder implementation, and
environment. Its literal flags prohibit model fitting and protected-outcome
scientific use.

`SourcePlaneDerivationReceipt` records, for each source, the selected row count,
selected nonzeros, selected-row hash, `(source_index, source_row)` pair hash,
scanner implementation, and parent source hash. A separate amendment decision
receipt derives pass/fail from parent, source, derivation, and v2-parent checks.

## Source-plane recomputation

Routine v2 virtual-store verification now recomputes or verifies:

- the little-endian int64 eligible-row hash;
- per-source selected-row counts;
- uniqueness and bounds of physical source rows;
- crosswalk eligible-cell and observed-source counts;
- exact guide and target catalog sets;
- row-level guide-to-target assignments;
- numerical-audit equality including finite counts; and
- the construction derivation receipt against locator rows and immutable sources.

The expensive selected-nonzero scan remains construction-time evidence. Routine
verification checks its content-addressed receipt rather than rescanning the
entire source plane.

## G00C result authority

The G00C protocol is outcome-free. Feature candidates are exactly
`256, 512, 1024, 2048, 4096`; the selected prefix exists only in the execution
bundle. The verifier reconstructs candidate NLL means, 59 paired-refit draws,
the p95 difference to 4,096, and the smallest-prefix decision. It also enforces
the `CUSTOM001_PuroR` technical sidecar outside the primary metric.

The sample-size verifier reconstructs the 59-refit saturation curve and the
smallest eligible cell count. A two-million-cell extension is legal only when
all candidates through one million fail the frozen equivalence threshold.

`validate_g00c_execution()` additionally proves:

- exact coverage and disjoint assignment of all four positive row roles;
- donor/checkpoint membership for every row;
- held-out Rest-only source queries;
- exclusion of held-out 8 h/48 h rows from compact payloads;
- exact compact row and feature order hashes;
- deterministic sampler microbatch, RNG, thinning, weight, and resume evidence;
- immutable publication; and
- full compact-payload reload.

## G00D evidence authority

The G00D receipt now binds and checks GPU count, eight microbatches per update,
the exact measurement-protocol hash, raw measurement/telemetry/parity/memory
artifacts, process and aggregate worker RSS, open file handles, and all eight
per-gate parity records. Pass/fail is recomputed from exact hashes or numerical
tolerances, performance thresholds, bounded-memory evidence, and zero error
counts.

## Historical proposed next operation

The draft proposed Dev30-B0: build the no-model v2 source-plane
amendment from the accepted Dev29 v1 parents and immutable source hashes. G00C
materialization remains forbidden until that amendment passes. G00D and G04
remain downstream-blocked. This proposal was superseded before execution.
