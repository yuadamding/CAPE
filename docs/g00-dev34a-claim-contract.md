# Dev34-A G00C claim-contract finalization

Last verified: 2026-08-20. Status: contract-only engineering release for
`4.0.0.dev34`. Authority: the generated JSON Schemas and strict package
contracts are normative; this page explains their intended use.

## Decision boundary

Dev33-B is closed as `pass_engineering_canary`. It remains non-promotable and
cannot parent claim-bearing G00C, G00D, or G04. Dev34-A does not read a count or
expression matrix, execute a feature or cell curve, launch a GPU, or produce a
model or biological result. It freezes the first promotion-bearing G00C fold
before any result can be observed.

The accepted Dev33-B decision text has SHA-256
`ef92fdb6061e0376db866ed235d5fe275d6fbc780998884999f578f9311ea56b`.
The committed canary record and provenance index remain immutable.

## Four closed specification gaps

### Expanded paired-refit seed schedule

`G00CRefitSeedScheduleV1` contains draws `0..58` and six independent uint64
streams per draw:

- initialization;
- training sampler;
- count thinning;
- validation evaluation;
- stochastic optimizer or augmentation; and
- restart interruption point.

Every value is derived from a hash-bound namespace, fold, stage, draw, and
stream name. Every candidate in a paired draw uses the same tuple. The selected
and reference candidates require replay of all 59 draws; all other candidates
require draws `{0,6,12,18,24,30,36,42,48,58}`. Stream collisions, missing
draws, altered replay subsets, or a mismatched schedule identity fail closed.

### Serial feature-to-cell parent edge

The selection order is frozen as `feature_then_cell_budget_v1`:

1. rank the 4,096-feature reference surface on the first 1,000,000 rows of the
   immutable nested training-row order;
2. select feature width at exactly that row scale;
3. freeze the feature result, selected width, and ordered-feature hash;
4. evaluate cell budgets using that unchanged feature surface.

`G00CSampleSizeSelectionResultV3` must bind the feature-result SHA-256, selected
feature count, and selected feature-order SHA-256. A smaller selected model-fit
budget therefore remains conditional on one million training-only cells having
been used for feature selection.

### Common 4,096-feature scoring support

Every prefix candidate is scored on the same frozen 4,096-feature validation
alphabet. A candidate models its first `K` features plus one residual category.
That residual mass is expanded over omitted features with a frozen,
checkpoint-specific, training-only frequency vector. Consequently:

- predictive distributions always contain 4,096 primary features;
- the weighted validation count denominator is identical across candidates;
- inverse-probability weights are paired across candidates;
- the metric unit is nats per weighted validation count; and
- `CUSTOM001_PuroR` remains outside the primary metric.

The selected feature width is the smallest prefix whose q95 absolute paired NLL
difference from 4,096 is no greater than `1e-4`.

### Terminal no-saturation state

The base cell grid is `50k, 100k, 250k, 500k, 1M`. If no candidate below 1M is
equivalent to 1M, the only valid status is `extension_required`; selected rows
and selected cell count remain null. A new extension contract adds exactly 2M.

If no candidate below 2M is equivalent to 2M, the terminal status is
`fail_no_saturation`. Selected rows remain null, G00C fails, a third extension
is forbidden, and G00D/G04 remain blocked. The cell equivalence epsilon is
fixed at `1e-4` nats per weighted validation count.

## Publication surfaces

The complete seed schedule, row hashes, feature and cell curves, all refit
records, replay audit, support audit, sampler restart evidence, decision
receipt, manifest, commit marker, and checksums are required for every terminal
status. A passing result additionally requires fresh selected feature/row
artifacts, compact and PuroR payloads, physical-run evidence, bounded source
verification, and writer restart evidence.

`extension_required` must publish a base-grid stop receipt.
`fail_no_saturation` must publish both the base stop and extension result.
Selected artifacts are forbidden for every non-pass status.

## Current authorization

| Component | Status |
| --- | --- |
| Dev33-B A3 | Closed, `pass_engineering_canary` |
| Dev34-A contract implementation | Qualified by local schema/adversarial tests |
| Claim-bearing G00C execution | Not run |
| G00D | Blocked |
| G04/G07/G08 | Blocked |
| Biological evidence | None |

The next permissible operation after independent review is one CPU-only,
promotion-bearing G00C fold under a fresh immutable attempt identity. No G00D
or GPU/model process is authorized until that fold passes.
