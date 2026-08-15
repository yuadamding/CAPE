# Component-wise qualification program

Status: authoritative development order for `4.0.0.dev19`. This page defines
software promotion, not a biological claim.

CREDO V4 is qualified as a dependency graph rather than one large training
run:

```text
T00 pooled data contract
  → T01 count-native representation
  → T02 empirical noise floors and baseline registry
  → T03 shared control and target hierarchy
  → T05 drift → T06 diffusion ┐
                  T07 reaction ├→ T08 joint finite measure
T04 particle engine ───────────┘
  → T09 ecology → T10 weak form → T11 same-start counterfactuals
  → T12 decoder/program recovery → T13 pooled biological validation
```

T04 is an independent numerical prerequisite for learned dynamics. A failed
component is disabled downstream. More updates, particles, decoder width, or
GPU memory cannot substitute for a failed scientific gate.

## Universal promotion rule

For lower-is-better loss `L`, a new component must satisfy all of:

1. genuine component-specific null refits control false selection;
2. the target-clustered 95% upper bound of `L(new)-L(previous)` is below the
   negative frozen margin;
3. channel activity exceeds a frozen nonzero floor;
4. every previously passed protected metric remains within tolerance;
5. update 0, the previous model, remains selectable;
6. selection is followed by a fresh all-training-data refit.

Cells estimate guide-level empirical laws. Guides are held-out prediction
units within targets. Targets are the primary uncertainty units. Cells,
guides from one target, and folds sharing targets are never treated as
independent biological replicates.

## Channel isolation

`ComponentTestContract` enforces the frozen channel matrix:

| Stage | Drift | Diffusion | Reaction | Ecology | Decoder |
|---|---|---|---|---|---|
| T03 | off | off | off | off | off |
| T04 | fixed | fixed | fixed | off | off |
| T05 | trainable | fixed | off | off | off |
| T06 | fixed | trainable | off | off | off |
| T07 | fixed | fixed | trainable | off | off |
| T08 | trainable | trainable | trainable | off | off |
| T09 | fixed | fixed | fixed | trainable | off |
| T10 | fixed | fixed | fixed | off or fixed | off |
| T11 | fixed | fixed | fixed | fixed | off |
| T12 | fixed | fixed | fixed | fixed | trainable |
| T13 | fixed | fixed | fixed | fixed | fixed |

T00–T03 expose no drift, diffusion, reaction, ecology, or decoder channel.

## T00 pooled finite-measure contract

The public cohort-neutral API is:

```python
from credo_count_sde_v4.data import build_pooled_finite_measures
```

and the CLI surface is:

```bash
credo-v4 pool-data \
  --cells cells.parquet \
  --guide-catalog guide-catalog.parquet \
  --feature-hashes feature-hashes.json \
  --source-checkpoint P4 \
  --terminal-checkpoint P60 \
  --minimum-source-cells 20 \
  --output T00_pooled_data_contract
```

The input cell table contains `row_id`, `cell_id`, source `sample_id`,
`guide_id`, and `checkpoint`. The catalog contains one immutable
`guide_id → target_id, is_control` mapping. The adapter:

- emits the sole model-facing `sample_id="pooled"`;
- preserves source sample identity only as provenance;
- makes row order irrelevant through canonical sorting;
- freezes eligibility from source counts only;
- fails, rather than changing eligibility, when an eligible guide lacks a
  terminal empirical law;
- requires one exact feature-order hash at both checkpoints;
- assigns every retained cell to exactly one guide/checkpoint measure;
- uses `(n + 0.5) / sum_g(n_g + 0.5)` relative masses;
- verifies atom weights sum to their declared finite-measure mass;
- publishes through manifest-last atomic directory creation.

The generated directory includes `TEST_CONTRACT.json`, `INPUTS.sha256`,
`CONFIG.yaml`, null/bootstrap/channel/model status files, per-guide and
per-target metrics, `TEST_RECEIPT.json`, and `SHA256SUMS`. T00 has no learned
model; its null calibration and bootstrap entries are explicitly
`not_applicable` rather than silently omitted.

## Current status

| Stage | Package status | Pooled real-data status | Promotion |
|---|---|---|---|
| T00 | implemented; eight focused tests pass | external Renz receipt passes | T00 only |
| T01 | v1/v2 count-native gates implemented | both failed; dimension 0 in 4/4 folds | blocked |
| T02 | baseline primitives exist; complete noise-floor receipt absent | not run | blocked |
| T03 | identifiability/null hardening exists as engineering code | not run under this ladder | blocked |
| T04–T13 | partial numerical/model primitives exist | not run under this ladder | blocked |

The external T00 Renz receipt retains 495 source-eligible guides (445
targeting and 50 controls), 150 perturbation targets, and 277,200 cells. Both
checkpoint mass sums are exactly one. This establishes only data semantics.
It does not qualify representation, transport, mass reaction, ecology,
counterfactuals, decoding, or biological claims.

The first T01 candidate used multinomial/Hellinger low-rank factors with
dimensions 8, 16, 32, and 48. The global gene-frequency decoder (dimension 0)
was selectable and won every fold. The learned candidates were worse by
0.204–0.256 nats per count on P4-only inner validation. T01 v1 is therefore
`fail_retired`; it was not refit, P60 support was not inspected, and T02 remains
blocked. The failure receipt is retained outside the package in
`credo_v4_renz_t01_representation_20260815/`.

The separately frozen centered successor represented the global square-root gene
frequency explicitly and fit only residual Hellinger structure. It also
selected dimension 0 in every fold: learned dimensions 8, 16, 32, and 48 were
0.178–0.238 nats per count worse than the global gene-frequency decoder. T01
v2 is also `fail_retired`; no post-selection refit or P60-support diagnostic
was eligible. The two immutable failures block T02. Changing the loss,
threshold, or candidate family would be a new T01 protocol, not continuation
of either failed test. See the
[detailed T01 implementation and result record](t01-representation-qualification.md).

Synthetic values reported outside a committed component directory are useful
design evidence but are not promotion evidence. Every subsequent stage must
publish the same typed receipt surface before downstream use.

## Resource and stop policy

T00–T04 run on CPU. The first real T03 fold is limited to one GPU and at most
250 updates. T05–T11 synthetic tests use CPU or one small GPU. T12 is sized
independently only after dynamics pass. Four-fold out-of-fold scale-out is
reserved for T13 after every required parent component passes.

There is no target VRAM requirement. Peak memory is an engineering consequence
of the qualified model, never a selection metric.
