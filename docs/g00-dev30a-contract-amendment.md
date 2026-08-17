# G00 Dev30-A contract amendment

> Historical first amendment. The r2 draft was also superseded; use the active
> [Dev31 provenance finalization](g00-dev31-provenance-finalization.md).

Last verified: 2026-08-16. Status: authoritative contract-only amendment for
`4.0.0.dev30`. Authority: the accepted Dev29 G00A/G00B evidence remains the
only executed GSE314342 source-plane result.

## Scope and decision

Dev30-A hardens the next G00 evidence format. It does not read protected
expression values, select features, fit a representation or model, execute a
loader benchmark, or evaluate a held-out stimulated outcome. The statuses are
therefore unchanged:

| Component | Status after Dev30-A |
|---|---|
| G00A source authority | **PASS**, accepted Dev29 v1 evidence |
| G00B virtual canonical plane | **PASS**, accepted Dev29 v1 evidence |
| G00C fold-native compact view | **not run** |
| G00D integrated H100 qualification | **not run** |
| G04/G07/G08 real-data work | **blocked on G00D** |
| Biological evidence | **none** |

The v1 contracts remain read-only compatible. New evidence must use v2.

## Protected-source access semantics

The old `protected_outer_endpoints_read=false` field was too broad: source
authority legitimately hashes protected source bytes, reads observation
metadata, scans CSR structure, and scans CSR values for numerical integrity. V2
records those operations separately from
forbidden scientific use:

- source bytes hashed: true;
- observation metadata read for authority: true;
- CSR structure scanned for authority: true;
- CSR values scanned for numerical authority: true;
- protected expression used for feature selection: false;
- protected expression used for model fitting: false;
- protected expression used for model selection: false;
- protected expression used for evaluation: false.

This prevents byte-integrity work from being mislabeled as no access while
keeping model-facing outcome use fail-closed.

## G00A and G00B v2 authority

Every source record now binds CSR encoding, stored value/index/indptr dtypes,
nonnegativity and integrality checks, maximum observed count, index bounds,
monotone indptr, and terminal-offset reconciliation. The same records are
materialized in a hash-bound `SOURCE_NUMERIC_AUDIT.parquet`.

The hash-bound `GUIDE_TARGET_CROSSWALK.parquet` has exactly these columns:

```text
guide_id, target_id, is_control, raw_guide_group,
eligible_cell_count, observed_source_count
```

Verification requires unique guide identity, one target/control assignment per
guide, no `multi_sgRNA` eligible row, disjoint control and targeting target
sets, positive eligible/source support, complete guide coverage, exact
row-locator agreement, and the declared guide and target/control counts. For
GSE314342, future Dev30-B evidence must reconcile exactly 25,956 guides and
12,732 target/control categories without using held-out-donor stimulated
outcomes for eligibility.

`validate_g00_source_plane()` requires G00B to reproduce G00A's ordered source
records and hashes, canonical feature hash, guide and target hashes, crosswalk,
numeric audit, eligibility rule, eligible row hash/count/nonzeros, and access
semantics. A path, timestamp, or matching total is insufficient.

## Frozen G00C contract

The v2 fold contract binds the exact G00A/G00B parents, eligible row universe,
guide-target crosswalk, four row roles, row-role derivation implementation,
training-only feature selection, training-cell scale selection, and sampler.

Feature selection uses a deterministic training-only ranked-prefix method with
an implementation hash, fit-row hash, ordered candidate counts through 4,096,
training-validation negative log likelihood, a minimum improvement margin, 59
paired refits, and a hash-bound ordered feature table. `CUSTOM001_PuroR` is a
technical-assay feature: it is excluded from the primary biological metric and
must be reported as a sidecar sensitivity.

Sample-size selection compares `NLL_N - NLL_Nmax`. The equivalence statistic is
the 95th percentile absolute paired difference over 59 refits. The allowed grid
is 50k, 100k, 250k, 500k, and 1m cells; 2m is allowed only as the final extension
when the smaller grid has not saturated.

The initial sampler is exactly 512 cells per microbatch, eight microbatches per
update, and 4,096 cells per macrobatch. Donor, checkpoint, target, and guide
within target are equally weighted. RNG, thinning, resume cursor, and sampler
implementation are hash-bound. `validate_g00_fold_view_parent()` rejects any
parent, row-universe, or crosswalk drift.

## Frozen G00D contract

G00D must report the GPU name and UUID, CUDA and Torch versions, container
digest, worker/CPU/storage identity, micro/macro batch, prefetch depth,
warmup/measured updates, cold and steady-state measurements, cache policy,
measurement hash, telemetry interval, median/p95 compute and wait times, GPU
utilization, open shards, and process RSS.

All parity gates are mandatory: row IDs, raw counts, sample weights, thinning
RNG, loss, gradients, parameters, and interrupted/resumed execution. Memory is
qualified numerically using peak RSS, RSS slope, the upper confidence bound of
that slope, maximum excursion, and the configured bounds. A passing receipt
also requires data wait at most 10%, GPU utilization at least 85%, p95 readiness
within threshold, bounded LRU/open shards, zero loader/CUDA/monitor errors, and
all absolute/relative parity errors within tolerance. The validator derives the
decision; a self-declared `pass` is rejected when any gate fails.

## Next permitted work

Dev30-B may construct one miniature G00C fold from the accepted Dev29 source
plane under these v2 contracts. Only after its row-role, feature, sample-size,
and sampler receipts verify may scale curves or G00D run. No G00D or downstream
component status is implied by this amendment.
