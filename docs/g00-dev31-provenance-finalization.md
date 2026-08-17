# G00 Dev31 provenance finalization

Last verified: 2026-08-16
Status: contract-only engineering amendment; no G00 execution
Authority: G00 v2 source, compact-view, and loader contracts in `4.0.0.dev31`

> Dev31 is frozen. B0 A1 correctly rejected the historical checksum-only
> Dev29 publication at its native-parent gate. The separately versioned
> [Dev32 legacy-parent boundary](g00-dev32-legacy-parent-attestation.md) is the
> only authorized compatibility path; this Dev31 record remains unchanged
> evidence of the stronger native default.

## Decision

Dev31 supersedes the Dev30-A-r2 contract draft and closes the provenance and
verification blockers identified in its external review. It does not scan the
12 source matrices, build a compact CountStore, inspect feature or scale
outcomes, run an H100 benchmark, or open protected held-out stimulated
expression for scientific inference.

The scientific state therefore remains unchanged:

| Component | Status |
|---|---|
| accepted Dev29 G00A v1 | pass, immutable |
| accepted Dev29 G00B v1 | pass, immutable |
| v1-to-v2 source-plane amendment | contract and verifier ready; not executed |
| G00C compact fold | contract and verifier ready; not executed |
| G00D integrated H100 qualification | contract and verifier ready; not executed |
| G04/G07/G08 | blocked |
| biological evidence | none |

## Exact v1-to-v2 projection

`validate_g00_source_plane_amendment()` first verifies the accepted v1 G00A and
G00B parent relationship. It then requires:

- exact parent authority and manifest identifiers and file `ArtifactRef`s;
- the same ordered source identifiers and immutable source hashes;
- field-for-field equality of every v1 source record, permitting only the new
  `numeric_integrity` member in v2;
- equality of the inherited feature, guide, target, eligibility, eligible-row,
  eligible-nonzero, and row-universe authorities;
- exact amendment wiring for the crosswalk, numerical audit, source-derivation
  receipt, and row locator in both derived G00A and G00B; and
- a passed amendment receipt whose derived authority and manifest identifiers
  and file `ArtifactRef`s equal the files actually supplied to verification.

Routine virtual-store verification now parses and validates the amendment
artifact itself. A malformed amendment, wrong amendment identity, stale file
hash, or cross-wired derived artifact fails before the virtual source plane can
be accepted.

## G00C parent and execution authority

Every v2 fold contract binds both the amendment and its passed receipt by
identifier and `ArtifactRef`. A failed or unrelated receipt blocks G00C.

The execution verifier no longer accepts selection summaries as sufficient
evidence. It verifies:

- feature-selection fit and validation row hashes against the frozen row roles;
- the selected feature prefix, canonical indices, and technical sidecar;
- the exact nested training-row order and selected prefix used at each scale;
- a literal compact CSR/HDF5 schema, dtypes, dimensions, row universe, selected
  feature order, and count limits;
- chunked equality between every compact CSR value and the corresponding
  virtual-source value; and
- actual uninterrupted and resumed sampler draw/state traces. Row IDs, sample
  weights, thinning draws, RNG state JSON, and cursor state JSON are rehashed
  independently into the epoch index, and interrupted replay must equal the
  uninterrupted trace.

The frozen sample-size maximum remains one million cells unless the separately
specified two-million extension is opened by the nonsaturation rule. A compact
payload may contain only the selected training prefix, training validation
rows, and held-out source-query rows. Protected held-out stimulated rows remain
forbidden.

## G00D artifact-derived qualification

`verify_integrated_loader_qualification()` verifies the bytes of the
measurement, telemetry, parity, and memory artifacts before deriving the
receipt decision. It recomputes compute/data-wait summaries, GPU utilization,
peak RSS and file-handle bounds, RSS slope and confidence limit, memory
excursion, and LRU status.

Parity semantics are gate-specific and frozen:

- row IDs, raw counts, thinning RNG, and interrupted/resumed state require exact
  hashes;
- sample weights permit exact hashes or the declared numerical tolerances; and
- loss, gradient, and parameter parity use numerical tolerances.

An exact gate cannot pass merely because its reported absolute and relative
errors are zero. Artifact/receipt disagreement, copied summaries, missing
measurements, or a status inconsistent with recomputed evidence fails closed.

## Adversarial coverage

The Dev31 regression tests reject source relabeling and reordering, wrong parent
files, cross-wired amendment and receipt files, failed amendment receipts,
malformed or misidentified amendments, compact-count drift, non-nested training
rows, sampler trace/index disagreement, exact-hash parity disagreement with
zero numerical error, telemetry-summary drift, and artifact tampering.

## Authorized next operation

After package validation and review, the next permissible external operation is
Dev31 B0: construct the no-model v2 source-plane amendment from the accepted
Dev29 v1 parents and the same 12 immutable source files. G00C remains blocked
until the B0 amendment and receipt pass. G00D remains blocked until an accepted
G00C artifact exists. No downstream model or biological gate is authorized by
this contract-only release.
