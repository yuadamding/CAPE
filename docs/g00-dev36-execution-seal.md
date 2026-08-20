# Dev36 G00C execution seal

Last verified: 2026-08-20. Status: expression-free engineering hardening.
Authority: strict package contracts, executable verifiers, generated schemas,
release checks, and the separately published D1 A2 metadata authority.

## Boundary

Dev36 preserves Dev35 and closes the five independent-review blockers that
could have allowed coherent but false evidence to reach a terminal G00C
decision. This release does not read an expression value, fit a cohort model,
use a GPU, run the G00C curve, authorize G00D, or make a biological claim.

The literal monitored process-tree ceiling is 68,719,476,736 bytes (64 GiB).
It is a deliberate claim-bearing full-grid RSS ceiling, not an estimate and not
a GPU-memory target.

## Sealed chain

`G00CExecutionBundleV4` replaces opaque evidence with the following typed
chain:

```text
Dev36 D1 authority V2
  -> exact hierarchical sampler plan and uninterrupted/resumed trace replay
  -> sampler-derived support audit
  -> executable closed-form feature and cell-budget refit replay
  -> artifact-derived feature and sample selection
  -> direct source-backed compact/PuroR materialization verification on pass
  -> exact status-dependent manifest-last publication inventory
  -> G00CDecisionReceiptV4
```

`may_parent_g00d=true` is possible only for a fully reconstructed `pass` with
typed materialization evidence. There is no callback parameter and no trusted
`implementation_sha256` attribute on a caller-supplied object.

## Materialization authority

The D1 authority now has a distinct `materialization_verifier` role bound to
`g00c_materialization_v3.py`. The verifier opens both HDF5 payloads and checks:

- exact selected feature prefix and selected training-row prefix;
- permitted row set: selected training rows plus training validation and D1
  Rest query rows, with zero protected D1 stimulated rows;
- donor/checkpoint/target/guide physical order and contiguous-run table;
- primary-feature and `CUSTOM001_PuroR` sidecar separation;
- bounded blockwise equality to the accepted G00B virtual source plane;
- uninterrupted/resumed artifact byte identity;
- strict zero-protected-read and full-reload receipts.

The source store is a direct function argument. An arbitrary object cannot
substitute for this verifier.

## Sampler and support evidence

The sampler is fixed to 512-cell microbatches, eight microbatches per 4,096-cell
macro-update, and the hierarchy donor/checkpoint → target → guide → row.
`verify_g00c_sampler_v3()` regenerates ordered row IDs, raw inverse draw
probabilities, thinning draws, macro/micro cursors, and PCG64DXSM state hashes
from the frozen six-stream schedule. It requires exact equality between an
uninterrupted execution and a serialized-state interruption/resume.

Support is no longer trusted from a supplied table. It is recomputed from the
verified trace and weights over every stratum present in the frozen candidate
prefix. Zero support, cell counts, effective sample size, maximum-to-median
weight ratio, control/targeting balance, donor/checkpoint coverage, target
coverage, guide coverage, and complete sampler-stratum coverage determine
candidate eligibility.

## Refit replay

Each replay receipt binds deterministic 4,096-gene training and validation
sufficient statistics. The verifier independently reconstructs the closed-form
0.5-per-gene multinomial probabilities, validation NLL sum, NLL per count, and
little-endian float64 state hash. It replays all 59 selected and reference
draws and draws 0, 6, 12, 18, 24, 30, 36, 42, 48, and 58 for every other
candidate. A coherent alternative curve or state-hash surface is rejected.

## Publication seal

`G00CPublicationManifestV3` is bound to the exact authority, selection freeze,
feature result, sample result, and terminal status without a circular bundle
identity. The verifier derives the legal inventory from the status and
requires exact equality among:

```text
expected payload names = manifest entries = SHA256SUMS entries
```

Every payload byte and size is checked. Non-pass publications cannot contain
selected or compact artifacts. Publication uses a fresh sibling directory,
no-clobber creation, durable control files, atomic rename, and parent-directory
`fsync`.

## Concrete A2 freeze and archive

`scripts/freeze_g00c_d1_authority_v2.py` builds a fresh metadata-only D1 A2
authority from the finalized V2 row locator and accepted guide/control
crosswalk. In addition to the Dev35 role and nested-order records it freezes a
16,924,672-row sampler hierarchy containing only row ID, source index, target
code, guide code, and control identity. It opens no source H5AD and reads no
expression value.

`scripts/archive_g00c_d1_authority_v2.py` verifies exact manifest coverage,
creates a deterministic regular-file-only archive, reopens every member,
rechecks the inner SHA-256 manifest, reparses the authority, reruns metadata
verification, and emits one outer SHA-256 plus an independent audit receipt.
The concrete IDs and archive digest are recorded in the follow-up provenance
commit after the release bytes are fixed.

## Test and release semantics

The validation receipt reports separate collected, passed, skipped, and failed
counts. A CUDA-only skip is not counted as a pass. Adversarial tests cover the
former callback escape hatch, altered sampler weights/resume traces, derived
zero support, coherent false refit surfaces, incomplete publication surfaces,
and incomplete materialization receipts.

## Status

| Surface | Status |
| --- | --- |
| Dev33-B and Dev34-A | Preserved |
| Dev35 | Accepted expression-free predecessor; not rewritten |
| Dev36 typed execution seal | Implemented |
| D1 A2 metadata authority and immutable archive | Published in follow-up provenance commit |
| Expression access / G00C curve | Not run; requires separate reviewed authorization |
| 2M extension | Blocked until a verified base `extension_required` stop |
| G00D and later real-data components | Blocked |
| GPU, model, and biology | None |
