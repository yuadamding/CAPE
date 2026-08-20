# Dev35 G00C execution-authority finalization

Last verified: 2026-08-20. Status: expression-free engineering authority.
Authority: strict package contracts, generated JSON Schemas, the concrete D1
authority root, and its SHA-256 manifest.

## Boundary

Dev35 preserves Dev34-A unchanged and closes the independent review's
execution-authority gaps. It does not open a cohort expression matrix, compute
a feature or cell curve, train a model, use a GPU, authorize G00D, or produce a
biological result. The concrete record has status
`finalized_preaccess_not_executed`; a separate reviewed authorization is still
required before expression access.

## Authoritative chain

`G00CExecutionBundleV3` and `G00CDecisionReceiptV3` define one chain:

```text
D1 execution authority
  -> Dev34 selection freeze + exact 59 x 6 seed schedule
  -> V3 feature result + common-support receipt
  -> V3 sample-size result + stage-scoped support audit
  -> pass | extension_required | fail_no_saturation | failed_integrity
  -> may_parent_g00d = true only for an independently materialized pass
```

`verify_g00c_execution_v3()` reads every curve and all 59 paired refits. It
recomputes means and q95 paired differences, checks seed streams and common
validation denominators, applies support eligibility before selection, requires
the reference difference to be no greater than `1e-12`, and verifies that the
selected feature and row artifacts are true prefixes. A self-declared status or
selected value cannot create a pass.

A pass additionally requires a separately supplied materialization verifier
whose source hash is frozen by the D1 authority. This isolates selection logic
from compact count-byte verification.

## Common-support estimand

Every candidate is scored on the same ordered 4,096-gene alphabet. Dev35
resolves the prior asymmetry by applying pseudocount `0.5` to every primary
gene, independent of prefix width. For a width `K` candidate:

- the modeled genes receive `0.5` each;
- the residual category receives total prior `0.5 * (4096 - K)`;
- omitted-gene frequencies are estimated from training counts after adding
  `0.5` to every omitted gene; and
- expansion therefore produces finite, strictly positive probabilities.

The helper `checkpoint_multinomial_refit_common_support()` implements this
identity directly. The execution receipt still requires a prior-sensitivity
audit, but the pre-access contract does not rely on a different category-level
prior.

## Single extension rule

Base support evidence is restricted to `50k, 100k, 250k, 500k, 1M`. If no
sub-million candidate qualifies, the base bundle must stop as
`extension_required`. `G00CSampleSizeExtensionFreezeV1` then binds that fully
verified base bundle and decision receipt before any 2M access, preserves the
feature surface and 59-draw schedule, adds exactly one 2M candidate, and
forbids a third extension. The 2M support audit is a separate artifact.

## Concrete D1 authority

`scripts/freeze_g00c_d1_authority.py` constructs the literal D1 authority from
the accepted V2 metadata plane. It reads only `row_ids_sorted` and
`source_indices_sorted` from the finalized row locator. It does not open any
source H5AD or count dataset.

The frozen population is 21,996,842 eligible rows:

| Role | Rows |
| --- | ---: |
| D2-D4 training fit | 16,924,672 |
| D2-D4 training validation | 65,536 |
| D1 Rest held-out source/query | 1,752,037 |
| D1 stimulated protected endpoints | 3,254,597 |

The validation set is the exact smallest 65,536 rows under a fixed independent
SplitMix64 priority. A second fixed priority creates a concrete two-million-row
nested training order; feature ranking uses its literal first million rows.
Both ordered and role-set hashes are frozen before expression access.

The authority also binds the six accepted B0-A2 parents, the Dev33-B canary
manifest as a non-promotable prerequisite, the Dev35 release commit/wheel/
normalized sdist/source tree, all seven role-labelled implementation files,
and `locks/tested-environment.v1.json`.

The concrete A1 authority is finalized at the workspace-relative publication
`../credo_v4_gse314342_g00_20260816/G00C_D1_AUTHORITY_DEV35_A1`. Its authority
ID is `dbfa9b0b7fe65f636c07ab920c55826ac961d6385a847b5ec77367c6b03228a5`;
the selection-freeze ID is
`26ab03780bb5846c80e39f452162f35e0e983b5af74df83bd38e5b7fd05bec98`;
and the seed-schedule ID is
`51ac2ad1140303734e7ee45aacd4c46c9174444ec75849934d29741a2f554c4c`.
The compact committed pointer under
`provenance/g00/dev35-d1-authority/PROVENANCE_INDEX.json` binds those IDs,
release hashes, row counts, the external manifest, and the no-expression
access receipt.

## Environment and provenance

The tested-environment record includes the Python build, OS and CPU identity,
NumPy BLAS/SIMD configuration, Torch MKL/OpenMP/build configuration, and each
direct dependency's installed `RECORD` and `METADATA` hashes. It is a CPU
contract-validation authority, not a GPU/container claim.

Historical conversation reviews are represented by the portable committed
hash authority under `provenance/reviews/`. Receipt generation contains no
user-specific attachment path.

## Status

| Surface | Status |
| --- | --- |
| Dev33-B canary | Closed, non-promotable |
| Dev34-A contract milestone | Preserved unchanged |
| Dev35 contracts/verifiers | Implemented and CPU-tested |
| Concrete D1 metadata freeze | Finalized and independently reverified; no expression read |
| G00C feature/cell execution | Not run |
| 2M extension | Not authorized unless base stops correctly |
| G00D and downstream real-data components | Blocked |
| GPU/model/biology | None |
