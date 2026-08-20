# Dev37 G00C source-derived execution seal

Last verified: 2026-08-20. Status: finalized expression-free hardening and
metadata-only D1 authority. Authority: strict Dev37 contracts, executable
verifiers, generated schemas, focused real-store tests, and the concrete A3
authority/archive recorded under `provenance/g00/dev37-d1-authority/`.

## Decision boundary

Dev37 preserves Dev36 commit `163f4d01dd66062697d366f165837214b22cfbee`
and its A2 authority as immutable historical evidence. It changes the contract
and verifier bytes, so it is a new `4.0.0.dev37` lineage. This implementation
does not read cohort expression, execute a G00C curve, authorize the 2M
extension, construct G00D, train a model, use a GPU, or make a biological
claim.

The independent review of Dev36 accepted its typed execution architecture but
found six remaining claim-bearing defects: caller-supplied source substitution,
execution-supplied sufficient statistics, semantically unbound publication,
legacy extension parents, self-attested monitoring/access/restart evidence,
and a sampler plan chosen after access. Dev37 addresses exactly those defects.

## Closed source authority

`G00CSourcePlaneBindingV4` freezes:

- the exact accepted G00B V2 manifest bytes and SHA-256;
- virtual-store and source-authority identities;
- canonical feature-index, row-locator, feature-permutation, and guide-target
  crosswalk hashes;
- every ordered source ID, checkpoint, and source-file SHA-256;
- the canonical feature count and exact `CUSTOM001_PuroR` index.

The claim-bearing verifier accepts source-plane and source-file roots, not a
preconstructed store object. It constructs `VirtualCanonicalCountStore`
itself, calls `verify(full=True)`, verifies the accepted manifest bytes, checks
every source hash, and independently compares each source H5AD `var/_index`
through its frozen permutation to the canonical index. A caller-defined
`TinyStore` or coherent alternative source surface cannot satisfy this path.

## Independent hierarchy and ranking

The metadata-only authority freezes a complete sampler hierarchy and a
`G00CHierarchyDerivationReceiptV4`. The verifier independently reconstructs
`source_index`, `target_code`, `guide_code`, and `is_control` from the accepted
G00B locator and crosswalk, then checks ordered column hashes and exact table
bytes.

At execution, `G00CFeatureRankingReceiptV4` is not trusted as a score table.
The verifier streams the exact first-million training rows, excludes the PuroR
sidecar, and recomputes checkpoint-conditioned Poisson deviance with the frozen
tie order:

1. deviance descending;
2. total UMI descending;
3. detection count descending;
4. UTF-8 feature ID ascending.

The complete ranking, 4,096-feature prefix, canonical indices, feature IDs,
PuroR identity, and ordered hashes must all match source-derived values.

## Pre-access sampler authority

`G00CSamplerPlanV4` is now a parent of expression access. Its base plan fixes
all 590 `(candidate kind, candidate value, draw)` entries, the macro-update
budget, 512-cell microbatches, eight microbatches per macro-update, 4,096-cell
macrobatches, resume cursor, six-stream seed-schedule identity, hierarchy, raw
inverse-probability weighting, and exact total trace size. The plan namespace
is the already frozen Dev34 selection identity, avoiding an authority-ID cycle.

`G00CSamplerEvidenceV4` binds the pre-access plan and distinct uninterrupted
and resumed traces. The verifier regenerates all draw rows and PCG64DXSM state
transitions. `G00CDurableRestartReceiptV4` additionally requires a durable
checkpoint, different attempt IDs, different PIDs, different output paths, no
final payload from the interrupted attempt, and byte-identical paired outputs.

## Source-derived refits

`G00CRefitReplayReceiptV4` binds source-derived arrays with exact shapes:

```text
training:   [59, candidate, Rest|Stim8hr|Stim48hr, 4096]
validation: [59, Rest|Stim8hr|Stim48hr, 4096]
```

For each plan entry, the verifier reads the exact hierarchical trace rows,
applies raw inverse draw weights, replays the draw's PCG64DXSM binomial
half-count stream in deterministic CSR order, and accumulates per-checkpoint
gene counts. Feature candidates share the same first-million row surface before
prefix collapse. Cell candidates use exact nested row prefixes. Validation
vectors are derived once per frozen draw and must be byte-identical across
candidates.

Only after deriving these arrays does the verifier replay checkpoint-specific
common-support intercepts, NLL sums, per-count NLLs, and state hashes. A
coherently replaced statistics-plus-refit package is rejected because its
arrays differ from source-derived values.

## Monitoring, access, and materialization

`G00CSourceAccessLedgerReceiptV4` binds an ordered, monotonic, role-labelled
row ledger including source index and checkpoint. The verifier checks every row
against G00B and its frozen role, compares the complete expected access order,
and derives zero protected D1 stimulated reads from the full ledger.

`G00CProcessTreeMonitorReceiptV4` binds an out-of-process trace containing the
root PID, aggregated process-tree RSS, descendants, readability, and monotonic
time. Unreadable-sample limits, maximum gap, complete source-access time
coverage, and the literal 68,719,476,736-byte ceiling are executable terminal
gates.

On a pass, `G00CMaterializationReceiptV4` wraps the complete Dev36 HDF5 source
equality checks but replaces same-reference writer evidence with the durable,
distinct-path restart receipt. It also binds the same access-ledger and monitor
receipts used by the execution bundle.

## Two-layer publication

The former final-decision cycle is removed:

```text
typed execution artifacts
  -> decision-free inner semantic inventory and manifest
  -> immutable G00CExecutionBundleV5
  -> independently recomputed G00CDecisionReceiptV5
  -> G00CFinalSealV1
```

Each inner filename maps to one semantic role and the exact `ArtifactRef`
reachable from the verified typed models. The on-disk `artifacts.json` must be
canonical-byte-identical to the bound inventory. Payload bytes, sizes,
`SHA256SUMS`, publication event, commit marker, no-clobber status, and absence
of extra files are checked. The final decision is forbidden from the inner
inventory.

The outer seal contains only the immutable bundle, inner manifest, final
decision, exact outer inventory, checksum file, the fixed `sealed` commit
marker, and seal. The fixed marker avoids a self-reference between the marker
hash and the final seal identity. A
claim-bearing G00D parent must pass the combined source-derived decision and
outer-seal verifier; an unsealed decision is insufficient operationally.

## V4 extension closure

`G00CSampleSizeExtensionFreezeV2` requires a sealed V5 base
`extension_required` stop, the same authority/source/feature/seed identities,
the exact base support receipt, a fresh extension sampler plan, one 2M
candidate, and no third extension. The extension has its own sampler evidence,
fresh-process restart, and support receipt. Both selected-extension and
`fail_no_saturation` paths remain blocked until a separately authorized base
execution actually produces the required sealed parent.

## Test evidence

Focused Dev37 tests use a real `VirtualCanonicalCountStore`, real HDF5 CSR
sources, real V2 source-plane contracts, all three checkpoints, 4,097 canonical
features, and a physical PuroR sidecar. They verify:

- full store/source hash and exact canonical ID/index mapping;
- source-derived ranking and PuroR exclusion;
- rejection after source-feature substitution;
- checkpoint-indexed `[59, candidate, 3, 4096]` statistics;
- rejection of a coherently altered sufficient-statistics artifact;
- complete pre-access plan validation;
- distinct fresh-process restart evidence;
- semantic publication rejection of arbitrary replacement bytes.

The test uses no monkeypatch, `SimpleNamespace`, cohort data, or GPU.

## Concrete A3 release

The first integration freeze stopped before publication because the Dev37
freezer replaced the bootstrap implementation directory without preserving its
two non-code semantic parents. Commit
`d82f5f8b3587cd1c2bd1ccaaaf40c463824d5807` preserves
`refit-model-config.json` and `common-support-prior.json`, while still replacing
the executable verifier files. The unpublished failed attempt left no authority
directory and accessed no expression source.

The corrected release was rebuilt twice with a fixed `SOURCE_DATE_EPOCH`; its
wheels and normalized sdists were byte-identical. Clean validation produced
291 passing tests, one intentional CUDA skip, zero failures, and 85.00% total
coverage. Ruff, mypy over 75 source files, generated-schema checks,
documentation-link checks, and wheel ZIP integrity passed.

The finalized metadata-only release is:

| Field | Value |
| --- | --- |
| Attempt | `G00C_D1_AUTHORITY_DEV37_A3` |
| Authority ID | `5ee82382643ed872d1a6ff191d07d2c7cc0c5793cc005e09db3466a059e49755` |
| Code commit | `d82f5f8b3587cd1c2bd1ccaaaf40c463824d5807` |
| Implementation tree | `6e0d2a566fc6c8e7aea1539f5ac8b09b0be90f8a0aa2270e5967a75269f19a0d` |
| Wheel SHA-256 | `3b9e24e014b145a3c7460354f29f660043bccd66a9fb8b8c6c51b5927e4d8924` |
| Normalized sdist SHA-256 | `66200da7c1eaccbc1570e214c994dd88464166fec413ea12064c4485557bb111` |
| Source binding ID | `40ef5a2789494a58015b5d67b1f9f025335c7718931249821f494eadf065ef07` |
| Pre-access sampler-plan ID | `003ac92ddc251fe3ef410a439a1dff460510efb98f800184056004204ef51ad4` |
| Planned trace rows | 19,333,120 |
| Hierarchy rows | 16,924,672 |
| Archive SHA-256 | `2f3085c9af3eebc6b5f567b163272011a79f54d1046c344d57cc216f6e5cf062` |
| Archive audit | `pass_independent_archive_audit`, 38 members |

The A3 source-access receipt states `raw_source_files_opened=false` and
`expression_values_accessed=false`. It binds all 12 source-file hashes and the
18,130-feature index, including `CUSTOM001_PuroR` at canonical index 18,129,
without reading a count matrix. It authorizes no G00C execution by itself.

## Current status

| Surface | Status |
| --- | --- |
| Frozen CREDO and accepted Dev33–Dev36 milestones | Preserved |
| Dev37 contracts and source-derived verifiers | Finalized at `d82f5f8…` |
| Real synthetic V2 store rehearsal | Passed |
| Fresh Dev37 D1 metadata authority | Finalized as A3; archive re-audit passed |
| Cohort expression access / G00C curve | Blocked |
| 2M extension | Blocked pending a sealed base stop |
| G00D/G04/G07/G08 | Blocked |
| GPU, model, biological evidence | None |
