# Contract reference

All public contracts are strict, frozen Pydantic models with committed JSON
Schemas under `schemas/`. Unknown fields, unsafe relative paths, non-finite
numbers, incomplete intent dependencies, and incompatible topology are rejected.

The run compiler binds exact hashes for:

- recipe implementation, environment lock, and frozen CREDO artifact;
- semantic snapshot, row universe, features, split, and information set;
- exposure, eligibility, hierarchy, topology, denominator, and physical pools;
- correction, encoder/decoder, latent cache, model, objective, and loss scales;
- preregistration, multiplicity, candidate selection, baselines, budgets, and quota.

Source × target pilots additionally bind a `StateSelectionCalibration`: its
null method, repeated seeds, false-interaction bound, two advancement margins,
early checkpoint grid, and calibration-result hash. The resulting typed
`SelectionManifest` records the global-null, shrunk-target, full-interaction,
and incremental scores; inner and refit checkpoint identities; selected
family; and calibration hash. Inference contains an artifact reference to the
exact manifest, and evaluation must report the same family.

`compiled_run_id` hashes this fully resolved surface. Checkpoints and inference
bundles retain it as a parent. `run_id`, `evaluation_id`, and `sealed_id` add
their own artifact identities and never replace the compiled parent.

Feature identity is always the composite
`(namespace, namespace_version, feature_id)`. Signed 64-bit cell-row IDs are
stable and unique. Endpoint existence cannot determine abundance eligibility.

The Inference/Evaluation bundle split is intentional: inference weights remain
immutable and evaluator-free; a `SealedRun` binds the inference bundle, one or
more evaluation bundles, and the claim audit. This is the repository ADR in
[`0002`](adr/0002-sealed-bundle-split.md).
