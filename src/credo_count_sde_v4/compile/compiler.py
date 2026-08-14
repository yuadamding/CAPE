"""Bind prepared representation and protocol artifacts into one run identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..canonical import (
    canonical_json_bytes,
    contract_id,
    sha256_bytes,
    sha256_file,
)
from ..contracts import (
    BaselineRegistry,
    CandidateSelectionPlan,
    CompiledRunContract,
    DenominatorContract,
    EffectHierarchyContract,
    EligibilityManifest,
    MultiplicityPlan,
    PoolContract,
    PreregistrationRef,
    ResolvedRunCapabilities,
    SemanticStudySnapshot,
    SplitContract,
    TransportTopologyContract,
)
from ..errors import ContractError
from ..persistence import publish_directory
from ..prepare.pipeline import load_config, load_prepared_arrays
from ..runtime_identity import (
    environment_lock_hash,
    implementation_tree_hash,
    recipe_distribution_hash,
)
from ..version import RECIPE_ID, RECIPE_VERSION

FROZEN_CREDO_SHA256 = "2020b0f2549bb98b2b0fbb9384783d39be3ce11aeef9ae2bfecf232ae07fc684"


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _problem_hash(arrays: dict[str, np.ndarray]) -> str:
    rows = []
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        rows.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": sha256_bytes(value.tobytes(order="C")),
            }
        )
    return _hash({"schema_version": 1, "arrays": rows})


def _lookup_means(
    snapshot: SemanticStudySnapshot,
    row_ids: np.ndarray,
    latents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {int(row_id): index for index, row_id in enumerate(row_ids)}
    source: list[np.ndarray] = []
    terminal: list[np.ndarray] = []
    for series in snapshot.series:
        missing_source = [row for row in series.source_rows if row not in lookup]
        missing_terminal = [row for row in series.terminal_rows if row not in lookup]
        if missing_source or missing_terminal:
            raise ContractError(
                f"Series {series.series_id!r} references missing latent rows: "
                f"source={missing_source[:5]}, terminal={missing_terminal[:5]}."
            )
        if not series.source_rows:
            raise ContractError(f"Series {series.series_id!r} has no source state.")
        source.append(np.mean([latents[lookup[row]] for row in series.source_rows], axis=0))
        terminal.append(
            np.mean([latents[lookup[row]] for row in series.terminal_rows], axis=0)
            if series.terminal_rows
            else np.full(latents.shape[1], np.nan, dtype=np.float32)
        )
    return np.asarray(source, dtype=np.float32), np.asarray(terminal, dtype=np.float32)


def compile_problem(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / config.workspace).resolve()
    destination = workspace / "compiled"
    prepared, row_ids, latents = load_prepared_arrays(workspace)
    snapshot_path = (root / config.semantic_snapshot).resolve()
    snapshot = SemanticStudySnapshot.model_validate_json(snapshot_path.read_text())
    split_path = (root / config.split_contract).resolve()
    eligibility_path = (root / config.eligibility_manifest).resolve()
    hierarchy_path = (root / config.effect_hierarchy).resolve()
    topology_path = (root / config.topology_contract).resolve()
    preregistration_path = (root / config.preregistration).resolve()
    multiplicity_path = (root / config.multiplicity_plan).resolve()
    candidate_path = (root / config.candidate_selection_plan).resolve()
    baseline_path = (root / config.baseline_registry).resolve()
    source_manifest_path = (root / config.source_manifest).resolve()
    feature_permutation_path = (root / config.feature_permutation).resolve()
    _split = SplitContract.model_validate_json(split_path.read_text())
    eligibility = EligibilityManifest.model_validate_json(eligibility_path.read_text())
    hierarchy = EffectHierarchyContract.model_validate_json(hierarchy_path.read_text())
    topology = TransportTopologyContract.model_validate_json(topology_path.read_text())
    preregistration = PreregistrationRef.model_validate_json(preregistration_path.read_text())
    _multiplicity = MultiplicityPlan.model_validate_json(multiplicity_path.read_text())
    candidates = CandidateSelectionPlan.model_validate_json(candidate_path.read_text())
    baselines = BaselineRegistry.model_validate_json(baseline_path.read_text())
    denominator = (
        DenominatorContract.model_validate_json((root / config.denominator_contract).read_text())
        if config.denominator_contract
        else None
    )
    pools = (
        PoolContract.model_validate_json((root / config.pool_contract).read_text())
        if config.pool_contract
        else None
    )
    if snapshot.feature_index_hash != prepared.feature_index.sha256:
        # The feature artifact hash and logical ordered-key hash have distinct
        # domains. A semantic snapshot may bind either only when explicitly
        # generated from the prepared feature artifact.
        feature_rows = json.loads((workspace / prepared.feature_index.relative_uri).read_text())
        logical = _hash(feature_rows)
        if snapshot.feature_index_hash != logical:
            raise ContractError("Semantic snapshot feature identity does not match preparation.")
    source_z, terminal_z = _lookup_means(snapshot, row_ids, latents)
    if not np.isfinite(terminal_z).all(axis=1).any():
        raise ContractError("Every v4 intent requires at least one evaluable terminal state.")
    target_index = np.asarray([series.target_index for series in snapshot.series], dtype=np.int64)
    pool_index = np.asarray([series.pool_index for series in snapshot.series], dtype=np.int64)
    is_control = np.asarray([series.is_control for series in snapshot.series], dtype=np.uint8)
    duration = np.asarray([series.duration for series in snapshot.series], dtype=np.float32)
    grid_steps = np.ceil(duration / config.evaluation.max_step_duration).astype(np.int64)
    grid_step_size = duration / grid_steps
    source_counts = np.asarray([series.source_count for series in snapshot.series], dtype=np.int64)
    terminal_counts = np.asarray(
        [series.terminal_count for series in snapshot.series], dtype=np.int64
    )
    if target_index.max(initial=-1) >= config.model.target_count:
        raise ContractError("Configured target_count does not cover semantic target indices.")
    if pool_index.max(initial=-1) >= config.model.pool_count:
        raise ContractError("Configured pool_count does not cover semantic pool indices.")
    series_ids = tuple(series.series_id for series in snapshot.series)
    if config.intent.value != "count_state" and set(eligibility.abundance_eligible_units) != set(
        series_ids
    ):
        raise ContractError(
            "Measure/context abundance eligibility must retain the complete semantic catalog."
        )
    declared_state = set(eligibility.state_evaluable_units)
    actual_state = {series.series_id for series in snapshot.series if series.terminal_rows}
    if not declared_state <= actual_state:
        raise ContractError("State-evaluable eligibility includes a series without endpoint state.")
    hierarchy_rows = {row.perturbation_id: row for row in hierarchy.rows}
    if set(hierarchy_rows) != set(series_ids):
        raise ContractError("Effect hierarchy must cover the semantic series exactly.")
    for series in snapshot.series:
        row = hierarchy_rows[series.series_id]
        if row.target_index != series.target_index or row.is_control != series.is_control:
            raise ContractError(f"Effect hierarchy disagrees for {series.series_id}.")
    topology_series = {row.series_id for row in topology.support}
    if topology_series and topology_series != set(series_ids):
        raise ContractError("Topology support must cover the semantic series exactly.")
    if denominator is not None:
        blocks = {block.block_id: set(block.category_ids) for block in denominator.blocks}
        expected_blocks = {
            f"pool-{pool}": {
                series.series_id for series in snapshot.series if series.pool_index == pool
            }
            for pool in range(config.model.pool_count)
        }
        if blocks != expected_blocks:
            raise ContractError("Denominator blocks do not exactly match complete pool catalogs.")
    if pools is not None:
        if len(pools.physical_pool_ids) != config.model.pool_count:
            raise ContractError("Physical pool contract count differs from model pool_count.")
        contributor_series = {row.series_id for row in pools.contributors}
        if contributor_series != set(series_ids):
            raise ContractError("Physical pool contributors must cover every semantic series.")
        if len(set(float(value) for value in duration)) != 1:
            raise ContractError(
                "Dynamic context currently requires one common physical integration interval."
            )
    if candidates.information_set_hash != sha256_file(root / config.information_set):
        raise ContractError("Candidate selection plan uses a different information set.")
    if eligibility.source_information_set_hash != sha256_file(root / config.information_set):
        raise ContractError("Eligibility manifest uses a different information set.")
    if not preregistration.frozen_before_evaluation:
        raise ContractError("Preregistration must be frozen before evaluation.")
    implementation_hash = implementation_tree_hash()
    environment_hash = environment_lock_hash()
    config_hash = _hash(config.model_dump(mode="json"))
    exposure_rows = {
        (row.unit_type, row.unit_id): row.model_dump(mode="json")
        for row in snapshot.exposure_registry.records
    }
    for series_id in series_ids:
        exposure_rows.setdefault(
            ("series", series_id),
            {
                "unit_type": "series",
                "unit_id": series_id,
                "endpoint_seen": False,
                "used_for_architecture": False,
                "used_for_hyperparameters": False,
                "used_for_thresholds": False,
                "used_for_biological_story": False,
                "first_exposure_date": None,
                "source_artifact_hash": None,
                "exposure_role": "development",
            },
        )
    exposure_hash = _hash(
        {"schema_version": 1, "records": [exposure_rows[key] for key in sorted(exposure_rows)]}
    )
    split_hash = sha256_file(split_path)
    denominator_hash = (
        sha256_file(root / config.denominator_contract)
        if config.denominator_contract is not None
        else None
    )
    pool_hash = (
        sha256_file(root / config.pool_contract) if config.pool_contract is not None else None
    )
    capabilities = ResolvedRunCapabilities.for_intent(config.intent).model_copy(
        update={"decode_gene_composition": bool(config.model.gene_decoder_features)}
    )
    problem_arrays = {
        "source_z": source_z,
        "terminal_z": terminal_z,
        "target_index": target_index,
        "pool_index": pool_index,
        "is_control": is_control,
        "duration": duration,
        "grid_steps": grid_steps,
        "grid_step_size": grid_step_size,
        "source_counts": source_counts,
        "terminal_counts": terminal_counts,
        "series_ids": np.asarray([series.series_id for series in snapshot.series]),
    }
    payload = {
        "schema_version": 1,
        "compiled_run_id": "pending",
        "recipe_id": RECIPE_ID,
        "recipe_version": RECIPE_VERSION,
        "recipe_wheel_hash": recipe_distribution_hash(),
        "frozen_credo_artifact_hash": FROZEN_CREDO_SHA256,
        "environment_lock_hash": environment_hash,
        "source_manifest_hash": sha256_file(source_manifest_path),
        "count_store_merkle_root": prepared.count_store.sha256,
        "row_universe_hash": snapshot.row_universe_hash,
        "feature_index_hash": snapshot.feature_index_hash,
        "feature_permutation_hash": sha256_file(feature_permutation_path),
        "exposure_registry_hash": exposure_hash,
        "split_manifest_hash": split_hash,
        "information_set_hash": prepared.information_set.sha256,
        "eligibility_manifest_hash": sha256_file(eligibility_path),
        "target_hierarchy_hash": sha256_file(hierarchy_path),
        "denominator_manifest_hash": denominator_hash,
        "experimental_topology_hash": sha256_file(topology_path),
        "physical_pool_manifest_hash": pool_hash,
        "preregistration_hash": sha256_file(preregistration_path),
        "multiplicity_plan_hash": sha256_file(multiplicity_path),
        "candidate_selection_plan_hash": sha256_file(candidate_path),
        "correction_contract_hash": prepared.input_view.sha256,
        "representation_id": prepared.prepared_id,
        "latent_cache_index_hash": prepared.latent_cache.sha256,
        "compiled_problem_hash": _problem_hash(problem_arrays),
        "resolved_config_hash": config_hash,
        "mathematical_contract_hash": _hash(
            {
                "selection_gauge": "weighted_zero",
                "fitness_gauge": "pool_weighted_zero",
                "physical_grid": {
                    "maximum_step": config.evaluation.max_step_duration,
                    "steps": grid_steps.tolist(),
                    "step_sizes": grid_step_size.tolist(),
                },
                "m2": False,
                "capture": False,
            }
        ),
        "count_estimator_contract_hash": _hash(
            {"likelihood": "exact_full_block_dm", "smoothing": 0.5, "lgamma": "float64"}
        )
        if denominator_hash
        else None,
        "loss_scale_hash": _hash({"state": 1.0, "count": 1.0, "context": 1.0}),
        "baseline_registry_hash": _hash(
            {
                "registry": baselines.model_dump(mode="json"),
                "representation_id": prepared.prepared_id,
            }
        ),
        "compute_budget_hash": _hash({"max_updates": config.training.max_updates}),
        "output_quota_hash": _hash({"bytes": config.evaluation.output_bytes_limit}),
        "implementation_tree_hash": implementation_hash,
        "capabilities": capabilities.model_dump(mode="json"),
    }
    payload["compiled_run_id"] = contract_id(payload, id_field="compiled_run_id")
    contract = CompiledRunContract.model_validate(payload)

    def writer(temp: Path) -> None:
        (temp / "contract.json").write_bytes(
            canonical_json_bytes(contract.model_dump(mode="json")) + b"\n"
        )
        (temp / "config.json").write_bytes(
            canonical_json_bytes(config.model_dump(mode="json")) + b"\n"
        )
        (temp / "snapshot.json").write_bytes(
            canonical_json_bytes(snapshot.model_dump(mode="json")) + b"\n"
        )
        with (temp / "problem.npz").open("xb") as handle:
            np.savez(handle, **problem_arrays)

    publish_directory(destination, writer)
    return destination


def load_compiled_problem(workspace: Path) -> tuple[CompiledRunContract, dict[str, np.ndarray]]:
    root = workspace / "compiled"
    contract = CompiledRunContract.model_validate_json((root / "contract.json").read_text())
    with np.load(root / "problem.npz", allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    if _problem_hash(arrays) != contract.compiled_problem_hash:
        raise ContractError("Compiled problem arrays do not match the bound semantic hash.")
    return contract, arrays
