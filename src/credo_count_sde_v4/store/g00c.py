"""Fail-closed verification of G00C materialization evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..canonical import canonical_json_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    FoldNativeCompactViewContractV2,
    G00CDecisionReceipt,
    G00CExecutionBundle,
)
from ..errors import IntegrityError
from .qualification import validate_g00c_decision
from .virtual import VirtualCanonicalCountStore


def _path(root: Path, artifact: ArtifactRef) -> Path:
    path = root / artifact.relative_uri
    if not path.is_file() or sha256_file(path) != artifact.sha256:
        raise IntegrityError(f"G00C artifact failed verification: {artifact.relative_uri}.")
    return path


def _row_hash(values: np.ndarray) -> str:
    import hashlib

    ordered = np.sort(np.asarray(values, dtype="<i8"), kind="stable")
    return hashlib.sha256(ordered.tobytes(order="C")).hexdigest()


def _verify_row_roles(
    root: Path,
    store: VirtualCanonicalCountStore,
    contract: FoldNativeCompactViewContractV2,
    bundle: G00CExecutionBundle,
) -> None:
    table = pd.read_parquet(_path(root, bundle.row_roles))
    expected_columns = (
        "row_id",
        "role",
        "donor_id",
        "checkpoint",
        "in_compact_payload",
    )
    if tuple(table.columns) != expected_columns or table["row_id"].duplicated().any():
        raise IntegrityError("G00C row-role artifact has an invalid schema or duplicate rows.")
    locator_ids, source_indices, _ = store._locator()
    observed_ids = table["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(observed_ids), locator_ids):
        raise IntegrityError("G00C row roles do not cover the exact G00B eligible universe.")
    locator_position = np.searchsorted(locator_ids, observed_ids)
    source_ids = source_indices[locator_position]
    expected_donors = np.asarray(
        [store.manifest.sources[int(index)].donor_id for index in source_ids]
    )
    expected_checkpoints = np.asarray(
        [store.manifest.sources[int(index)].checkpoint for index in source_ids]
    )
    expected_times = np.asarray(
        [store.manifest.sources[int(index)].physical_time_hours for index in source_ids]
    )
    if not np.array_equal(table["donor_id"].astype(str).to_numpy(), expected_donors):
        raise IntegrityError("G00C row-role donor identities differ from G00B.")
    if not np.array_equal(table["checkpoint"].astype(str).to_numpy(), expected_checkpoints):
        raise IntegrityError("G00C row-role checkpoints differ from G00B.")
    roles = table["role"].astype(str).to_numpy()
    donors = table["donor_id"].astype(str).to_numpy()
    training = np.isin(donors, contract.training_donor_ids)
    heldout = donors == contract.heldout_donor_id
    valid = (
        (np.isin(roles, ("training_fit", "training_validation")) & training)
        | ((roles == "heldout_source_query") & heldout & (expected_times == 0))
        | ((roles == "protected_heldout_stimulated") & heldout & (expected_times > 0))
    )
    if not np.all(valid):
        raise IntegrityError("G00C row roles violate donor/checkpoint permissions.")
    compact = table["in_compact_payload"].to_numpy(dtype=bool)
    if np.any(compact[roles == "protected_heldout_stimulated"]):
        raise IntegrityError("Protected held-out stimulated rows entered the compact payload.")
    if _row_hash(observed_ids[compact]) != bundle.compact_payload_row_ids_hash:
        raise IntegrityError("G00C compact payload row universe differs from its row roles.")
    for record in contract.row_roles:
        selected = observed_ids[roles == record.role]
        if len(selected) != record.rows or _row_hash(selected) != record.row_ids_hash:
            raise IntegrityError(f"G00C row-role receipt differs for {record.role}.")


def _verify_feature_selection(
    root: Path,
    contract: FoldNativeCompactViewContractV2,
    bundle: G00CExecutionBundle,
) -> None:
    result = bundle.feature_selection
    curve = pd.read_parquet(_path(root, result.curve))
    draws = pd.read_parquet(_path(root, result.paired_refit_draws))
    ordered = pd.read_parquet(_path(root, result.ordered_features))
    candidates = contract.feature_selection.candidate_feature_counts
    if tuple(curve.columns) != (
        "feature_count",
        "mean_validation_nll",
        "p95_absolute_difference_to_4096",
    ) or tuple(draws.columns) != ("draw_id", "feature_count", "validation_nll"):
        raise IntegrityError("G00C feature-selection evidence has an invalid schema.")
    if tuple(curve["feature_count"].astype(int)) != candidates:
        raise IntegrityError("G00C feature curve differs from the frozen candidate grid.")
    pivot = draws.pivot(index="draw_id", columns="feature_count", values="validation_nll")
    if (
        tuple(pivot.columns.astype(int)) != candidates
        or len(pivot) != 59
        or pivot.isna().any().any()
    ):
        raise IntegrityError("G00C feature refits are incomplete or duplicated.")
    means = pivot.mean(axis=0).to_numpy()
    p95 = np.quantile(np.abs(pivot.to_numpy() - pivot[4096].to_numpy()[:, None]), 0.95, axis=0)
    if not np.allclose(curve["mean_validation_nll"], means, atol=1e-12, rtol=0) or not np.allclose(
        curve["p95_absolute_difference_to_4096"], p95, atol=1e-12, rtol=0
    ):
        raise IntegrityError("G00C feature curve is not derived from paired refits.")
    eligible = [
        count
        for count, value in zip(candidates, p95, strict=True)
        if value <= contract.feature_selection.minimum_improvement_margin
    ]
    if not eligible or result.selected_feature_count != min(eligible):
        raise IntegrityError("G00C selected feature prefix violates the frozen rule.")
    expected_order_columns = ("rank", "feature_id", "in_primary_metric", "is_sidecar")
    if tuple(ordered.columns) != expected_order_columns or ordered["feature_id"].duplicated().any():
        raise IntegrityError("G00C ordered-feature table has an invalid schema.")
    primary = ordered.loc[~ordered["is_sidecar"].astype(bool)]
    sidecar = ordered.loc[ordered["is_sidecar"].astype(bool)]
    if (
        len(primary) != 4096
        or not np.array_equal(primary["rank"].to_numpy(dtype=int), np.arange(1, 4097))
        or not primary["in_primary_metric"].astype(bool).all()
        or len(sidecar) != 1
        or str(sidecar.iloc[0]["feature_id"]) != "CUSTOM001_PuroR"
        or bool(sidecar.iloc[0]["in_primary_metric"])
    ):
        raise IntegrityError("G00C ordered features violate the primary/technical-sidecar rule.")
    import hashlib

    feature_order_hash = hashlib.sha256(
        canonical_json_bytes(primary["feature_id"].astype(str).tolist())
    ).hexdigest()
    if feature_order_hash != bundle.compact_payload_feature_order_hash:
        raise IntegrityError("G00C compact payload feature order differs from selection evidence.")


def _verify_sample_size(
    root: Path,
    contract: FoldNativeCompactViewContractV2,
    bundle: G00CExecutionBundle,
) -> None:
    result = bundle.sample_size_selection
    curve = pd.read_parquet(_path(root, result.curve))
    draws = pd.read_parquet(_path(root, result.paired_refit_draws))
    candidates = contract.sample_size_selection.candidate_cells
    if tuple(curve.columns) != (
        "training_cells",
        "mean_validation_nll",
        "p95_absolute_difference_to_nmax",
    ) or tuple(draws.columns) != ("draw_id", "training_cells", "validation_nll"):
        raise IntegrityError("G00C sample-size evidence has an invalid schema.")
    if tuple(curve["training_cells"].astype(int)) != candidates:
        raise IntegrityError("G00C sample-size curve differs from the frozen grid.")
    pivot = draws.pivot(index="draw_id", columns="training_cells", values="validation_nll")
    if (
        tuple(pivot.columns.astype(int)) != candidates
        or len(pivot) != 59
        or pivot.isna().any().any()
    ):
        raise IntegrityError("G00C sample-size refits are incomplete or duplicated.")
    maximum = candidates[-1]
    means = pivot.mean(axis=0).to_numpy()
    p95 = np.quantile(np.abs(pivot.to_numpy() - pivot[maximum].to_numpy()[:, None]), 0.95, axis=0)
    if not np.allclose(curve["mean_validation_nll"], means, atol=1e-12, rtol=0) or not np.allclose(
        curve["p95_absolute_difference_to_nmax"], p95, atol=1e-12, rtol=0
    ):
        raise IntegrityError("G00C sample-size curve is not derived from paired refits.")
    epsilon = contract.sample_size_selection.equivalence_epsilon
    eligible = [count for count, value in zip(candidates, p95, strict=True) if value <= epsilon]
    if not eligible or result.selected_training_cells != min(eligible):
        raise IntegrityError("G00C selected training-cell count violates the saturation rule.")
    opened = 2_000_000 in candidates
    if result.two_million_extension_opened != opened:
        raise IntegrityError("G00C two-million-cell extension evidence is inconsistent.")
    if opened and np.any(p95[:5] <= epsilon):
        raise IntegrityError("G00C opened two million cells before proving nonsaturation.")


def _verify_sampler(
    root: Path,
    contract: FoldNativeCompactViewContractV2,
    bundle: G00CExecutionBundle,
) -> None:
    plan_path = _path(root, bundle.sampler.sampler_plan)
    index = pd.read_parquet(_path(root, bundle.sampler.epoch_index))
    resume = json.loads(_path(root, bundle.sampler.resume_test).read_text())
    plan = json.loads(plan_path.read_text())
    expected_plan = {
        "implementation_sha256": contract.sampler.implementation_sha256,
        "microbatch_cells": 512,
        "microbatches_per_update": 8,
        "macrobatch_cells": 4096,
        "rng_algorithm": contract.sampler.rng_algorithm,
        "rng_seed": contract.sampler.rng_seed,
        "thinning_rule": contract.sampler.thinning_rule,
        "resume_cursor_schema": contract.sampler.resume_cursor_schema,
        "resume_cursor_initial_hash": contract.sampler.resume_cursor_initial_hash,
        "donor_weighting": "equal",
        "checkpoint_weighting": "equal",
        "target_weighting": "equal",
        "guide_weighting_within_target": "equal",
    }
    if plan != expected_plan:
        raise IntegrityError("G00C sampler plan differs from its frozen contract.")
    expected_columns = (
        "update",
        "microbatch",
        "row_ids_hash",
        "sample_weights_hash",
        "thinning_draws_hash",
        "rng_state_hash",
        "resume_cursor_hash",
    )
    if tuple(index.columns) != expected_columns or index.duplicated(["update", "microbatch"]).any():
        raise IntegrityError("G00C sampler epoch index has an invalid schema.")
    counts = index.groupby("update", sort=True)["microbatch"].agg(list)
    if counts.empty or any(sorted(values) != list(range(8)) for values in counts):
        raise IntegrityError("G00C sampler trace lacks exact microbatch boundaries.")
    if resume != {
        "status": "pass",
        "row_sequence_equal": True,
        "sample_weights_equal": True,
        "thinning_draws_equal": True,
        "rng_state_sequence_equal": True,
        "resume_cursor_sequence_equal": True,
    }:
        raise IntegrityError("G00C sampler resume evidence failed.")


def validate_g00c_execution(
    root: Path,
    store: VirtualCanonicalCountStore,
    contract: FoldNativeCompactViewContractV2,
    bundle: G00CExecutionBundle,
    receipt: G00CDecisionReceipt,
) -> None:
    """Recompute every G00C selection, row, sampler, and publication gate."""

    validate_g00c_decision(contract, bundle, receipt)
    _verify_row_roles(root, store, contract, bundle)
    _verify_feature_selection(root, contract, bundle)
    _verify_sample_size(root, contract, bundle)
    _verify_sampler(root, contract, bundle)
    _path(root, bundle.compact_payload)
    if bundle.compact_payload_counts_sha256 != bundle.compact_payload.sha256:
        raise IntegrityError("G00C compact count hash differs from its payload artifact.")
    publication = json.loads(_path(root, bundle.publication_manifest).read_text())
    reload_receipt = json.loads(_path(root, bundle.reload_receipt).read_text())
    if (
        publication.get("status") != "pass"
        or publication.get("fold_view_id") != contract.fold_view_id
    ):
        raise IntegrityError("G00C immutable-publication evidence failed.")
    if (
        reload_receipt.get("status") != "pass"
        or reload_receipt.get("compact_payload_sha256") != bundle.compact_payload.sha256
    ):
        raise IntegrityError("G00C full-reload evidence failed.")
    if not all(
        (
            receipt.row_roles_verified,
            receipt.feature_selection_verified,
            receipt.sample_size_selection_verified,
            receipt.sampler_sequence_verified,
            receipt.compact_counts_verified,
            receipt.protected_rows_absent,
            receipt.immutable_publication_verified,
            receipt.full_reload_verified,
        )
    ):
        raise IntegrityError("G00C decision flags disagree with verified evidence.")
