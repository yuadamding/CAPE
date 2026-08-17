from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from credo_count_sde_v4.canonical import canonical_json_bytes, contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    CompactSamplerContract,
    FoldNativeCompactViewContractV2,
    FoldRowRoleRecord,
    G00CDecisionReceipt,
    G00CExecutionBundle,
    G00CFeatureSelectionResult,
    G00CSamplerEvidence,
    G00CSampleSizeSelectionResult,
    G00SourcePlaneV2Amendment,
    G00SourcePlaneV2AmendmentReceipt,
    SourceHashBinding,
    StrictModel,
    TrainingOnlyFeatureSelectionContract,
    TrainingOnlySampleSizeSelectionContract,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import validate_g00c_execution

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized = model.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in model.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _artifact(path: Path, root: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.artifact",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.relative_to(root).as_posix(),
    )


def _row_hash(values: list[int]) -> str:
    return hashlib.sha256(np.sort(np.asarray(values, dtype="<i8")).tobytes()).hexdigest()


def test_source_plane_amendment_requires_exact_twelve_sources_and_derived_status() -> None:
    sources = tuple(
        SourceHashBinding(source_id=f"D{index}", source_file_sha256=f"{index:x}" * 64)
        for index in range(12)
    )
    amendment = _identified(
        G00SourcePlaneV2Amendment,
        {
            "schema_version": 1,
            "parent_g00a_v1_authority_id": "g00a-v1",
            "parent_g00a_v1": ArtifactRef(
                schema_id="g00a",
                schema_version=1,
                sha256="a" * 64,
                size_bytes=1,
                media_type="application/json",
                relative_uri="g00a.json",
            ).model_dump(mode="json"),
            "parent_g00b_v1_virtual_store_id": "g00b-v1",
            "parent_g00b_v1_manifest": ArtifactRef(
                schema_id="g00b",
                schema_version=1,
                sha256="b" * 64,
                size_bytes=1,
                media_type="application/json",
                relative_uri="g00b.json",
            ).model_dump(mode="json"),
            "immutable_source_hashes": [source.model_dump(mode="json") for source in sources],
            "v2_guide_target_crosswalk": ArtifactRef(
                schema_id="crosswalk",
                schema_version=1,
                sha256="c" * 64,
                size_bytes=1,
                media_type="application/x-parquet",
                relative_uri="crosswalk.parquet",
            ).model_dump(mode="json"),
            "v2_numerical_audit": ArtifactRef(
                schema_id="numeric",
                schema_version=1,
                sha256="d" * 64,
                size_bytes=1,
                media_type="application/x-parquet",
                relative_uri="numeric.parquet",
            ).model_dump(mode="json"),
            "v2_source_derivation_receipt": ArtifactRef(
                schema_id="derivation",
                schema_version=1,
                sha256="e" * 64,
                size_bytes=1,
                media_type="application/json",
                relative_uri="derivation.json",
            ).model_dump(mode="json"),
            "v2_row_locator": ArtifactRef(
                schema_id="locator",
                schema_version=1,
                sha256="f" * 64,
                size_bytes=1,
                media_type="application/x-hdf5",
                relative_uri="locator.h5",
            ).model_dump(mode="json"),
            "builder_implementation_sha256": "1" * 64,
            "environment_hash": "2" * 64,
        },
        "amendment_id",
    )
    assert not amendment.model_fitting_performed
    with pytest.raises(ValidationError, match="exactly 12"):
        _identified(
            G00SourcePlaneV2Amendment,
            {
                **amendment.model_dump(mode="json"),
                "amendment_id": "pending",
                "immutable_source_hashes": amendment.immutable_source_hashes[:-1],
            },
            "amendment_id",
        )
    receipt_payload = {
        "schema_version": 1,
        "amendment_id": amendment.amendment_id,
        "derived_g00a_v2_authority_id": "g00a-v2",
        "derived_g00a_v2": amendment.parent_g00a_v1.model_dump(mode="json"),
        "derived_g00b_v2_virtual_store_id": "g00b-v2",
        "derived_g00b_v2": amendment.parent_g00b_v1_manifest.model_dump(mode="json"),
        "parent_files_verified": True,
        "all_source_hashes_verified": True,
        "derivation_receipt_verified": True,
        "v2_parent_equality_verified": True,
        "status": "pass",
    }
    receipt = _identified(G00SourcePlaneV2AmendmentReceipt, receipt_payload, "receipt_id")
    assert receipt.status == "pass"
    with pytest.raises(ValidationError, match="status must be fail"):
        _identified(
            G00SourcePlaneV2AmendmentReceipt,
            {
                **receipt.model_dump(mode="json"),
                "receipt_id": "pending",
                "parent_files_verified": False,
            },
            "receipt_id",
        )


def _write_g00c_evidence(
    root: Path,
) -> tuple[Any, FoldNativeCompactViewContractV2, G00CExecutionBundle, G00CDecisionReceipt]:
    role_ids = {
        "training_fit": [10],
        "training_validation": [20],
        "heldout_source_query": [30],
        "protected_heldout_stimulated": [40],
    }
    feature_contract = TrainingOnlyFeatureSelectionContract(
        implementation_sha256="1" * 64,
        fit_rows_hash="2" * 64,
        minimum_improvement_margin=0.001,
        ordered_feature_table=ArtifactRef(
            schema_id="ordered",
            schema_version=1,
            sha256="3" * 64,
            size_bytes=1,
            media_type="application/x-parquet",
            relative_uri="ORDERED_FEATURES.parquet",
        ),
    )
    sample_contract = TrainingOnlySampleSizeSelectionContract(
        equivalence_epsilon=0.001,
        candidate_cells=(50_000, 100_000, 250_000, 500_000, 1_000_000),
    )
    sampler_contract = CompactSamplerContract(
        implementation_sha256="4" * 64,
        rng_algorithm="PCG64DXSM",
        rng_seed=7,
        thinning_rule="without_replacement_within_microbatch",
        resume_cursor_schema="weighted_sampler_cursor_v1",
        resume_cursor_initial_hash="5" * 64,
    )
    contract = _identified(
        FoldNativeCompactViewContractV2,
        {
            "schema_version": 2,
            "parent_source_authority_id": "authority",
            "parent_virtual_store_id": "store",
            "parent_eligible_row_ids_hash": _row_hash([10, 20, 30, 40]),
            "parent_guide_target_crosswalk_hash": "6" * 64,
            "outer_split_id": "lodo-D2",
            "training_donor_ids": ["D1"],
            "heldout_donor_id": "D2",
            "row_roles": [
                FoldRowRoleRecord(role=role, rows=1, row_ids_hash=_row_hash(ids)).model_dump(
                    mode="json"
                )
                for role, ids in role_ids.items()
            ],
            "row_role_audit": ArtifactRef(
                schema_id="roles",
                schema_version=1,
                sha256="7" * 64,
                size_bytes=1,
                media_type="application/x-parquet",
                relative_uri="FOLD_ROW_ROLES.parquet",
            ).model_dump(mode="json"),
            "row_role_assignment_implementation_sha256": "8" * 64,
            "feature_selection": feature_contract.model_dump(mode="json"),
            "sample_size_selection": sample_contract.model_dump(mode="json"),
            "sampler": sampler_contract.model_dump(mode="json"),
            "count_dtype": "uint16",
            "index_dtype": "uint16",
            "maximum_observed_count": 60,
            "physical_layout": "donor_checkpoint_target_guide_row",
        },
        "fold_view_id",
    )
    roles = pd.DataFrame(
        {
            "row_id": [10, 20, 30, 40],
            "role": list(role_ids),
            "donor_id": ["D1", "D1", "D2", "D2"],
            "checkpoint": ["Rest", "Stim8hr", "Rest", "Stim8hr"],
            "in_compact_payload": [True, True, True, False],
        }
    )
    roles_path = root / "FOLD_ROW_ROLES.parquet"
    roles.to_parquet(roles_path, index=False)
    feature_candidates = feature_contract.candidate_feature_counts
    feature_penalties = dict(
        zip(feature_candidates, (0.01, 0.005, 0.0005, 0.0002, 0.0), strict=True)
    )
    feature_draws = pd.DataFrame(
        [
            (draw, count, 1.0 + feature_penalties[count])
            for draw in range(59)
            for count in feature_candidates
        ],
        columns=("draw_id", "feature_count", "validation_nll"),
    )
    feature_draws_path = root / "FEATURE_SELECTION_REFIT_DRAWS.parquet"
    feature_draws.to_parquet(feature_draws_path, index=False)
    feature_curve = pd.DataFrame(
        {
            "feature_count": feature_candidates,
            "mean_validation_nll": [1.0 + feature_penalties[count] for count in feature_candidates],
            "p95_absolute_difference_to_4096": [
                feature_penalties[count] for count in feature_candidates
            ],
        }
    )
    feature_curve_path = root / "FEATURE_SELECTION_CURVE.parquet"
    feature_curve.to_parquet(feature_curve_path, index=False)
    ordered = pd.DataFrame(
        {
            "rank": [*range(1, 4097), 0],
            "feature_id": [*[f"gene-{index}" for index in range(4096)], "CUSTOM001_PuroR"],
            "in_primary_metric": [*([True] * 4096), False],
            "is_sidecar": [*([False] * 4096), True],
        }
    )
    ordered_path = root / "ORDERED_FEATURES.parquet"
    ordered.to_parquet(ordered_path, index=False)
    sample_candidates = sample_contract.candidate_cells
    sample_penalties = dict(zip(sample_candidates, (0.01, 0.005, 0.0009, 0.0002, 0.0), strict=True))
    sample_draws = pd.DataFrame(
        [
            (draw, count, 1.0 + sample_penalties[count])
            for draw in range(59)
            for count in sample_candidates
        ],
        columns=("draw_id", "training_cells", "validation_nll"),
    )
    sample_draws_path = root / "SAMPLE_SIZE_REFIT_DRAWS.parquet"
    sample_draws.to_parquet(sample_draws_path, index=False)
    sample_curve = pd.DataFrame(
        {
            "training_cells": sample_candidates,
            "mean_validation_nll": [1.0 + sample_penalties[count] for count in sample_candidates],
            "p95_absolute_difference_to_nmax": [
                sample_penalties[count] for count in sample_candidates
            ],
        }
    )
    sample_curve_path = root / "SAMPLE_SIZE_CURVE.parquet"
    sample_curve.to_parquet(sample_curve_path, index=False)
    plan = {
        "implementation_sha256": sampler_contract.implementation_sha256,
        "microbatch_cells": 512,
        "microbatches_per_update": 8,
        "macrobatch_cells": 4096,
        "rng_algorithm": sampler_contract.rng_algorithm,
        "rng_seed": sampler_contract.rng_seed,
        "thinning_rule": sampler_contract.thinning_rule,
        "resume_cursor_schema": sampler_contract.resume_cursor_schema,
        "resume_cursor_initial_hash": sampler_contract.resume_cursor_initial_hash,
        "donor_weighting": "equal",
        "checkpoint_weighting": "equal",
        "target_weighting": "equal",
        "guide_weighting_within_target": "equal",
    }
    plan_path = root / "SAMPLER_PLAN.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    epoch_path = root / "SAMPLER_EPOCH_INDEX.parquet"
    pd.DataFrame(
        [(0, microbatch, *([f"{microbatch + 1:x}" * 64] * 5)) for microbatch in range(8)],
        columns=(
            "update",
            "microbatch",
            "row_ids_hash",
            "sample_weights_hash",
            "thinning_draws_hash",
            "rng_state_hash",
            "resume_cursor_hash",
        ),
    ).to_parquet(epoch_path, index=False)
    resume_path = root / "SAMPLER_RESUME_TEST.json"
    resume_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "row_sequence_equal": True,
                "sample_weights_equal": True,
                "thinning_draws_equal": True,
                "rng_state_sequence_equal": True,
                "resume_cursor_sequence_equal": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    compact_path = root / "compact-counts.bin"
    compact_path.write_bytes(b"compact")
    publication_path = root / "PUBLICATION.json"
    publication_path.write_text(
        json.dumps({"status": "pass", "fold_view_id": contract.fold_view_id}) + "\n"
    )
    reload_path = root / "RELOAD.json"
    reload_path.write_text(
        json.dumps({"status": "pass", "compact_payload_sha256": sha256_file(compact_path)}) + "\n"
    )
    primary_ids = ordered.loc[~ordered["is_sidecar"], "feature_id"].tolist()
    bundle = _identified(
        G00CExecutionBundle,
        {
            "schema_version": 1,
            "fold_view_id": contract.fold_view_id,
            "row_roles": _artifact(roles_path, root, "application/x-parquet").model_dump(
                mode="json"
            ),
            "feature_selection": G00CFeatureSelectionResult(
                curve=_artifact(feature_curve_path, root, "application/x-parquet"),
                paired_refit_draws=_artifact(feature_draws_path, root, "application/x-parquet"),
                ordered_features=_artifact(ordered_path, root, "application/x-parquet"),
                selected_feature_count=1024,
            ).model_dump(mode="json"),
            "sample_size_selection": G00CSampleSizeSelectionResult(
                curve=_artifact(sample_curve_path, root, "application/x-parquet"),
                paired_refit_draws=_artifact(sample_draws_path, root, "application/x-parquet"),
                selected_training_cells=250_000,
                two_million_extension_opened=False,
            ).model_dump(mode="json"),
            "sampler": G00CSamplerEvidence(
                sampler_plan=_artifact(plan_path, root, "application/json"),
                epoch_index=_artifact(epoch_path, root, "application/x-parquet"),
                resume_test=_artifact(resume_path, root, "application/json"),
            ).model_dump(mode="json"),
            "compact_payload": _artifact(compact_path, root, "application/octet-stream").model_dump(
                mode="json"
            ),
            "publication_manifest": _artifact(
                publication_path, root, "application/json"
            ).model_dump(mode="json"),
            "reload_receipt": _artifact(reload_path, root, "application/json").model_dump(
                mode="json"
            ),
            "compact_payload_row_ids_hash": _row_hash([10, 20, 30]),
            "compact_payload_feature_order_hash": hashlib.sha256(
                canonical_json_bytes(primary_ids)
            ).hexdigest(),
            "compact_payload_counts_sha256": sha256_file(compact_path),
            "maximum_observed_count": 60,
            "count_dtype": "uint16",
            "index_dtype": "uint16",
        },
        "bundle_id",
    )
    receipt = _identified(
        G00CDecisionReceipt,
        {
            "schema_version": 1,
            "fold_view_id": contract.fold_view_id,
            "execution_bundle_id": bundle.bundle_id,
            "row_roles_verified": True,
            "feature_selection_verified": True,
            "sample_size_selection_verified": True,
            "sampler_sequence_verified": True,
            "compact_counts_verified": True,
            "protected_rows_absent": True,
            "immutable_publication_verified": True,
            "full_reload_verified": True,
            "status": "pass",
        },
        "receipt_id",
    )
    fake_store = SimpleNamespace(
        manifest=SimpleNamespace(
            sources=(
                SimpleNamespace(donor_id="D1", checkpoint="Rest", physical_time_hours=0.0),
                SimpleNamespace(donor_id="D1", checkpoint="Stim8hr", physical_time_hours=8.0),
                SimpleNamespace(donor_id="D2", checkpoint="Rest", physical_time_hours=0.0),
                SimpleNamespace(donor_id="D2", checkpoint="Stim8hr", physical_time_hours=8.0),
            )
        ),
        _locator=lambda: (
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([0, 1, 2, 3], dtype=np.int16),
            np.asarray([0, 0, 0, 0], dtype=np.int64),
        ),
    )
    return fake_store, contract, bundle, receipt


def test_g00c_execution_recomputes_selection_rows_sampler_and_protected_boundary(
    tmp_path: Path,
) -> None:
    store, contract, bundle, receipt = _write_g00c_evidence(tmp_path)
    validate_g00c_execution(tmp_path, store, contract, bundle, receipt)
    roles_path = tmp_path / bundle.row_roles.relative_uri
    roles = pd.read_parquet(roles_path)
    roles.loc[roles["role"] == "protected_heldout_stimulated", "in_compact_payload"] = True
    roles.to_parquet(roles_path, index=False)
    with pytest.raises(IntegrityError, match="artifact failed verification"):
        validate_g00c_execution(tmp_path, store, contract, bundle, receipt)


def test_g00c_contract_rejects_nonpositive_feature_prefix_and_empty_role() -> None:
    with pytest.raises(ValidationError, match="frozen positive prefix grid"):
        TrainingOnlyFeatureSelectionContract(
            implementation_sha256="1" * 64,
            fit_rows_hash="2" * 64,
            candidate_feature_counts=(-1, 0, 4096),
            minimum_improvement_margin=0.001,
            ordered_feature_table=ArtifactRef(
                schema_id="ordered",
                schema_version=1,
                sha256="3" * 64,
                size_bytes=1,
                media_type="application/x-parquet",
                relative_uri="ordered.parquet",
            ),
        )
    with pytest.raises(ValidationError):
        FoldRowRoleRecord(role="training_fit", rows=0, row_ids_hash="4" * 64)
