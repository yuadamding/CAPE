from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError
from scipy import sparse

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
    G00SourceAuthorityV1,
    G00SourceAuthorityV2,
    G00SourcePlaneV2Amendment,
    G00SourcePlaneV2AmendmentReceipt,
    ProtectedSourceAccessSemantics,
    SourceHashBinding,
    SourceNumericIntegrity,
    StrictModel,
    TrainingOnlyFeatureSelectionContract,
    TrainingOnlySampleSizeSelectionContract,
    VirtualCanonicalCountStoreManifestV1,
    VirtualCanonicalCountStoreManifestV2,
    VirtualCountSourceV1,
    VirtualCountSourceV2,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import (
    validate_g00_source_plane_amendment,
    validate_g00c_execution,
)

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


def _row_hash(values: Any) -> str:
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


def _source_projection_fixture(tmp_path: Path) -> tuple[Any, ...]:
    v1_sources = tuple(
        VirtualCountSourceV1(
            source_id=f"D{index // 3 + 1}_{('Rest', 'Stim8hr', 'Stim48hr')[index % 3]}",
            donor_id=f"D{index // 3 + 1}",
            checkpoint=("Rest", "Stim8hr", "Stim48hr")[index % 3],
            physical_time_hours=(0.0, 8.0, 48.0)[index % 3],
            relative_uri=f"source-{index}.h5ad",
            source_file_sha256=hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            rows=10,
            features=3,
            nnz=10,
            eligible_rows=10,
            eligible_nnz=10,
            source_feature_order_hash=hashlib.sha256(f"features-{index}".encode()).hexdigest(),
            canonical_permutation_hash=hashlib.sha256(f"permutation-{index}".encode()).hexdigest(),
        )
        for index in range(12)
    )
    parent_authority = _identified(
        G00SourceAuthorityV1,
        {
            "schema_version": 1,
            "sources": [source.model_dump(mode="json") for source in v1_sources],
            "canonical_feature_index_hash": "1" * 64,
            "guide_catalog_hash": "2" * 64,
            "target_catalog_hash": "3" * 64,
            "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
            "eligible_row_ids_hash": "4" * 64,
            "eligible_rows": 120,
            "eligible_nnz": 120,
        },
        "authority_id",
    )
    locator = tmp_path / "row-locator.h5"
    locator.write_bytes(b"locator")
    permutations = tmp_path / "feature-permutations.npz"
    permutations.write_bytes(b"permutations")
    parent_manifest = _identified(
        VirtualCanonicalCountStoreManifestV1,
        {
            "schema_version": 1,
            "source_authority_id": parent_authority.authority_id,
            "canonical_feature_index_hash": "1" * 64,
            "guide_catalog_hash": "2" * 64,
            "target_catalog_hash": "3" * 64,
            "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
            "eligible_rows": 120,
            "features": 3,
            "row_locator": _artifact(locator, tmp_path, "application/x-hdf5").model_dump(
                mode="json"
            ),
            "feature_permutations": _artifact(
                permutations, tmp_path, "application/x-npz"
            ).model_dump(mode="json"),
            "sources": [source.model_dump(mode="json") for source in v1_sources],
        },
        "virtual_store_id",
    )
    parent_authority_path = tmp_path / "G00A-v1.json"
    parent_authority_path.write_text(parent_authority.model_dump_json() + "\n")
    parent_manifest_path = tmp_path / "G00B-v1.json"
    parent_manifest_path.write_text(parent_manifest.model_dump_json() + "\n")
    parent_authority_ref = _artifact(parent_authority_path, tmp_path, "application/json")
    parent_manifest_ref = _artifact(parent_manifest_path, tmp_path, "application/json")
    crosswalk_path = tmp_path / "crosswalk.parquet"
    crosswalk_path.write_bytes(b"crosswalk")
    numeric_path = tmp_path / "numeric.parquet"
    numeric_path.write_bytes(b"numeric")
    derivation_path = tmp_path / "derivation.json"
    derivation_path.write_bytes(b"derivation")
    crosswalk_ref = _artifact(crosswalk_path, tmp_path, "application/x-parquet")
    numeric_ref = _artifact(numeric_path, tmp_path, "application/x-parquet")
    derivation_ref = _artifact(derivation_path, tmp_path, "application/json")
    locator_ref = _artifact(locator, tmp_path, "application/x-hdf5")
    amendment = _identified(
        G00SourcePlaneV2Amendment,
        {
            "schema_version": 1,
            "parent_g00a_v1_authority_id": parent_authority.authority_id,
            "parent_g00a_v1": parent_authority_ref.model_dump(mode="json"),
            "parent_g00b_v1_virtual_store_id": parent_manifest.virtual_store_id,
            "parent_g00b_v1_manifest": parent_manifest_ref.model_dump(mode="json"),
            "immutable_source_hashes": [
                SourceHashBinding(
                    source_id=source.source_id,
                    source_file_sha256=source.source_file_sha256,
                ).model_dump(mode="json")
                for source in v1_sources
            ],
            "v2_guide_target_crosswalk": crosswalk_ref.model_dump(mode="json"),
            "v2_numerical_audit": numeric_ref.model_dump(mode="json"),
            "v2_source_derivation_receipt": derivation_ref.model_dump(mode="json"),
            "v2_row_locator": locator_ref.model_dump(mode="json"),
            "builder_implementation_sha256": "5" * 64,
            "environment_hash": "6" * 64,
        },
        "amendment_id",
    )
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text(amendment.model_dump_json() + "\n")
    amendment_ref = _artifact(amendment_path, tmp_path, "application/json")
    integrity = SourceNumericIntegrity(
        storage_value_dtype="int32",
        indices_dtype="int32",
        indptr_dtype="int64",
        maximum_observed_count=10,
    )
    v2_sources = tuple(
        VirtualCountSourceV2(**source.model_dump(mode="python"), numeric_integrity=integrity)
        for source in v1_sources
    )
    shared = {
        "source_plane_amendment_id": amendment.amendment_id,
        "source_plane_amendment": amendment_ref.model_dump(mode="json"),
        "canonical_feature_index_hash": "1" * 64,
        "guide_catalog_hash": "2" * 64,
        "target_catalog_hash": "3" * 64,
        "guide_target_crosswalk_hash": crosswalk_ref.sha256,
        "guide_target_crosswalk": crosswalk_ref.model_dump(mode="json"),
        "source_numeric_audit": numeric_ref.model_dump(mode="json"),
        "source_derivation_receipt": derivation_ref.model_dump(mode="json"),
        "guide_count": 2,
        "target_control_count": 2,
        "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
        "eligible_row_ids_hash": "4" * 64,
        "eligible_rows": 120,
        "eligible_nnz": 120,
        "sources": [source.model_dump(mode="json") for source in v2_sources],
        "access_semantics": ProtectedSourceAccessSemantics().model_dump(mode="json"),
    }
    authority = _identified(
        G00SourceAuthorityV2,
        {"schema_version": 2, **shared},
        "authority_id",
    )
    manifest = _identified(
        VirtualCanonicalCountStoreManifestV2,
        {
            "schema_version": 2,
            "source_authority_id": authority.authority_id,
            **shared,
            "features": 3,
            "row_locator": locator_ref.model_dump(mode="json"),
            "feature_permutations": _artifact(
                permutations, tmp_path, "application/x-npz"
            ).model_dump(mode="json"),
        },
        "virtual_store_id",
    )
    authority_path = tmp_path / "G00A-v2.json"
    authority_path.write_text(authority.model_dump_json() + "\n")
    manifest_path = tmp_path / "G00B-v2.json"
    manifest_path.write_text(manifest.model_dump_json() + "\n")
    authority_ref = _artifact(authority_path, tmp_path, "application/json")
    manifest_ref = _artifact(manifest_path, tmp_path, "application/json")
    receipt = _identified(
        G00SourcePlaneV2AmendmentReceipt,
        {
            "schema_version": 1,
            "amendment_id": amendment.amendment_id,
            "derived_g00a_v2_authority_id": authority.authority_id,
            "derived_g00a_v2": authority_ref.model_dump(mode="json"),
            "derived_g00b_v2_virtual_store_id": manifest.virtual_store_id,
            "derived_g00b_v2": manifest_ref.model_dump(mode="json"),
            "parent_files_verified": True,
            "all_source_hashes_verified": True,
            "derivation_receipt_verified": True,
            "v2_parent_equality_verified": True,
            "status": "pass",
        },
        "receipt_id",
    )
    return (
        amendment,
        receipt,
        parent_authority,
        parent_manifest,
        authority,
        manifest,
        parent_authority_ref,
        parent_manifest_ref,
        amendment_ref,
        authority_ref,
        manifest_ref,
    )


def test_source_plane_amendment_rejects_relabeling_and_artifact_crosswiring(
    tmp_path: Path,
) -> None:
    values = _source_projection_fixture(tmp_path)
    validate_g00_source_plane_amendment(
        *values[:6],
        parent_authority_artifact=values[6],
        parent_manifest_artifact=values[7],
        amendment_artifact=values[8],
        authority_artifact=values[9],
        manifest_artifact=values[10],
    )
    authority = values[4]
    relabeled = authority.model_copy(
        update={
            "sources": (
                authority.sources[0].model_copy(update={"donor_id": "relabeled"}),
                *authority.sources[1:],
            )
        }
    )
    with pytest.raises(IntegrityError, match="changes accepted v1 fields"):
        validate_g00_source_plane_amendment(
            *values[:4],
            relabeled,
            values[5],
            parent_authority_artifact=values[6],
            parent_manifest_artifact=values[7],
            amendment_artifact=values[8],
            authority_artifact=values[9],
            manifest_artifact=values[10],
        )
    mismatched = values[0].model_copy(update={"v2_numerical_audit": values[0].parent_g00a_v1})
    with pytest.raises(IntegrityError, match="artifact wiring differs"):
        validate_g00_source_plane_amendment(
            mismatched,
            *values[1:6],
            parent_authority_artifact=values[6],
            parent_manifest_artifact=values[7],
            amendment_artifact=values[8],
            authority_artifact=values[9],
            manifest_artifact=values[10],
        )
    reordered = authority.model_copy(update={"sources": tuple(reversed(authority.sources))})
    with pytest.raises(IntegrityError, match="reordered"):
        validate_g00_source_plane_amendment(
            *values[:4],
            reordered,
            values[5],
            parent_authority_artifact=values[6],
            parent_manifest_artifact=values[7],
            amendment_artifact=values[8],
            authority_artifact=values[9],
            manifest_artifact=values[10],
        )
    with pytest.raises(IntegrityError, match="different G00A v1 file bytes"):
        validate_g00_source_plane_amendment(
            *values[:6],
            parent_authority_artifact=values[7],
            parent_manifest_artifact=values[7],
            amendment_artifact=values[8],
            authority_artifact=values[9],
            manifest_artifact=values[10],
        )
    receipt_crosswire = values[1].model_copy(update={"derived_g00a_v2": values[6]})
    with pytest.raises(IntegrityError, match="different derived evidence"):
        validate_g00_source_plane_amendment(
            values[0],
            receipt_crosswire,
            *values[2:6],
            parent_authority_artifact=values[6],
            parent_manifest_artifact=values[7],
            amendment_artifact=values[8],
            authority_artifact=values[9],
            manifest_artifact=values[10],
        )


def _write_g00c_evidence(
    root: Path,
) -> tuple[Any, FoldNativeCompactViewContractV2, G00CExecutionBundle, G00CDecisionReceipt]:
    training_fit = np.arange(1, 1_000_001, dtype=np.int64)
    training_validation = np.asarray([1_000_001], dtype=np.int64)
    heldout_source = np.asarray([1_000_002], dtype=np.int64)
    protected = np.asarray([1_000_003], dtype=np.int64)
    role_ids = {
        "training_fit": training_fit,
        "training_validation": training_validation,
        "heldout_source_query": heldout_source,
        "protected_heldout_stimulated": protected,
    }
    all_row_ids = np.concatenate(tuple(role_ids.values()))
    compact_row_ids = np.concatenate((training_fit[:50_000], training_validation, heldout_source))
    compact_membership = np.isin(all_row_ids, compact_row_ids)
    roles = pd.DataFrame(
        {
            "row_id": all_row_ids,
            "role": np.concatenate(
                tuple(np.repeat(role, len(ids)) for role, ids in role_ids.items())
            ),
            "donor_id": np.concatenate(
                (
                    np.repeat("D1", len(training_fit) + len(training_validation)),
                    np.repeat("D2", len(heldout_source) + len(protected)),
                )
            ),
            "checkpoint": np.concatenate(
                (
                    np.repeat("Rest", len(training_fit)),
                    np.repeat("Stim8hr", len(training_validation)),
                    np.repeat("Rest", len(heldout_source)),
                    np.repeat("Stim8hr", len(protected)),
                )
            ),
            "in_compact_payload": compact_membership,
        }
    )
    roles_path = root / "FOLD_ROW_ROLES.parquet"
    roles.to_parquet(roles_path, index=False)
    ordered = pd.DataFrame(
        {
            "rank": [*range(1, 4097), 0],
            "canonical_index": [*range(4096), 4096],
            "feature_id": [*[f"gene-{index}" for index in range(4096)], "CUSTOM001_PuroR"],
            "in_primary_metric": [*([True] * 4096), False],
            "is_sidecar": [*([False] * 4096), True],
        }
    )
    ordered_path = root / "ORDERED_FEATURES.parquet"
    ordered.to_parquet(ordered_path, index=False)
    feature_contract = TrainingOnlyFeatureSelectionContract(
        implementation_sha256="1" * 64,
        fit_rows_hash=_row_hash(training_fit),
        validation_rows_hash=_row_hash(training_validation),
        minimum_improvement_margin=0.001,
        ordered_feature_table=_artifact(ordered_path, root, "application/x-parquet"),
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
    amendment_artifact = ArtifactRef(
        schema_id="test.amendment",
        schema_version=1,
        sha256="a" * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri="amendment.json",
    )
    amendment_receipt_artifact = ArtifactRef(
        schema_id="test.amendment-receipt",
        schema_version=1,
        sha256="b" * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri="amendment-receipt.json",
    )
    contract = _identified(
        FoldNativeCompactViewContractV2,
        {
            "schema_version": 2,
            "parent_source_authority_id": "authority",
            "parent_virtual_store_id": "store",
            "source_plane_amendment_id": "amendment",
            "source_plane_amendment": amendment_artifact.model_dump(mode="json"),
            "source_plane_amendment_receipt_id": "amendment-receipt",
            "source_plane_amendment_receipt": amendment_receipt_artifact.model_dump(mode="json"),
            "parent_eligible_row_ids_hash": _row_hash(all_row_ids),
            "parent_guide_target_crosswalk_hash": "6" * 64,
            "outer_split_id": "lodo-D2",
            "training_donor_ids": ["D1"],
            "heldout_donor_id": "D2",
            "row_roles": [
                FoldRowRoleRecord(role=role, rows=len(ids), row_ids_hash=_row_hash(ids)).model_dump(
                    mode="json"
                )
                for role, ids in role_ids.items()
            ],
            "row_role_audit": _artifact(roles_path, root, "application/x-parquet").model_dump(
                mode="json"
            ),
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
    feature_candidates = feature_contract.candidate_feature_counts
    feature_penalties = dict(
        zip(feature_candidates, (0.0005, 0.0004, 0.0003, 0.0002, 0.0), strict=True)
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
    sample_candidates = sample_contract.candidate_cells
    sample_penalties = dict(
        zip(sample_candidates, (0.0005, 0.0004, 0.0003, 0.0002, 0.0), strict=True)
    )
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
    selected_rows_path = root / "SELECTED_TRAINING_ROWS.parquet"
    pd.DataFrame({"row_id": training_fit[:50_000]}).to_parquet(selected_rows_path, index=False)
    scale_order_path = root / "TRAINING_SCALE_ROW_ORDER.parquet"
    pd.DataFrame({"rank": np.arange(1, len(training_fit) + 1), "row_id": training_fit}).to_parquet(
        scale_order_path, index=False
    )
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
    sampler_draws = pd.DataFrame(
        {
            "update": np.zeros(4096, dtype=np.int64),
            "microbatch": np.repeat(np.arange(8, dtype=np.int64), 512),
            "position": np.tile(np.arange(512, dtype=np.int64), 8),
            "row_id": training_fit[:4096],
            "sample_weight": np.full(4096, 1.0 / 512.0, dtype=np.float64),
            "thinning_draw": np.arange(4096, dtype=np.int64) % 2,
        }
    )
    sampler_states = pd.DataFrame(
        {
            "update": np.zeros(8, dtype=np.int64),
            "microbatch": np.arange(8, dtype=np.int64),
            "rng_state_json": [
                json.dumps({"algorithm": "philox4x32_10", "counter": index}) for index in range(8)
            ],
            "resume_cursor_json": [
                json.dumps({"microbatch": index, "update": 0}) for index in range(8)
            ],
        }
    )
    uninterrupted_draws_path = root / "SAMPLER_UNINTERRUPTED_DRAWS.parquet"
    resumed_draws_path = root / "SAMPLER_RESUMED_DRAWS.parquet"
    uninterrupted_states_path = root / "SAMPLER_UNINTERRUPTED_STATES.parquet"
    resumed_states_path = root / "SAMPLER_RESUMED_STATES.parquet"
    sampler_draws.to_parquet(uninterrupted_draws_path, index=False)
    sampler_draws.to_parquet(resumed_draws_path, index=False)
    sampler_states.to_parquet(uninterrupted_states_path, index=False)
    sampler_states.to_parquet(resumed_states_path, index=False)
    epoch_path = root / "SAMPLER_EPOCH_INDEX.parquet"
    epoch_rows = []
    for microbatch, frame in sampler_draws.groupby("microbatch", sort=True):
        state = sampler_states.loc[sampler_states["microbatch"] == microbatch].iloc[0]
        epoch_rows.append(
            (
                0,
                int(microbatch),
                hashlib.sha256(frame["row_id"].to_numpy(dtype="<i8").tobytes()).hexdigest(),
                hashlib.sha256(frame["sample_weight"].to_numpy(dtype="<f8").tobytes()).hexdigest(),
                hashlib.sha256(frame["thinning_draw"].to_numpy(dtype="<i8").tobytes()).hexdigest(),
                hashlib.sha256(
                    canonical_json_bytes(json.loads(state["rng_state_json"]))
                ).hexdigest(),
                hashlib.sha256(
                    canonical_json_bytes(json.loads(state["resume_cursor_json"]))
                ).hexdigest(),
            )
        )
    pd.DataFrame(
        epoch_rows,
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
    uninterrupted_draws_ref = _artifact(uninterrupted_draws_path, root, "application/x-parquet")
    resumed_draws_ref = _artifact(resumed_draws_path, root, "application/x-parquet")
    uninterrupted_states_ref = _artifact(uninterrupted_states_path, root, "application/x-parquet")
    resumed_states_ref = _artifact(resumed_states_path, root, "application/x-parquet")
    resume_path = root / "SAMPLER_RESUME_TEST.json"
    resume_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "resume_update": 0,
                "resume_microbatch": 4,
                "uninterrupted_draw_trace_sha256": uninterrupted_draws_ref.sha256,
                "resumed_draw_trace_sha256": resumed_draws_ref.sha256,
                "uninterrupted_state_trace_sha256": uninterrupted_states_ref.sha256,
                "resumed_state_trace_sha256": resumed_states_ref.sha256,
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
    compact_path = root / "compact-counts.h5"
    compact_values = (compact_row_ids % 60 + 1).astype(np.uint16)
    with h5py.File(compact_path, "x") as handle:
        handle.create_dataset("row_ids", data=compact_row_ids)
        handle.create_dataset(
            "feature_ids",
            data=np.asarray([f"gene-{index}" for index in range(256)], dtype=h5py.string_dtype()),
        )
        handle.create_dataset("indptr", data=np.arange(len(compact_row_ids) + 1, dtype=np.int64))
        handle.create_dataset("indices", data=np.zeros(len(compact_row_ids), dtype=np.uint16))
        handle.create_dataset("data", data=compact_values)
    publication_path = root / "PUBLICATION.json"
    publication_path.write_text(
        json.dumps({"status": "pass", "fold_view_id": contract.fold_view_id}) + "\n"
    )
    reload_path = root / "RELOAD.json"
    reload_path.write_text(
        json.dumps({"status": "pass", "compact_payload_sha256": sha256_file(compact_path)}) + "\n"
    )
    primary_ids = ordered.loc[~ordered["is_sidecar"], "feature_id"].tolist()[:256]
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
                fit_rows_hash=feature_contract.fit_rows_hash,
                validation_rows_hash=feature_contract.validation_rows_hash,
                selected_feature_count=256,
            ).model_dump(mode="json"),
            "sample_size_selection": G00CSampleSizeSelectionResult(
                curve=_artifact(sample_curve_path, root, "application/x-parquet"),
                paired_refit_draws=_artifact(sample_draws_path, root, "application/x-parquet"),
                selected_training_rows=_artifact(selected_rows_path, root, "application/x-parquet"),
                training_scale_row_order=_artifact(scale_order_path, root, "application/x-parquet"),
                selected_training_cells=50_000,
                two_million_extension_opened=False,
            ).model_dump(mode="json"),
            "sampler": G00CSamplerEvidence(
                sampler_plan=_artifact(plan_path, root, "application/json"),
                epoch_index=_artifact(epoch_path, root, "application/x-parquet"),
                uninterrupted_draw_trace=uninterrupted_draws_ref,
                resumed_draw_trace=resumed_draws_ref,
                uninterrupted_state_trace=uninterrupted_states_ref,
                resumed_state_trace=resumed_states_ref,
                resume_test=_artifact(resume_path, root, "application/json"),
            ).model_dump(mode="json"),
            "compact_payload": _artifact(compact_path, root, "application/x-hdf5").model_dump(
                mode="json"
            ),
            "publication_manifest": _artifact(
                publication_path, root, "application/json"
            ).model_dump(mode="json"),
            "reload_receipt": _artifact(reload_path, root, "application/json").model_dump(
                mode="json"
            ),
            "compact_payload_rows": len(compact_row_ids),
            "compact_payload_features": 256,
            "compact_payload_row_ids_hash": _row_hash(compact_row_ids),
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
    source_indices = np.concatenate(
        (
            np.zeros(len(training_fit), dtype=np.int16),
            np.ones(len(training_validation), dtype=np.int16),
            np.full(len(heldout_source), 2, dtype=np.int16),
            np.full(len(protected), 3, dtype=np.int16),
        )
    )

    def rows(row_ids: np.ndarray) -> SimpleNamespace:
        ids = np.asarray(row_ids, dtype=np.int64)
        values = (ids % 60 + 1).astype(np.int32)
        matrix = sparse.csr_matrix(
            (
                values,
                np.zeros(len(ids), dtype=np.int32),
                np.arange(len(ids) + 1, dtype=np.int64),
            ),
            shape=(len(ids), 4097),
        )
        return SimpleNamespace(matrix=matrix, row_ids=ids)

    fake_store = SimpleNamespace(
        manifest=SimpleNamespace(
            features=4097,
            sources=(
                SimpleNamespace(donor_id="D1", checkpoint="Rest", physical_time_hours=0.0),
                SimpleNamespace(donor_id="D1", checkpoint="Stim8hr", physical_time_hours=8.0),
                SimpleNamespace(donor_id="D2", checkpoint="Rest", physical_time_hours=0.0),
                SimpleNamespace(donor_id="D2", checkpoint="Stim8hr", physical_time_hours=8.0),
            ),
        ),
        _locator=lambda: (
            all_row_ids,
            source_indices,
            np.arange(len(all_row_ids), dtype=np.int64),
        ),
        rows=rows,
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


def test_g00c_execution_rejects_sampler_trace_not_derived_by_epoch_index(
    tmp_path: Path,
) -> None:
    store, contract, bundle, receipt = _write_g00c_evidence(tmp_path)
    draws = pd.read_parquet(tmp_path / bundle.sampler.uninterrupted_draw_trace.relative_uri)
    draws.loc[0, "row_id"] += 99
    uninterrupted_path = tmp_path / bundle.sampler.uninterrupted_draw_trace.relative_uri
    resumed_path = tmp_path / bundle.sampler.resumed_draw_trace.relative_uri
    draws.to_parquet(uninterrupted_path, index=False)
    draws.to_parquet(resumed_path, index=False)
    uninterrupted_ref = _artifact(uninterrupted_path, tmp_path, "application/x-parquet")
    resumed_ref = _artifact(resumed_path, tmp_path, "application/x-parquet")
    resume_path = tmp_path / bundle.sampler.resume_test.relative_uri
    resume = json.loads(resume_path.read_text())
    resume["uninterrupted_draw_trace_sha256"] = uninterrupted_ref.sha256
    resume["resumed_draw_trace_sha256"] = resumed_ref.sha256
    resume_path.write_text(json.dumps(resume, sort_keys=True) + "\n")
    sampler = bundle.sampler.model_copy(
        update={
            "uninterrupted_draw_trace": uninterrupted_ref,
            "resumed_draw_trace": resumed_ref,
            "resume_test": _artifact(resume_path, tmp_path, "application/json"),
        }
    )
    bundle_payload = bundle.model_dump(mode="json")
    bundle_payload["sampler"] = sampler.model_dump(mode="json")
    changed_bundle = _identified(G00CExecutionBundle, bundle_payload, "bundle_id")
    receipt_payload = receipt.model_dump(mode="json")
    receipt_payload["execution_bundle_id"] = changed_bundle.bundle_id
    changed_receipt = _identified(G00CDecisionReceipt, receipt_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="not derived from its actual trace"):
        validate_g00c_execution(tmp_path, store, contract, changed_bundle, changed_receipt)


def test_g00c_contract_rejects_nonpositive_feature_prefix_and_empty_role() -> None:
    with pytest.raises(ValidationError, match="frozen positive prefix grid"):
        TrainingOnlyFeatureSelectionContract(
            implementation_sha256="1" * 64,
            fit_rows_hash="2" * 64,
            validation_rows_hash="5" * 64,
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
