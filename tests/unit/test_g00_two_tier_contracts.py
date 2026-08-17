from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import TypeAdapter, ValidationError

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    CompactSamplerContract,
    FoldNativeCompactViewContractV1,
    FoldNativeCompactViewContractV2,
    FoldRowRoleRecord,
    G00DParityGateContract,
    G00DParityGateEvidence,
    G00SourceAuthorityV1,
    G00SourceAuthorityV2,
    G00SourcePlaneV2Amendment,
    G00SourcePlaneV2AmendmentReceipt,
    IntegratedLoaderQualificationContractV1,
    IntegratedLoaderQualificationContractV2,
    IntegratedLoaderQualificationReceiptV1,
    IntegratedLoaderQualificationReceiptV2,
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
    validate_g00_fold_view_parent,
    validate_g00_source_plane,
    validate_integrated_loader_qualification,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized_fields = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in model.model_fields.items()
        if name in payload
    }
    normalized = model.model_construct(**normalized_fields).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def test_g00a_c_d_contracts_are_strict_and_publicly_dispatchable(tmp_path: Path) -> None:
    source = VirtualCountSourceV1(
        source_id="D1_Rest",
        donor_id="D1",
        checkpoint="Rest",
        physical_time_hours=0.0,
        relative_uri="D1_Rest.h5ad",
        source_file_sha256="a" * 64,
        rows=100,
        features=18_130,
        nnz=10_000,
        eligible_rows=100,
        eligible_nnz=10_000,
        source_feature_order_hash="b" * 64,
        canonical_permutation_hash="c" * 64,
    )
    authority = _identified(
        G00SourceAuthorityV1,
        {
            "schema_version": 1,
            "sources": [source.model_dump(mode="json")],
            "canonical_feature_index_hash": "d" * 64,
            "guide_catalog_hash": "4" * 64,
            "target_catalog_hash": "5" * 64,
            "eligibility_rule": ("guide_group == targeting single sgRNA AND low_quality == false"),
            "eligible_row_ids_hash": "e" * 64,
            "eligible_rows": 100,
            "eligible_nnz": 10_000,
        },
        "authority_id",
    )
    fold_view = _identified(
        FoldNativeCompactViewContractV1,
        {
            "schema_version": 1,
            "parent_source_authority_id": authority.authority_id,
            "parent_virtual_store_id": "virtual-1",
            "outer_split_id": "lodo-D4",
            "training_donor_ids": ["D1", "D2", "D3"],
            "heldout_donor_id": "D4",
            "training_only_feature_selection_hash": "f" * 64,
            "ordered_selected_feature_ids_hash": "0" * 64,
            "selected_features": 4096,
            "sample_size_candidates": [50_000, 250_000, 1_000_000],
            "selected_training_cells": 250_000,
            "training_rows_hash": "1" * 64,
            "heldout_source_rows_hash": "2" * 64,
            "protected_outer_endpoint_rows_hash": "3" * 64,
            "count_dtype": "int32",
            "index_dtype": "uint16",
            "count_maximum_audit_pass": False,
            "physical_layout": "donor_checkpoint_target_guide_row",
            "microbatch_cells": 512,
            "microbatches_per_update": 8,
        },
        "fold_view_id",
    )
    loader = _identified(
        IntegratedLoaderQualificationContractV1,
        {
            "schema_version": 1,
            "fold_view_id": fold_view.fold_view_id,
            "prefetch_depth": 4,
            "maximum_loader_rss_bytes": 16_000_000_000,
            "maximum_open_shards": 8,
        },
        "qualification_contract_id",
    )
    receipt = _identified(
        IntegratedLoaderQualificationReceiptV1,
        {
            "schema_version": 1,
            "qualification_contract_id": loader.qualification_contract_id,
            "data_wait_fraction": 0.05,
            "steady_state_gpu_utilization": 0.90,
            "p95_batch_ready_seconds": 0.2,
            "covered_compute_seconds": 0.3,
            "peak_loader_rss_bytes": 8_000_000_000,
            "peak_open_shards": 4,
            "row_count_order_errors": 0,
            "unbounded_memory_growth_detected": False,
            "training_metric_max_absolute_error": 5e-7,
            "training_metric_max_relative_error": 5e-6,
            "training_metric_parity_pass": True,
            "status": "pass",
        },
        "receipt_id",
    )
    expected = (
        ("authority.json", authority, "G00SourceAuthority"),
        ("fold.json", fold_view, "FoldNativeCompactViewContract"),
        ("loader.json", loader, "IntegratedLoaderQualificationContract"),
        ("receipt.json", receipt, "IntegratedLoaderQualificationReceipt"),
    )
    for name, contract, contract_type in expected:
        path = tmp_path / name
        path.write_text(contract.model_dump_json() + "\n")
        assert validate_contract(path)["contract_type"] == contract_type
    validate_integrated_loader_qualification(loader, receipt)
    dishonest = receipt.model_copy(update={"data_wait_fraction": 0.2})
    with pytest.raises(IntegrityError, match="status must be fail"):
        validate_integrated_loader_qualification(loader, dishonest)
    with pytest.raises(IntegrityError, match="different qualification contract"):
        validate_integrated_loader_qualification(
            loader,
            receipt.model_copy(update={"qualification_contract_id": "wrong"}),
        )
    honest_failure = dishonest.model_copy(update={"status": "fail"})
    validate_integrated_loader_qualification(loader, honest_failure)
    false_parity = receipt.model_copy(update={"training_metric_max_absolute_error": 2e-6})
    with pytest.raises(IntegrityError, match="flag differs"):
        validate_integrated_loader_qualification(loader, false_parity)
    honest_parity_failure = false_parity.model_copy(
        update={"training_metric_parity_pass": False, "status": "fail"}
    )
    validate_integrated_loader_qualification(loader, honest_parity_failure)


def test_fold_view_rejects_heldout_donor_leakage_and_unsafe_uint16_counts() -> None:
    base = {
        "schema_version": 1,
        "fold_view_id": "pending",
        "parent_source_authority_id": "authority",
        "parent_virtual_store_id": "virtual",
        "outer_split_id": "lodo-D4",
        "training_donor_ids": ["D1", "D4"],
        "heldout_donor_id": "D4",
        "training_only_feature_selection_hash": hashlib.sha256(b"f").hexdigest(),
        "ordered_selected_feature_ids_hash": hashlib.sha256(b"g").hexdigest(),
        "selected_features": 4096,
        "sample_size_candidates": [50_000],
        "selected_training_cells": 50_000,
        "training_rows_hash": "1" * 64,
        "heldout_source_rows_hash": "2" * 64,
        "protected_outer_endpoint_rows_hash": "3" * 64,
        "count_dtype": "uint16",
        "index_dtype": "uint16",
        "count_maximum_audit_pass": False,
        "physical_layout": "donor_checkpoint_target_guide_row",
        "microbatch_cells": 512,
        "microbatches_per_update": 8,
    }
    base["fold_view_id"] = contract_id(base, id_field="fold_view_id")
    with pytest.raises(ValidationError, match="disjoint"):
        FoldNativeCompactViewContractV1.model_validate(base)
    base["training_donor_ids"] = ["D1", "D2", "D3"]
    base["fold_view_id"] = "pending"
    base["fold_view_id"] = contract_id(base, id_field="fold_view_id")
    with pytest.raises(ValidationError, match="uint16"):
        FoldNativeCompactViewContractV1.model_validate(base)


def test_g00_source_authority_requires_reconciled_source_totals() -> None:
    source = VirtualCountSourceV1(
        source_id="D1_Rest",
        donor_id="D1",
        checkpoint="Rest",
        physical_time_hours=0.0,
        relative_uri="D1_Rest.h5ad",
        source_file_sha256="a" * 64,
        rows=100,
        features=18_130,
        nnz=10_000,
        eligible_rows=99,
        eligible_nnz=9_999,
        source_feature_order_hash="b" * 64,
        canonical_permutation_hash="c" * 64,
    )
    with pytest.raises(ValidationError, match="eligible rows do not reconcile"):
        _identified(
            G00SourceAuthorityV1,
            {
                "schema_version": 1,
                "sources": (source,),
                "canonical_feature_index_hash": "d" * 64,
                "guide_catalog_hash": "4" * 64,
                "target_catalog_hash": "5" * 64,
                "eligibility_rule": (
                    "guide_group == targeting single sgRNA AND low_quality == false"
                ),
                "eligible_row_ids_hash": "e" * 64,
                "eligible_rows": 100,
                "eligible_nnz": 10_000,
            },
            "authority_id",
        )


def test_dev29_v1_source_plane_remains_cross_validatable() -> None:
    source = VirtualCountSourceV1(
        source_id="D1_Rest",
        donor_id="D1",
        checkpoint="Rest",
        physical_time_hours=0.0,
        relative_uri="D1_Rest.h5ad",
        source_file_sha256="a" * 64,
        rows=100,
        features=18_130,
        nnz=10_000,
        eligible_rows=90,
        eligible_nnz=9_000,
        source_feature_order_hash="b" * 64,
        canonical_permutation_hash="c" * 64,
    )
    shared = {
        "canonical_feature_index_hash": "d" * 64,
        "guide_catalog_hash": "4" * 64,
        "target_catalog_hash": "5" * 64,
        "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
        "eligible_rows": 90,
        "sources": [source.model_dump(mode="json")],
    }
    authority = _identified(
        G00SourceAuthorityV1,
        {
            "schema_version": 1,
            **shared,
            "eligible_row_ids_hash": "e" * 64,
            "eligible_nnz": 9_000,
        },
        "authority_id",
    )
    manifest = _identified(
        VirtualCanonicalCountStoreManifestV1,
        {
            "schema_version": 1,
            "source_authority_id": authority.authority_id,
            **shared,
            "features": 18_130,
            "row_locator": _artifact("8", "row-locator.h5").model_dump(mode="json"),
            "feature_permutations": _artifact("9", "feature-permutations.npz").model_dump(
                mode="json"
            ),
        },
        "virtual_store_id",
    )
    validate_g00_source_plane(authority, manifest)
    with pytest.raises(IntegrityError, match="same schema version"):
        validate_g00_source_plane(authority, _dev30_source_plane()[1])


def _artifact(digit: str, uri: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.artifact",
        schema_version=1,
        sha256=digit * 64,
        size_bytes=100,
        media_type="application/octet-stream",
        relative_uri=uri,
    )


def _dev30_source_plane() -> tuple[
    G00SourceAuthorityV2,
    VirtualCanonicalCountStoreManifestV2,
    G00SourcePlaneV2Amendment,
    G00SourcePlaneV2AmendmentReceipt,
    ArtifactRef,
    ArtifactRef,
]:
    access = ProtectedSourceAccessSemantics()
    source = VirtualCountSourceV2(
        source_id="D1_Rest",
        donor_id="D1",
        checkpoint="Rest",
        physical_time_hours=0.0,
        relative_uri="D1_Rest.h5ad",
        source_file_sha256="a" * 64,
        rows=100,
        features=18_130,
        nnz=10_000,
        eligible_rows=90,
        eligible_nnz=9_000,
        source_feature_order_hash="b" * 64,
        canonical_permutation_hash="c" * 64,
        numeric_integrity=SourceNumericIntegrity(
            storage_value_dtype="float32",
            indices_dtype="int64",
            indptr_dtype="int64",
            maximum_observed_count=312,
        ),
    )
    crosswalk = _artifact("6", "GUIDE_TARGET_CROSSWALK.parquet")
    numeric = _artifact("7", "SOURCE_NUMERIC_AUDIT.parquet")
    derivation = _artifact("f", "SOURCE_PLANE_DERIVATION_RECEIPT.json")
    locator = _artifact("8", "row-locator.h5")
    amendment = _identified(
        G00SourcePlaneV2Amendment,
        {
            "schema_version": 1,
            "parent_g00a_v1_authority_id": "g00a-v1",
            "parent_g00a_v1": _artifact("1", "G00A-v1.json").model_dump(mode="json"),
            "parent_g00b_v1_virtual_store_id": "g00b-v1",
            "parent_g00b_v1_manifest": _artifact("2", "G00B-v1.json").model_dump(mode="json"),
            "immutable_source_hashes": [
                SourceHashBinding(
                    source_id=(source.source_id if index == 0 else f"source-{index}"),
                    source_file_sha256=(
                        source.source_file_sha256 if index == 0 else f"{index:x}" * 64
                    ),
                ).model_dump(mode="json")
                for index in range(12)
            ],
            "v2_guide_target_crosswalk": crosswalk.model_dump(mode="json"),
            "v2_numerical_audit": numeric.model_dump(mode="json"),
            "v2_source_derivation_receipt": derivation.model_dump(mode="json"),
            "v2_row_locator": locator.model_dump(mode="json"),
            "builder_implementation_sha256": "3" * 64,
            "environment_hash": "4" * 64,
        },
        "amendment_id",
    )
    amendment_artifact = ArtifactRef(
        schema_id="test.g00-amendment",
        schema_version=1,
        sha256=hashlib.sha256(amendment.model_dump_json().encode()).hexdigest(),
        size_bytes=len(amendment.model_dump_json()),
        media_type="application/json",
        relative_uri="G00_SOURCE_PLANE_V2_AMENDMENT.json",
    )
    shared = {
        "source_plane_amendment_id": amendment.amendment_id,
        "source_plane_amendment": amendment_artifact.model_dump(mode="json"),
        "canonical_feature_index_hash": "d" * 64,
        "guide_catalog_hash": "4" * 64,
        "target_catalog_hash": "5" * 64,
        "guide_target_crosswalk_hash": crosswalk.sha256,
        "guide_target_crosswalk": crosswalk.model_dump(mode="json"),
        "source_numeric_audit": numeric.model_dump(mode="json"),
        "source_derivation_receipt": derivation.model_dump(mode="json"),
        "guide_count": 25_956,
        "target_control_count": 12_732,
        "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
        "eligible_row_ids_hash": "e" * 64,
        "eligible_rows": 90,
        "eligible_nnz": 9_000,
        "sources": [source.model_dump(mode="json")],
        "access_semantics": access.model_dump(mode="json"),
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
            "features": 18_130,
            "row_locator": locator.model_dump(mode="json"),
            "feature_permutations": _artifact("9", "feature-permutations.npz").model_dump(
                mode="json"
            ),
        },
        "virtual_store_id",
    )
    authority_artifact = ArtifactRef(
        schema_id="test.g00a-v2",
        schema_version=2,
        sha256=hashlib.sha256(authority.model_dump_json().encode()).hexdigest(),
        size_bytes=len(authority.model_dump_json()),
        media_type="application/json",
        relative_uri="G00A-v2.json",
    )
    manifest_artifact = ArtifactRef(
        schema_id="test.g00b-v2",
        schema_version=2,
        sha256=hashlib.sha256(manifest.model_dump_json().encode()).hexdigest(),
        size_bytes=len(manifest.model_dump_json()),
        media_type="application/json",
        relative_uri="G00B-v2.json",
    )
    receipt = _identified(
        G00SourcePlaneV2AmendmentReceipt,
        {
            "schema_version": 1,
            "amendment_id": amendment.amendment_id,
            "derived_g00a_v2_authority_id": authority.authority_id,
            "derived_g00a_v2": authority_artifact.model_dump(mode="json"),
            "derived_g00b_v2_virtual_store_id": manifest.virtual_store_id,
            "derived_g00b_v2": manifest_artifact.model_dump(mode="json"),
            "parent_files_verified": True,
            "all_source_hashes_verified": True,
            "derivation_receipt_verified": True,
            "v2_parent_equality_verified": True,
            "status": "pass",
        },
        "receipt_id",
    )
    receipt_artifact = ArtifactRef(
        schema_id="test.g00-amendment-receipt",
        schema_version=1,
        sha256=hashlib.sha256(receipt.model_dump_json().encode()).hexdigest(),
        size_bytes=len(receipt.model_dump_json()),
        media_type="application/json",
        relative_uri="G00_SOURCE_PLANE_V2_AMENDMENT_RECEIPT.json",
    )
    return authority, manifest, amendment, receipt, amendment_artifact, receipt_artifact


def test_dev30_source_plane_binds_access_numeric_crosswalk_and_parent(tmp_path: Path) -> None:
    authority, manifest, _, _, _, _ = _dev30_source_plane()
    validate_g00_source_plane(authority, manifest)
    for filename, value in (("authority.json", authority), ("manifest.json", manifest)):
        path = tmp_path / filename
        path.write_text(value.model_dump_json() + "\n")
        assert validate_contract(path)["schema_version"] == 2
    with pytest.raises(IntegrityError, match="eligible_rows"):
        validate_g00_source_plane(authority, manifest.model_copy(update={"eligible_rows": 89}))
    with pytest.raises(ValidationError, match="protected_expression_values_used_for_evaluation"):
        ProtectedSourceAccessSemantics(
            protected_expression_values_used_for_evaluation=True  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="ArtifactRef hash"):
        _identified(
            G00SourceAuthorityV2,
            {
                **authority.model_dump(mode="json"),
                "authority_id": "pending",
                "guide_target_crosswalk_hash": "f" * 64,
            },
            "authority_id",
        )


def _dev30_fold_view(
    *,
    authority_id: str = "authority",
    virtual_store_id: str = "virtual",
    amendment_id: str = "amendment",
    amendment_artifact: ArtifactRef | None = None,
    amendment_receipt_id: str = "amendment-receipt",
    amendment_receipt_artifact: ArtifactRef | None = None,
) -> FoldNativeCompactViewContractV2:
    amendment_artifact = amendment_artifact or _artifact("a", "amendment.json")
    amendment_receipt_artifact = amendment_receipt_artifact or _artifact(
        "b", "amendment-receipt.json"
    )
    feature = TrainingOnlyFeatureSelectionContract(
        implementation_sha256="1" * 64,
        fit_rows_hash="2" * 64,
        validation_rows_hash="0" * 64,
        minimum_improvement_margin=0.001,
        ordered_feature_table=_artifact("3", "ORDERED_FEATURES.parquet"),
    )
    sample = TrainingOnlySampleSizeSelectionContract(
        equivalence_epsilon=0.002,
        candidate_cells=(50_000, 100_000, 250_000, 500_000, 1_000_000),
    )
    sampler = CompactSamplerContract(
        implementation_sha256="4" * 64,
        rng_algorithm="PCG64DXSM",
        rng_seed=20260816,
        thinning_rule="without_replacement_within_microbatch",
        resume_cursor_schema="weighted_sampler_cursor_v1",
        resume_cursor_initial_hash="5" * 64,
    )
    roles = tuple(
        FoldRowRoleRecord(role=role, rows=rows, row_ids_hash=digit * 64)
        for role, rows, digit in (
            ("training_fit", 1000, "6"),
            ("training_validation", 100, "7"),
            ("heldout_source_query", 200, "8"),
            ("protected_heldout_stimulated", 300, "9"),
        )
    )
    return _identified(
        FoldNativeCompactViewContractV2,
        {
            "schema_version": 2,
            "parent_source_authority_id": authority_id,
            "parent_virtual_store_id": virtual_store_id,
            "source_plane_amendment_id": amendment_id,
            "source_plane_amendment": amendment_artifact.model_dump(mode="json"),
            "source_plane_amendment_receipt_id": amendment_receipt_id,
            "source_plane_amendment_receipt": amendment_receipt_artifact.model_dump(mode="json"),
            "parent_eligible_row_ids_hash": "e" * 64,
            "parent_guide_target_crosswalk_hash": "6" * 64,
            "outer_split_id": "lodo-D4",
            "training_donor_ids": ["D1", "D2", "D3"],
            "heldout_donor_id": "D4",
            "row_roles": [role.model_dump(mode="json") for role in roles],
            "row_role_audit": _artifact("a", "FOLD_ROW_ROLES.parquet").model_dump(mode="json"),
            "row_role_assignment_implementation_sha256": "b" * 64,
            "feature_selection": feature.model_dump(mode="json"),
            "sample_size_selection": sample.model_dump(mode="json"),
            "sampler": sampler.model_dump(mode="json"),
            "count_dtype": "int32",
            "index_dtype": "uint16",
            "maximum_observed_count": 312,
            "physical_layout": "donor_checkpoint_target_guide_row",
        },
        "fold_view_id",
    )


def test_dev30_g00c_freezes_selection_roles_sampler_and_technical_feature() -> None:
    fold = _dev30_fold_view()
    assert fold.sampler.macrobatch_cells == 4096
    assert not fold.feature_selection.custom001_puror_in_primary_biological_metric
    with pytest.raises(ValidationError, match="frozen G00C grid"):
        TrainingOnlySampleSizeSelectionContract(
            equivalence_epsilon=0.001,
            candidate_cells=(50_000, 75_000),
        )
    with pytest.raises(ValidationError, match="documented nonsaturation"):
        TrainingOnlySampleSizeSelectionContract(
            equivalence_epsilon=0.001,
            candidate_cells=(
                50_000,
                100_000,
                250_000,
                500_000,
                1_000_000,
                2_000_000,
            ),
        )
    bad = fold.model_dump(mode="json")
    bad["fold_view_id"] = "pending"
    bad["row_roles"] = bad["row_roles"][:-1]
    bad["fold_view_id"] = contract_id(bad, id_field="fold_view_id")
    with pytest.raises(ValidationError, match="every role exactly once"):
        FoldNativeCompactViewContractV2.model_validate(bad)


def test_dev30_g00c_parent_binding_rejects_row_universe_drift() -> None:
    authority, manifest, amendment, receipt, amendment_artifact, receipt_artifact = (
        _dev30_source_plane()
    )
    fold = _dev30_fold_view(
        authority_id=authority.authority_id,
        virtual_store_id=manifest.virtual_store_id,
        amendment_id=amendment.amendment_id,
        amendment_artifact=amendment_artifact,
        amendment_receipt_id=receipt.receipt_id,
        amendment_receipt_artifact=receipt_artifact,
    )
    validate_g00_fold_view_parent(
        fold,
        authority,
        manifest,
        amendment,
        receipt,
        amendment_artifact=amendment_artifact,
        amendment_receipt_artifact=receipt_artifact,
    )
    with pytest.raises(IntegrityError, match="eligible row universe"):
        validate_g00_fold_view_parent(
            fold.model_copy(update={"parent_eligible_row_ids_hash": "0" * 64}),
            authority,
            manifest,
            amendment,
            receipt,
            amendment_artifact=amendment_artifact,
            amendment_receipt_artifact=receipt_artifact,
        )
    failed_receipt = receipt.model_copy(update={"status": "fail", "parent_files_verified": False})
    with pytest.raises(IntegrityError, match="passed amendment receipt"):
        validate_g00_fold_view_parent(
            fold,
            authority,
            manifest,
            amendment,
            failed_receipt,
            amendment_artifact=amendment_artifact,
            amendment_receipt_artifact=receipt_artifact,
        )


def _dev30_g00d() -> tuple[
    IntegratedLoaderQualificationContractV2, IntegratedLoaderQualificationReceiptV2
]:
    contract = _identified(
        IntegratedLoaderQualificationContractV2,
        {
            "schema_version": 2,
            "fold_view_id": _dev30_fold_view().fold_view_id,
            "expected_gpu_name": "NVIDIA H100 80GB HBM3",
            "expected_cuda_version": "13.0",
            "expected_torch_version": "2.13.0+cu130",
            "expected_container_digest": "a" * 64,
            "worker_count": 8,
            "cpu_count": 32,
            "storage_authority_hash": "b" * 64,
            "prefetch_depth": 4,
            "warmup_updates": 50,
            "measured_updates": 500,
            "cache_policy": "bounded_lru_v1",
            "measurement_protocol_sha256": "c" * 64,
            "telemetry_interval_seconds": 0.1,
            "maximum_p95_batch_ready_seconds": 0.05,
            "maximum_loader_rss_bytes": 16_000_000_000,
            "maximum_process_loader_rss_bytes": 10_000_000_000,
            "maximum_aggregate_worker_rss_bytes": 16_000_000_000,
            "maximum_open_shards": 8,
            "maximum_open_file_handles": 128,
            "maximum_rss_slope_upper_bytes_per_second": 1000.0,
            "maximum_rss_excursion_fraction": 0.10,
            "parity_gates": [
                G00DParityGateContract(gate=gate, comparison_mode=mode).model_dump(mode="json")
                for gate, mode in (
                    ("row_ids", "exact_hash"),
                    ("raw_counts", "exact_hash"),
                    ("sample_weights", "exact_or_numerical"),
                    ("thinning_rng", "exact_hash"),
                    ("loss", "numerical_tolerance"),
                    ("gradient", "numerical_tolerance"),
                    ("parameter", "numerical_tolerance"),
                    ("interrupted_resume", "exact_hash"),
                )
            ],
        },
        "qualification_contract_id",
    )
    receipt = _identified(
        IntegratedLoaderQualificationReceiptV2,
        {
            "schema_version": 2,
            "qualification_contract_id": contract.qualification_contract_id,
            "gpu_name": contract.expected_gpu_name,
            "gpu_uuid": "GPU-0123",
            "gpu_count": 1,
            "cuda_version": contract.expected_cuda_version,
            "torch_version": contract.expected_torch_version,
            "container_digest": contract.expected_container_digest,
            "worker_count": contract.worker_count,
            "cpu_count": contract.cpu_count,
            "storage_authority_hash": contract.storage_authority_hash,
            "microbatch_cells": 512,
            "microbatches_per_update": 8,
            "macrobatch_cells": 4096,
            "prefetch_depth": contract.prefetch_depth,
            "warmup_updates": contract.warmup_updates,
            "measured_updates": contract.measured_updates,
            "cold_start_measured": True,
            "steady_state_measured": True,
            "cache_policy": contract.cache_policy,
            "measurement_protocol_sha256": contract.measurement_protocol_sha256,
            "measurement_evidence": _artifact("d", "MEASUREMENT.json").model_dump(mode="json"),
            "telemetry_artifact": _artifact("e", "TELEMETRY.parquet").model_dump(mode="json"),
            "parity_artifact": _artifact("f", "PARITY.parquet").model_dump(mode="json"),
            "memory_trace_artifact": _artifact("0", "MEMORY.parquet").model_dump(mode="json"),
            "telemetry_interval_seconds": contract.telemetry_interval_seconds,
            "median_compute_seconds": 0.2,
            "p95_compute_seconds": 0.25,
            "median_data_wait_seconds": 0.01,
            "p95_batch_ready_seconds": 0.04,
            "data_wait_fraction": 0.05,
            "steady_state_gpu_utilization": 0.90,
            "peak_loader_rss_bytes": 8_000_000_000,
            "peak_process_loader_rss_bytes": 7_000_000_000,
            "peak_aggregate_worker_rss_bytes": 8_000_000_000,
            "peak_open_shards": 4,
            "peak_open_file_handles": 64,
            "rss_slope_bytes_per_second": 100.0,
            "rss_slope_upper_ci_bytes_per_second": 500.0,
            "maximum_rss_excursion_bytes": 500_000_000,
            "parity_gates": [
                G00DParityGateEvidence(
                    gate=gate,
                    comparison_mode=mode,
                    reference_sha256="1" * 64,
                    observed_sha256="1" * 64,
                    maximum_absolute_error=5e-7,
                    maximum_relative_error=5e-6,
                ).model_dump(mode="json")
                for gate, mode in (
                    ("row_ids", "exact_hash"),
                    ("raw_counts", "exact_hash"),
                    ("sample_weights", "exact_or_numerical"),
                    ("thinning_rng", "exact_hash"),
                    ("loss", "numerical_tolerance"),
                    ("gradient", "numerical_tolerance"),
                    ("parameter", "numerical_tolerance"),
                    ("interrupted_resume", "exact_hash"),
                )
            ],
            "lru_bound_pass": True,
            "loader_error_count": 0,
            "cuda_error_count": 0,
            "monitor_error_count": 0,
            "maximum_parity_absolute_error": 5e-7,
            "maximum_parity_relative_error": 5e-6,
            "memory_growth_pass": True,
            "status": "pass",
        },
        "receipt_id",
    )
    return contract, receipt


def test_dev30_g00d_is_fail_closed_for_parity_memory_and_environment() -> None:
    contract, receipt = _dev30_g00d()
    validate_integrated_loader_qualification(contract, receipt)
    failed_gate = receipt.parity_gates[0].model_copy(
        update={
            "observed_sha256": "2" * 64,
            "maximum_absolute_error": 2e-6,
            "maximum_relative_error": 2e-5,
        }
    )
    with pytest.raises(IntegrityError, match="status must be fail"):
        validate_integrated_loader_qualification(
            contract,
            receipt.model_copy(update={"parity_gates": (failed_gate, *receipt.parity_gates[1:])}),
        )
    with pytest.raises(IntegrityError, match="memory-growth flag"):
        validate_integrated_loader_qualification(
            contract,
            receipt.model_copy(update={"rss_slope_upper_ci_bytes_per_second": 2000.0}),
        )
    with pytest.raises(IntegrityError, match="gpu_name differs"):
        validate_integrated_loader_qualification(
            contract, receipt.model_copy(update={"gpu_name": "Different GPU"})
        )
    with pytest.raises(IntegrityError, match="gpu_count differs"):
        validate_integrated_loader_qualification(
            contract, receipt.model_copy(update={"gpu_count": 2})
        )
    with pytest.raises(IntegrityError, match="measurement_protocol_sha256 differs"):
        validate_integrated_loader_qualification(
            contract,
            receipt.model_copy(update={"measurement_protocol_sha256": "9" * 64}),
        )
