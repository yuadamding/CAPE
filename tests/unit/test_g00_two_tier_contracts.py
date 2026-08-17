from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import TypeAdapter, ValidationError

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.contracts import (
    FoldNativeCompactViewContract,
    G00SourceAuthority,
    IntegratedLoaderQualificationContract,
    IntegratedLoaderQualificationReceipt,
    StrictModel,
    VirtualCountSource,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import validate_integrated_loader_qualification

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
    source = VirtualCountSource(
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
        G00SourceAuthority,
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
        FoldNativeCompactViewContract,
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
        IntegratedLoaderQualificationContract,
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
        IntegratedLoaderQualificationReceipt,
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
        FoldNativeCompactViewContract.model_validate(base)
    base["training_donor_ids"] = ["D1", "D2", "D3"]
    base["fold_view_id"] = "pending"
    base["fold_view_id"] = contract_id(base, id_field="fold_view_id")
    with pytest.raises(ValidationError, match="uint16"):
        FoldNativeCompactViewContract.model_validate(base)


def test_g00_source_authority_requires_reconciled_source_totals() -> None:
    source = VirtualCountSource(
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
            G00SourceAuthority,
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
