from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

import credo_count_sde_v4.store.g00c_v3 as g00c_v3
from credo_count_sde_v4.canonical import canonical_json_bytes, contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    FoldRowRoleRecord,
    G00CCanaryPrerequisiteV1,
    G00CCommonSupportFeatureMetricV1,
    G00CCommonSupportMetricReceiptV1,
    G00CCommonSupportPriorV2,
    G00CD1ExecutionAuthorityFreezeV1,
    G00CDecisionReceiptV3,
    G00CExecutionBundleV3,
    G00CFeatureRankingFreezeV1,
    G00CFeatureSelectionResultV3,
    G00CImplementationAuthorityV1,
    G00CImplementationBindingV1,
    G00CMonitorFreezeV1,
    G00CParentBindingV1,
    G00CPublicationFreezeV1,
    G00CRefitFreezeV1,
    G00CRefitSeedScheduleV1,
    G00CSampleSizeExtensionFreezeV1,
    G00CSampleSizeSelectionResultV3,
    G00CSelectionFreezeContractV1,
    G00CSelectionMarginFreezeV1,
    G00CSerialSelectionFreezeV1,
    G00CSupportAuditContractV2,
    G00CSupportAuditFreezeV1,
    StrictModel,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import (
    build_g00c_decision_receipt_v3,
    checkpoint_multinomial_refit_common_support,
    derive_refit_seed_schedule,
    feature_selection_decision_v3,
    sample_size_decision_v3,
    verify_g00c_decision_v3,
    verify_g00c_execution_v3,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "0" * 64
    normalized = model.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in model.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _write(path: Path, content: bytes) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef(
        schema_id="test.dev35",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type="application/octet-stream",
        relative_uri=path.name,
    )


def _write_json(root: Path, name: str, payload: Any) -> ArtifactRef:
    content = (
        payload.model_dump_json().encode("utf-8")
        if isinstance(payload, StrictModel)
        else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    artifact = _write(root / name, content)
    return artifact.model_copy(update={"media_type": "application/json"})


def _write_parquet(root: Path, name: str, table: pd.DataFrame) -> ArtifactRef:
    path = root / name
    table.to_parquet(path, index=False)
    artifact = ArtifactRef(
        schema_id="test.dev35",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type="application/vnd.apache.parquet",
        relative_uri=name,
    )
    return artifact


def _publication() -> G00CPublicationFreezeV1:
    return G00CPublicationFreezeV1(
        always_required_artifacts=(
            "G00C_REFIT_SEED_SCHEDULE.json",
            "ROW_HASHES.json",
            "FEATURE_SELECTION_CURVE.parquet",
            "CELL_SELECTION_CURVE.parquet",
            "REFIT_PROVENANCE.parquet",
            "REPLAY_AUDIT.json",
            "SUPPORT_AUDIT.parquet",
            "SAMPLER_RESTART.json",
            "DECISION_RECEIPT.json",
            "artifacts.json",
            "COMMITTED",
            "SHA256SUMS",
        ),
        pass_only_required_artifacts=(
            "SELECTED_FEATURES.parquet",
            "SELECTED_ROWS.parquet",
            "compact.h5",
            "puroR-sidecar.h5",
            "PHYSICAL_RUNS.parquet",
            "COMPACT_VERIFICATION.json",
            "WRITER_RESTART.json",
        ),
        extension_required_artifacts=("BASE_GRID_STOP_RECEIPT.json",),
        fail_no_saturation_required_artifacts=(
            "BASE_GRID_STOP_RECEIPT.json",
            "EXTENSION_RESULT.json",
        ),
    )


def _freeze(root: Path, schedule_artifact: ArtifactRef) -> G00CSelectionFreezeContractV1:
    schedule = derive_refit_seed_schedule("0" * 64)
    roles = tuple(
        FoldRowRoleRecord(role=role, rows=rows, row_ids_hash=digit * 64)
        for role, rows, digit in (
            ("training_fit", 16_924_672, "1"),
            ("training_validation", 65_536, "2"),
            ("heldout_source_query", 1_752_037, "3"),
            ("protected_heldout_stimulated", 3_254_597, "4"),
        )
    )
    parent_roles = (
        "g00a_v2",
        "g00b_v2",
        "source_plane_amendment",
        "source_plane_amendment_receipt",
        "source_plane_decision_receipt",
        "b0_a2_execution_amendment",
    )
    parent_bindings = []
    for index, role in enumerate(parent_roles):
        artifact = _write_json(root, f"parent-{index}.json", {"role": role})
        parent_bindings.append(
            G00CParentBindingV1(
                role=role,
                identity_field="identity_id",
                identity_value=f"parent-{index}",
                artifact=artifact,
            )
        )
    archive = _write(root / "dev33b-authority.tar.zst", b"closed-dev33b")
    row_roles_artifact = _write(root / "row-roles-placeholder.parquet", b"metadata")
    ranking = G00CFeatureRankingFreezeV1(
        tie_breaks=("total_umi_desc", "detection_count_desc", "feature_id_utf8_asc"),
        fit_reference_rows_hash="5" * 64,
        validation_rows_hash="2" * 64,
    )
    payload = {
        "parent_bindings": parent_bindings,
        "dev33_canary": G00CCanaryPrerequisiteV1(
            dev33_code_commit="8" * 40,
            dev33_wheel_sha256="9" * 64,
            canary_contract_id="a" * 64,
            execution_record_commit="b" * 40,
            independent_verification_id="c" * 64,
            final_audit_id="d" * 64,
            authority_archive_uri=archive.relative_uri,
            authority_archive_sha256=archive.sha256,
        ),
        "row_roles": roles,
        "row_role_freeze": row_roles_artifact,
        "training_scale_row_order_hash": "5" * 64,
        "seed_schedule": schedule,
        "seed_schedule_artifact": schedule_artifact,
        "feature_ranking": ranking,
        "common_support_metric": G00CCommonSupportFeatureMetricV1(
            residual_frequency_artifact=_write(root / "residual-plan.json", b"plan"),
            residual_frequency_fit_rows_hash="5" * 64,
        ),
        "serial_selection": G00CSerialSelectionFreezeV1(
            frozen_nested_training_row_order_hash="5" * 64,
            feature_selection_reference_rows_hash="5" * 64,
        ),
        "selection_margins": G00CSelectionMarginFreezeV1(),
        "refits": G00CRefitFreezeV1(
            seed_schedule_id=schedule.schedule_id,
            seed_schedule_artifact=schedule_artifact,
            model_config_hash="8" * 64,
            replay_implementation_sha256="9" * 64,
        ),
        "support_audit": G00CSupportAuditFreezeV1(
            dimensions=(
                "donor_checkpoint",
                "target",
                "guide",
                "control_vs_targeting",
                "sampler_stratum",
            )
        ),
        "monitor": G00CMonitorFreezeV1(
            implementation_sha256="f" * 64,
            polling_interval_milliseconds=50,
            maximum_unreadable_samples=10,
            maximum_unreadable_fraction=0.001,
            maximum_consecutive_unreadable_samples=3,
            maximum_temporal_gap_milliseconds=500,
            maximum_process_tree_rss_bytes=16 * 1024**3,
        ),
        "publication": _publication(),
    }
    return _identified(G00CSelectionFreezeContractV1, payload, "freeze_id")


def _refit_records(
    *,
    kind: str,
    candidates: tuple[int, ...],
    schedule: Any,
    validation_hash: str,
    fit_hashes: dict[int, str],
    differences: dict[int, float],
) -> pd.DataFrame:
    rows = []
    for record in schedule.records:
        total = 10_000 + record.draw_id
        reference_nll = 7.0 + record.draw_id * 1e-6
        for candidate in candidates:
            nll = reference_nll + differences[candidate]
            rows.append(
                {
                    "candidate_kind": kind,
                    "draw_id": record.draw_id,
                    "candidate_value": candidate,
                    "initialization": record.initialization,
                    "training_sampler": record.training_sampler,
                    "thinning": record.thinning,
                    "validation_evaluation": record.validation_evaluation,
                    "stochastic_optimizer_or_augmentation": (
                        record.stochastic_optimizer_or_augmentation
                    ),
                    "restart_interruption_point": record.restart_interruption_point,
                    "fit_row_hash": fit_hashes[candidate],
                    "validation_row_hash": validation_hash,
                    "model_config_hash": "8" * 64,
                    "final_state_hash": hashlib.sha256(
                        f"{kind}|{record.draw_id}|{candidate}".encode()
                    ).hexdigest(),
                    "validation_total_count": total,
                    "validation_nll_sum": nll * total,
                    "validation_nll_per_count": nll,
                    "fit_status": "pass",
                }
            )
    return pd.DataFrame(rows, columns=g00c_v3.REFIT_V3_COLUMNS)


def _support_table() -> pd.DataFrame:
    rows = []
    dimensions = (
        "donor_checkpoint",
        "target",
        "guide",
        "control_vs_targeting",
        "sampler_stratum",
    )
    grids = {
        "feature_count": (256, 512, 1024, 2048, 4096),
        "training_cells": (50_000, 100_000, 250_000, 500_000, 1_000_000),
    }
    for kind, candidates in grids.items():
        for candidate in candidates:
            for dimension in dimensions:
                rows.append(
                    {
                        "candidate_kind": kind,
                        "candidate_value": candidate,
                        "dimension": dimension,
                        "stratum_id": f"{dimension}-all",
                        "cells": candidate,
                        "weighted_effective_sample_size": float(candidate),
                        "maximum_to_median_weight_ratio": 1.0,
                        "zero_support": False,
                        "selection_eligible": True,
                    }
                )
    return pd.DataFrame(rows, columns=g00c_v3.SUPPORT_V1_COLUMNS)


def _authority_chain(
    tmp_path: Path,
    *,
    order_count: int = 1_000_000,
) -> tuple[
    G00CSelectionFreezeContractV1,
    G00CD1ExecutionAuthorityFreezeV1,
    G00CExecutionBundleV3,
]:
    schedule = derive_refit_seed_schedule("0" * 64)
    schedule_artifact = _write_json(tmp_path, "seed-schedule.json", schedule)
    freeze = _freeze(tmp_path, schedule_artifact)
    freeze_artifact = _write_json(tmp_path, "selection-freeze.json", freeze)

    order_rows = np.arange(order_count, dtype=np.int64)
    order_table = pd.DataFrame({"rank": np.arange(1, order_count + 1), "row_id": order_rows})
    order_artifact = _write_parquet(tmp_path, "training-row-order.parquet", order_table)
    order_hash = hashlib.sha256(np.asarray(order_rows, dtype="<i8").tobytes()).hexdigest()
    support_contract = _identified(
        G00CSupportAuditContractV2,
        {
            "stage": "base",
            "feature_candidate_counts": (256, 512, 1024, 2048, 4096),
            "cell_candidate_counts": (50_000, 100_000, 250_000, 500_000, 1_000_000),
        },
        "contract_id",
    )
    support_contract_artifact = _write_json(
        tmp_path, "base-support-contract.json", support_contract
    )
    implementation_file = _write(tmp_path / "implementation.py", b"# test implementation\n")
    implementation_roles = (
        "feature_ranking",
        "residual_frequency",
        "refit",
        "sampler",
        "monitor",
        "execution_verifier",
        "decision_verifier",
    )
    implementation = G00CImplementationAuthorityV1(
        dev35_code_commit="1" * 40,
        wheel=_write(tmp_path / "dev35.whl", b"wheel"),
        normalized_sdist=_write(tmp_path / "dev35.tar.gz", b"sdist"),
        implementation_tree_sha256="2" * 64,
        environment_lock=_write_json(tmp_path, "environment.json", {"exact": True}),
        environment_kind="exact_local_lock",
        execution_environment_digest=f"sha256:{'3' * 64}",
        implementations=tuple(
            G00CImplementationBindingV1(role=role, artifact=implementation_file)
            for role in implementation_roles
        ),
    )
    authority = _identified(
        G00CD1ExecutionAuthorityFreezeV1,
        {
            "selection_freeze": freeze_artifact,
            "selection_freeze_id": freeze.freeze_id,
            "row_role_freeze": freeze.row_role_freeze,
            "row_roles": freeze.row_roles,
            "nested_training_row_order": order_artifact,
            "nested_training_row_order_hash": order_hash,
            "feature_reference_rows": order_artifact,
            "feature_reference_rows_hash": order_hash,
            "seed_schedule": schedule_artifact,
            "seed_schedule_id": schedule.schedule_id,
            "common_support_prior": G00CCommonSupportPriorV2(),
            "base_support_audit_contract": support_contract_artifact,
            "implementation": implementation,
            "fresh_attempt_id": "dev35-test-a1",
            "publication_root_uri": "publication/dev35-test-a1",
        },
        "authority_id",
    )
    authority_artifact = _write_json(tmp_path, "execution-authority.json", authority)

    feature_candidates = (256, 512, 1024, 2048, 4096)
    feature_refits = _refit_records(
        kind="feature_count",
        candidates=feature_candidates,
        schedule=schedule,
        validation_hash=freeze.feature_ranking.validation_rows_hash,
        fit_hashes={
            candidate: freeze.feature_ranking.fit_reference_rows_hash
            for candidate in feature_candidates
        },
        differences={256: 0.00005, 512: 0.0002, 1024: 0.0003, 2048: 0.0004, 4096: 0.0},
    )
    feature_refits_artifact = _write_parquet(tmp_path, "feature-refits.parquet", feature_refits)
    feature_curve_artifact = _write_parquet(
        tmp_path,
        "feature-curve.parquet",
        pd.DataFrame(
            {
                "feature_count": feature_candidates,
                "mean_validation_nll": [
                    feature_refits.loc[
                        feature_refits["candidate_value"] == candidate,
                        "validation_nll_per_count",
                    ].mean()
                    for candidate in feature_candidates
                ],
                "q95_absolute_paired_nll_difference_to_4096": (
                    0.00005,
                    0.0002,
                    0.0003,
                    0.0004,
                    0.0,
                ),
                "support_eligible": (True,) * 5,
            },
            columns=g00c_v3.FEATURE_CURVE_V3_COLUMNS,
        ),
    )
    ordered = pd.DataFrame(
        {
            "rank": np.arange(1, 4097),
            "canonical_index": np.arange(4096),
            "feature_id": [f"gene-{index:04d}" for index in range(4096)],
        }
    )
    ordered_artifact = _write_parquet(tmp_path, "ordered-features.parquet", ordered)
    residual_path = tmp_path / "residual-frequencies.npz"
    np.savez(
        residual_path,
        **{
            str(width): np.full((3, 4096 - width), 1.0 / (4096 - width))
            for width in (256, 512, 1024, 2048)
        },
    )
    residual_artifact = ArtifactRef(
        schema_id="test.dev35",
        schema_version=1,
        sha256=sha256_file(residual_path),
        size_bytes=residual_path.stat().st_size,
        media_type="application/x-npz",
        relative_uri=residual_path.name,
    )
    validation_counts = _write(tmp_path / "validation-counts.npz", b"bound-counts")
    sensitivity = _write_json(
        tmp_path,
        "prior-sensitivity.json",
        {"selection_threshold": 0.0001, "status": "pass"},
    )
    totals = (
        feature_refits.loc[feature_refits["candidate_value"] == 4096]
        .sort_values("draw_id")["validation_total_count"]
        .to_numpy(dtype="<i8")
    )
    common = _identified(
        G00CCommonSupportMetricReceiptV1,
        {
            "selection_freeze_id": freeze.freeze_id,
            "seed_schedule_id": schedule.schedule_id,
            "seed_schedule_artifact": schedule_artifact,
            "refit_records": feature_refits_artifact,
            "validation_counts": validation_counts,
            "residual_frequencies": residual_artifact,
            "prior_sensitivity": sensitivity,
            "prior": G00CCommonSupportPriorV2(),
            "validation_total_count_hash": hashlib.sha256(totals.tobytes()).hexdigest(),
        },
        "receipt_id",
    )
    common_artifact = _write_json(tmp_path, "common-support-receipt.json", common)
    selected_feature_ids = tuple(ordered.iloc[:256]["feature_id"].astype(str))
    feature = _identified(
        G00CFeatureSelectionResultV3,
        {
            "curve": feature_curve_artifact,
            "refit_records": feature_refits_artifact,
            "ordered_features": ordered_artifact,
            "common_support_metric_receipt": common_artifact,
            "fit_rows_hash": freeze.feature_ranking.fit_reference_rows_hash,
            "validation_rows_hash": freeze.feature_ranking.validation_rows_hash,
            "selected_feature_count": 256,
            "selected_feature_order_sha256": hashlib.sha256(
                canonical_json_bytes(selected_feature_ids)
            ).hexdigest(),
            "selected_at_reference": False,
        },
        "result_id",
    )
    feature_artifact = _write_json(tmp_path, "feature-result.json", feature)

    cell_candidates = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    sample_refits = _refit_records(
        kind="training_cells",
        candidates=cell_candidates,
        schedule=schedule,
        validation_hash=freeze.feature_ranking.validation_rows_hash,
        fit_hashes={
            candidate: hashlib.sha256(
                np.asarray(order_rows[:candidate], dtype="<i8").tobytes()
            ).hexdigest()
            for candidate in cell_candidates
        },
        differences={
            50_000: 0.001,
            100_000: 0.0009,
            250_000: 0.0008,
            500_000: 0.0007,
            1_000_000: 0.0,
        },
    )
    sample_refits_artifact = _write_parquet(tmp_path, "sample-refits.parquet", sample_refits)
    sample_curve_artifact = _write_parquet(
        tmp_path,
        "sample-curve.parquet",
        pd.DataFrame(
            {
                "training_cells": cell_candidates,
                "mean_validation_nll": [
                    sample_refits.loc[
                        sample_refits["candidate_value"] == candidate,
                        "validation_nll_per_count",
                    ].mean()
                    for candidate in cell_candidates
                ],
                "q95_absolute_paired_nll_difference_to_reference": (
                    0.001,
                    0.0009,
                    0.0008,
                    0.0007,
                    0.0,
                ),
                "support_eligible": (True,) * 5,
            },
            columns=g00c_v3.SAMPLE_CURVE_V3_COLUMNS,
        ),
    )
    sample = _identified(
        G00CSampleSizeSelectionResultV3,
        {
            "grid_stage": "base",
            "curve": sample_curve_artifact,
            "refit_records": sample_refits_artifact,
            "training_scale_row_order": order_artifact,
            "parent_feature_selection_result_sha256": feature_artifact.sha256,
            "selected_feature_count": 256,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "selection_status": "extension_required",
        },
        "result_id",
    )
    sample_artifact = _write_json(tmp_path, "sample-result.json", sample)
    support_artifact = _write_parquet(tmp_path, "base-support.parquet", _support_table())
    publication = _write_json(
        tmp_path, "publication.json", {"terminal_status": "extension_required"}
    )
    bundle = _identified(
        G00CExecutionBundleV3,
        {
            "execution_authority": authority_artifact,
            "execution_authority_id": authority.authority_id,
            "selection_freeze": freeze_artifact,
            "selection_freeze_id": freeze.freeze_id,
            "seed_schedule": schedule_artifact,
            "seed_schedule_id": schedule.schedule_id,
            "row_roles": authority.row_role_freeze,
            "feature_selection_result": feature_artifact,
            "feature_selection_result_id": feature.result_id,
            "sample_size_selection_result": sample_artifact,
            "sample_size_selection_result_id": sample.result_id,
            "common_support_receipt": common_artifact,
            "base_support_audit_contract": support_contract_artifact,
            "base_support_audit": support_artifact,
            "sampler_evidence": _write_json(tmp_path, "sampler.json", {"status": "pass"}),
            "publication_manifest": publication,
            "terminal_status": "extension_required",
        },
        "bundle_id",
    )
    return freeze, authority, bundle


def test_dev35_common_support_prior_is_per_gene_and_strictly_positive() -> None:
    training = np.zeros((3, 4096), dtype=np.int64)
    training[:, 0] = 100
    fit = checkpoint_multinomial_refit_common_support(training, modeled_features=256)
    assert fit.candidate_probabilities.shape == (3, 257)
    assert fit.residual_frequencies is not None
    assert np.all(fit.residual_frequencies > 0.0)
    assert np.all(fit.expanded_probabilities > 0.0)
    np.testing.assert_allclose(fit.expanded_probabilities.sum(axis=1), 1.0)
    expected_residual_prior = 0.5 * (4096 - 256)
    denominator = training.sum(axis=1) + 0.5 * 4096
    np.testing.assert_allclose(
        fit.candidate_probabilities[:, -1], expected_residual_prior / denominator
    )


def test_dev35_decisions_require_zero_reference_and_support() -> None:
    feature_values = np.asarray([0.00001, 0.00002, 0.2, 0.3, 0.0])
    selected = feature_selection_decision_v3(
        candidates=(256, 512, 1024, 2048, 4096),
        paired_absolute_difference_q95=feature_values,
        support_eligible=(False, True, True, True, True),
    )
    assert selected == 512
    with pytest.raises(ValueError, match="Feature-selection evidence"):
        feature_selection_decision_v3(
            candidates=(256, 512, 1024, 2048, 4096),
            paired_absolute_difference_q95=np.asarray([0.1, 0.1, 0.1, 0.1, 1e-6]),
            support_eligible=(True,) * 5,
        )

    with pytest.raises(ValueError, match="Sample-size decision"):
        sample_size_decision_v3(
            grid_stage="base",
            candidates=(50_000, 100_000, 250_000, 500_000, 1_000_000),
            paired_absolute_difference_q95=np.asarray([0.1, 0.1, 0.1, 0.1, 1e-6]),
            support_eligible=(True,) * 5,
        )


def test_dev35_v3_execution_and_decision_are_artifact_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, authority, bundle = _authority_chain(tmp_path)
    monkeypatch.setattr(
        g00c_v3,
        "verify_g00c_d1_freeze_v1",
        lambda root, observed: freeze if observed == authority else None,
    )
    verified = verify_g00c_execution_v3(tmp_path, bundle)
    assert verified.terminal_status == "extension_required"
    assert verified.grid_stage == "base"
    assert len(verified.support_audit_sha256s) == 1
    decision_hash = next(
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == "decision_verifier"
    )
    receipt = build_g00c_decision_receipt_v3(verified, verifier_implementation_sha256=decision_hash)
    verify_g00c_decision_v3(tmp_path, bundle, receipt)
    assert not receipt.may_parent_g00d

    corrupted = receipt.model_dump(mode="json")
    corrupted["verifier_implementation_sha256"] = "f" * 64
    altered = _identified(G00CDecisionReceiptV3, corrupted, "receipt_id")
    with pytest.raises(IntegrityError, match="differs from recomputed"):
        verify_g00c_decision_v3(tmp_path, bundle, altered)


def test_dev35_d1_freeze_verifies_all_metadata_relations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(g00c_v3, "MAXIMUM_FROZEN_TRAINING_ROWS", 4)
    monkeypatch.setattr(g00c_v3, "FEATURE_REFERENCE_ROW_COUNT", 2)
    schedule = derive_refit_seed_schedule("0" * 64)
    schedule_ref = _write_json(tmp_path, "small-schedule.json", schedule)
    original = _freeze(tmp_path, schedule_ref)
    role_rows = {
        "training_fit": np.asarray([1, 2, 9, 10], dtype=np.int64),
        "training_validation": np.asarray([3, 4], dtype=np.int64),
        "heldout_source_query": np.asarray([5, 6], dtype=np.int64),
        "protected_heldout_stimulated": np.asarray([7, 8], dtype=np.int64),
    }
    roles_frame = pd.DataFrame(
        [
            {"row_id": int(row_id), "role": role}
            for role, row_ids in role_rows.items()
            for row_id in row_ids
        ]
    )
    roles_ref = _write_parquet(tmp_path, "small-row-roles.parquet", roles_frame)
    role_records = tuple(
        FoldRowRoleRecord(
            role=role,
            rows=len(row_ids),
            row_ids_hash=hashlib.sha256(np.sort(row_ids).astype("<i8").tobytes()).hexdigest(),
        )
        for role, row_ids in role_rows.items()
    )
    order_rows = np.asarray([10, 1, 9, 2], dtype=np.int64)
    order_ref = _write_parquet(
        tmp_path,
        "small-order.parquet",
        pd.DataFrame({"rank": np.arange(1, 5), "row_id": order_rows}),
    )
    reference_rows = order_rows[:2]
    reference_ref = _write_parquet(
        tmp_path,
        "small-reference.parquet",
        pd.DataFrame({"rank": np.arange(1, 3), "row_id": reference_rows}),
    )
    order_hash = hashlib.sha256(order_rows.astype("<i8").tobytes()).hexdigest()
    reference_hash = hashlib.sha256(reference_rows.astype("<i8").tobytes()).hexdigest()
    prior = G00CCommonSupportPriorV2()
    prior_ref = _write_json(tmp_path, "small-prior.json", prior)
    freeze = original.model_copy(
        update={
            "parent_eligible_rows": len(roles_frame),
            "row_roles": role_records,
            "row_role_freeze": roles_ref,
            "training_scale_row_order_hash": order_hash,
            "feature_ranking": original.feature_ranking.model_copy(
                update={
                    "fit_reference_cells": 2,
                    "fit_reference_rows_hash": reference_hash,
                    "validation_rows_hash": role_records[1].row_ids_hash,
                }
            ),
            "common_support_metric": original.common_support_metric.model_copy(
                update={
                    "residual_frequency_artifact": prior_ref,
                    "residual_frequency_fit_rows_hash": reference_hash,
                }
            ),
            "serial_selection": original.serial_selection.model_copy(
                update={
                    "frozen_nested_training_row_order_hash": order_hash,
                    "feature_selection_reference_rows_hash": reference_hash,
                    "feature_selection_training_cells": 2,
                }
            ),
        }
    )
    freeze_ref = _write_json(tmp_path, "small-freeze.json", {"test": "dispatch"})
    support = _identified(
        G00CSupportAuditContractV2,
        {
            "stage": "base",
            "feature_candidate_counts": (256, 512, 1024, 2048, 4096),
            "cell_candidate_counts": (50_000, 100_000, 250_000, 500_000, 1_000_000),
        },
        "contract_id",
    )
    support_ref = _write_json(tmp_path, "small-support-contract.json", support)
    environment_ref = _write_json(
        tmp_path,
        "small-environment.json",
        {
            "environment_kind": "exact_local_lock",
            "execution_environment_digest": f"sha256:{'3' * 64}",
        },
    )
    implementation_ref = _write(tmp_path / "small-implementation.py", b"# frozen\n")
    implementation = G00CImplementationAuthorityV1(
        dev35_code_commit="1" * 40,
        wheel=_write(tmp_path / "small.whl", b"wheel"),
        normalized_sdist=_write(tmp_path / "small.tar.gz", b"sdist"),
        implementation_tree_sha256="2" * 64,
        environment_lock=environment_ref,
        environment_kind="exact_local_lock",
        execution_environment_digest=f"sha256:{'3' * 64}",
        implementations=tuple(
            G00CImplementationBindingV1(role=role, artifact=implementation_ref)
            for role in (
                "feature_ranking",
                "residual_frequency",
                "refit",
                "sampler",
                "monitor",
                "execution_verifier",
                "decision_verifier",
            )
        ),
    )
    authority = _identified(
        G00CD1ExecutionAuthorityFreezeV1,
        {
            "selection_freeze": freeze_ref,
            "selection_freeze_id": freeze.freeze_id,
            "row_role_freeze": roles_ref,
            "row_roles": role_records,
            "nested_training_row_order": order_ref,
            "nested_training_row_order_hash": order_hash,
            "feature_reference_rows": reference_ref,
            "feature_reference_rows_hash": reference_hash,
            "seed_schedule": schedule_ref,
            "seed_schedule_id": schedule.schedule_id,
            "common_support_prior": prior,
            "base_support_audit_contract": support_ref,
            "implementation": implementation,
            "fresh_attempt_id": "small-a1",
            "publication_root_uri": "small-a1",
        },
        "authority_id",
    )
    real_read_model = g00c_v3._read_model

    def dispatch(root: Path, artifact: ArtifactRef, model: type[Any]) -> Any:
        if artifact == freeze_ref:
            return freeze
        return real_read_model(root, artifact, model)

    monkeypatch.setattr(g00c_v3, "_read_model", dispatch)
    assert g00c_v3.verify_g00c_d1_freeze_v1(tmp_path, authority) == freeze


def test_dev35_pass_requires_independent_materialization_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, authority, base_bundle = _authority_chain(tmp_path)
    monkeypatch.setattr(
        g00c_v3,
        "verify_g00c_d1_freeze_v1",
        lambda root, observed: freeze if observed == authority else None,
    )
    payload = base_bundle.model_dump(mode="json")
    payload["terminal_status"] = "pass"
    for field, name in (
        ("compact_payload", "compact.h5"),
        ("compact_verification_receipt", "compact-verification.json"),
        ("reload_receipt", "reload.json"),
    ):
        payload[field] = _write_json(tmp_path, name, {"status": "pass"}).model_dump(mode="json")
    # The underlying sample result is still extension_required, so a self-declared pass fails
    # before any compact callback can confer parent eligibility.
    bundle = _identified(G00CExecutionBundleV3, payload, "bundle_id")
    with pytest.raises(IntegrityError, match="publication status"):
        verify_g00c_execution_v3(tmp_path, bundle)


def test_dev35_verified_pass_requires_and_runs_frozen_materializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, authority, base_bundle = _authority_chain(tmp_path)
    monkeypatch.setattr(
        g00c_v3,
        "verify_g00c_d1_freeze_v1",
        lambda root, observed: freeze if observed == authority else None,
    )
    schedule = derive_refit_seed_schedule("0" * 64)
    order_table = pd.read_parquet(tmp_path / "training-row-order.parquet")
    order_rows = order_table["row_id"].to_numpy(dtype=np.int64)
    candidates = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    records = _refit_records(
        kind="training_cells",
        candidates=candidates,
        schedule=schedule,
        validation_hash=freeze.feature_ranking.validation_rows_hash,
        fit_hashes={
            candidate: hashlib.sha256(
                np.asarray(order_rows[:candidate], dtype="<i8").tobytes()
            ).hexdigest()
            for candidate in candidates
        },
        differences={
            50_000: 0.00005,
            100_000: 0.0002,
            250_000: 0.0003,
            500_000: 0.0004,
            1_000_000: 0.0,
        },
    )
    records_ref = _write_parquet(tmp_path, "pass-sample-refits.parquet", records)
    curve_ref = _write_parquet(
        tmp_path,
        "pass-sample-curve.parquet",
        pd.DataFrame(
            {
                "training_cells": candidates,
                "mean_validation_nll": [
                    records.loc[
                        records["candidate_value"] == candidate,
                        "validation_nll_per_count",
                    ].mean()
                    for candidate in candidates
                ],
                "q95_absolute_paired_nll_difference_to_reference": (
                    0.00005,
                    0.0002,
                    0.0003,
                    0.0004,
                    0.0,
                ),
                "support_eligible": (True,) * 5,
            },
            columns=g00c_v3.SAMPLE_CURVE_V3_COLUMNS,
        ),
    )
    selected_rows_ref = _write_parquet(
        tmp_path,
        "selected-rows.parquet",
        pd.DataFrame({"row_id": order_rows[:50_000]}),
    )
    feature = G00CFeatureSelectionResultV3.model_validate_json(
        (tmp_path / base_bundle.feature_selection_result.relative_uri).read_text()
    )
    sample = _identified(
        G00CSampleSizeSelectionResultV3,
        {
            "grid_stage": "base",
            "curve": curve_ref,
            "refit_records": records_ref,
            "training_scale_row_order": authority.nested_training_row_order,
            "parent_feature_selection_result_sha256": (base_bundle.feature_selection_result.sha256),
            "selected_feature_count": feature.selected_feature_count,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "selection_status": "selected",
            "selected_training_rows": selected_rows_ref,
            "selected_training_cells": 50_000,
        },
        "result_id",
    )
    sample_ref = _write_json(tmp_path, "pass-sample-result.json", sample)
    payload = base_bundle.model_dump(mode="json")
    payload.update(
        {
            "sample_size_selection_result": sample_ref.model_dump(mode="json"),
            "sample_size_selection_result_id": sample.result_id,
            "publication_manifest": _write_json(
                tmp_path, "pass-publication.json", {"terminal_status": "pass"}
            ).model_dump(mode="json"),
            "compact_payload": _write(tmp_path / "compact.h5", b"compact").model_dump(mode="json"),
            "compact_verification_receipt": _write_json(
                tmp_path, "compact-verification.json", {"status": "pass"}
            ).model_dump(mode="json"),
            "reload_receipt": _write_json(tmp_path, "reload.json", {"status": "pass"}).model_dump(
                mode="json"
            ),
            "terminal_status": "pass",
        }
    )
    bundle = _identified(G00CExecutionBundleV3, payload, "bundle_id")

    class Materializer:
        implementation_sha256 = authority.implementation.implementations[5].artifact.sha256
        called = False

        def __call__(
            self,
            root: Path,
            observed: G00CExecutionBundleV3,
            selected_features: tuple[str, ...],
            selected_rows: np.ndarray,
        ) -> None:
            self.called = True
            assert root == tmp_path
            assert observed == bundle
            assert len(selected_features) == 256
            np.testing.assert_array_equal(selected_rows, order_rows[:50_000])

    materializer = Materializer()
    with pytest.raises(IntegrityError, match="requires independent compact"):
        verify_g00c_execution_v3(tmp_path, bundle)

    class WrongMaterializer(Materializer):
        implementation_sha256 = "0" * 64

    with pytest.raises(IntegrityError, match="differs from the frozen implementation"):
        verify_g00c_execution_v3(tmp_path, bundle, materialized_pass_verifier=WrongMaterializer())

    verified = verify_g00c_execution_v3(tmp_path, bundle, materialized_pass_verifier=materializer)
    assert verified.terminal_status == "pass"
    assert materializer.called
    decision = build_g00c_decision_receipt_v3(
        verified,
        verifier_implementation_sha256=authority.implementation.implementations[6].artifact.sha256,
    )
    assert decision.may_parent_g00d
    verify_g00c_decision_v3(tmp_path, bundle, decision, materialized_pass_verifier=materializer)


def test_dev35_common_support_receipt_corruptions_fail_closed(tmp_path: Path) -> None:
    freeze, _, bundle = _authority_chain(tmp_path)
    schedule = G00CRefitSeedScheduleV1.model_validate_json(
        (tmp_path / bundle.seed_schedule.relative_uri).read_text()
    )
    common = G00CCommonSupportMetricReceiptV1.model_validate_json(
        (tmp_path / bundle.common_support_receipt.relative_uri).read_text()
    )
    records = pd.read_parquet(tmp_path / common.refit_records.relative_uri)

    with pytest.raises(IntegrityError, match="cross-wired"):
        g00c_v3._verify_common_support_receipt(
            tmp_path,
            common.model_copy(update={"selection_freeze_id": "f" * 64}),
            freeze=freeze,
            schedule=schedule,
            refit_records=records,
        )

    bad_sensitivity = _write_json(
        tmp_path, "bad-prior-sensitivity.json", {"status": "fail", "selection_threshold": 0.0001}
    )
    with pytest.raises(IntegrityError, match="sensitivity did not pass"):
        g00c_v3._verify_common_support_receipt(
            tmp_path,
            common.model_copy(update={"prior_sensitivity": bad_sensitivity}),
            freeze=freeze,
            schedule=schedule,
            refit_records=records,
        )

    wrong_grid_path = tmp_path / "wrong-grid-residuals.npz"
    np.savez(wrong_grid_path, **{"256": np.ones((3, 4096 - 256))})
    wrong_grid = ArtifactRef(
        schema_id="test.dev35",
        schema_version=1,
        sha256=sha256_file(wrong_grid_path),
        size_bytes=wrong_grid_path.stat().st_size,
        media_type="application/x-npz",
        relative_uri=wrong_grid_path.name,
    )
    with pytest.raises(IntegrityError, match="another prefix grid"):
        g00c_v3._verify_common_support_receipt(
            tmp_path,
            common.model_copy(update={"residual_frequencies": wrong_grid}),
            freeze=freeze,
            schedule=schedule,
            refit_records=records,
        )

    zero_path = tmp_path / "zero-residuals.npz"
    np.savez(
        zero_path,
        **{str(width): np.zeros((3, 4096 - width)) for width in (256, 512, 1024, 2048)},
    )
    zero = ArtifactRef(
        schema_id="test.dev35",
        schema_version=1,
        sha256=sha256_file(zero_path),
        size_bytes=zero_path.stat().st_size,
        media_type="application/x-npz",
        relative_uri=zero_path.name,
    )
    with pytest.raises(IntegrityError, match="not strictly positive"):
        g00c_v3._verify_common_support_receipt(
            tmp_path,
            common.model_copy(update={"residual_frequencies": zero}),
            freeze=freeze,
            schedule=schedule,
            refit_records=records,
        )

    unreadable = _write(tmp_path / "unreadable-residuals.npz", b"not an npz")
    with pytest.raises(IntegrityError, match="cannot be read"):
        g00c_v3._verify_common_support_receipt(
            tmp_path,
            common.model_copy(update={"residual_frequencies": unreadable}),
            freeze=freeze,
            schedule=schedule,
            refit_records=records,
        )

    with pytest.raises(IntegrityError, match="denominator hash differs"):
        g00c_v3._verify_common_support_receipt(
            tmp_path,
            common.model_copy(update={"validation_total_count_hash": "e" * 64}),
            freeze=freeze,
            schedule=schedule,
            refit_records=records,
        )


def test_dev35_execution_cross_wiring_and_failure_receipt_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, authority, bundle = _authority_chain(tmp_path)
    monkeypatch.setattr(
        g00c_v3,
        "verify_g00c_d1_freeze_v1",
        lambda root, observed: freeze if observed == authority else None,
    )
    with pytest.raises(IntegrityError, match="cross-wired"):
        verify_g00c_execution_v3(
            tmp_path, bundle.model_copy(update={"execution_authority_id": "f" * 64})
        )
    with pytest.raises(IntegrityError, match="schedule identity differs"):
        verify_g00c_execution_v3(tmp_path, bundle.model_copy(update={"seed_schedule_id": "f" * 64}))
    with pytest.raises(IntegrityError, match="result artifact identities differ"):
        verify_g00c_execution_v3(
            tmp_path, bundle.model_copy(update={"feature_selection_result_id": "f" * 64})
        )

    failed_publication = _write_json(
        tmp_path, "failed-publication.json", {"terminal_status": "failed_integrity"}
    )
    wrong_failure = _write_json(tmp_path, "wrong-failure.json", {"status": "pass"})
    failed = bundle.model_copy(
        update={
            "terminal_status": "failed_integrity",
            "publication_manifest": failed_publication,
            "failure_receipt": wrong_failure,
        }
    )
    with pytest.raises(IntegrityError, match="failure receipt has another status"):
        verify_g00c_execution_v3(tmp_path, failed)


def test_dev35_feature_selection_artifacts_fail_closed(tmp_path: Path) -> None:
    freeze, _, bundle = _authority_chain(tmp_path)
    schedule = G00CRefitSeedScheduleV1.model_validate_json(
        (tmp_path / bundle.seed_schedule.relative_uri).read_text()
    )
    feature = G00CFeatureSelectionResultV3.model_validate_json(
        (tmp_path / bundle.feature_selection_result.relative_uri).read_text()
    )

    def verify(result: G00CFeatureSelectionResultV3) -> None:
        g00c_v3._verify_feature_selection_v3(
            tmp_path,
            freeze=freeze,
            bundle=bundle,
            result=result,
            schedule=schedule,
        )

    with pytest.raises(IntegrityError, match="other fit or validation rows"):
        verify(feature.model_copy(update={"fit_rows_hash": "f" * 64}))

    curve = pd.read_parquet(tmp_path / feature.curve.relative_uri)
    curve.loc[0, "mean_validation_nll"] += 1.0
    bad_curve = _write_parquet(tmp_path, "bad-feature-curve.parquet", curve)
    with pytest.raises(IntegrityError, match="curve is not derived"):
        verify(feature.model_copy(update={"curve": bad_curve}))

    with pytest.raises(IntegrityError, match="smallest supported prefix"):
        verify(feature.model_copy(update={"selected_feature_count": 1024}))

    ordered = pd.read_parquet(tmp_path / feature.ordered_features.relative_uri)
    bad_schema = _write_parquet(
        tmp_path,
        "bad-feature-schema.parquet",
        ordered.rename(columns={"canonical_index": "wrong"}),
    )
    with pytest.raises(IntegrityError, match="invalid schema"):
        verify(feature.model_copy(update={"ordered_features": bad_schema}))

    duplicate = ordered.copy()
    duplicate.loc[1, "feature_id"] = duplicate.loc[0, "feature_id"]
    duplicate_ref = _write_parquet(tmp_path, "duplicate-features.parquet", duplicate)
    with pytest.raises(IntegrityError, match="unique 4,096-gene ranking"):
        verify(feature.model_copy(update={"ordered_features": duplicate_ref}))

    with pytest.raises(IntegrityError, match="not the true ordered prefix"):
        verify(feature.model_copy(update={"selected_feature_order_sha256": "f" * 64}))

    common = G00CCommonSupportMetricReceiptV1.model_validate_json(
        (tmp_path / feature.common_support_metric_receipt.relative_uri).read_text()
    )
    alternate_common = _write_json(tmp_path, "alternate-common.json", common)
    with pytest.raises(IntegrityError, match="binds other feature refits"):
        verify(feature.model_copy(update={"common_support_metric_receipt": alternate_common}))


def test_dev35_extension_is_preaccess_parented_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, authority, base_bundle = _authority_chain(tmp_path, order_count=2_000_000)
    monkeypatch.setattr(
        g00c_v3,
        "verify_g00c_d1_freeze_v1",
        lambda root, observed: freeze if observed == authority else None,
    )
    base_verified = verify_g00c_execution_v3(tmp_path, base_bundle)
    base_decision = build_g00c_decision_receipt_v3(
        base_verified,
        verifier_implementation_sha256=authority.implementation.implementations[6].artifact.sha256,
    )
    base_bundle_ref = _write_json(tmp_path, "base-execution-bundle.json", base_bundle)
    base_decision_ref = _write_json(tmp_path, "base-stop-decision.json", base_decision)
    feature = G00CFeatureSelectionResultV3.model_validate_json(
        (tmp_path / base_bundle.feature_selection_result.relative_uri).read_text()
    )
    ordered = pd.read_parquet(tmp_path / feature.ordered_features.relative_uri)
    surface_ref = _write_parquet(
        tmp_path,
        "selected-feature-surface.parquet",
        ordered.iloc[: feature.selected_feature_count],
    )
    extension_support_contract = _identified(
        G00CSupportAuditContractV2,
        {
            "stage": "extension",
            "feature_candidate_counts": (),
            "cell_candidate_counts": (2_000_000,),
        },
        "contract_id",
    )
    extension_support_contract_ref = _write_json(
        tmp_path, "extension-support-contract.json", extension_support_contract
    )
    extension = _identified(
        G00CSampleSizeExtensionFreezeV1,
        {
            "base_selection_freeze_id": freeze.freeze_id,
            "base_execution_bundle": base_bundle_ref,
            "base_extension_required_receipt": base_decision_ref,
            "base_feature_selection_result": base_bundle.feature_selection_result,
            "selected_feature_count": feature.selected_feature_count,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "selected_feature_surface": surface_ref,
            "seed_schedule_id": base_bundle.seed_schedule_id,
            "seed_schedule_artifact": base_bundle.seed_schedule,
            "base_support_audit": base_bundle.base_support_audit,
            "extension_support_audit_contract": extension_support_contract_ref,
            "fresh_attempt_id": "dev35-extension-a1",
            "publication_root_uri": "publication/dev35-extension-a1",
        },
        "extension_freeze_id",
    )
    extension_ref = _write_json(tmp_path, "extension-freeze.json", extension)
    dimensions = (
        "donor_checkpoint",
        "target",
        "guide",
        "control_vs_targeting",
        "sampler_stratum",
    )
    extension_support_ref = _write_parquet(
        tmp_path,
        "extension-support.parquet",
        pd.DataFrame(
            [
                {
                    "candidate_kind": "training_cells",
                    "candidate_value": 2_000_000,
                    "dimension": dimension,
                    "stratum_id": f"{dimension}-all",
                    "cells": 2_000_000,
                    "weighted_effective_sample_size": 2_000_000.0,
                    "maximum_to_median_weight_ratio": 1.0,
                    "zero_support": False,
                    "selection_eligible": True,
                }
                for dimension in dimensions
            ],
            columns=g00c_v3.SUPPORT_V1_COLUMNS,
        ),
    )
    schedule = derive_refit_seed_schedule("0" * 64)
    order_rows = pd.read_parquet(tmp_path / "training-row-order.parquet")["row_id"].to_numpy(
        dtype=np.int64
    )
    candidates = (50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000)
    differences = {
        50_000: 0.001,
        100_000: 0.0009,
        250_000: 0.0008,
        500_000: 0.0007,
        1_000_000: 0.0006,
        2_000_000: 0.0,
    }
    records = _refit_records(
        kind="training_cells",
        candidates=candidates,
        schedule=schedule,
        validation_hash=freeze.feature_ranking.validation_rows_hash,
        fit_hashes={
            candidate: hashlib.sha256(
                np.asarray(order_rows[:candidate], dtype="<i8").tobytes()
            ).hexdigest()
            for candidate in candidates
        },
        differences=differences,
    )
    records_ref = _write_parquet(tmp_path, "extension-refits.parquet", records)
    curve_ref = _write_parquet(
        tmp_path,
        "extension-curve.parquet",
        pd.DataFrame(
            {
                "training_cells": candidates,
                "mean_validation_nll": [
                    records.loc[
                        records["candidate_value"] == candidate,
                        "validation_nll_per_count",
                    ].mean()
                    for candidate in candidates
                ],
                "q95_absolute_paired_nll_difference_to_reference": tuple(
                    differences[candidate] for candidate in candidates
                ),
                "support_eligible": (True,) * 6,
            },
            columns=g00c_v3.SAMPLE_CURVE_V3_COLUMNS,
        ),
    )
    sample = _identified(
        G00CSampleSizeSelectionResultV3,
        {
            "grid_stage": "extension",
            "curve": curve_ref,
            "refit_records": records_ref,
            "training_scale_row_order": authority.nested_training_row_order,
            "parent_feature_selection_result_sha256": (base_bundle.feature_selection_result.sha256),
            "selected_feature_count": feature.selected_feature_count,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "base_grid_extension_required_receipt": base_decision_ref,
            "selection_status": "fail_no_saturation",
        },
        "result_id",
    )
    sample_ref = _write_json(tmp_path, "extension-sample-result.json", sample)
    payload = base_bundle.model_dump(mode="json")
    payload.update(
        {
            "sample_size_selection_result": sample_ref.model_dump(mode="json"),
            "sample_size_selection_result_id": sample.result_id,
            "extension_freeze": extension_ref.model_dump(mode="json"),
            "extension_support_audit_contract": extension_support_contract_ref.model_dump(
                mode="json"
            ),
            "extension_support_audit": extension_support_ref.model_dump(mode="json"),
            "publication_manifest": _write_json(
                tmp_path,
                "extension-publication.json",
                {"terminal_status": "fail_no_saturation"},
            ).model_dump(mode="json"),
            "terminal_status": "fail_no_saturation",
        }
    )
    bundle = _identified(G00CExecutionBundleV3, payload, "bundle_id")
    verified = verify_g00c_execution_v3(tmp_path, bundle)
    assert verified.terminal_status == "fail_no_saturation"
    assert len(verified.support_audit_sha256s) == 2
    decision = build_g00c_decision_receipt_v3(
        verified,
        verifier_implementation_sha256=authority.implementation.implementations[6].artifact.sha256,
    )
    assert not decision.may_parent_g00d
    verify_g00c_decision_v3(tmp_path, bundle, decision)


def test_dev35_extension_contract_forbids_third_grid_and_unparented_access() -> None:
    with pytest.raises(ValidationError, match="exactly one two-million"):
        _identified(
            G00CSampleSizeExtensionFreezeV1,
            {
                "base_selection_freeze_id": "1" * 64,
                "base_execution_bundle": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="a" * 64,
                    size_bytes=1,
                    media_type="application/json",
                    relative_uri="base-bundle.json",
                ),
                "base_extension_required_receipt": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="2" * 64,
                    size_bytes=1,
                    media_type="application/json",
                    relative_uri="base.json",
                ),
                "base_feature_selection_result": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="3" * 64,
                    size_bytes=1,
                    media_type="application/json",
                    relative_uri="feature.json",
                ),
                "selected_feature_count": 4096,
                "selected_feature_order_sha256": "4" * 64,
                "selected_feature_surface": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="5" * 64,
                    size_bytes=1,
                    media_type="application/octet-stream",
                    relative_uri="surface.bin",
                ),
                "seed_schedule_id": "6" * 64,
                "seed_schedule_artifact": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="7" * 64,
                    size_bytes=1,
                    media_type="application/json",
                    relative_uri="seeds.json",
                ),
                "added_candidate_cells": (2_000_000, 4_000_000),
                "base_support_audit": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="8" * 64,
                    size_bytes=1,
                    media_type="application/octet-stream",
                    relative_uri="base-support.parquet",
                ),
                "extension_support_audit_contract": ArtifactRef(
                    schema_id="test",
                    schema_version=1,
                    sha256="9" * 64,
                    size_bytes=1,
                    media_type="application/json",
                    relative_uri="extension-support.json",
                ),
                "fresh_attempt_id": "extension-a1",
                "publication_root_uri": "publication/extension-a1",
            },
            "extension_freeze_id",
        )


def test_dev35_artifact_and_refit_corruption_paths_fail_closed(tmp_path: Path) -> None:
    missing = ArtifactRef(
        schema_id="test.dev35",
        schema_version=1,
        sha256="0" * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri="missing.json",
    )
    with pytest.raises(IntegrityError, match="failed verification"):
        g00c_v3._path(tmp_path, missing)
    valid = _write_json(tmp_path, "valid.json", {"status": "pass"})
    with pytest.raises(IntegrityError, match="failed verification"):
        g00c_v3._path(tmp_path, valid.model_copy(update={"sha256": "1" * 64}))
    with pytest.raises(IntegrityError, match="size differs"):
        g00c_v3._path(tmp_path, valid.model_copy(update={"size_bytes": valid.size_bytes + 1}))
    invalid_model = _write(tmp_path / "invalid-model.json", b"not-json")
    with pytest.raises(IntegrityError, match="model artifact is invalid"):
        g00c_v3._read_model(tmp_path, invalid_model, G00CRefitSeedScheduleV1)

    schedule = derive_refit_seed_schedule("0" * 64)
    candidates = (256, 512, 1024, 2048, 4096)
    records = _refit_records(
        kind="feature_count",
        candidates=candidates,
        schedule=schedule,
        validation_hash="2" * 64,
        fit_hashes={candidate: "1" * 64 for candidate in candidates},
        differences={256: 0.1, 512: 0.08, 1024: 0.04, 2048: 0.01, 4096: 0.0},
    )

    def load(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        artifact = _write_parquet(tmp_path, "mutated-refits.parquet", frame)
        return g00c_v3._load_refits_v3(
            tmp_path,
            artifact,
            kind="feature_count",
            candidates=candidates,
            validation_row_hash="2" * 64,
            schedule=schedule,
        )

    loaded, pivot = load(records)
    assert len(loaded) == 59 * 5
    assert pivot.shape == (59, 5)
    with pytest.raises(IntegrityError, match="duplicate"):
        load(pd.concat((records, records.iloc[[0]]), ignore_index=True))
    with pytest.raises(IntegrityError, match="incomplete"):
        load(records.iloc[:-1])
    mutated = records.copy()
    mutated.loc[0, "final_state_hash"] = "bad"
    with pytest.raises(IntegrityError, match="invalid content hash"):
        load(mutated)
    mutated = records.copy()
    mutated.loc[0, "validation_row_hash"] = "3" * 64
    with pytest.raises(IntegrityError, match="another validation-row"):
        load(mutated)
    mutated = records.copy()
    mutated.loc[0, "fit_status"] = "failed"
    with pytest.raises(IntegrityError, match="invalid fit or NLL"):
        load(mutated)
    mutated = records.copy()
    mutated.loc[0, "validation_nll_sum"] += 1.0
    with pytest.raises(IntegrityError, match="not derived"):
        load(mutated)
    mutated = records.copy()
    mutated.loc[0, "validation_total_count"] += 1
    mutated.loc[0, "validation_nll_sum"] = (
        mutated.loc[0, "validation_nll_per_count"] * mutated.loc[0, "validation_total_count"]
    )
    with pytest.raises(IntegrityError, match="same count denominator"):
        load(mutated)
    mutated = records.copy()
    mutated.loc[0, "initialization"] = np.uint64(7)
    with pytest.raises(IntegrityError, match="initialization seeds"):
        load(mutated)


def test_dev35_support_corruption_and_ineligible_candidates(tmp_path: Path) -> None:
    support = _support_table()
    artifact = _write_parquet(tmp_path, "support.parquet", support)
    candidates = (256, 512, 1024, 2048, 4096)
    assert (
        g00c_v3._support_eligibility(
            tmp_path, (artifact,), kind="feature_count", candidates=candidates
        )
        == (True,) * 5
    )
    with pytest.raises(IntegrityError, match="absent"):
        g00c_v3._support_eligibility(tmp_path, (), kind="feature_count", candidates=candidates)
    bad_columns = support.rename(columns={"cells": "wrong"})
    with pytest.raises(IntegrityError, match="invalid schema"):
        g00c_v3._support_eligibility(
            tmp_path,
            (_write_parquet(tmp_path, "bad-support.parquet", bad_columns),),
            kind="feature_count",
            candidates=candidates,
        )
    omitted = support.loc[
        ~((support["candidate_kind"] == "feature_count") & (support["candidate_value"] == 256))
    ]
    with pytest.raises(IntegrityError, match="omits"):
        g00c_v3._support_eligibility(
            tmp_path,
            (_write_parquet(tmp_path, "omitted-support.parquet", omitted),),
            kind="feature_count",
            candidates=candidates,
        )
    zero = support.copy()
    index = zero.index[
        (zero["candidate_kind"] == "feature_count") & (zero["candidate_value"] == 256)
    ][0]
    zero.loc[index, "zero_support"] = True
    zero.loc[index, "selection_eligible"] = False
    assert (
        g00c_v3._support_eligibility(
            tmp_path,
            (_write_parquet(tmp_path, "zero-support.parquet", zero),),
            kind="feature_count",
            candidates=candidates,
        )[0]
        is False
    )
    duplicate = pd.concat((support, support.iloc[[0]]), ignore_index=True)
    with pytest.raises(IntegrityError, match="incomplete or nonfinite"):
        g00c_v3._support_eligibility(
            tmp_path,
            (_write_parquet(tmp_path, "duplicate-support.parquet", duplicate),),
            kind="feature_count",
            candidates=candidates,
        )


def test_dev35_contract_validators_reject_weakened_authority() -> None:
    with pytest.raises(ValidationError, match="exactly frozen"):
        G00CCommonSupportPriorV2(per_feature_pseudocount=0.25)
    with pytest.raises(ValidationError, match="exactly frozen"):
        G00CCommonSupportPriorV2(selection_threshold=0.001)
    with pytest.raises(ValidationError, match="base support surface"):
        _identified(
            G00CSupportAuditContractV2,
            {
                "stage": "base",
                "feature_candidate_counts": (256, 4096),
                "cell_candidate_counts": (50_000, 1_000_000),
            },
            "contract_id",
        )
    with pytest.raises(ValidationError, match="extension support surface"):
        _identified(
            G00CSupportAuditContractV2,
            {
                "stage": "extension",
                "feature_candidate_counts": (4096,),
                "cell_candidate_counts": (2_000_000,),
            },
            "contract_id",
        )
