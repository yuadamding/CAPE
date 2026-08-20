from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    FoldRowRoleRecord,
    G00CCanaryPrerequisiteV1,
    G00CCommonSupportFeatureMetricV1,
    G00CFeatureRankingFreezeV1,
    G00CFeatureSelectionResultV3,
    G00CMonitorFreezeV1,
    G00CParentBindingV1,
    G00CPublicationFreezeV1,
    G00CRefitFreezeV1,
    G00CSampleSizeSelectionResultV3,
    G00CSelectionFreezeContractV1,
    G00CSelectionMarginFreezeV1,
    G00CSerialSelectionFreezeV1,
    G00CSupportAuditFreezeV1,
    StrictModel,
)
from credo_count_sde_v4.store import (
    checkpoint_multinomial_refit,
    derive_refit_seed_schedule,
    expand_common_support_probabilities,
    sample_size_decision_v3,
    weighted_multinomial_nll_per_count,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _artifact(name: str, digit: str = "a") -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.dev34",
        schema_version=1,
        sha256=digit * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri=name,
    )


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


def _payload() -> dict[str, Any]:
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
    parents = tuple(
        G00CParentBindingV1(
            role=role,
            identity_field="identity_id",
            identity_value=f"parent-{index}",
            artifact=_artifact(f"parents/{role}.json", hex(index + 10)[-1]),
        )
        for index, role in enumerate(parent_roles)
    )
    seed_schedule = derive_refit_seed_schedule("0" * 64)
    schedule_artifact = _artifact("G00C_REFIT_SEED_SCHEDULE.json", "6")
    ranking = G00CFeatureRankingFreezeV1(
        tie_breaks=("total_umi_desc", "detection_count_desc", "feature_id_utf8_asc"),
        fit_reference_rows_hash="5" * 64,
        validation_rows_hash="2" * 64,
    )
    common_support = G00CCommonSupportFeatureMetricV1(
        residual_frequency_artifact=_artifact("RESIDUAL_FREQUENCIES.parquet", "7"),
        residual_frequency_fit_rows_hash="5" * 64,
    )
    refits = G00CRefitFreezeV1(
        seed_schedule_id=seed_schedule.schedule_id,
        seed_schedule_artifact=schedule_artifact,
        model_config_hash="8" * 64,
        replay_implementation_sha256="9" * 64,
    )
    return {
        "schema_version": 1,
        "parent_bindings": [parent.model_dump(mode="json") for parent in parents],
        "dev33_canary": G00CCanaryPrerequisiteV1(
            dev33_code_commit="8" * 40,
            dev33_wheel_sha256="9" * 64,
            canary_contract_id="a" * 64,
            execution_record_commit="b" * 40,
            independent_verification_id="c" * 64,
            final_audit_id="d" * 64,
            authority_archive_uri="immutable://g00/dev33b-authority.tar.zst",
            authority_archive_sha256="e" * 64,
        ).model_dump(mode="json"),
        "row_roles": [role.model_dump(mode="json") for role in roles],
        "row_role_freeze": _artifact("ROW_ROLE_FREEZE.json", "e").model_dump(mode="json"),
        "training_scale_row_order_hash": "5" * 64,
        "seed_schedule": seed_schedule.model_dump(mode="json"),
        "seed_schedule_artifact": schedule_artifact.model_dump(mode="json"),
        "feature_ranking": ranking.model_dump(mode="json"),
        "common_support_metric": common_support.model_dump(mode="json"),
        "serial_selection": G00CSerialSelectionFreezeV1(
            frozen_nested_training_row_order_hash="5" * 64,
            feature_selection_reference_rows_hash="5" * 64,
        ).model_dump(mode="json"),
        "selection_margins": G00CSelectionMarginFreezeV1().model_dump(mode="json"),
        "refits": refits.model_dump(mode="json"),
        "support_audit": G00CSupportAuditFreezeV1(
            dimensions=(
                "donor_checkpoint",
                "target",
                "guide",
                "control_vs_targeting",
                "sampler_stratum",
            )
        ).model_dump(mode="json"),
        "monitor": G00CMonitorFreezeV1(
            implementation_sha256="f" * 64,
            polling_interval_milliseconds=50,
            maximum_unreadable_samples=10,
            maximum_unreadable_fraction=0.001,
            maximum_consecutive_unreadable_samples=3,
            maximum_temporal_gap_milliseconds=500,
            maximum_process_tree_rss_bytes=16 * 1024**3,
        ).model_dump(mode="json"),
        "publication": _publication().model_dump(mode="json"),
    }


def test_dev34_contract_freezes_the_four_authorized_gaps() -> None:
    contract = _identified(G00CSelectionFreezeContractV1, _payload(), "freeze_id")
    assert contract.seed_schedule.other_candidate_replay_draw_ids == (
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        58,
    )
    assert contract.serial_selection.serial_dependency == "feature_then_cell_budget_v1"
    assert contract.common_support_metric.reference_feature_count == 4096
    assert contract.selection_margins.feature_equivalence_epsilon == 0.0001
    assert contract.selection_margins.cell_equivalence_epsilon == 0.0001
    assert "fail_no_saturation" in contract.publication.terminal_statuses
    assert contract.dev33_canary.parent_eligible is False
    assert contract.execution_backend == "cpu_only"
    assert contract.g00c_status == "contract_frozen_not_run"
    assert contract.g00d_status == "blocked"
    assert not contract.biological_claims


def test_dev34_rejects_parent_drift_canary_reuse_and_row_role_incompleteness() -> None:
    payload = _payload()
    payload["parent_bindings"] = list(reversed(payload["parent_bindings"]))
    with pytest.raises(ValidationError, match="six ordered B0-A2"):
        _identified(G00CSelectionFreezeContractV1, payload, "freeze_id")

    payload = _payload()
    payload["dev33_canary"]["compact_payload_reuse_permitted"] = True
    with pytest.raises(ValidationError, match="Input should be False"):
        _identified(G00CSelectionFreezeContractV1, payload, "freeze_id")

    payload = _payload()
    payload["row_roles"][0]["rows"] -= 1
    with pytest.raises(ValidationError, match="reconcile to G00B"):
        _identified(G00CSelectionFreezeContractV1, payload, "freeze_id")


def test_dev34_seed_schedule_is_expanded_deterministic_and_fail_closed() -> None:
    first = derive_refit_seed_schedule("0" * 64)
    second = derive_refit_seed_schedule("0" * 64)
    different = derive_refit_seed_schedule("1" * 64)
    assert first == second
    assert first.schedule_id != different.schedule_id
    assert len(first.records) == 59
    assert first.records[0].initialization != first.records[0].thinning

    corrupted = first.model_dump(mode="json")
    corrupted["records"][1]["initialization"] = corrupted["records"][0]["initialization"]
    with pytest.raises(ValidationError, match="must not collide"):
        type(first).model_validate(corrupted)

    weakened = first.model_dump(mode="json")
    weakened["other_candidate_replay_draw_ids"] = [0, 58]
    with pytest.raises(ValidationError, match="frozen subset"):
        type(first).model_validate(weakened)


def test_dev34_common_support_expansion_preserves_mass_and_denominator() -> None:
    candidate = np.asarray([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]], dtype=np.float64)
    residual = np.asarray([[0.25, 0.75], [0.6, 0.4]], dtype=np.float64)
    expanded = expand_common_support_probabilities(candidate, residual, reference_feature_count=4)
    np.testing.assert_allclose(expanded.sum(axis=1), 1.0)
    np.testing.assert_allclose(expanded[:, :2], candidate[:, :2])
    np.testing.assert_allclose(expanded[:, 2:], candidate[:, -1:] * residual)

    counts = np.asarray([[4, 3, 2, 1], [1, 2, 3, 4]], dtype=np.int64)
    weights = np.asarray([1.0, 2.0], dtype=np.float64)
    score = weighted_multinomial_nll_per_count(counts, expanded, weights)
    assert np.isfinite(score)

    with pytest.raises(ValueError, match="do not match"):
        expand_common_support_probabilities(
            candidate,
            residual[:, :1],
            reference_feature_count=4,
        )
    with pytest.raises(ValueError, match="invalid"):
        weighted_multinomial_nll_per_count(counts[:, :2], expanded, weights)


def test_dev34_sample_size_rule_has_terminal_no_saturation() -> None:
    base = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    status, selected = sample_size_decision_v3(
        grid_stage="base",
        candidates=base,
        paired_absolute_difference_q95=np.asarray([0.5, 0.4, 0.3, 0.2, 0.0]),
        support_eligible=(True,) * 5,
    )
    assert (status, selected) == ("extension_required", None)

    status, selected = sample_size_decision_v3(
        grid_stage="base",
        candidates=base,
        paired_absolute_difference_q95=np.asarray([0.5, 0.4, 0.3, 0.00005, 0.0]),
        support_eligible=(True,) * 5,
    )
    assert (status, selected) == ("selected", 500_000)

    extension = (*base, 2_000_000)
    status, selected = sample_size_decision_v3(
        grid_stage="extension",
        candidates=extension,
        paired_absolute_difference_q95=np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        support_eligible=(True,) * 6,
    )
    assert (status, selected) == ("fail_no_saturation", None)

    status, selected = sample_size_decision_v3(
        grid_stage="extension",
        candidates=extension,
        paired_absolute_difference_q95=np.asarray([0.5, 0.4, 0.3, 0.2, 0.00005, 0.0]),
        support_eligible=(True,) * 6,
    )
    assert (status, selected) == ("selected", 1_000_000)

    with pytest.raises(ValueError, match="frozen Dev35"):
        sample_size_decision_v3(
            grid_stage="base",
            candidates=base,
            paired_absolute_difference_q95=np.zeros(5),
            support_eligible=(True,) * 5,
            epsilon=0.01,
        )


def test_dev34_result_contracts_bind_serial_parent_and_stop_states() -> None:
    feature = _identified(
        G00CFeatureSelectionResultV3,
        {
            "schema_version": 3,
            "curve": _artifact("feature-curve.parquet").model_dump(mode="json"),
            "refit_records": _artifact("feature-refits.parquet", "b").model_dump(mode="json"),
            "ordered_features": _artifact("ordered-features.parquet", "c").model_dump(mode="json"),
            "common_support_metric_receipt": _artifact("common-support.json", "d").model_dump(
                mode="json"
            ),
            "fit_rows_hash": "1" * 64,
            "validation_rows_hash": "2" * 64,
            "selected_feature_count": 1024,
            "selected_feature_order_sha256": "3" * 64,
            "selected_at_reference": False,
        },
        "result_id",
    )
    base_stop = _identified(
        G00CSampleSizeSelectionResultV3,
        {
            "schema_version": 3,
            "grid_stage": "base",
            "curve": _artifact("cell-curve.parquet", "4").model_dump(mode="json"),
            "refit_records": _artifact("cell-refits.parquet", "5").model_dump(mode="json"),
            "training_scale_row_order": _artifact("row-order.json", "6").model_dump(mode="json"),
            "parent_feature_selection_result_sha256": feature.result_id,
            "selected_feature_count": feature.selected_feature_count,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "selection_status": "extension_required",
        },
        "result_id",
    )
    assert base_stop.selected_training_cells is None

    extension_failure = _identified(
        G00CSampleSizeSelectionResultV3,
        {
            "schema_version": 3,
            "grid_stage": "extension",
            "curve": _artifact("extension-curve.parquet", "7").model_dump(mode="json"),
            "refit_records": _artifact("extension-refits.parquet", "8").model_dump(mode="json"),
            "training_scale_row_order": _artifact("row-order.json", "6").model_dump(mode="json"),
            "parent_feature_selection_result_sha256": feature.result_id,
            "selected_feature_count": feature.selected_feature_count,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "base_grid_extension_required_receipt": _artifact("base-stop.json", "9").model_dump(
                mode="json"
            ),
            "selection_status": "fail_no_saturation",
        },
        "result_id",
    )
    assert extension_failure.selection_status == "fail_no_saturation"

    invalid = extension_failure.model_dump(mode="json")
    invalid["selection_status"] = "selected"
    invalid["selected_training_cells"] = 2_000_000
    invalid["selected_training_rows"] = _artifact("selected.parquet").model_dump(mode="json")
    with pytest.raises(ValidationError, match="reference alone"):
        _identified(G00CSampleSizeSelectionResultV3, invalid, "result_id")


def test_dev34_rejects_feature_grid_row_hash_or_publication_weakening() -> None:
    with pytest.raises(ValidationError, match="frozen prefix grid"):
        G00CFeatureRankingFreezeV1(
            tie_breaks=("total_umi_desc", "detection_count_desc", "feature_id_utf8_asc"),
            fit_reference_rows_hash="1" * 64,
            validation_rows_hash="2" * 64,
            candidate_feature_counts=(256, 512),
        )

    payload = _payload()
    payload["serial_selection"]["frozen_nested_training_row_order_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="frozen nested row order"):
        _identified(G00CSelectionFreezeContractV1, payload, "freeze_id")

    payload = _payload()
    payload["common_support_metric"]["residual_frequency_fit_rows_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="residual frequencies"):
        _identified(G00CSelectionFreezeContractV1, payload, "freeze_id")

    publication = _publication().model_dump(mode="json")
    publication["always_required_artifacts"].remove("REFIT_PROVENANCE.parquet")
    with pytest.raises(ValidationError, match="status surfaces"):
        G00CPublicationFreezeV1.model_validate(publication)


def test_dev34_closed_form_refit_is_finite_and_rejects_invalid_counts() -> None:
    counts = np.asarray([[100, 30, 10], [25, 50, 75]], dtype=np.int64)
    refit = checkpoint_multinomial_refit(counts, counts)
    assert refit.validation_total_count == 290
    assert refit.validation_excess_nll_per_count >= 0.0
    assert np.isfinite(refit.validation_nll_per_count)

    with pytest.raises(ValueError, match="invalid"):
        checkpoint_multinomial_refit(
            np.asarray([[1, -1]], dtype=np.int64),
            np.asarray([[1, 1]], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="positive"):
        checkpoint_multinomial_refit(
            np.asarray([[0, 0]], dtype=np.int64),
            np.asarray([[1, 1]], dtype=np.int64),
        )
