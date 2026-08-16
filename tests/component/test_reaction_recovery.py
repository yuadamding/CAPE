from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from credo_count_sde_v4.canonical import canonical_json_bytes, contract_id, path_manifest
from credo_count_sde_v4.contracts import (
    ComponentTestContract,
    ComponentTestReceiptV2,
    ReactionRecoveryMetricAmendment,
    ReactionRecoveryQualificationBundle,
    ReactionRecoveryTestReceipt,
    ReactionRecoveryTestReceiptV1,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.persistence import artifact_ref
from credo_count_sde_v4.reaction import (
    amend_reaction_recovery_metrics,
    qualify_reaction_recovery,
    verify_reaction_recovery_metric_amendment,
    verify_reaction_recovery_qualification,
)
from credo_count_sde_v4.reaction.qualification import (
    _bootstrap_from_draws,
    _losses,
    _new_model,
    _recovery_rows,
    _target_metrics,
)


@pytest.fixture(scope="module")
def qualified_t07s(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("t07s") / "reaction-recovery"
    qualify_reaction_recovery(destination)
    return destination


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _make_legacy_parent(current: Path, destination: Path) -> Path:
    """Construct a manifest-valid v1 parent from the corrected test fixture."""

    shutil.copytree(current, destination)
    for name in ("artifacts.json", "COMMITTED", "SHA256SUMS"):
        (destination / name).unlink()
    current_bundle = ReactionRecoveryQualificationBundle.model_validate_json(
        (destination / "reaction-recovery.json").read_text()
    )
    current_receipt = ReactionRecoveryTestReceipt.model_validate_json(
        (destination / "TEST_RECEIPT.json").read_text()
    )
    series = pd.read_parquet(destination / "RECOVERY_SERIES.parquet")
    old = series.drop(
        columns=[
            "observed_centered_interval_log_frequency_change",
            "predicted_centered_interval_log_frequency_change",
            "baseline_centered_interval_log_frequency_change",
            "predicted_centered_relative_fitness_rate",
            "new_squared_error",
            "baseline_squared_error",
        ]
    )
    old["observed_relative_fitness"] = series[
        "observed_centered_interval_log_frequency_change"
    ]
    old["predicted_relative_fitness"] = series["predicted_centered_relative_fitness_rate"]
    old["baseline_relative_fitness"] = 0.0
    old["new_squared_error"] = np.square(
        old.predicted_relative_fitness - old.observed_relative_fitness
    )
    old["baseline_squared_error"] = np.square(old.observed_relative_fitness)
    old.to_parquet(destination / "RECOVERY_SERIES.parquet", index=False)
    target = _target_metrics(old)
    target.to_parquet(destination / "TARGET_METRICS.parquet", index=False)
    new_loss, baseline_loss, point_delta = _losses(target)
    del new_loss, baseline_loss
    draws = pd.read_parquet(destination / "BOOTSTRAP_TARGET_DRAWS.parquet")
    interval, _ = _bootstrap_from_draws(target, draws)
    legacy_receipt_payload = {
        "schema_version": 1,
        "receipt_id": "pending",
        "test_contract_id": current_receipt.test_contract_id,
        "status": "pass",
        "r0_calibration_repeats": current_receipt.r0_calibration_repeats,
        "r0_audit_repeats": current_receipt.r0_audit_repeats,
        "r0_required_margin": current_receipt.r0_required_margin,
        "r0_audit_false_promotions": current_receipt.r0_audit_false_promotions,
        "r0_audit_false_promotion_upper_95": current_receipt.r0_audit_false_promotion_upper_95,
        "r0_false_selection_guard_pass": True,
        "r1_selected_update": current_receipt.r1_selected_update,
        "r1_post_selection_refit_pass": True,
        "r1_point_delta": point_delta,
        "r1_target_bootstrap_interval": interval,
        "r1_margin_pass": interval[1] < -current_receipt.r0_required_margin,
        "r1_reaction_rmse": current_receipt.r1_reaction_rmse,
        "r1_sign_accuracy": current_receipt.r1_sign_accuracy,
        "r1_channel_activity": current_receipt.r1_channel_activity,
        "r1_channel_activity_pass": current_receipt.r1_channel_activity_pass,
        "weighted_gauge_max_abs_error": current_receipt.weighted_gauge_max_abs_error,
        "rollout_mass_max_relative_error": current_receipt.rollout_mass_max_relative_error,
        "probability_normalization_max_abs_error": (
            current_receipt.probability_normalization_max_abs_error
        ),
        "control_target_mask_max_abs_error": current_receipt.control_target_mask_max_abs_error,
        "fixed_channel_max_abs_change": current_receipt.fixed_channel_max_abs_change,
        "protected_metrics_pass": current_receipt.protected_metrics_pass,
        "update_zero_selectable": True,
        "config_hash": current_receipt.config_hash,
        "implementation_hash": current_receipt.implementation_hash,
        "environment_hash": current_receipt.environment_hash,
    }
    legacy_receipt_payload["receipt_id"] = contract_id(
        legacy_receipt_payload, id_field="receipt_id"
    )
    legacy_receipt = ReactionRecoveryTestReceiptV1.model_validate(legacy_receipt_payload)
    _write_json(destination / "TEST_RECEIPT.json", legacy_receipt.model_dump(mode="json"))
    bundle_payload = current_bundle.model_dump(mode="json")
    bundle_payload["qualification_id"] = "pending"
    bundle_payload["method"] = "complete_denominator_dm_reaction_recovery_v1"
    for field, name, schema_id, media_type in (
        (
            "recovery_series",
            "RECOVERY_SERIES.parquet",
            "credo.t07s_series",
            "application/x-parquet",
        ),
        (
            "target_metrics",
            "TARGET_METRICS.parquet",
            "credo.t07s_target_metrics",
            "application/x-parquet",
        ),
        (
            "test_receipt",
            "TEST_RECEIPT.json",
            "credo.t07s_test_receipt",
            "application/json",
        ),
    ):
        bundle_payload[field] = artifact_ref(
            destination,
            destination / name,
            schema_id=schema_id,
            media_type=media_type,
        ).model_dump(mode="json")
    bundle_payload["qualification_id"] = contract_id(
        bundle_payload, id_field="qualification_id"
    )
    bundle = ReactionRecoveryQualificationBundle.model_validate(bundle_payload)
    _write_json(destination / "reaction-recovery.json", bundle.model_dump(mode="json"))
    _write_json(
        destination / "QUALIFICATION_LINK.json",
        {
            "schema_version": 1,
            "qualification_id": bundle.qualification_id,
            "receipt_id": legacy_receipt.receipt_id,
        },
    )
    checksums = path_manifest(destination)
    (destination / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
    )
    manifest = path_manifest(destination)
    _write_json(destination / "artifacts.json", {"schema_version": 1, "files": manifest})
    (destination / "COMMITTED").write_bytes(b"CREDO-V4-COMMITTED\n")
    return destination


@pytest.fixture(scope="module")
def legacy_t07s(qualified_t07s: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_legacy_parent(
        qualified_t07s, tmp_path_factory.mktemp("t07s-parent") / "dev23-parent"
    )


@pytest.fixture(scope="module")
def amended_t07s(legacy_t07s: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("t07s-amendment") / "dev24-amendment"
    amend_reaction_recovery_metrics(destination, parent=legacy_t07s)
    return destination


def test_t07s_passes_null_and_nonzero_recovery_gates(qualified_t07s: Path) -> None:
    bundle = verify_reaction_recovery_qualification(qualified_t07s)
    receipt = ReactionRecoveryTestReceipt.model_validate_json(
        (qualified_t07s / "TEST_RECEIPT.json").read_text()
    )
    component = ComponentTestReceiptV2.model_validate_json(
        (qualified_t07s / "COMPONENT_RECEIPT.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json(
        (qualified_t07s / "TEST_CONTRACT.json").read_text()
    )
    assert isinstance(bundle, ReactionRecoveryQualificationBundle)
    assert bundle.null_calibration_repeats == 59
    assert bundle.null_audit_repeats == 60
    assert bundle.candidate_updates == (0, 25, 50, 100, 200)
    assert contract.test_id == "T07S_REACTION_RECOVERY"
    assert (contract.drift, contract.diffusion, contract.reaction) == (
        "fixed",
        "fixed",
        "trainable",
    )
    assert contract.ecology == contract.decoder == "off"
    assert receipt.status == component.status == "pass"
    assert receipt.schema_version == 2
    assert receipt.r0_false_promotion_guard_pass
    assert receipt.r0_calibration_nonzero_checkpoint_selections == 9
    assert receipt.r0_audit_nonzero_checkpoint_selections == 4
    assert receipt.r0_audit_nonzero_checkpoint_selection_rate == 4 / 60
    assert receipt.r0_audit_false_promotions == 0
    assert receipt.r0_audit_false_promotion_upper_95 < 0.05
    assert receipt.r1_selected_update == 100
    assert receipt.r1_metric_estimand == "centered_interval_log_frequency_change"
    assert receipt.r1_interval_effect_target_balanced_rmse < 0.04
    assert receipt.r1_point_delta < -0.6
    assert receipt.r1_target_bootstrap_interval[1] < -receipt.r0_required_margin
    assert receipt.r1_reaction_rmse < 0.02
    assert receipt.r1_sign_accuracy == 1.0
    assert component.receipt_role == "model_comparison"


def test_t07s_retains_replay_grade_null_and_target_draws(qualified_t07s: Path) -> None:
    null = pd.read_parquet(qualified_t07s / "NULL_REFITS.parquet")
    effects = pd.read_parquet(qualified_t07s / "NULL_MODEL_EFFECTS.parquet")
    draws = pd.read_parquet(qualified_t07s / "BOOTSTRAP_TARGET_DRAWS.parquet")
    curve = pd.read_parquet(qualified_t07s / "RECOVERY_CURVE.parquet")
    assert len(null) == 119
    assert null.repeat.nunique() == 119
    assert set(null.partition.value_counts().to_dict().items()) == {
        ("calibration", 59),
        ("audit", 60),
    }
    assert len(effects) == 119 * 13
    assert effects.groupby(["partition", "repeat"]).target_index.nunique().eq(13).all()
    assert len(draws) == 4000 * 12
    assert draws.groupby("draw").position.nunique().eq(12).all()
    assert tuple(curve.loc[curve.selected, "update"]) == (100,)


def test_t07s_interval_prediction_multiplies_rate_by_duration() -> None:
    model = _new_model(3)
    with torch.no_grad():
        model.target_fitness[1] = 0.4
    frame = pd.DataFrame(
        {
            "catalog": ["duration_regression"] * 6,
            "pool_index": [0, 0, 1, 1, 2, 2],
            "series_index": range(6),
            "target_index": [0, 1, 0, 1, 0, 1],
            "guide_id": [f"guide{index}" for index in range(6)],
            "is_control": [True, False] * 3,
            "duration": [0.5, 0.5, 1.0, 1.0, 2.0, 2.0],
            "source_count": [1000] * 6,
            "terminal_count": [1000] * 6,
            "truth_raw_fitness": [0.0, 0.4] * 3,
        }
    )
    rows = _recovery_rows(model, frame)
    targeting = rows.loc[~rows.is_control]
    ratio = (
        targeting.predicted_centered_interval_log_frequency_change
        / targeting.predicted_centered_relative_fitness_rate
    )
    np.testing.assert_allclose(ratio, targeting.duration, atol=0.0, rtol=0.0)


def test_dev24_amendment_reuses_parent_model_and_corrects_metric(
    amended_t07s: Path, legacy_t07s: Path
) -> None:
    amendment = verify_reaction_recovery_metric_amendment(
        amended_t07s, parent=legacy_t07s
    )
    receipt = ReactionRecoveryTestReceipt.model_validate_json(
        (amended_t07s / "TEST_RECEIPT.json").read_text()
    )
    assert isinstance(amendment, ReactionRecoveryMetricAmendment)
    assert receipt.r0_false_promotion_guard_pass
    assert receipt.r0_calibration_nonzero_checkpoint_selections == 9
    assert receipt.r0_audit_nonzero_checkpoint_selections == 4
    assert receipt.r0_audit_false_promotion_upper_95 < 0.05
    assert receipt.r1_interval_effect_target_balanced_rmse < 0.04
    assert receipt.r1_point_delta < -0.6
    assert receipt.r1_target_bootstrap_interval[1] < -0.45
    parent_link = json.loads((amended_t07s / "PARENT_LINK.json").read_text())
    assert parent_link["optimizer_rerun"] is False
    assert (amended_t07s / "SELECTED_MODEL.safetensors").read_bytes() == (
        legacy_t07s / "SELECTED_MODEL.safetensors"
    ).read_bytes()


def test_dev24_amendment_is_no_clobber_and_parent_tamper_evident(
    amended_t07s: Path, legacy_t07s: Path
) -> None:
    with pytest.raises(FileExistsError):
        amend_reaction_recovery_metrics(amended_t07s, parent=legacy_t07s)
    parent_model = legacy_t07s / "SELECTED_MODEL.safetensors"
    original = parent_model.read_bytes()
    parent_model.write_bytes(original + b"\n")
    try:
        with pytest.raises(IntegrityError, match="Artifact mismatch"):
            verify_reaction_recovery_metric_amendment(amended_t07s, parent=legacy_t07s)
    finally:
        parent_model.write_bytes(original)
    verify_reaction_recovery_metric_amendment(amended_t07s, parent=legacy_t07s)


def test_t07s_protects_gauge_controls_and_fixed_channels(qualified_t07s: Path) -> None:
    receipt = json.loads((qualified_t07s / "TEST_RECEIPT.json").read_text())
    assert receipt["weighted_gauge_max_abs_error"] <= 1e-12
    assert receipt["probability_normalization_max_abs_error"] <= 1e-12
    assert receipt["rollout_mass_max_relative_error"] <= 2e-12
    assert receipt["control_target_mask_max_abs_error"] == 0.0
    assert receipt["fixed_channel_max_abs_change"] == 0.0
    assert receipt["protected_metrics_pass"] is True


def test_t07s_publication_is_no_clobber_and_tamper_evident(qualified_t07s: Path) -> None:
    with pytest.raises(FileExistsError):
        qualify_reaction_recovery(qualified_t07s)
    target = qualified_t07s / "TARGET_METRICS.parquet"
    original = target.read_bytes()
    target.write_bytes(original + b"\n")
    try:
        with pytest.raises(IntegrityError, match="Artifact mismatch"):
            verify_reaction_recovery_qualification(qualified_t07s)
    finally:
        target.write_bytes(original)
    verify_reaction_recovery_qualification(qualified_t07s)
