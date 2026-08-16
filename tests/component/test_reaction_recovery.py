from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credo_count_sde_v4.contracts import (
    ComponentTestContract,
    ComponentTestReceiptV2,
    ReactionRecoveryQualificationBundle,
    ReactionRecoveryTestReceipt,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.reaction import (
    qualify_reaction_recovery,
    verify_reaction_recovery_qualification,
)


@pytest.fixture(scope="module")
def qualified_t07s(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("t07s") / "reaction-recovery"
    qualify_reaction_recovery(destination)
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
    assert receipt.r0_false_selection_guard_pass
    assert receipt.r0_audit_false_promotions == 0
    assert receipt.r1_selected_update == 100
    assert receipt.r1_point_delta < -0.4
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
