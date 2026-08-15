from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from credo_count_sde_v4.contracts import (
    ArtifactRef,
    StateCalibrationCheckpointScore,
    StateSelectionCalibration,
    StateSelectionCalibrationResults,
    StateSelectionCalibrationRow,
)
from credo_count_sde_v4.training.calibration import (
    _conditional_interaction_null,
    _global_target_main_null,
    run_state_selection_calibration,
)


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        schema_id="credo.state_selection_calibration_results",
        schema_version=1,
        sha256="0" * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri="results.json",
    )


def _zero_failure_upper(repetitions: int) -> float:
    return 1.0 - 0.05 ** (1.0 / repetitions)


def _calibration_payload(
    *, stage: str = "development", repetitions: int = 119
) -> dict[str, object]:
    upper = _zero_failure_upper(repetitions)
    return {
        "calibration_id": "calibration",
        "calibration_stage": stage,
        "development_calibration_sha256": "9" * 64 if stage == "locked_audit" else None,
        "repeated_per_null": repetitions,
        "false_target_main_count": 0,
        "false_interaction_count": 0,
        "false_joint_interaction_count": 0,
        "false_target_main_rate_upper_bound": upper,
        "false_interaction_rate_upper_bound": upper,
        "false_joint_interaction_rate_upper_bound": upper,
        "target_minimum_improvement": 0.01,
        "interaction_minimum_improvement": 0.01,
        "checkpoint_updates": (1, 2),
        "results_artifact": _artifact(),
        "calibration_protocol_hash": "1" * 64,
    }


def _scores() -> tuple[StateCalibrationCheckpointScore, ...]:
    def row(update: int) -> StateCalibrationCheckpointScore:
        return StateCalibrationCheckpointScore(
            update=update,
            global_null_score=1.0,
            shrunk_target_only_score=1.0,
            empirical_bayes_target_score=1.0,
            target_terminal_score=1.0,
            target_delta_score=1.0,
            linear_source_plus_target_score=1.0,
            best_target_only_score=1.0,
            best_target_only_baseline="shrunk_target_only",
            best_noninteraction_score=1.0,
            best_noninteraction_baseline="shrunk_target_only",
            interaction_score=1.0,
        )

    return tuple(row(update) for update in (0, 1, 2))


def _row(index: int, family: str, local_index: int) -> StateSelectionCalibrationRow:
    return StateSelectionCalibrationRow(
        replicate_index=index,
        null_family=family,
        outer_fold_id="outer-fold-0",
        inner_split_id="inner-split-0",
        permutation_seed=10_000 + local_index,
        optimizer_seed=20_000 + local_index,
        initialization_seed=30_000 + local_index,
        permutation_sha256=f"{local_index + 1:064x}",
        fit_series_sha256="1" * 64,
        validation_series_sha256="2" * 64,
        checkpoint_scores=_scores(),
        selected_update=0,
        selected_family="global_terminal_null",
        maximum_target_main_gain=0.0,
        maximum_interaction_gain=0.0,
        false_target_main_selected=False,
        false_interaction_selected=False,
        false_joint_interaction_selected=False,
    )


def _results_payload(*, repetitions: int = 119) -> dict[str, object]:
    families = ("global_target_main", "conditional_interaction", "joint_nested")
    rows: list[StateSelectionCalibrationRow] = []
    for family_index, family in enumerate(families):
        for local_index in range(repetitions):
            rows.append(_row(len(rows), family, family_index * repetitions + local_index))
    return {
        "calibration_id": "calibration",
        "calibration_stage": "development",
        "repeated_per_null": repetitions,
        "pooled_estimand": "pooled_known_target_heldout_guide",
        "outer_fold_id": "outer-fold-0",
        "inner_split_id": "inner-split-0",
        "pooled_outer_fold_ids": ("outer-fold-0", "outer-fold-1"),
        "pooled_inner_split_ids": ("inner-split-0", "inner-split-1"),
        "pooled_optimization_seeds": (0, 1, 2),
        "state_split_seed": 7,
        "implementation_tree_hash": "0" * 64,
        "calibration_code_hash": "1" * 64,
        "environment_lock_hash": "2" * 64,
        "optimizer_fingerprint": "3" * 64,
        "device_type": "cpu",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "representation_id": "prepared",
        "split_manifest_hash": "4" * 64,
        "compiled_problem_hash": "5" * 64,
        "interaction_rank": 2,
        "interaction_scale": 0.25,
        "learning_rate": 0.01,
        "state_batch_size": 8,
        "state_full_batch": True,
        "noninteraction_linear_ridge": 1.0,
        "source_target_main_penalty": 1.0,
        "source_target_interaction_penalty": 1.0,
        "target_minimum_improvement": 0.01,
        "interaction_minimum_improvement": 0.01,
        "checkpoint_updates": (1, 2),
        "guide_per_target_distribution_hash": "6" * 64,
        "support_distribution_hash": "7" * 64,
        "rows": tuple(rows),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_updates", (), "nonempty"),
        ("checkpoint_updates", (2, 1), "increasing"),
        ("false_interaction_rate_upper_bound", 0.01, "inconsistent"),
    ],
)
def test_calibration_summary_fails_closed(field: str, value: object, message: str) -> None:
    payload = _calibration_payload()
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        StateSelectionCalibration.model_validate(payload)


def test_locked_calibration_requires_199_fits_per_null() -> None:
    with pytest.raises(ValidationError, match="at least 199"):
        StateSelectionCalibration.model_validate(
            _calibration_payload(stage="locked_audit", repetitions=119)
        )


def test_calibration_row_false_selection_label_is_exact() -> None:
    payload = _row(0, "conditional_interaction", 0).model_dump(mode="python")
    payload.update(
        {
            "selected_family": "target_plus_source_target_interaction",
            "false_interaction_selected": False,
        }
    )
    with pytest.raises(ValidationError, match="disagree"):
        StateSelectionCalibrationRow.model_validate(payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint_updates": (2, 1)}, "increasing"),
        ({"rows": ()}, "repeated_per_null"),
        (
            {
                "rows": (
                    _row(1, "global_target_main", 0),
                    *tuple(_results_payload()["rows"])[1:],
                )
            },
            "contiguous",
        ),
    ],
)
def test_calibration_results_fails_closed(change: dict[str, object], message: str) -> None:
    payload = _results_payload()
    payload.update(change)
    with pytest.raises(ValidationError, match=message):
        StateSelectionCalibrationResults.model_validate(payload)


def test_conditional_null_preserves_each_target_source_mean() -> None:
    arrays = {
        "source_z": np.asarray([[0.0], [2.0], [10.0], [14.0], [50.0]], dtype=np.float32),
        "terminal_z": np.arange(5, dtype=np.float32)[:, None],
        "target_index": np.asarray([1, 1, 2, 2, 0], dtype=np.int64),
        "is_control": np.asarray([0, 0, 0, 0, 1], dtype=np.uint8),
    }
    result, mapping = _conditional_interaction_null(arrays, seed=9)
    assert mapping["null"] == "conditional_interaction"
    for target in (1, 2):
        local = arrays["target_index"] == target
        np.testing.assert_allclose(
            result["source_z"][local].mean(0), arrays["source_z"][local].mean(0)
        )


def test_target_main_null_permutes_complete_equal_multiplicity_blocks() -> None:
    arrays = {
        "source_z": np.zeros((5, 1), dtype=np.float32),
        "terminal_z": np.asarray([[1.0], [2.0], [10.0], [20.0], [99.0]], dtype=np.float32),
        "target_index": np.asarray([1, 1, 2, 2, 0], dtype=np.int64),
        "is_control": np.asarray([0, 0, 0, 0, 1], dtype=np.uint8),
    }
    result, mapping = _global_target_main_null(arrays, seed=5)
    assert mapping["null"] == "global_target_main"
    np.testing.assert_allclose(result["terminal_z"][:2], arrays["terminal_z"][2:4])
    np.testing.assert_allclose(result["terminal_z"][2:4], arrays["terminal_z"][:2])
    np.testing.assert_allclose(result["terminal_z"][4], arrays["terminal_z"][4])


def test_target_main_null_removes_unpermutable_singleton_stratum_mean() -> None:
    arrays = {
        "source_z": np.zeros((5, 1), dtype=np.float32),
        "terminal_z": np.asarray([[1.0], [3.0], [10.0], [20.0], [99.0]], dtype=np.float32),
        "target_index": np.asarray([1, 1, 2, 2, 3], dtype=np.int64),
        "is_control": np.zeros(5, dtype=np.uint8),
    }
    result, mapping = _global_target_main_null(arrays, seed=5)
    pooled_target_mean = np.mean([2.0, 15.0, 99.0])
    assert result["terminal_z"][4, 0] == pytest.approx(pooled_target_mean)
    assert any(row.get("fallback") == "pooled_centered_residual" for row in mapping["mapping"])


def test_calibration_runner_rejects_insufficient_repeats_and_existing_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least 119"):
        run_state_selection_calibration(
            tmp_path / "missing.yaml",
            tmp_path / "out",
            repeats_per_null=118,
            permutation_seed_start=1,
            optimizer_seed_start=2,
            initialization_seed_start=3,
        )
    with pytest.raises(ValueError, match="at least 199"):
        run_state_selection_calibration(
            tmp_path / "missing.yaml",
            tmp_path / "out",
            repeats_per_null=198,
            permutation_seed_start=1,
            optimizer_seed_start=2,
            initialization_seed_start=3,
            calibration_stage="locked_audit",
        )
    with pytest.raises(ValueError, match="development receipt"):
        run_state_selection_calibration(
            tmp_path / "missing.yaml",
            tmp_path / "out",
            repeats_per_null=199,
            permutation_seed_start=1,
            optimizer_seed_start=2,
            initialization_seed_start=3,
            calibration_stage="locked_audit",
        )
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run_state_selection_calibration(
            tmp_path / "missing.yaml",
            tmp_path / "out",
            repeats_per_null=119,
            permutation_seed_start=1,
            optimizer_seed_start=2,
            initialization_seed_start=3,
        )
