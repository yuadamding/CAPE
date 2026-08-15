from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from credo_count_sde_v4.contracts import (
    ArtifactRef,
    StateSelectionCalibration,
    StateSelectionCalibrationResults,
    StateSelectionCalibrationRow,
)
from credo_count_sde_v4.training.calibration import run_state_selection_calibration


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        schema_id="credo.state_selection_calibration_results",
        schema_version=1,
        sha256="0" * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri="results.json",
    )


def _calibration_payload() -> dict[str, object]:
    repetitions = 59
    return {
        "calibration_id": "calibration",
        "method": "conditional_source_permutation",
        "repeated_seeds": repetitions,
        "false_interaction_count": 0,
        "false_interaction_rate_upper_bound": 1.0 - 0.05 ** (1.0 / repetitions),
        "target_minimum_improvement": 0.01,
        "interaction_minimum_improvement": 0.01,
        "checkpoint_updates": (1, 2),
        "results_artifact": _artifact(),
        "calibration_protocol_hash": "1" * 64,
    }


def _row(index: int, seed: int) -> StateSelectionCalibrationRow:
    return StateSelectionCalibrationRow(
        replicate_index=index,
        seed=seed,
        selected_update=0,
        selected_family="global_terminal_null",
        global_null_score=1.0,
        shrunk_target_only_score=1.0,
        interaction_score=1.0,
        false_interaction_selected=False,
    )


def _results_payload() -> dict[str, object]:
    seeds = (10, 11)
    return {
        "calibration_id": "calibration",
        "method": "conditional_source_permutation",
        "implementation_tree_hash": "0" * 64,
        "calibration_code_hash": "1" * 64,
        "representation_id": "prepared",
        "split_manifest_hash": "2" * 64,
        "compiled_problem_hash": "3" * 64,
        "interaction_rank": 2,
        "interaction_scale": 0.25,
        "learning_rate": 0.01,
        "state_batch_size": 8,
        "source_target_main_penalty": 1.0,
        "source_target_interaction_penalty": 1.0,
        "target_minimum_improvement": 0.01,
        "interaction_minimum_improvement": 0.01,
        "checkpoint_updates": (1, 2),
        "guide_per_target_distribution_hash": "4" * 64,
        "support_distribution_hash": "5" * 64,
        "seeds": seeds,
        "rows": tuple(_row(index, seed) for index, seed in enumerate(seeds)),
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


def test_calibration_row_false_selection_label_is_exact() -> None:
    payload = _row(0, 10).model_dump(mode="python")
    payload["false_interaction_selected"] = True
    with pytest.raises(ValidationError, match="disagrees"):
        StateSelectionCalibrationRow.model_validate(payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint_updates": (2, 1)}, "increasing"),
        ({"seeds": (10, 10)}, "unique"),
        ({"rows": (_row(0, 10),)}, "exactly one"),
        ({"rows": (_row(1, 10), _row(0, 11))}, "contiguous"),
        ({"rows": (_row(0, 11), _row(1, 10))}, "differ"),
    ],
)
def test_calibration_results_fails_closed(change: dict[str, object], message: str) -> None:
    payload = _results_payload()
    payload.update(change)
    with pytest.raises(ValidationError, match=message):
        StateSelectionCalibrationResults.model_validate(payload)


def test_calibration_runner_rejects_insufficient_or_unimplemented_repeats(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least 59"):
        run_state_selection_calibration(
            tmp_path / "missing.yaml", tmp_path / "out", seeds=tuple(range(58))
        )
    with pytest.raises(NotImplementedError, match="Only conditional"):
        run_state_selection_calibration(
            tmp_path / "missing.yaml",
            tmp_path / "out",
            seeds=tuple(range(59)),
            method="synthetic_null_repeats",
        )
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run_state_selection_calibration(
            tmp_path / "missing.yaml", tmp_path / "out", seeds=tuple(range(59))
        )
