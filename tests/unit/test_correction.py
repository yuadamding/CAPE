from __future__ import annotations

import json
from pathlib import Path

import h5py
import yaml

from credo_count_sde_v4.prepare import prepare_representation
from credo_count_sde_v4.synthetic import create_synthetic_project


def test_matched_control_view_persists_estimability_audit(tmp_path: Path) -> None:
    config_path = create_synthetic_project(tmp_path / "corrected", updates=2)
    with h5py.File(config_path.parent / "work/input/counts.h5", "r") as handle:
        rows = handle["row_ids"][:].tolist()
    fit_rows = set(
        json.loads((config_path.parent / "work/input/information-set.json").read_text())["fit_rows"]
    )
    # Batch and protected labels form a crossed, balanced design within the
    # source-only fit rows. Endpoint rows are present in metadata but never fit.
    metadata = {
        "row_ids": rows,
        "batch_ids": ["A" if index % 2 == 0 else "B" for index in range(192)],
        "protected_ids": ["P" if (index // 2) % 2 == 0 else "Q" for index in range(192)],
        "is_reference": [row_id in fit_rows for row_id in rows],
        "validation": {
            "reconstruction_delta": -0.01,
            "state_prediction_delta": -0.01,
            "technical_separability_delta": -0.10,
            "protected_program_correlation": 0.99,
        },
    }
    metadata_path = config_path.parent / "correction.json"
    metadata_path.write_text(json.dumps(metadata))
    config = yaml.safe_load(config_path.read_text())
    config["input_view"] = "matched_control_offset_v1"
    config["correction_metadata"] = "correction.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    prepared = prepare_representation(config_path)
    audit = json.loads((prepared / "input_view.json").read_text())
    assert audit["full_rank"]
    assert audit["minimum_support"]
    assert audit["overlap_connected"]
    assert audit["selected"] == "matched_control_offset_v1"


def test_correction_fails_closed_to_identity_when_support_is_too_small(tmp_path: Path) -> None:
    config_path = create_synthetic_project(tmp_path / "fallback", updates=2)
    with h5py.File(config_path.parent / "work/input/counts.h5", "r") as handle:
        rows = handle["row_ids"][:].tolist()
    fit_rows = set(
        json.loads((config_path.parent / "work/input/information-set.json").read_text())["fit_rows"]
    )
    metadata = {
        "row_ids": rows,
        "batch_ids": ["rare" if index == 0 else "common" for index in range(192)],
        "protected_ids": ["P" if index % 2 else "Q" for index in range(192)],
        "is_reference": [row_id in fit_rows for row_id in rows],
        "validation": {
            "reconstruction_delta": -0.01,
            "state_prediction_delta": -0.01,
            "technical_separability_delta": -0.10,
            "protected_program_correlation": 0.99,
        },
    }
    (config_path.parent / "correction.json").write_text(json.dumps(metadata))
    config = yaml.safe_load(config_path.read_text())
    config["input_view"] = "matched_control_offset_v1"
    config["correction_metadata"] = "correction.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    prepared = prepare_representation(config_path)
    audit = json.loads((prepared / "input_view.json").read_text())
    assert audit["selected"] == "identity_library_normalized"
    assert audit["reason"] == "fail_closed_identity"
