from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from credo_count_sde_v4.compile import compile_problem
from credo_count_sde_v4.contracts import RunIntent
from credo_count_sde_v4.inference import finalize_inference
from credo_count_sde_v4.persistence import load_tensor_file
from credo_count_sde_v4.prepare import prepare_representation
from credo_count_sde_v4.synthetic import create_synthetic_project
from credo_count_sde_v4.training import resume_training, train_model


def _latest_model(config: Path) -> dict[str, torch.Tensor]:
    workspace = config.parent / "work"
    latest = __import__("json").loads(
        (workspace / "training" / "checkpoints" / "latest.json").read_text()
    )
    return load_tensor_file(workspace / "training" / latest["relative_uri"] / "model.safetensors")


def test_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path) -> None:
    full = create_synthetic_project(tmp_path / "full", updates=8)
    split = create_synthetic_project(tmp_path / "split", updates=8)
    for config in (full, split):
        prepare_representation(config)
        compile_problem(config)
    train_model(full, device="cpu")
    train_model(split, device="cpu", stop_after=4)
    resume_training(split, device="cpu")
    full_state, split_state = _latest_model(full), _latest_model(split)
    assert full_state.keys() == split_state.keys()
    assert all(torch.equal(full_state[key], split_state[key]) for key in full_state)


def test_finalize_honors_preselected_checkpoint_generation(tmp_path: Path) -> None:
    config = create_synthetic_project(tmp_path / "selected", updates=6)
    payload = yaml.safe_load(config.read_text())
    payload["training"]["checkpoint_every"] = 2
    payload["training"]["selected_update"] = 4
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    prepare_representation(config)
    compile_problem(config)
    train_model(config, device="cpu")
    finalized = finalize_inference(config)
    manifest = __import__("json").loads((finalized / "inference.json").read_text())
    selected = __import__("json").loads(
        (
            config.parent / "work/training/checkpoints/generation-000000004/checkpoint.json"
        ).read_text()
    )
    assert manifest["selected_checkpoint_id"] == selected["checkpoint_id"]


def test_null_guarded_selection_survives_interrupted_training(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "null-guarded-resume", intent=RunIntent.COUNT_STATE, updates=4
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "source_target_interaction_scale": 0.25,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "checkpoint_every": 2,
            "state_checkpoint_updates": [1, 2, 4],
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_validation_max_per_target": 1,
            "source_target_interaction_penalty": 1.0,
            "state_validation_minimum_improvement": 1_000.0,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    prepare_representation(config)
    compile_problem(config)
    train_model(config, device="cpu", stop_after=2)
    resume_training(config, device="cpu")
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["candidate_updates"] == [0, 1, 2, 4]
    assert selection["null_selected"] is True
    assert selection["post_selection_refit"] is True
    assert selection["selected_checkpoint_relative_uri"].endswith(
        "refit/checkpoints/generation-000000000"
    )
