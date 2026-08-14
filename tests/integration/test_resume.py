from __future__ import annotations

from pathlib import Path

import torch
import yaml

from credo_count_sde_v4.compile import compile_problem
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
