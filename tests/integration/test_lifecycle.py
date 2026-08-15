from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from credo_count_sde_v4 import api
from credo_count_sde_v4.compile import load_compiled_problem
from credo_count_sde_v4.contracts import LifecycleState, RunIntent
from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.inference import open_inference_run
from credo_count_sde_v4.persistence import LifecycleLedger, load_tensor_file
from credo_count_sde_v4.prepare.pipeline import load_config
from credo_count_sde_v4.store import CountStore
from credo_count_sde_v4.synthetic import create_synthetic_project
from credo_count_sde_v4.training import load_training_state
from credo_count_sde_v4.training.trainer import _post_selection_refit, train_model


@pytest.mark.parametrize("intent", list(RunIntent))
def test_complete_synthetic_lifecycle(tmp_path: Path, intent: RunIntent) -> None:
    config = create_synthetic_project(tmp_path / intent.value, intent=intent, updates=6)
    api.prepare(config)
    api.compile_run(config)
    api.train(config, device="cpu")
    api.finalize(config)
    api.evaluate(config)
    sealed = api.seal(config)
    assert (
        LifecycleLedger(sealed.parent / "ledger" / "events.jsonl").state() is LifecycleState.SEALED
    )
    result = api.verify(sealed, level="full")
    assert result["reload"]["finite_mass"]
    metrics = json.loads((sealed.parent / "evaluation" / "metrics.json").read_text())
    assert metrics["evaluable_series"] == metrics["total_series"] == 6


def test_prepare_rejects_reordered_feature_contract(tmp_path: Path) -> None:
    config = create_synthetic_project(tmp_path / "feature-order", updates=2)
    feature_path = config.parent / "work/input/features.json"
    features = json.loads(feature_path.read_text())
    feature_path.write_text(json.dumps(list(reversed(features))))
    with pytest.raises(ContractError, match="Ordered feature contract"):
        api.prepare(config)


def test_trained_gene_decoder_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = create_synthetic_project(tmp_path / "decoder", intent=RunIntent.COUNT_STATE, updates=3)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["gene_decoder_features"] = 12
    payload["model"]["gene_decoder_hidden_dim"] = 8
    payload["model"]["terminal_anchor_drift"] = True
    payload["model"]["source_carryover_alpha"] = 0.0
    payload["model"]["target_anchor_weight"] = 0.0
    payload["training"]["gene_decoder_batch_size"] = 64
    payload["training"]["gene_decoder_loss_weight"] = 0.05
    payload["training"]["selected_update"] = None
    payload["training"]["checkpoint_every"] = 1
    payload["training"]["gene_decoder_validation_fraction"] = 0.2
    payload["training"]["gene_decoder_validation_max_rows"] = 16
    payload["training"]["checkpoint_selection"] = "minimum_gene_decoder_validation"
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    calls = 0
    original_rows = CountStore.rows

    def tracked_rows(self: CountStore, row_ids: np.ndarray):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original_rows(self, row_ids)

    monkeypatch.setattr(CountStore, "rows", tracked_rows)
    api.train(config, device="cpu")
    assert calls == 1
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["policy"] == "minimum_gene_decoder_validation"
    assert selection["selected_update"] in {1, 2, 3}
    assert np.isfinite(selection["score"])
    _, arrays, model, _ = load_training_state(config, device="cpu")
    np.testing.assert_allclose(
        model.terminal_anchor.detach().numpy(), arrays["terminal_z"].mean(axis=0), atol=1e-7
    )
    api.finalize(config)
    run = open_inference_run(config.parent / "work/inference", device="cpu", verify="full")
    mean, _, _ = run.terminal(particles=2)
    composition = run.decode_composition(mean)
    assert composition.shape == (6, 12)
    np.testing.assert_allclose(composition.sum(axis=1), 1.0, atol=1e-6)


def test_state_validation_selects_a_scientific_checkpoint_and_excludes_capacity_probe(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(tmp_path / "state-selection", updates=4)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["state_dependent_drift"] = True
    payload["training"].update(
        {
            "checkpoint_every": 2,
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_validation_max_per_target": 1,
            "support_weight_power": 0.5,
            "target_drift_penalty": 0.25,
            "source_drift_penalty": 0.1,
            "checkpoint_selection": "minimum_state_validation",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    train_model(config, device="cpu", stop_after=1)
    api.resume(config, device="cpu")
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["policy"] == "minimum_state_validation"
    assert selection["candidate_updates"] == [2, 4]
    assert selection["selected_update"] in {2, 4}
    assert np.isfinite(selection["score"])


def test_joint_decoder_training_updates_null_nested_state_channel(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "joint-state-decoder", intent=RunIntent.COUNT_STATE, updates=4
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_conditioned_anchor": True,
            "source_anchor_residual_scale": 0.5,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
            "gene_decoder_features": 12,
            "gene_decoder_hidden_dim": 8,
        }
    )
    payload["training"].update(
        {
            "checkpoint_every": 2,
            "selected_update": None,
            "gene_decoder_batch_size": 64,
            "gene_decoder_loss_weight": 0.05,
            "gene_decoder_validation_fraction": 0.2,
            "train_state_with_gene_decoder": True,
            "state_validation_fraction": 0.34,
            "state_validation_max_per_target": 1,
            "source_drift_penalty": 0.01,
            "checkpoint_selection": "minimum_state_validation",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    api.train(config, device="cpu")
    _, _, model, _ = load_training_state(config, device="cpu")
    assert model.source_anchor_output is not None
    assert torch.count_nonzero(model.source_anchor_output.weight).item() > 0
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["policy"] == "minimum_state_validation"
    assert selection["candidate_updates"] == [2, 4]


def test_null_guarded_interaction_persists_zero_and_refits_selected_null(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path / "null-guarded-interaction", intent=RunIntent.COUNT_STATE, updates=4
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
    api.prepare(config)
    api.compile_run(config)
    api.train(config, device="cpu")
    training = config.parent / "work/training"
    selection = json.loads((training / "selection.json").read_text())
    assert selection["candidate_updates"] == [0, 1, 2, 4]
    assert selection["inner_selected_checkpoint_id"] != selection["selected_checkpoint_id"]
    assert selection["selected_update"] == 0
    assert selection["null_selected"] is True
    assert selection["post_selection_refit"] is True
    assert selection["selected_checkpoint_relative_uri"].endswith(
        "refit/checkpoints/generation-000000000"
    )
    _, arrays, model, checkpoint = load_training_state(config, device="cpu")
    assert checkpoint.checkpoint_id == selection["selected_checkpoint_id"]
    assert checkpoint.update == 0
    np.testing.assert_allclose(
        model.terminal_anchor.detach().numpy(), arrays["terminal_z"].mean(axis=0), atol=1e-7
    )
    assert model.source_target_output is not None
    assert torch.count_nonzero(model.source_target_output.weight).item() == 0
    api.finalize(config)
    inference = json.loads((config.parent / "work/inference/inference.json").read_text())
    assert inference["selected_checkpoint_id"] == selection["selected_checkpoint_id"]


def test_source_target_pilot_requires_a_positive_selection_margin(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "zero-selection-margin", intent=RunIntent.COUNT_STATE, updates=2
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "source_target_interaction_penalty": 1.0,
            "state_validation_minimum_improvement": 0.0,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="positive selection margin"):
        api.prepare(config)


def test_target_only_candidate_is_materialized_by_post_selection_refit(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "target-only-refit", intent=RunIntent.COUNT_STATE, updates=2
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "source_target_interaction_penalty": 1.0,
            "state_validation_minimum_improvement": 0.01,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    workspace = config.parent / "work"
    contract, arrays = load_compiled_problem(workspace)
    training = workspace / "training"
    training.mkdir()
    selection = {
        "schema_version": 1,
        "compiled_run_id": contract.compiled_run_id,
        "selected_update": 0,
        "selected_checkpoint_id": "0" * 64,
        "selected_family": "sister_guide_target_terminal",
    }
    _post_selection_refit(
        workspace,
        training,
        contract,
        arrays,
        load_config(config),
        torch.device("cpu"),
        selection,
    )
    state = load_tensor_file(training / "refit/checkpoints/generation-000000000/model.safetensors")
    assert torch.equal(state["source_target_main_weight"], torch.tensor(1.0))
    assert torch.count_nonzero(state["source_target_main_offset"]).item() > 0
    assert torch.count_nonzero(state["source_target_output.weight"]).item() == 0
