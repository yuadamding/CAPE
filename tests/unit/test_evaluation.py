from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy import sparse

from credo_count_sde_v4.canonical import (
    canonical_json_bytes,
    contract_id,
    sha256_bytes,
    sha256_file,
)
from credo_count_sde_v4.contracts import (
    BaselineInformationSet,
    BaselineRegistry,
    FeatureKey,
    ModelConfig,
    RunIntent,
    SelectionManifest,
    SemanticStudySnapshot,
    SeriesRecord,
)
from credo_count_sde_v4.evaluation.evaluator import (
    _evaluate_bound_outer,
    _hash_rows,
    _independent_shrunk_target_prediction,
    _interaction_advancement_pass,
    _observed_gene_compositions,
    _target_balanced_bootstrap_differences,
)
from credo_count_sde_v4.model import CountSDEModel
from credo_count_sde_v4.store import CountStore, build_count_store


def test_target_bootstrap_preserves_equal_target_weighting() -> None:
    terminal = np.zeros((4, 1), dtype=np.float32)
    model = np.asarray([[1.0], [1.0], [1.0], [3.0]], dtype=np.float32)
    baseline = np.asarray([[2.0], [2.0], [2.0], [2.0]], dtype=np.float32)
    targets = np.asarray([0, 0, 0, 1], dtype=np.int64)
    values = _target_balanced_bootstrap_differences(
        model, baseline, terminal, targets, seed=7, draws=5_000
    )
    middle = np.sqrt((1.0 + 9.0) / 2.0) - 2.0
    expected = 0.25 * (-1.0) + 0.5 * middle + 0.25 * 1.0
    assert abs(float(values.mean()) - expected) < 0.04


def test_interaction_advancement_requires_interaction_family_and_effect_floor() -> None:
    common = {
        "interaction_bootstrap_upper": -0.02,
        "overall_bootstrap_upper": -0.03,
        "target_main_bootstrap_upper": -0.015,
        "required_interaction_improvement": 0.01,
        "required_overall_improvement": 0.02,
        "required_target_main_improvement": 0.01,
        "interaction_displacement_rms": 0.2,
        "minimum_interaction_displacement_rms": 0.1,
        "target_win_fraction": 0.75,
        "minimum_target_win_fraction": 0.6,
        "maximum_leave_one_target_out_delta": -0.005,
        "top_target_absolute_contribution_fraction": 0.15,
        "maximum_single_target_contribution_fraction": 0.25,
    }
    assert _interaction_advancement_pass(
        selected_family="target_plus_source_target_interaction", **common
    )
    assert not _interaction_advancement_pass(
        selected_family="shrunk_sister_guide_target_terminal", **common
    )
    assert not _interaction_advancement_pass(
        selected_family="target_plus_source_target_interaction",
        **{**common, "interaction_displacement_rms": 0.05},
    )
    assert not _interaction_advancement_pass(
        selected_family="target_plus_source_target_interaction",
        **{**common, "overall_bootstrap_upper": -0.01},
    )


def test_shrunk_target_baseline_is_materialized_independently() -> None:
    train_terminal = np.asarray([[0.0], [0.0], [1.0], [1.2], [2.0], [2.2]], dtype=np.float32)
    train_target = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    train_control = np.asarray([True, True, False, False, False, False])
    prediction, alpha = _independent_shrunk_target_prediction(
        train_terminal=train_terminal,
        train_target=train_target,
        train_control=train_control,
        evaluation_target=np.asarray([0, 1, 2], dtype=np.int64),
        evaluation_control=np.asarray([True, False, False]),
        maximum_weight=1.0,
        scalar_ridge=1.0,
    )
    global_terminal = float(np.mean([train_terminal[2:4].mean(), train_terminal[4:6].mean()]))
    assert 0.0 < alpha < 1.0
    assert prediction[0, 0] == global_terminal
    assert prediction[1, 0] != global_terminal
    assert prediction[2, 0] != global_terminal


def test_observed_gene_compositions_aggregate_terminal_cells(tmp_path) -> None:
    features = tuple(
        FeatureKey(namespace="test", namespace_version="v1", feature_id=str(index))
        for index in range(2)
    )
    path = tmp_path / "counts.h5"
    manifest = build_count_store(
        path,
        sparse.csr_matrix(np.asarray([[1, 0], [0, 3], [1, 1]], dtype=np.int32)),
        row_ids=np.asarray([10, 11, 12], dtype=np.int64),
        features=features,
    )
    records = (
        SeriesRecord(
            series_id="a",
            target_index=0,
            pool_index=0,
            is_control=True,
            source_rows=(10,),
            terminal_rows=(10, 11),
            source_count=1,
            terminal_count=2,
            duration=1.0,
        ),
        SeriesRecord(
            series_id="b",
            target_index=1,
            pool_index=0,
            is_control=False,
            source_rows=(12,),
            terminal_rows=(12,),
            source_count=1,
            terminal_count=1,
            duration=1.0,
        ),
    )
    result = _observed_gene_compositions(records, CountStore(path, manifest))
    np.testing.assert_allclose(result, [[0.25, 0.75], [0.5, 0.5]])


def test_bound_pooled_outer_evaluation_exercises_nested_gate(tmp_path) -> None:
    workspace = tmp_path / "work"
    input_root = workspace / "input"
    compiled = workspace / "compiled"
    prepared = workspace / "prepared"
    inference = workspace / "inference"
    for path in (input_root, compiled, prepared, inference):
        path.mkdir(parents=True)

    train_records: list[SeriesRecord] = []
    train_source_rows: list[int] = []
    train_terminal_rows: list[int] = []
    cursor = 1_000
    for index, target in enumerate((0, 0, 1, 1, 1, 2, 2, 2)):
        source_rows = (cursor, cursor + 1)
        terminal_rows = (cursor + 2, cursor + 3)
        cursor += 4
        train_source_rows.extend(source_rows)
        train_terminal_rows.extend(terminal_rows)
        train_records.append(
            SeriesRecord(
                series_id=f"train-{index}",
                target_index=target,
                pool_index=0,
                is_control=target == 0,
                source_rows=source_rows,
                terminal_rows=terminal_rows,
                source_count=20,
                terminal_count=20,
                duration=1.0,
            )
        )
    snapshot_payload = {
        "study_id": "pending",
        "series": [row.model_dump(mode="json") for row in train_records],
        "observed_edges": [["P4", "P60"]],
        "feature_index_hash": "a" * 64,
        "row_universe_hash": "b" * 64,
        "exposure_registry": {"schema_version": 1, "records": []},
    }
    snapshot_payload["study_id"] = contract_id(snapshot_payload, id_field="study_id")
    snapshot = SemanticStudySnapshot.model_validate(snapshot_payload)
    (compiled / "snapshot.json").write_bytes(
        canonical_json_bytes(snapshot.model_dump(mode="json")) + b"\n"
    )

    outer_records: list[SeriesRecord] = []
    latent_rows: list[int] = []
    latent_values: list[np.ndarray] = []
    cursor = 10_000
    for index, target in enumerate((0, 0, 1, 1, 2, 2)):
        source_rows = (cursor, cursor + 1)
        terminal_rows = (cursor + 2, cursor + 3)
        cursor += 4
        base = np.asarray([target, 0.2 * index], dtype=np.float32)
        latent_rows.extend((*source_rows, *terminal_rows))
        latent_values.extend([base - 0.05, base + 0.05, base + 0.15, base + 0.25])
        outer_records.append(
            SeriesRecord(
                series_id=f"outer-{index}",
                target_index=target,
                pool_index=0,
                is_control=target == 0,
                source_rows=source_rows,
                terminal_rows=terminal_rows,
                source_count=20,
                terminal_count=20,
                duration=1.0,
            )
        )
    with h5py.File(prepared / "latents.h5", "w") as handle:
        handle.create_dataset("row_ids", data=np.asarray(latent_rows, dtype=np.int64))
        handle.create_dataset("z", data=np.asarray(latent_values, dtype=np.float32))
    outer_payload = {
        "schema_version": 1,
        "series": [row.model_dump(mode="json") for row in outer_records],
    }
    (input_root / "outer-evaluation.json").write_bytes(canonical_json_bytes(outer_payload) + b"\n")
    plan = {
        "schema_version": 1,
        "primary_population": "interaction_eligible_targets",
        "primary_baseline": "best_preregistered_noninteraction",
        "primary_metric": "target_balanced_rmse",
        "noninteraction_baselines": [
            "shrunk_target_only",
            "empirical_bayes_target",
            "target_terminal",
            "target_delta",
            "linear_source_plus_target",
        ],
        "linear_source_target_ridge": 1.0,
        "bootstrap_seed": 41,
        "bootstrap_draws": 100,
        "numerical_tolerance": 0.0,
        "scientific_minimum_improvement": 0.001,
        "interaction_scientific_minimum_improvement": 0.001,
        "overall_scientific_minimum_improvement": 0.001,
        "target_main_scientific_minimum_improvement": 0.001,
        "minimum_interaction_displacement_rms": 1e-6,
        "minimum_target_win_fraction": 0.5,
        "maximum_single_target_contribution_fraction": 1.0,
        "historically_exposed": True,
        "endpoint_used_for_eligibility": False,
        "eligibility_rule": "source_only",
    }
    (input_root / "evaluation-plan.json").write_bytes(canonical_json_bytes(plan) + b"\n")

    evaluator_hash = sha256_file(
        Path(__import__("credo_count_sde_v4.evaluation.evaluator", fromlist=["__file__"]).__file__)
    )
    train_order = tuple(train_source_rows + train_terminal_rows)
    outer_source = tuple(row for record in outer_records for row in record.source_rows)
    outer_terminal = tuple(row for record in outer_records for row in record.terminal_rows)
    baseline_ids = (
        "shrunk_target_only",
        "empirical_bayes_target",
        "target_terminal",
        "target_delta",
        "linear_source_plus_target",
    )
    registry = BaselineRegistry(
        registry_id="pooled-test-baselines",
        baselines=tuple(
            BaselineInformationSet(
                baseline_id=name,
                code_hash=evaluator_hash,
                allowed_training_rows_hash=_hash_rows(train_order),
                allowed_test_source_rows_hash=_hash_rows(outer_source),
                forbidden_endpoint_rows_hash=_hash_rows(outer_terminal),
                split_hash="c" * 64,
                representation_id="mock-representation",
                aggregation_hash="d" * 64,
                seed_plan_hash="e" * 64,
            )
            for name in baseline_ids
        ),
    )
    (input_root / "baseline-registry.json").write_bytes(
        canonical_json_bytes(registry.model_dump(mode="json")) + b"\n"
    )
    source_manifest = {
        "schema_version": 1,
        "outer_evaluation_sha256": sha256_file(input_root / "outer-evaluation.json"),
        "evaluation_plan_sha256": sha256_file(input_root / "evaluation-plan.json"),
    }
    (input_root / "source-manifest.json").write_bytes(canonical_json_bytes(source_manifest) + b"\n")

    selection_payload = {
        "schema_version": 1,
        "selection_id": "pending",
        "compiled_run_id": "f" * 64,
        "policy": "minimum_state_validation_null_guarded",
        "metric": "state_validation_full_interaction_rmse",
        "score": 0.2,
        "selected_update": 1,
        "selected_checkpoint_id": "1" * 64,
        "selected_checkpoint_relative_uri": "training/checkpoints/generation-000000001",
        "candidate_updates": (0, 1),
        "selected_family": "target_plus_source_target_interaction",
        "global_null_score": 0.4,
        "shrunk_target_only_score": 0.3,
        "best_target_only_score": 0.3,
        "best_target_only_baseline": "shrunk_target_only",
        "best_noninteraction_score": 0.3,
        "best_noninteraction_baseline": "shrunk_target_only",
        "interaction_score": 0.2,
        "interaction_incremental_gain": 0.1,
        "target_incremental_gain": 0.1,
        "target_minimum_required_improvement": 0.01,
        "interaction_minimum_required_improvement": 0.01,
        "selection_calibration_hash": "2" * 64,
    }
    selection_payload = SelectionManifest.model_construct(**selection_payload).model_dump(
        mode="json"
    )
    selection_payload["selection_id"] = contract_id(selection_payload, id_field="selection_id")
    selection = SelectionManifest.model_validate(selection_payload)
    (inference / "selection-manifest.json").write_bytes(
        canonical_json_bytes(selection.model_dump(mode="json")) + b"\n"
    )

    model_config = ModelConfig(
        state_dim=2,
        target_count=3,
        pool_count=1,
        hidden_dim=4,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        source_target_interaction_rank=1,
        source_target_main_max_weight=1.0,
    )
    torch.manual_seed(3)
    model = CountSDEModel(model_config, RunIntent.COUNT_STATE)
    train_source_array = np.asarray(
        [[row.target_index, 0.1 * index] for index, row in enumerate(train_records)],
        dtype=np.float32,
    )
    train_terminal_array = train_source_array + np.asarray([0.2, 0.1], dtype=np.float32)
    with torch.no_grad():
        model.terminal_anchor.copy_(torch.tensor([1.5, 0.5]))
        model.source_target_main_weight.copy_(torch.tensor(0.2))
        model.source_target_main_offset[1].copy_(torch.tensor([0.2, 0.1]))
        model.source_target_main_offset[2].copy_(torch.tensor([-0.1, 0.2]))
    model.refresh_source_target_interaction_mean(
        torch.from_numpy(train_source_array),
        torch.tensor([row.target_index for row in train_records]),
        torch.tensor([row.is_control for row in train_records]),
    )
    logical_baseline_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "registry": registry.model_dump(mode="json"),
                "representation_id": "mock-representation",
            }
        )
    )
    run = SimpleNamespace(
        contract=SimpleNamespace(
            source_manifest_hash=sha256_file(input_root / "source-manifest.json"),
            baseline_registry_hash=logical_baseline_hash,
            representation_id="mock-representation",
            state_selection_calibration_stage="development",
        ),
        config=SimpleNamespace(
            model=model_config,
            training=SimpleNamespace(
                noninteraction_linear_ridge=1.0,
                source_target_main_penalty=1.0,
                seed=17,
            ),
            evaluation=SimpleNamespace(particles=1, steps=2, seed=23),
            pooled_estimand="pooled_known_target_heldout_guide",
            outer_fold_id="fold0",
            inner_split_id="inner0",
            pooled_outer_fold_ids=("fold0", "fold1"),
            pooled_inner_split_ids=("inner0", "inner1"),
            pooled_optimization_seeds=(17, 18, 19),
        ),
        arrays={
            "source_z": train_source_array,
            "terminal_z": train_terminal_array,
            "target_index": np.asarray([row.target_index for row in train_records]),
            "is_control": np.asarray([row.is_control for row in train_records]),
        },
        model=model,
        device=torch.device("cpu"),
        manifest=SimpleNamespace(
            run_id="mock-run", selected_family="target_plus_source_target_interaction"
        ),
        capabilities=SimpleNamespace(decode_gene_composition=False),
    )
    metrics, bound_plan, audit, frame = _evaluate_bound_outer(
        workspace, run, workspace / "evaluation"
    )
    assert metrics["interaction_eligible_targets"]["series"] == 4
    assert metrics["interaction_outer_gate"]["best_noninteraction_baseline"] in baseline_ids
    assert bound_plan["outer_evaluation_sha256"] == source_manifest["outer_evaluation_sha256"]
    assert audit["pooled_estimand"] == "pooled_known_target_heldout_guide"
    assert audit["state_selection_calibration_stage"] == "development"
    assert frame.interaction_eligible.sum() == 4
