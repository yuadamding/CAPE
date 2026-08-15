from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import gammaln

from credo_count_sde_v4.contracts import (
    ModelConfig,
    PhysicalGrid,
    RunIntent,
    SeriesRecord,
    TopologySupport,
    TrainingConfig,
    TransportTopologyContract,
)
from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.evaluation.evaluator import _training_row_order
from credo_count_sde_v4.model import CountSDEModel, DynamicPoolBank
from credo_count_sde_v4.numerics import rollout
from credo_count_sde_v4.objectives import count_probabilities, dirichlet_multinomial_log_prob
from credo_count_sde_v4.training.trainer import (
    _csr_selected_logit_sum,
    _fit_shrunk_target_main_weight,
    _initialize_state_channels,
    _state_split,
    _target_balanced_mse,
    _write_selection,
)


def test_baseline_training_rows_are_sources_then_endpoints() -> None:
    records = (
        SeriesRecord(
            series_id="a",
            target_index=0,
            pool_index=0,
            is_control=True,
            source_rows=(1, 2),
            terminal_rows=(5,),
            source_count=2,
            terminal_count=1,
            duration=1.0,
        ),
        SeriesRecord(
            series_id="b",
            target_index=1,
            pool_index=0,
            is_control=False,
            source_rows=(3,),
            terminal_rows=(6, 7),
            source_count=1,
            terminal_count=2,
            duration=1.0,
        ),
    )
    assert _training_row_order(records) == (1, 2, 3, 5, 6, 7)


def test_dm_matches_direct_reference_and_has_finite_gradients() -> None:
    counts = torch.tensor([3.0, 0.0, 7.0], dtype=torch.float64)
    logits = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64, requires_grad=True)
    log_concentration = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    actual = dirichlet_multinomial_log_prob(counts, logits, log_concentration)
    concentration = torch.nn.functional.softplus(log_concentration).item()
    probability = torch.softmax(logits.detach(), dim=-1).numpy()
    alpha = concentration * probability
    y = counts.numpy()
    total = y.sum()
    expected = (
        gammaln(total + 1)
        - gammaln(y + 1).sum()
        + gammaln(concentration)
        - gammaln(total + concentration)
        + (gammaln(y + alpha) - gammaln(alpha)).sum()
    )
    assert np.isclose(actual.item(), expected, atol=1e-12)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(log_concentration.grad)
    assert concentration > 0
    epsilon = 1e-5
    upper = dirichlet_multinomial_log_prob(
        counts, logits.detach(), torch.tensor(1.3 + epsilon, dtype=torch.float64)
    )
    lower = dirichlet_multinomial_log_prob(
        counts, logits.detach(), torch.tensor(1.3 - epsilon, dtype=torch.float64)
    )
    finite_difference = (upper - lower) / (2 * epsilon)
    assert torch.allclose(log_concentration.grad, finite_difference, atol=1e-7, rtol=1e-6)


def test_dm_large_total_and_truncated_denominator_negative_control() -> None:
    full = dirichlet_multinomial_log_prob(
        torch.tensor([100_000_000.0, 1.0, 0.0]),
        torch.tensor([2.0, 1.0, -1.0]),
        torch.tensor(1.0),
    )
    truncated = dirichlet_multinomial_log_prob(
        torch.tensor([100_000_000.0, 1.0]),
        torch.tensor([2.0, 1.0]),
        torch.tensor(1.0),
    )
    assert torch.isfinite(full)
    assert not torch.isclose(full, truncated)


def test_selection_and_fitness_gauges_and_positive_mass() -> None:
    config = ModelConfig(
        state_dim=3,
        target_count=2,
        pool_count=1,
        centered_selection=True,
        selection_inner_validation_pass=True,
        shared_diffusion=False,
    )
    model = CountSDEModel(config, RunIntent.COUNT_MEASURE)
    with torch.no_grad():
        model.target_selection[1] = torch.tensor([1.0, -0.5, 0.25])
        model.target_fitness[1] = 0.7
    target = torch.tensor([0, 1])
    pool = torch.tensor([0, 0])
    control = torch.tensor([True, False])
    z = torch.randn(2, 5, 3)
    weights = torch.softmax(torch.randn(2, 5), dim=1)
    selection = model.centered_selection(z, weights, target, control)
    assert torch.allclose((weights * selection).sum(dim=1), torch.zeros(2), atol=1e-6)
    exposure = torch.tensor([3.0, 7.0])
    fitness = model.relative_fitness(target, pool, control, exposure)
    normalized = exposure / exposure.sum()
    assert torch.allclose((normalized * fitness).sum(), torch.tensor(0.0), atol=1e-6)
    drift = model.drift(target, pool, control)
    assert torch.equal(drift[0], model.base_drift)
    _, rollout_weights, mass = rollout(
        model,
        torch.zeros(2, 3),
        torch.ones(2),
        target,
        pool,
        control,
        exposure,
        particles=8,
        steps=2,
        seed=1,
    )
    assert torch.all(mass > 0)
    assert torch.allclose(rollout_weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_count_probabilities_grid_and_topology_invariants() -> None:
    probabilities = count_probabilities(
        torch.tensor([0.0, 10.0, 2.0]),
        torch.tensor([0.0, 0.3, -0.1]),
        torch.ones(3),
    )
    assert torch.all(probabilities > 0)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    grid = PhysicalGrid.resolve(axis_unit="hour", duration=48.0, maximum_step=2.0)
    assert grid.steps == 24 and grid.step_size == 2.0
    with pytest.raises(ValueError, match="immigration"):
        TransportTopologyContract(
            topology_id="invalid",
            partitions=("T",),
            allowed_edges=(("T", "T"),),
            support=(
                TopologySupport(series_id="g", partition_id="T", source_count=0, terminal_count=2),
            ),
        )


def test_deterministic_configuration_collapses_particles_exactly() -> None:
    config = ModelConfig(state_dim=3, target_count=2, pool_count=1)
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    source = torch.randn(2, 3)
    target = torch.tensor([0, 1])
    pool = torch.zeros(2, dtype=torch.long)
    control = torch.tensor([True, False])
    exposure = torch.ones(2)
    one, one_weights, _ = rollout(
        model,
        source,
        torch.ones(2),
        target,
        pool,
        control,
        exposure,
        particles=1,
        steps=3,
        seed=1,
    )
    many, many_weights, _ = rollout(
        model,
        source,
        torch.ones(2),
        target,
        pool,
        control,
        exposure,
        particles=7,
        steps=3,
        seed=999,
    )
    assert torch.equal(many, one.expand(-1, 7, -1))
    assert torch.equal(one_weights, torch.ones_like(one_weights))
    assert torch.allclose(many_weights, torch.full_like(many_weights, 1.0 / 7.0))


def test_zero_carryover_terminal_anchor_contains_global_terminal_null() -> None:
    config = ModelConfig(
        state_dim=2,
        target_count=2,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        target_anchor_weight=0.0,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    with torch.no_grad():
        model.terminal_anchor.copy_(torch.tensor([2.0, -1.0]))
    source = torch.tensor([[100.0, 100.0], [-100.0, -100.0]])
    states, weights, _ = rollout(
        model,
        source,
        torch.ones(2),
        torch.tensor([0, 1]),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([True, False]),
        torch.ones(2),
        particles=3,
        steps=24,
        seed=1,
    )
    expected = torch.tensor([2.0, -1.0]).expand(2, 3, 2)
    assert torch.equal(states, expected)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_source_conditioned_anchor_is_null_nested_and_source_sensitive() -> None:
    config = ModelConfig(
        state_dim=2,
        target_count=1,
        hidden_dim=3,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        source_conditioned_anchor=True,
        source_anchor_residual_scale=0.5,
        trainable_terminal_anchor=False,
        trainable_target_anchor=False,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    source = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    target = torch.zeros(2, dtype=torch.long)
    control = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        model.terminal_anchor.copy_(torch.tensor([0.25, -0.5]))
    null = model.anchor(target, control, source_z=source)
    assert torch.equal(null, model.terminal_anchor.expand_as(null))
    assert model.source_anchor_hidden is not None
    assert model.source_anchor_output is not None
    with torch.no_grad():
        model.source_anchor_hidden.weight.zero_()
        model.source_anchor_hidden.bias.zero_()
        model.source_anchor_hidden.weight[0, 0] = 1.0
        model.source_anchor_output.weight.zero_()
        model.source_anchor_output.weight[0, 0] = 1.0
    shifted = model.anchor(target, control, source_z=source)
    assert shifted[0, 0] > null[0, 0]
    assert shifted[1, 0] < null[1, 0]
    assert torch.allclose(shifted[:, 1], null[:, 1])


def test_source_conditioned_anchor_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="require terminal_anchor_drift"):
        ModelConfig(state_dim=2, target_count=1, source_conditioned_anchor=True)
    with pytest.raises(ValueError, match="zero recurrent carryover"):
        ModelConfig(
            state_dim=2,
            target_count=1,
            terminal_anchor_drift=True,
            source_conditioned_anchor=True,
        )
    with pytest.raises(ValueError, match="requires an enabled gene decoder"):
        TrainingConfig(train_state_with_gene_decoder=True)


def test_source_target_interaction_is_null_nested_target_specific_and_ablatable() -> None:
    config = ModelConfig(
        state_dim=2,
        target_count=3,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        source_target_interaction_rank=2,
        source_target_interaction_scale=0.5,
        trainable_terminal_anchor=False,
        trainable_target_anchor=False,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    source = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([0, 1, 2])
    control = torch.tensor([True, False, False])
    with torch.no_grad():
        model.terminal_anchor.copy_(torch.tensor([0.25, -0.5]))
    null = model.anchor(target, control, source_z=source)
    assert torch.equal(null, model.terminal_anchor.expand_as(null))
    assert model.source_interaction_projection is not None
    assert model.target_interaction_embedding is not None
    assert model.source_target_output is not None
    with torch.no_grad():
        model.source_interaction_projection.weight.copy_(torch.eye(2))
        model.target_interaction_embedding.copy_(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        )
        model.source_target_output.weight.copy_(torch.eye(2))
    factual = model.anchor(target, control, source_z=source)
    reference = model.anchor(target, control, source_z=source, effect_mode="reference")
    assert torch.equal(factual[0], null[0])
    assert factual[1, 0] > null[1, 0]
    assert factual[2, 0] < null[2, 0]
    assert torch.equal(reference, null)
    swapped = model.anchor(torch.tensor([0, 2, 1]), control, source_z=source)
    assert not torch.equal(swapped[1:], factual[1:])
    with torch.no_grad():
        model.source_target_output.weight.zero_()
    assert torch.equal(model.anchor(target, control, source_z=source), null)
    assert model.source_target_main_weight is not None
    with torch.no_grad():
        model.source_target_main_offset[1].copy_(torch.tensor([0.2, -0.1]))
        model.source_target_main_weight.fill_(1.0)
    target_only = model.anchor(target, control, source_z=source)
    assert torch.equal(target_only[0], null[0])
    assert not torch.equal(target_only[1], null[1])
    assert torch.equal(
        model.anchor(target, control, source_z=source, effect_mode="reference"), null
    )


def test_source_target_interaction_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="zero-carryover terminal anchor"):
        ModelConfig(state_dim=2, target_count=2, source_target_interaction_rank=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelConfig(
            state_dim=2,
            target_count=2,
            terminal_anchor_drift=True,
            source_carryover_alpha=0.0,
            source_conditioned_anchor=True,
            source_target_interaction_rank=2,
        )
    with pytest.raises(ValueError, match="post-selection refitting"):
        TrainingConfig(
            state_validation_fraction=0.25,
            checkpoint_selection="minimum_state_validation_null_guarded",
        )


def test_source_target_interaction_can_learn_a_sign_controlled_signal() -> None:
    config = ModelConfig(
        state_dim=2,
        target_count=3,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        source_target_interaction_rank=2,
        source_target_interaction_scale=1.0,
        trainable_terminal_anchor=False,
        trainable_target_anchor=False,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    source = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    target = torch.tensor([1, 1, 2, 2])
    control = torch.zeros(4, dtype=torch.bool)
    expected = torch.tensor([[0.5, 0.0], [-0.5, 0.0], [-0.5, 0.0], [0.5, 0.0]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = torch.mean((model.anchor(target, control, source_z=source) - expected) ** 2)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.anchor(target, control, source_z=source)
        loss = torch.mean((prediction - expected) ** 2)
        loss.backward()
        optimizer.step()
    final = torch.mean((model.anchor(target, control, source_z=source) - expected) ** 2)
    interaction_off = torch.mean(
        (model.anchor(target, control, source_z=source, effect_mode="target_only") - expected) ** 2
    )
    assert final < initial * 0.05
    assert final < interaction_off * 0.05


def test_sister_guide_target_weight_is_leave_one_out_bounded_and_shrunk() -> None:
    model_config = ModelConfig(
        state_dim=2,
        target_count=3,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        source_target_interaction_rank=2,
        trainable_terminal_anchor=False,
        trainable_target_anchor=False,
    )
    problem = {
        "source_z": torch.zeros(8, 2),
        "terminal_z": torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.2, 0.0],
                [0.8, 0.0],
                [-1.0, 0.0],
                [-0.8, 0.0],
                [-1.2, 0.0],
            ]
        ),
        "target_index": torch.tensor([0, 0, 1, 1, 1, 2, 2, 2]),
        "pool_index": torch.zeros(8, dtype=torch.long),
        "is_control": torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.uint8),
        "duration": torch.ones(8),
        "grid_steps": torch.ones(8, dtype=torch.long),
        "source_counts": torch.full((8,), 100),
        "terminal_counts": torch.full((8,), 100),
    }

    def fitted(penalty: float) -> float:
        model = CountSDEModel(model_config, RunIntent.COUNT_STATE)
        runtime = SimpleNamespace(
            training=TrainingConfig(
                source_target_main_penalty=penalty,
                source_target_interaction_penalty=1.0,
            )
        )
        _initialize_state_channels(
            model,
            problem,
            torch.arange(8),
            runtime,  # type: ignore[arg-type]
        )
        return _fit_shrunk_target_main_weight(
            model,
            problem,
            torch.arange(8),
            runtime,  # type: ignore[arg-type]
            materialize=True,
        )

    weakly_regularized = fitted(1e-6)
    strongly_regularized = fitted(10.0)
    assert 0.0 <= strongly_regularized < weakly_regularized <= 1.0


def test_null_guarded_selection_requires_margin_over_best_baseline(tmp_path: Path) -> None:
    def run_case(root: Path, trained_score: float, target_score: float = 0.45) -> dict[str, object]:
        for update, score in ((0, 0.5), (5, trained_score)):
            generation = root / "checkpoints" / f"generation-{update:09d}"
            generation.mkdir(parents=True)
            diagnostics = {
                "state_validation_full_interaction_rmse": score,
                "state_validation_global_null_rmse": 0.5,
                "state_validation_shrunk_target_only_rmse": target_score,
                "state_validation_interaction_incremental_gain": target_score - score,
                "interaction_displacement_rms": abs(target_score - score),
                "target_main_displacement_rms": 0.1,
            }
            (generation / "training-state.json").write_text(
                json.dumps({"diagnostics": diagnostics})
            )
        training = SimpleNamespace(
            checkpoint_selection="minimum_state_validation_null_guarded",
            checkpoint_every=5,
            state_checkpoint_updates=(5,),
            max_updates=5,
            state_validation_target_minimum_improvement=0.01,
            state_validation_interaction_minimum_improvement=0.01,
            selected_update=None,
        )
        config = SimpleNamespace(training=training)
        checkpoints = [
            SimpleNamespace(update=0, checkpoint_id="null"),
            SimpleNamespace(update=5, checkpoint_id="trained"),
        ]
        return _write_selection(
            root,
            SimpleNamespace(
                compiled_run_id="compiled",
                state_selection_calibration_hash="0" * 64,
            ),
            config,  # type: ignore[arg-type]
            checkpoints,  # type: ignore[arg-type]
        )

    selected = run_case(tmp_path / "selected", 0.40)
    assert selected["selected_update"] == 5
    assert selected["selected_family"] == "target_plus_source_target_interaction"
    target = run_case(tmp_path / "target", 0.445)
    assert target["selected_update"] == 0
    assert target["selected_family"] == "shrunk_sister_guide_target_terminal"
    full_target = run_case(tmp_path / "full-target", 0.35, target_score=0.35)
    assert full_target["selected_family"] == "shrunk_sister_guide_target_terminal"
    rejected = run_case(tmp_path / "rejected", 0.49, target_score=0.495)
    assert rejected["selected_update"] == 0
    assert rejected["selected_family"] == "global_terminal_null"
    false_interaction_selections = 0
    for seed in range(20):
        # A shrunk-target truth may yield small numerical fluctuations, but a
        # calibrated 0.01 incremental margin must not relabel those as source
        # interactions.
        target_score = 0.44 + seed * 1e-5
        row = run_case(
            tmp_path / f"target-only-null-seed-{seed}",
            target_score - 0.005,
            target_score=target_score,
        )
        false_interaction_selections += int(
            row["selected_family"] == "target_plus_source_target_interaction"
        )
    assert false_interaction_selections == 0


def test_csr_decoder_reduction_matches_reference_and_has_finite_gradients() -> None:
    from scipy import sparse

    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        requires_grad=True,
    )
    counts = sparse.csr_matrix(np.asarray([[0.0, 2.0, 1.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]))
    observed = _csr_selected_logit_sum(logits, counts)
    expected = torch.tensor([7.0, 0.0, 57.0])
    assert torch.equal(observed, expected)
    observed.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_state_split_is_deterministic_and_target_stratified() -> None:
    arrays = {
        "terminal_z": np.zeros((6, 2), dtype=np.float32),
        "target_index": np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64),
        "series_ids": np.asarray(["c1", "c2", "a1", "a2", "a3", "singleton"]),
    }
    training = TrainingConfig(
        state_validation_fraction=0.34,
        state_validation_max_per_target=1,
        checkpoint_selection="minimum_state_validation",
    )
    config = SimpleNamespace(training=training)
    first = _state_split(arrays, config, torch.device("cpu"))  # type: ignore[arg-type]
    second = _state_split(arrays, config, torch.device("cpu"))  # type: ignore[arg-type]
    assert torch.equal(first.fit_indices, second.fit_indices)
    assert torch.equal(first.validation_indices, second.validation_indices)
    assert len(first.validation_indices) == 2
    assert 5 in first.fit_indices.tolist()
    held_targets = arrays["target_index"][first.validation_indices.numpy()]
    assert sorted(held_targets.tolist()) == [0, 1]


def test_target_balanced_support_weighting_preserves_equal_target_weight() -> None:
    error = torch.tensor([[1.0], [3.0], [2.0]])
    target = torch.tensor([0, 0, 1])
    unweighted = _target_balanced_mse(error, target)
    weighted = _target_balanced_mse(error, target, torch.tensor([9.0, 1.0, 1.0]))
    assert torch.allclose(unweighted, torch.tensor(((1.0 + 9.0) / 2.0 + 4.0) / 2.0))
    assert torch.allclose(weighted, torch.tensor((1.8 + 4.0) / 2.0))


def test_source_and_target_state_channels_have_positive_null_adversarial_ablations() -> None:
    config = ModelConfig(
        state_dim=2,
        target_count=2,
        hidden_dim=4,
        state_dependent_drift=True,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    source = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    target = torch.tensor([0, 1])
    pool = torch.zeros(2, dtype=torch.long)
    control = torch.tensor([True, False])
    with torch.no_grad():
        model.base_drift.zero_()
        model.target_drift.zero_()
        model.target_drift[1] = torch.tensor([0.25, -0.5])
        model.state_drift_output.weight.fill_(0.1)
    factual = model.drift(target, pool, control, z=source)
    reference = model.drift(target, pool, control, z=source, effect_mode="reference")
    assert torch.allclose(factual[0], reference[0])
    assert not torch.allclose(factual[1], reference[1])
    swapped = model.drift(target.flip(0), pool, control, z=source)
    assert not torch.allclose(factual, swapped)
    with torch.no_grad():
        model.target_drift.zero_()
        model.state_drift_output.weight.zero_()
    ablated = model.drift(target, pool, control, z=source)
    assert torch.equal(ablated, torch.zeros_like(ablated))


def test_adaptive_target_anchor_retains_coherent_and_rejects_adversarial_targets() -> None:
    config = ModelConfig(
        state_dim=2,
        target_count=3,
        terminal_anchor_drift=True,
        source_carryover_alpha=0.0,
        target_anchor_weight=0.0,
        adaptive_target_anchor=True,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    problem = {
        "source_z": torch.zeros(6, 2),
        "terminal_z": torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]
        ),
        "target_index": torch.tensor([0, 0, 1, 1, 2, 2]),
        "pool_index": torch.zeros(6, dtype=torch.long),
        "is_control": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8),
        "duration": torch.ones(6),
        "grid_steps": torch.ones(6, dtype=torch.long),
        "source_counts": torch.full((6,), 100),
        "terminal_counts": torch.full((6,), 100),
    }
    runtime = SimpleNamespace(training=TrainingConfig(support_weight_power=0.5))
    diagnostics = _initialize_state_channels(
        model,
        problem,
        torch.arange(6),
        runtime,  # type: ignore[arg-type]
    )
    assert model.target_anchor_gate[1] > 0
    assert model.target_anchor_gate[2] == 0
    assert diagnostics["adaptive_target_anchor_active_targets"] == 1.0


def test_trained_gene_decoder_has_finite_normalized_gradients() -> None:
    config = ModelConfig(
        state_dim=4,
        target_count=2,
        gene_decoder_features=7,
        gene_decoder_hidden_dim=8,
    )
    model = CountSDEModel(config, RunIntent.COUNT_STATE)
    state = torch.randn(5, 4)
    logits = model.decode_logits(state)
    probabilities = torch.softmax(logits, dim=-1)
    loss = -torch.log(probabilities[:, 0]).mean()
    loss.backward()
    assert logits.shape == (5, 7)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_gene_decoder_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="hidden gene decoder"):
        ModelConfig(state_dim=4, target_count=2, gene_decoder_hidden_dim=8)
    with pytest.raises(ValueError, match="enabled together"):
        TrainingConfig(gene_decoder_batch_size=4)
    with pytest.raises(ValueError, match="cannot train"):
        TrainingConfig(
            max_updates=1,
            selected_update=1,
            analytic_fit=True,
            gene_decoder_batch_size=4,
            gene_decoder_loss_weight=0.1,
        )
    with pytest.raises(ValueError, match="requires decoder training"):
        TrainingConfig(gene_decoder_validation_fraction=0.1)
    with pytest.raises(ValueError, match="requires a validation split"):
        TrainingConfig(
            gene_decoder_batch_size=4,
            gene_decoder_loss_weight=0.1,
            checkpoint_selection="minimum_gene_decoder_validation",
        )
    with pytest.raises(ValueError, match="resolves selected_update"):
        TrainingConfig(
            selected_update=100,
            gene_decoder_batch_size=4,
            gene_decoder_loss_weight=0.1,
            gene_decoder_validation_fraction=0.1,
            checkpoint_selection="minimum_gene_decoder_validation",
        )
    with pytest.raises(ValueError, match="State validation selection"):
        TrainingConfig(checkpoint_selection="minimum_state_validation")
    model = CountSDEModel(ModelConfig(state_dim=2, target_count=1), RunIntent.COUNT_STATE)
    with pytest.raises(RuntimeError, match="no trained gene decoder"):
        model.decode_logits(torch.zeros(1, 2))


def test_dynamic_pool_bank_is_complete_resumable_and_endpoint_guarded() -> None:
    state = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    mass = torch.tensor([1.0, 3.0, 2.0])
    pools = torch.tensor([0, 0, 1])
    bank = DynamicPoolBank.from_series(state, mass, pools, pool_count=2)
    bank.advance()
    restored = DynamicPoolBank.from_tensor_state(bank.tensor_state())
    assert restored.generation == 0
    assert restored.age.tolist() == [1, 1]
    with pytest.raises(ContractError, match="Protected endpoint"):
        bank.refresh(state, mass, pools, protected_endpoint_accessed=True)
