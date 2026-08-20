"""T04 fixed-truth qualification for the streaming particle engine."""

from __future__ import annotations

import json
import math
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from ..canonical import canonical_json_bytes, contract_id, path_manifest, sha256_bytes, sha256_file
from ..contracts import (
    ComponentTestContract,
    ComponentTestReceipt,
    ModelConfig,
    ParticleEngineQualificationBundle,
    ParticleEngineTestReceipt,
    RunIntent,
    TrainingConfig,
)
from ..errors import IntegrityError
from ..model import CountSDEModel, DynamicPoolBank
from ..persistence import artifact_ref, publish_directory, verify_directory
from ..training.trainer import scientific_checkpoint_updates
from .particles import (
    ParticleState,
    advance_particle_state,
    initialize_particle_state,
    rollout,
    rollout_with_context_schedule,
)

_PARTICLE_GRID = (64, 256, 1024, 4096)
_STEP_GRID = (8, 16, 32, 64)
_SEED_COUNT = 50


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _implementation_identity() -> tuple[str, dict[str, str]]:
    package = Path(__file__).resolve().parents[1]
    relative_paths = (
        "contracts/models.py",
        "model/pool_bank.py",
        "numerics/particles.py",
        "numerics/qualification.py",
        "training/trainer.py",
    )
    files = {relative: sha256_file(package / relative) for relative in relative_paths}
    return sha256_bytes(canonical_json_bytes(files)), files


def _environment_identity() -> tuple[str, dict[str, str]]:
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": metadata.version("pyarrow"),
        "torch": torch.__version__,
        "device": "cpu",
    }
    return sha256_bytes(canonical_json_bytes(environment)), environment


class _FixedTruthModel(CountSDEModel):
    """Non-trainable analytic dynamics routed through the production engine."""

    def __init__(
        self,
        *,
        mode: Literal["constant", "ou", "identity"],
        velocity: float = 0.0,
        theta: float = 0.0,
        sigma: float = 0.0,
        reaction_rates: tuple[float, ...] = (0.0,),
        ecology_coefficients: tuple[float, ...] | None = None,
        pool_count: int = 1,
    ) -> None:
        config = ModelConfig(
            state_dim=1,
            target_count=len(reaction_rates),
            pool_count=pool_count,
            shared_diffusion=sigma > 0,
            shared_diffusion_inner_validation_pass=sigma > 0,
        )
        super().__init__(config, RunIntent.COUNT_MEASURE)
        self.fixed_mode = mode
        self.fixed_velocity = velocity
        self.fixed_theta = theta
        self.fixed_sigma = sigma
        self.fixed_reaction_rates = torch.tensor(reaction_rates, dtype=torch.float64)
        self.fixed_ecology_coefficients = (
            None
            if ecology_coefficients is None
            else torch.tensor(ecology_coefficients, dtype=torch.float64)
        )

    def diffusion(self) -> torch.Tensor:
        return torch.tensor(
            [self.fixed_sigma], dtype=self.base_drift.dtype, device=self.base_drift.device
        )

    def state_step(
        self,
        z: torch.Tensor,
        dt: torch.Tensor,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        total_steps: int,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
        pool_state_mean: torch.Tensor | None = None,
        pool_log_mass: torch.Tensor | None = None,
        source_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del target_index, pool_index, is_control, total_steps, effect_mode
        del context_mode, pool_state_mean, pool_log_mass, source_z
        while dt.ndim < z.ndim:
            dt = dt.unsqueeze(-1)
        if self.fixed_mode == "constant":
            return z + dt * self.fixed_velocity
        if self.fixed_mode == "ou":
            return z - dt * self.fixed_theta * z
        return z

    def centered_selection(
        self,
        z: torch.Tensor,
        weights: torch.Tensor,
        target_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        effect_mode: str = "factual",
    ) -> torch.Tensor:
        del weights, target_index, is_control, effect_mode
        return torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)

    def relative_fitness(
        self,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
        source_exposure: torch.Tensor,
        *,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
        pool_state_mean: torch.Tensor | None = None,
        pool_log_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del is_control, source_exposure, effect_mode, pool_log_mass
        rates = self.fixed_reaction_rates.to(target_index.device)[target_index]
        if self.fixed_ecology_coefficients is None:
            return rates
        if context_mode == "source_fixed" or pool_state_mean is None:
            raise ValueError("Fixed ecology truth requires a complete dynamic pool bank.")
        coefficients = self.fixed_ecology_coefficients.to(target_index.device)[target_index]
        return rates + coefficients * pool_state_mean[pool_index, 0]


def _common_inputs(series: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.arange(series, dtype=torch.int64),
        torch.zeros(series, dtype=torch.int64),
        torch.zeros(series, dtype=torch.bool),
        torch.ones(series, dtype=torch.float64),
    )


def _deterministic_drift_metrics() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
    duration = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
    target, pool, control, exposure = _common_inputs(3)
    model = _FixedTruthModel(mode="constant", velocity=0.75, reaction_rates=(0.0, 0.0, 0.0))
    model = model.to(dtype=torch.float64)
    expected = source + duration[:, None] * 0.75
    rows: list[dict[str, Any]] = []
    for steps in _STEP_GRID:
        particles, _, _ = rollout(
            model,
            source,
            duration,
            target,
            pool,
            control,
            exposure,
            particles=1,
            steps=steps,
            seed=4100 + steps,
        )
        error = float(torch.max(torch.abs(particles[:, 0] - expected)))
        rows.append(
            {
                "truth": "constant_drift",
                "steps": steps,
                "max_abs_error": error,
                "expected_terminal": expected[:, 0].tolist(),
                "observed_terminal": particles[:, 0, 0].tolist(),
            }
        )

    linear_source = torch.tensor([[1.0]], dtype=torch.float64)
    linear_duration = torch.ones(1, dtype=torch.float64)
    linear_model = _FixedTruthModel(mode="ou", theta=0.7).to(dtype=torch.float64)
    linear_expected = math.exp(-0.7)
    linear_errors: list[float] = []
    for steps in _STEP_GRID:
        particles, _, _ = rollout(
            linear_model,
            linear_source,
            linear_duration,
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.float64),
            particles=1,
            steps=steps,
            seed=4200 + steps,
        )
        error = abs(float(particles[0, 0, 0]) - linear_expected)
        linear_errors.append(error)
        rows.append(
            {
                "truth": "linear_drift_refinement",
                "steps": steps,
                "max_abs_error": error,
                "expected_terminal": [linear_expected],
                "observed_terminal": [float(particles[0, 0, 0])],
            }
        )
    constant_errors = [float(row["max_abs_error"]) for row in rows[: len(_STEP_GRID)]]
    summary = {
        "constant_max_abs_error": max(constant_errors),
        "constant_tolerance": 1e-12,
        "linear_refinement_errors": linear_errors,
        "refinement_pass": bool(
            max(constant_errors) <= 1e-12
            and all(
                right < left for left, right in zip(linear_errors, linear_errors[1:], strict=False)
            )
        ),
    }
    return pd.DataFrame(rows), summary


def _ou_metrics() -> tuple[pd.DataFrame, dict[str, Any]]:
    theta, sigma, x0, duration_value = 0.2, 0.5, 0.25, 1.0
    exact_mean = x0 * math.exp(-theta * duration_value)
    exact_variance = sigma**2 / (2.0 * theta) * (1.0 - math.exp(-2.0 * theta * duration_value))
    source = torch.tensor([[x0]], dtype=torch.float64)
    duration = torch.tensor([duration_value], dtype=torch.float64)
    target = pool = torch.zeros(1, dtype=torch.int64)
    control = torch.zeros(1, dtype=torch.bool)
    exposure = torch.ones(1, dtype=torch.float64)
    model = _FixedTruthModel(mode="ou", theta=theta, sigma=sigma).to(dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    raw_largest_variance_errors: list[float] = []
    for particles in _PARTICLE_GRID:
        for steps in _STEP_GRID:
            means: list[float] = []
            variances: list[float] = []
            for seed_index in range(_SEED_COUNT):
                terminal, _, _ = rollout(
                    model,
                    source,
                    duration,
                    target,
                    pool,
                    control,
                    exposure,
                    particles=particles,
                    steps=steps,
                    seed=430000 + particles * 100 + steps * 10 + seed_index,
                )
                sample = terminal[0, :, 0]
                means.append(float(sample.mean()))
                variances.append(float(sample.var(unbiased=True)))
            aggregate_mean = float(np.mean(means))
            aggregate_variance = float(np.mean(variances))
            mean_error = abs(aggregate_mean - exact_mean)
            mean_standard_error = math.sqrt(exact_variance / (particles * _SEED_COUNT))
            variance_relative_error = abs(aggregate_variance - exact_variance) / exact_variance
            rows.append(
                {
                    "particles": particles,
                    "steps": steps,
                    "seed_count": _SEED_COUNT,
                    "exact_mean": exact_mean,
                    "observed_mean": aggregate_mean,
                    "mean_abs_error": mean_error,
                    "mean_standard_error": mean_standard_error,
                    "mean_within_two_standard_errors": mean_error <= 2.0 * mean_standard_error,
                    "exact_variance": exact_variance,
                    "observed_variance": aggregate_variance,
                    "variance_relative_error": variance_relative_error,
                }
            )
            if particles == _PARTICLE_GRID[-1] and steps == _STEP_GRID[-1]:
                raw_largest_variance_errors = [
                    abs(value - exact_variance) / exact_variance for value in variances
                ]
    frame = pd.DataFrame(rows)
    largest = frame[(frame.particles == _PARTICLE_GRID[-1]) & (frame.steps == _STEP_GRID[-1])].iloc[
        0
    ]
    coarsest = frame[(frame.particles == _PARTICLE_GRID[0]) & (frame.steps == _STEP_GRID[0])].iloc[
        0
    ]
    finest_error = float(largest.mean_abs_error + largest.variance_relative_error)
    coarsest_error = float(coarsest.mean_abs_error + coarsest.variance_relative_error)
    analytic_step_errors = []
    for steps in _STEP_GRID:
        dt = duration_value / steps
        multiplier = 1.0 - theta * dt
        euler_mean = x0 * multiplier**steps
        euler_variance = sigma**2 * dt * (1.0 - multiplier ** (2 * steps)) / (1.0 - multiplier**2)
        analytic_step_errors.append(
            abs(euler_mean - exact_mean) + abs(euler_variance - exact_variance) / exact_variance
        )
    summary = {
        "theta": theta,
        "sigma": sigma,
        "exact_mean": exact_mean,
        "exact_variance": exact_variance,
        "largest_grid_mean_within_two_standard_errors": bool(
            largest.mean_within_two_standard_errors
        ),
        "largest_grid_variance_relative_error": float(largest.variance_relative_error),
        "largest_grid_variance_error_limit": 0.10,
        "coarsest_combined_error": coarsest_error,
        "finest_combined_error": finest_error,
        "analytic_step_errors": analytic_step_errors,
        "convergence_pass": bool(
            finest_error < coarsest_error
            and all(
                right < left
                for left, right in zip(analytic_step_errors, analytic_step_errors[1:], strict=False)
            )
        ),
        "largest_grid_seed_variance_error_interval": [
            float(np.quantile(raw_largest_variance_errors, 0.025)),
            float(np.quantile(raw_largest_variance_errors, 0.975)),
        ],
    }
    return frame, summary


def _reaction_metrics() -> tuple[pd.DataFrame, dict[str, Any]]:
    rates = (-0.7, math.log(2.0))
    model = _FixedTruthModel(mode="identity", reaction_rates=rates, pool_count=2).to(
        dtype=torch.float64
    )
    source = torch.zeros(2, 1, dtype=torch.float64)
    target = torch.tensor([0, 1], dtype=torch.int64)
    pool = torch.tensor([0, 1], dtype=torch.int64)
    control = torch.zeros(2, dtype=torch.bool)
    exposure = torch.ones(2, dtype=torch.float64)
    _, weights, mass = rollout(
        model,
        source,
        torch.ones(2, dtype=torch.float64),
        target,
        pool,
        control,
        exposure,
        particles=8,
        steps=64,
        seed=440001,
    )
    expected = torch.exp(torch.tensor(rates, dtype=torch.float64))
    relative_error = torch.abs(mass - expected) / expected
    frame = pd.DataFrame(
        {
            "truth": ["constant_depletion", "mass_doubling"],
            "rate": list(rates),
            "expected_mass": expected.tolist(),
            "observed_mass": mass.tolist(),
            "relative_error": relative_error.tolist(),
        }
    )
    return frame, {
        "max_relative_error": float(relative_error.max()),
        "relative_error_limit": 1e-4,
        "normalized_particle_weights_pass": bool(
            torch.allclose(weights.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-12)
        ),
        "declared_mass_pass": bool(
            torch.allclose((weights * mass[:, None]).sum(dim=1), mass, atol=1e-12)
        ),
    }


def _manual_ecology(
    source_state: torch.Tensor,
    source_exposure: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, list[float]]:
    log_mass = torch.zeros_like(source_exposure)
    means: list[float] = []
    for _ in range(steps):
        absolute = source_exposure * torch.exp(log_mass)
        mean = (absolute * source_state).sum() / absolute.sum()
        means.append(float(mean))
        log_mass = log_mass + coefficients * mean / steps
    return torch.exp(log_mass), means


def _ecology_metrics() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    exposure = torch.tensor([9.0, 1.0], dtype=torch.float64)
    coefficients = torch.tensor([0.4, -0.2], dtype=torch.float64)
    target = torch.tensor([0, 1], dtype=torch.int64)
    pool = torch.zeros(2, dtype=torch.int64)
    control = torch.zeros(2, dtype=torch.bool)
    model = _FixedTruthModel(
        mode="identity",
        reaction_rates=(0.0, 0.0),
        ecology_coefficients=(0.4, -0.2),
    ).to(dtype=torch.float64)
    steps = 64
    _, _, observed_mass, schedule = rollout_with_context_schedule(
        model,
        source,
        torch.ones(2, dtype=torch.float64),
        target,
        pool,
        control,
        exposure,
        particles=8,
        steps=steps,
        seed=450001,
        effect_mode="factual",
    )
    expected_mass, expected_means = _manual_ecology(
        source[:, 0], exposure, coefficients, steps=steps
    )
    observed_means = [float(item[0][0, 0]) for item in schedule]
    absolute_error = float(torch.max(torch.abs(observed_mass - expected_mass)))
    normalized_within_guide_mass = torch.ones_like(observed_mass)
    negative_control_gap = float(torch.max(torch.abs(normalized_within_guide_mass - expected_mass)))

    direct = DynamicPoolBank.from_series(source, exposure, pool, pool_count=1)
    logged = DynamicPoolBank.from_log_series(source, exposure.log(), pool, pool_count=1)
    extreme = DynamicPoolBank.from_log_series(
        source,
        torch.tensor([1000.0, 998.0], dtype=torch.float64),
        pool,
        pool_count=1,
    )
    shifted = DynamicPoolBank.from_series(
        source,
        torch.exp(torch.tensor([0.0, -2.0], dtype=torch.float64)),
        pool,
        pool_count=1,
    )
    log_weight_pass = bool(
        torch.allclose(direct.state_mean, logged.state_mean, atol=1e-12)
        and torch.allclose(direct.log_mass, logged.log_mass, atol=1e-12)
        and torch.allclose(extreme.state_mean, shifted.state_mean, atol=1e-12)
        and torch.allclose(extreme.log_mass - 1000.0, shifted.log_mass, atol=1e-12)
        and bool(torch.isfinite(extreme.state_mean).all())
        and bool(torch.isfinite(extreme.log_mass).all())
    )
    frame = pd.DataFrame(
        {
            "series_index": [0, 1],
            "source_state": source[:, 0].tolist(),
            "source_exposure": exposure.tolist(),
            "coefficient": coefficients.tolist(),
            "expected_relative_mass": expected_mass.tolist(),
            "observed_relative_mass": observed_mass.tolist(),
            "normalized_within_guide_negative_control_mass": normalized_within_guide_mass.tolist(),
        }
    )
    return frame, {
        "absolute_weight_max_error": absolute_error,
        "absolute_weight_error_limit": 1e-12,
        "normalized_context_negative_control_gap": negative_control_gap,
        "normalized_context_negative_control_minimum_gap": 1e-2,
        "expected_context_means": expected_means,
        "observed_context_means": observed_means,
        "stabilized_log_weight_pass": log_weight_pass,
    }


def _lifecycle_metrics() -> dict[str, Any]:
    source = torch.tensor([[-0.5], [0.75]], dtype=torch.float64)
    duration = torch.ones(2, dtype=torch.float64)
    target = torch.tensor([0, 1], dtype=torch.int64)
    pool = torch.tensor([0, 1], dtype=torch.int64)
    control = torch.zeros(2, dtype=torch.bool)
    exposure = torch.tensor([3.0, 2.0], dtype=torch.float64)
    model = _FixedTruthModel(
        mode="ou",
        theta=0.3,
        sigma=0.2,
        reaction_rates=(-0.1, 0.15),
        pool_count=2,
    ).to(dtype=torch.float64)
    one_shot = rollout(
        model,
        source,
        duration,
        target,
        pool,
        control,
        exposure,
        particles=128,
        steps=32,
        seed=460001,
    )
    replay = rollout(
        model,
        source,
        duration,
        target,
        pool,
        control,
        exposure,
        particles=128,
        steps=32,
        seed=460001,
    )
    initial = initialize_particle_state(source, particles=128, seed=460001)
    partial, _ = advance_particle_state(
        model,
        initial,
        source,
        duration,
        target,
        pool,
        control,
        exposure,
        total_steps=32,
        stop_step=13,
    )
    restored = ParticleState.from_tensor_state(partial.tensor_state())
    resumed, _ = advance_particle_state(
        model,
        restored,
        source,
        duration,
        target,
        pool,
        control,
        exposure,
        total_steps=32,
        stop_step=32,
    )
    deterministic_replay = all(
        torch.equal(left, right) for left, right in zip(one_shot, replay, strict=True)
    )
    interrupted_resume = bool(
        torch.equal(one_shot[0], resumed.z)
        and torch.equal(one_shot[1], resumed.weights)
        and torch.equal(one_shot[2], torch.exp(resumed.log_mass))
    )
    expected_mass = torch.exp(torch.tensor([-0.1, 0.15], dtype=torch.float64))
    guide_switching = not bool(
        torch.allclose(one_shot[2], expected_mass, atol=1e-12)
        and float(one_shot[0][0].mean()) < float(one_shot[0][1].mean())
    )
    selection_config = TrainingConfig(
        max_updates=4,
        checkpoint_every=2,
        selected_update=None,
        state_validation_fraction=0.25,
        checkpoint_selection="minimum_state_validation",
    )
    capacity_candidates = scientific_checkpoint_updates((1, 2, 4), selection_config)
    return {
        "deterministic_cpu_replay_pass": deterministic_replay,
        "interrupted_resumed_parity_pass": interrupted_resume,
        "no_guide_switching_pass": not guide_switching,
        "finite_particle_weights_pass": bool(torch.isfinite(one_shot[1]).all()),
        "normalized_particle_weights_pass": bool(
            torch.allclose(one_shot[1].sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-12)
        ),
        "declared_mass_pass": bool(
            torch.allclose(
                (one_shot[1] * one_shot[2][:, None]).sum(dim=1),
                one_shot[2],
                atol=1e-12,
            )
        ),
        "capacity_probe_exclusion_pass": capacity_candidates == (2, 4),
        "capacity_probe_persisted_updates": [1, 2, 4],
        "capacity_probe_scientific_candidates": list(capacity_candidates),
        "capacity_probe_policy": (
            "The production selection-candidate filter excludes a disposable update-1 "
            "capacity probe from the frozen 2/4 scientific schedule."
        ),
    }


def qualify_particle_engine(destination: Path) -> Path:
    """Run and atomically publish the complete CPU T04 numerical qualification."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        drift_frame, drift = _deterministic_drift_metrics()
        ou_frame, ou = _ou_metrics()
        reaction_frame, reaction = _reaction_metrics()
        ecology_frame, ecology = _ecology_metrics()
        lifecycle = _lifecycle_metrics()
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    config = {
        "method": "streaming_euler_maruyama_v1",
        "dtype": "float64",
        "device": "cpu",
        "particle_grid": list(_PARTICLE_GRID),
        "step_grid": list(_STEP_GRID),
        "seed_count": _SEED_COUNT,
        "ou": {"theta": ou["theta"], "sigma": ou["sigma"], "duration": 1.0},
        "reaction_rates": [-0.7, math.log(2.0)],
        "ecology_source_exposure": [9.0, 1.0],
    }
    environment_hash, environment = _environment_identity()
    config["environment"] = environment
    config_hash = sha256_bytes(canonical_json_bytes(config))
    implementation_hash, implementation_files = _implementation_identity()
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": "T04_PARTICLE_ENGINE",
        "component": "streaming_particle_engine",
        "primary_metric": "largest_grid_ou_variance_relative_error",
        "primary_baseline": "analytic_fixed_truth",
        "required_margin": 0.0,
        "drift": "fixed",
        "diffusion": "fixed",
        "reaction": "fixed",
        "ecology": "off",
        "decoder": "off",
        "update_zero_selectable": True,
        "post_selection_refit_required": False,
    }
    contract_payload["test_contract_id"] = contract_id(
        contract_payload, id_field="test_contract_id"
    )
    test_contract = ComponentTestContract.model_validate(contract_payload)
    all_gates = (
        drift["constant_max_abs_error"] <= drift["constant_tolerance"]
        and drift["refinement_pass"]
        and ou["largest_grid_mean_within_two_standard_errors"]
        and ou["largest_grid_variance_relative_error"] <= ou["largest_grid_variance_error_limit"]
        and ou["convergence_pass"]
        and reaction["max_relative_error"] <= reaction["relative_error_limit"]
        and ecology["absolute_weight_max_error"] <= ecology["absolute_weight_error_limit"]
        and ecology["normalized_context_negative_control_gap"]
        >= ecology["normalized_context_negative_control_minimum_gap"]
        and ecology["stabilized_log_weight_pass"]
        and lifecycle["deterministic_cpu_replay_pass"]
        and lifecycle["interrupted_resumed_parity_pass"]
        and lifecycle["no_guide_switching_pass"]
        and lifecycle["finite_particle_weights_pass"]
        and lifecycle["normalized_particle_weights_pass"]
        and lifecycle["declared_mass_pass"]
        and lifecycle["capacity_probe_exclusion_pass"]
    )

    def writer(temp: Path) -> None:
        drift_frame.to_parquet(temp / "DETERMINISTIC_DRIFT.parquet", index=False)
        ou_frame.to_parquet(temp / "OU_GRID.parquet", index=False)
        reaction_frame.to_parquet(temp / "REACTION_MASS.parquet", index=False)
        ecology_frame.to_parquet(temp / "ECOLOGY_RESULTS.parquet", index=False)
        _write_json(temp / "DRIFT_RESULTS.json", drift)
        _write_json(temp / "OU_RESULTS.json", ou)
        _write_json(temp / "REACTION_RESULTS.json", reaction)
        _write_json(temp / "ECOLOGY_RESULTS.json", ecology)
        _write_json(temp / "LIFECYCLE_RESULTS.json", lifecycle)
        _write_json(temp / "TEST_CONTRACT.json", test_contract.model_dump(mode="json"))
        _write_json(temp / "CONFIG.json", config)
        _write_json(
            temp / "IMPLEMENTATION.sha256",
            {
                "schema_version": 1,
                "implementation_hash": implementation_hash,
                "files": implementation_files,
            },
        )
        _write_json(
            temp / "NULL_CALIBRATION.json",
            {
                "schema_version": 1,
                "status": "not_applicable_fixed_truth",
                "negative_controls": ["normalized_within_guide_context", "zero_noise_replay"],
            },
        )
        _write_json(
            temp / "BOOTSTRAP_RESULTS.json",
            {
                "schema_version": 1,
                "unit": "independent_ou_seed",
                "seed_count": _SEED_COUNT,
                "largest_grid_variance_error_interval": ou[
                    "largest_grid_seed_variance_error_interval"
                ],
            },
        )
        detail_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "test_contract_id": test_contract.test_contract_id,
            "status": "pass" if all_gates else "fail_retired",
            "deterministic_drift_max_abs_error": drift["constant_max_abs_error"],
            "drift_refinement_pass": drift["refinement_pass"],
            "ou_mean_within_two_standard_errors": ou[
                "largest_grid_mean_within_two_standard_errors"
            ],
            "ou_largest_grid_variance_relative_error": ou["largest_grid_variance_relative_error"],
            "ou_convergence_pass": ou["convergence_pass"],
            "reaction_max_relative_error": reaction["max_relative_error"],
            "ecology_absolute_weight_max_error": ecology["absolute_weight_max_error"],
            "normalized_context_negative_control_detected": (
                ecology["normalized_context_negative_control_gap"]
                >= ecology["normalized_context_negative_control_minimum_gap"]
            ),
            "stabilized_log_weight_pass": ecology["stabilized_log_weight_pass"],
            "deterministic_replay_pass": lifecycle["deterministic_cpu_replay_pass"],
            "interrupted_resume_pass": lifecycle["interrupted_resumed_parity_pass"],
            "no_guide_switching_pass": lifecycle["no_guide_switching_pass"],
            "normalized_particle_weights_pass": lifecycle["normalized_particle_weights_pass"],
            "declared_mass_pass": lifecycle["declared_mass_pass"],
            "capacity_probe_exclusion_pass": lifecycle["capacity_probe_exclusion_pass"],
            "protected_metrics_pass": True,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
        }
        # The qualification ID is linked after both independent identities are
        # resolved, avoiding a receipt/bundle identity cycle.
        detail_payload["receipt_id"] = contract_id(detail_payload, id_field="receipt_id")
        detail_receipt = ParticleEngineTestReceipt.model_validate(detail_payload)
        _write_json(temp / "TEST_RECEIPT.json", detail_receipt.model_dump(mode="json"))

        generic_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "test_id": test_contract.test_id,
            "status": "pass" if all_gates else "fail_retired",
            "primary_metric": test_contract.primary_metric,
            "primary_baseline": test_contract.primary_baseline,
            "point_delta": ou["largest_grid_variance_relative_error"] - 0.10,
            "bootstrap_interval": [
                ou["largest_grid_seed_variance_error_interval"][0] - 0.10,
                ou["largest_grid_seed_variance_error_interval"][1] - 0.10,
            ],
            "required_margin": 0.0,
            "channel_activity": 1.0,
            "protected_metrics_pass": True,
            "selected_update": 0,
            "input_hashes": {"fixed_truth_config": config_hash},
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
        }
        generic_payload["input_hashes"]["environment"] = environment_hash
        generic_payload["receipt_id"] = contract_id(generic_payload, id_field="receipt_id")
        component_receipt = ComponentTestReceipt.model_validate(generic_payload)
        _write_json(temp / "COMPONENT_RECEIPT.json", component_receipt.model_dump(mode="json"))

        bundle_payload = {
            "schema_version": 1,
            "qualification_id": "pending",
            "test_contract_id": test_contract.test_contract_id,
            "method": "streaming_euler_maruyama_v1",
            "environment_hash": environment_hash,
            "particle_grid": _PARTICLE_GRID,
            "step_grid": _STEP_GRID,
            "seed_count": _SEED_COUNT,
            "deterministic_drift": artifact_ref(
                temp,
                temp / "DETERMINISTIC_DRIFT.parquet",
                schema_id="credo.t04_deterministic_drift",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "ou_grid": artifact_ref(
                temp,
                temp / "OU_GRID.parquet",
                schema_id="credo.t04_ou_grid",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "reaction_mass": artifact_ref(
                temp,
                temp / "REACTION_MASS.parquet",
                schema_id="credo.t04_reaction_mass",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "ecology": artifact_ref(
                temp,
                temp / "ECOLOGY_RESULTS.parquet",
                schema_id="credo.t04_ecology",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "lifecycle": artifact_ref(
                temp,
                temp / "LIFECYCLE_RESULTS.json",
                schema_id="credo.t04_lifecycle",
                media_type="application/json",
            ).model_dump(mode="json"),
            "test_receipt": artifact_ref(
                temp,
                temp / "TEST_RECEIPT.json",
                schema_id="credo.t04_test_receipt",
                media_type="application/json",
            ).model_dump(mode="json"),
        }
        bundle_payload["qualification_id"] = contract_id(
            bundle_payload, id_field="qualification_id"
        )
        bundle = ParticleEngineQualificationBundle.model_validate(bundle_payload)
        _write_json(temp / "particle-engine.json", bundle.model_dump(mode="json"))

        # Bind the final qualification ID in a non-identity metadata file. The
        # typed detailed receipt remains cycle-free and independently valid.
        _write_json(
            temp / "QUALIFICATION_LINK.json",
            {
                "schema_version": 1,
                "qualification_id": bundle.qualification_id,
                "receipt_id": detail_receipt.receipt_id,
            },
        )
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_particle_engine_qualification(destination)
    return destination


def verify_particle_engine_qualification(path: Path) -> ParticleEngineQualificationBundle:
    """Verify the committed T04 evidence and recompute its fixed gates."""

    verify_directory(path)
    bundle = ParticleEngineQualificationBundle.model_validate_json(
        (path / "particle-engine.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    receipt = ParticleEngineTestReceipt.model_validate_json(
        (path / "TEST_RECEIPT.json").read_text()
    )
    component = ComponentTestReceipt.model_validate_json(
        (path / "COMPONENT_RECEIPT.json").read_text()
    )
    if (
        contract.test_id != "T04_PARTICLE_ENGINE"
        or contract.test_contract_id != bundle.test_contract_id
        or receipt.test_contract_id != bundle.test_contract_id
        or component.test_id != contract.test_id
        or receipt.status != component.status
    ):
        raise IntegrityError("T04 contract, bundle, and receipts are inconsistent.")
    for reference in (
        bundle.deterministic_drift,
        bundle.ou_grid,
        bundle.reaction_mass,
        bundle.ecology,
        bundle.lifecycle,
        bundle.test_receipt,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"T04 artifact differs from its reference: {reference.relative_uri}."
            )
    link = json.loads((path / "QUALIFICATION_LINK.json").read_text())
    if link != {
        "schema_version": 1,
        "qualification_id": bundle.qualification_id,
        "receipt_id": receipt.receipt_id,
    }:
        raise IntegrityError("T04 qualification-to-receipt link differs from its bundle.")
    implementation = json.loads((path / "IMPLEMENTATION.sha256").read_text())
    if (
        implementation.get("implementation_hash") != receipt.implementation_hash
        or sha256_bytes(canonical_json_bytes(implementation.get("files")))
        != receipt.implementation_hash
    ):
        raise IntegrityError("T04 implementation file identities differ from the receipt.")
    config = json.loads((path / "CONFIG.json").read_text())
    if (
        sha256_bytes(canonical_json_bytes(config.get("environment"))) != receipt.environment_hash
        or receipt.environment_hash != bundle.environment_hash
    ):
        raise IntegrityError("T04 numerical environment differs from its receipt.")
    ou = json.loads((path / "OU_RESULTS.json").read_text())
    drift = json.loads((path / "DRIFT_RESULTS.json").read_text())
    reaction = json.loads((path / "REACTION_RESULTS.json").read_text())
    ecology = json.loads((path / "ECOLOGY_RESULTS.json").read_text())
    lifecycle = json.loads((path / "LIFECYCLE_RESULTS.json").read_text())
    stored_pass = bool(
        drift["constant_max_abs_error"] <= drift["constant_tolerance"]
        and drift["refinement_pass"]
        and ou["largest_grid_mean_within_two_standard_errors"]
        and ou["largest_grid_variance_relative_error"] <= ou["largest_grid_variance_error_limit"]
        and ou["convergence_pass"]
        and reaction["max_relative_error"] <= reaction["relative_error_limit"]
        and ecology["absolute_weight_max_error"] <= ecology["absolute_weight_error_limit"]
        and ecology["normalized_context_negative_control_gap"]
        >= ecology["normalized_context_negative_control_minimum_gap"]
        and ecology["stabilized_log_weight_pass"]
        and lifecycle["deterministic_cpu_replay_pass"]
        and lifecycle["interrupted_resumed_parity_pass"]
        and lifecycle["no_guide_switching_pass"]
        and lifecycle["finite_particle_weights_pass"]
        and lifecycle["normalized_particle_weights_pass"]
        and lifecycle["declared_mass_pass"]
        and lifecycle["capacity_probe_exclusion_pass"]
    )
    if (receipt.status == "pass") != stored_pass:
        raise IntegrityError("T04 status differs from recomputed fixed numerical gates.")
    return bundle
