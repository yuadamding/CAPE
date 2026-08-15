"""One-shot evaluation, baseline audit, and final sealing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from ..canonical import canonical_json_bytes, contract_id, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    BaselineRegistry,
    EvaluationBundleManifest,
    InferenceBundleManifest,
    SealedRunManifest,
    SemanticStudySnapshot,
    SeriesRecord,
)
from ..errors import ContractError
from ..inference import open_inference_run
from ..numerics import rollout
from ..persistence import publish_directory, verify_directory
from ..store import CountStore


def _future_ref(workspace: Path, final: Path, temp: Path, schema: str, media: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema,
        schema_version=1,
        sha256=sha256_file(temp),
        size_bytes=temp.stat().st_size,
        media_type=media,
        relative_uri=final.relative_to(workspace).as_posix(),
    )


def _hash_rows(rows: tuple[int, ...]) -> str:
    return sha256_bytes(np.asarray(rows, dtype="<i8").tobytes())


def _training_row_order(records: tuple[SeriesRecord, ...]) -> tuple[int, ...]:
    """Canonical baseline order: every source row, then every endpoint row."""

    source = tuple(row for record in records for row in record.source_rows)
    terminal = tuple(row for record in records for row in record.terminal_rows)
    return source + terminal


def _interaction_advancement_pass(
    *,
    selected_family: str,
    interaction_bootstrap_upper: float,
    overall_bootstrap_upper: float,
    required_interaction_improvement: float,
    required_overall_improvement: float,
    interaction_displacement_rms: float,
    minimum_interaction_displacement_rms: float,
) -> bool:
    """Fail closed unless the deployed family and both outer gates pass."""

    return bool(
        selected_family == "target_plus_source_target_interaction"
        and interaction_bootstrap_upper < -required_interaction_improvement
        and overall_bootstrap_upper < -required_overall_improvement
        and interaction_displacement_rms >= minimum_interaction_displacement_rms
    )


def _series_means(
    records: tuple[SeriesRecord, ...], row_ids: np.ndarray, latents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {int(row): index for index, row in enumerate(row_ids)}
    source = np.asarray(
        [latents[[lookup[row] for row in record.source_rows]].mean(axis=0) for record in records],
        dtype=np.float32,
    )
    terminal = np.asarray(
        [latents[[lookup[row] for row in record.terminal_rows]].mean(axis=0) for record in records],
        dtype=np.float32,
    )
    return source, terminal


def _observed_gene_compositions(records: tuple[SeriesRecord, ...], store: CountStore) -> np.ndarray:
    row_ids = np.asarray(
        [row for record in records for row in record.terminal_rows], dtype=np.int64
    )
    batch = store.rows(row_ids).matrix
    compositions: list[np.ndarray] = []
    cursor = 0
    for record in records:
        end = cursor + len(record.terminal_rows)
        counts = np.asarray(batch[cursor:end].sum(axis=0)).reshape(-1).astype(np.float64)
        total = counts.sum()
        compositions.append(counts / total if total > 0 else np.zeros_like(counts))
        cursor = end
    return np.asarray(compositions, dtype=np.float32)


def _rmse(predicted: np.ndarray, terminal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - terminal))))


def _safe_correlation(kind: str, left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = (
        pearsonr(left, right).statistic if kind == "pearson" else spearmanr(left, right).statistic
    )
    return float(value) if np.isfinite(value) else None


def _population_metrics(
    frame: pd.DataFrame,
    terminal: np.ndarray,
    source: np.ndarray,
    predictions: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"series": int(mask.sum())}
    target = frame.loc[mask, "target_index"].to_numpy(dtype=np.int64)
    for name, values in predictions.items():
        result[f"{name}_rmse"] = _rmse(values[mask], terminal[mask])
        target_mse = [
            float(
                np.mean(np.square(values[mask][target == value] - terminal[mask][target == value]))
            )
            for value in np.unique(target)
        ]
        result[f"{name}_target_balanced_rmse"] = float(np.sqrt(np.mean(target_mse)))
    observed_delta = terminal[mask] - source[mask]
    model_delta = predictions["v4"][mask] - source[mask]
    result["v4_delta_flat_pearson"] = _safe_correlation(
        "pearson", model_delta.reshape(-1), observed_delta.reshape(-1)
    )
    result["v4_delta_norm_spearman"] = _safe_correlation(
        "spearman", np.linalg.norm(model_delta, axis=1), np.linalg.norm(observed_delta, axis=1)
    )
    return result


def _target_balanced_bootstrap_differences(
    model: np.ndarray,
    baseline: np.ndarray,
    terminal: np.ndarray,
    target_indices: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> np.ndarray:
    """Bootstrap targets while preserving the target-balanced RMSE estimand."""

    unique_targets = np.unique(target_indices)
    model_target_mse = np.asarray(
        [
            np.mean(np.square(model[target_indices == value] - terminal[target_indices == value]))
            for value in unique_targets
        ],
        dtype=np.float64,
    )
    baseline_target_mse = np.asarray(
        [
            np.mean(
                np.square(baseline[target_indices == value] - terminal[target_indices == value])
            )
            for value in unique_targets
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(unique_targets), size=(draws, len(unique_targets)))
    return np.sqrt(model_target_mse[sampled].mean(axis=1)) - np.sqrt(
        baseline_target_mse[sampled].mean(axis=1)
    )


def _target_balanced_rms(displacement: np.ndarray, target_indices: np.ndarray) -> float:
    target_mse = [
        float(np.mean(np.square(displacement[target_indices == value])))
        for value in np.unique(target_indices)
    ]
    return float(np.sqrt(np.mean(target_mse)))


def _independent_shrunk_target_prediction(
    *,
    train_terminal: np.ndarray,
    train_target: np.ndarray,
    train_control: np.ndarray,
    evaluation_target: np.ndarray,
    evaluation_control: np.ndarray,
    maximum_weight: float,
    scalar_ridge: float,
) -> tuple[np.ndarray, float]:
    """Materialize M1 independently of whichever family was deployed."""

    global_terminal = train_terminal.mean(axis=0)
    numerators: list[float] = []
    denominators: list[float] = []
    offsets: dict[int, np.ndarray] = {}
    for target_value in np.unique(train_target[~train_control]):
        local = (train_target == target_value) & ~train_control
        local_terminal = train_terminal[local]
        offsets[int(target_value)] = local_terminal.mean(axis=0) - global_terminal
        if len(local_terminal) < 2:
            continue
        residual = local_terminal - global_terminal
        other_mean = (local_terminal.sum(axis=0) - local_terminal) / (len(local_terminal) - 1)
        candidate = other_mean - global_terminal
        numerators.append(float(np.mean(residual * candidate)))
        denominators.append(float(np.mean(np.square(candidate))))
    alpha = (
        float(
            np.clip(
                np.mean(numerators) / (np.mean(denominators) + scalar_ridge),
                0,
                maximum_weight,
            )
        )
        if numerators
        else 0.0
    )
    prediction = np.broadcast_to(
        global_terminal, (len(evaluation_target), len(global_terminal))
    ).copy()
    for index, target_value in enumerate(evaluation_target):
        if not evaluation_control[index] and int(target_value) in offsets:
            prediction[index] += alpha * offsets[int(target_value)]
    return prediction.astype(np.float32), alpha


def _evaluate_bound_outer(
    workspace: Path, run: Any, destination: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    input_root = workspace / "input"
    outer_path = input_root / "outer-evaluation.json"
    plan_path = input_root / "evaluation-plan.json"
    source_manifest_path = input_root / "source-manifest.json"
    if sha256_file(source_manifest_path) != run.contract.source_manifest_hash:
        raise ContractError("Bound source manifest differs from the compiled run.")
    source_manifest = json.loads(source_manifest_path.read_text())
    for key, path in (
        ("outer_evaluation_sha256", outer_path),
        ("evaluation_plan_sha256", plan_path),
    ):
        if source_manifest.get(key) != sha256_file(path):
            raise ContractError(f"Bound source manifest does not match {path.name}.")
    outer = json.loads(outer_path.read_text())
    frozen_plan = json.loads(plan_path.read_text())
    records = tuple(SeriesRecord.model_validate(row) for row in outer["series"])
    if not records:
        raise ContractError("Outer evaluation catalog is empty.")
    baseline_registry = BaselineRegistry.model_validate_json(
        (input_root / "baseline-registry.json").read_text()
    )
    logical_baseline_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "registry": baseline_registry.model_dump(mode="json"),
                "representation_id": run.contract.representation_id,
            }
        )
    )
    if logical_baseline_hash != run.contract.baseline_registry_hash:
        raise ContractError("Baseline registry differs from the compiled run.")
    evaluator_hash = sha256_file(Path(__file__))
    if any(row.code_hash != evaluator_hash for row in baseline_registry.baselines):
        raise ContractError("Baseline implementation hash differs from the frozen evaluator.")
    train_snapshot = SemanticStudySnapshot.model_validate_json(
        (workspace / "compiled" / "snapshot.json").read_text()
    )
    train_rows = _training_row_order(train_snapshot.series)
    test_source = tuple(row for record in records for row in record.source_rows)
    test_terminal = tuple(row for record in records for row in record.terminal_rows)
    for baseline in baseline_registry.baselines:
        if (
            baseline.allowed_training_rows_hash != _hash_rows(train_rows)
            or baseline.allowed_test_source_rows_hash != _hash_rows(test_source)
            or baseline.forbidden_endpoint_rows_hash != _hash_rows(test_terminal)
        ):
            raise ContractError(f"Baseline information set is false: {baseline.baseline_id}.")

    with h5py.File(workspace / "prepared" / "latents.h5", "r") as handle:
        row_ids = handle["row_ids"][:]
        latents = handle["z"][:]
    source, terminal = _series_means(records, row_ids, latents)
    train_source = run.arrays["source_z"]
    train_terminal = run.arrays["terminal_z"]
    train_target = run.arrays["target_index"].astype(np.int64)
    train_control = run.arrays["is_control"].astype(bool)
    global_delta = (train_terminal - train_source).mean(axis=0)
    control_delta = (train_terminal[train_control] - train_source[train_control]).mean(axis=0)
    global_terminal = train_terminal.mean(axis=0)
    target_indices = np.asarray([record.target_index for record in records], dtype=np.int64)
    target_delta_rows: list[np.ndarray] = []
    target_terminal_rows: list[np.ndarray] = []
    for row, target_value in zip(source, target_indices, strict=True):
        local = train_target == target_value
        target_delta_rows.append(
            row
            + ((train_terminal - train_source)[local].mean(axis=0) if local.any() else global_delta)
        )
        target_terminal_rows.append(
            train_terminal[local].mean(axis=0) if local.any() else global_terminal
        )
    predictions: dict[str, np.ndarray] = {
        "persistence": source,
        "global_delta": source + global_delta,
        "control_delta": source + control_delta,
        "target_delta": np.asarray(target_delta_rows, dtype=np.float32),
        "global_terminal": np.broadcast_to(global_terminal, terminal.shape).copy(),
        "target_terminal": np.asarray(target_terminal_rows, dtype=np.float32),
    }
    interaction_pilot = bool(run.config.model.source_target_interaction_rank)
    shrunk_target_alpha: float | None = None
    if interaction_pilot:
        predictions["shrunk_target_only"], shrunk_target_alpha = (
            _independent_shrunk_target_prediction(
                train_terminal=train_terminal,
                train_target=train_target,
                train_control=train_control,
                evaluation_target=target_indices,
                evaluation_control=np.asarray(
                    [record.is_control for record in records], dtype=bool
                ),
                maximum_weight=run.config.model.source_target_main_max_weight,
                scalar_ridge=run.config.training.source_target_main_penalty,
            )
        )
    device = run.device
    source_tensor = torch.from_numpy(source).to(device)
    target_tensor = torch.from_numpy(target_indices).to(device)
    control_tensor = torch.tensor([record.is_control for record in records], device=device)
    states, weights, mass = rollout(
        run.model,
        source_tensor,
        torch.tensor([record.duration for record in records], device=device),
        target_tensor,
        torch.zeros_like(target_tensor),
        control_tensor,
        torch.tensor([record.source_count + 0.5 for record in records], device=device),
        particles=run.config.evaluation.particles,
        steps=run.config.evaluation.steps,
        seed=run.config.evaluation.seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    predictions["v4"] = (states * weights.unsqueeze(-1)).sum(dim=1).cpu().numpy()
    mass_np = mass.cpu().numpy()
    weights_np = weights.cpu().numpy()
    numerical_pass = bool(
        all(np.isfinite(value).all() for value in predictions.values())
        and np.isfinite(mass_np).all()
        and (mass_np > 0).all()
        and np.isfinite(weights_np).all()
        and np.max(np.abs(weights_np.sum(axis=1) - 1.0)) <= 1e-6
    )
    frame = pd.DataFrame(
        {
            "series_id": [record.series_id for record in records],
            "target_index": target_indices,
            "is_control": [record.is_control for record in records],
            "source_cells": [record.source_count for record in records],
            "terminal_cells": [record.terminal_count for record in records],
        }
    )
    for name, values in predictions.items():
        frame[f"{name}_rmse"] = np.sqrt(np.mean(np.square(values - terminal), axis=1))
    all_mask = np.ones(len(frame), dtype=bool)
    targeting = ~frame.is_control.to_numpy(dtype=bool)
    control = ~targeting
    population_masks = {
        "all_series": all_mask,
        "targeting_series": targeting,
        "control_series": control,
    }
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "all_series": _population_metrics(frame, terminal, source, predictions, all_mask),
        "targeting_series": _population_metrics(frame, terminal, source, predictions, targeting),
        "control_series": _population_metrics(frame, terminal, source, predictions, control),
    }
    if run.capabilities.decode_gene_composition:
        observed_composition = _observed_gene_compositions(
            records, CountStore(workspace / "input/counts.h5")
        )
        gene_metrics: dict[str, Any] = {}
        for name in ("v4", "global_terminal"):
            composition = run.decode_composition(predictions[name])
            cross_entropy = -np.sum(
                observed_composition * np.log(composition.clip(min=1e-12)), axis=1
            )
            frame[f"{name}_gene_cross_entropy"] = cross_entropy
            target_means = [
                float(cross_entropy[target_indices == value].mean())
                for value in np.unique(target_indices)
            ]
            gene_metrics[f"{name}_target_balanced_cross_entropy"] = float(np.mean(target_means))
            gene_metrics[f"{name}_median_effective_genes"] = float(
                np.median(1.0 / np.square(composition).sum(axis=1))
            )
        gene_metrics["v4_minus_global_terminal_cross_entropy"] = (
            gene_metrics["v4_target_balanced_cross_entropy"]
            - gene_metrics["global_terminal_target_balanced_cross_entropy"]
        )
        gene_metrics["observed_median_effective_genes"] = float(
            np.median(1.0 / np.square(observed_composition).sum(axis=1))
        )
        metrics["gene_composition_diagnostic"] = gene_metrics
    primary = str(frozen_plan["primary_baseline"])
    if interaction_pilot and primary != "shrunk_target_only":
        raise ContractError(
            "Source-target interaction evaluation requires shrunk_target_only as primary baseline."
        )
    primary_metric = str(frozen_plan.get("primary_metric", "target_balanced_rmse"))
    key = f"{primary}_{primary_metric}"
    model_key = f"v4_{primary_metric}"
    primary_population = str(frozen_plan["primary_population"])
    if primary_population not in metrics:
        raise ValueError(
            f"Primary population {primary_population!r} is absent from evaluation metrics."
        )
    primary_metrics = metrics[primary_population]
    if key not in primary_metrics or model_key not in primary_metrics:
        raise ValueError("Primary metric or baseline is absent from the frozen population.")
    numerical_tolerance = float(frozen_plan.get("numerical_tolerance", 0.0))
    scientific_minimum = float(frozen_plan.get("scientific_minimum_improvement", 0.0))
    delta = primary_metrics[model_key] - primary_metrics[key]
    metrics["primary_comparison"] = {
        "population": primary_population,
        "baseline": primary,
        "metric": primary_metric,
        "v4": primary_metrics[model_key],
        "baseline_value": primary_metrics[key],
        "delta": delta,
        "gate_type": "numerical_tie_or_better",
        "numerical_tolerance": numerical_tolerance,
        "pass": bool(delta <= numerical_tolerance),
    }
    differences = _target_balanced_bootstrap_differences(
        predictions["v4"][population_masks[primary_population]],
        predictions[primary][population_masks[primary_population]],
        terminal[population_masks[primary_population]],
        target_indices[population_masks[primary_population]],
        seed=int(frozen_plan["bootstrap_seed"]),
        draws=int(frozen_plan["bootstrap_draws"]),
    )
    interval = [float(value) for value in np.quantile(differences, [0.025, 0.975])]
    metrics["conditional_target_bootstrap"] = {
        "seed": int(frozen_plan["bootstrap_seed"]),
        "draws": int(frozen_plan["bootstrap_draws"]),
        "interval_95": interval,
        "aggregation": "target_balanced_RMSE",
        "biological_replicate_interval": False,
    }
    plan = {
        **frozen_plan,
        "run_id": run.manifest.run_id,
        "outer_evaluation_sha256": sha256_file(outer_path),
        "baseline_registry_hash": run.contract.baseline_registry_hash,
        "one_shot": bool(not frozen_plan.get("historically_exposed", False)),
    }
    scientific_threshold = scientific_minimum + numerical_tolerance
    minimum_interaction_rms = float(frozen_plan.get("minimum_interaction_displacement_rms", 0.0))
    if interaction_pilot and minimum_interaction_rms <= 0.0:
        raise ContractError(
            "Source-target interaction evaluation requires a positive frozen effect-size floor."
        )
    targeting_interaction_rms = (
        _target_balanced_rms(
            predictions["v4"][targeting] - predictions["shrunk_target_only"][targeting],
            target_indices[targeting],
        )
        if interaction_pilot and targeting.any()
        else 0.0
    )
    family_eligible = run.manifest.selected_family == "target_plus_source_target_interaction"
    overall_interval: list[float] | None = None
    if interaction_pilot:
        if (
            "interaction_scientific_minimum_improvement" not in frozen_plan
            or "overall_scientific_minimum_improvement" not in frozen_plan
        ):
            raise ContractError(
                "Interaction evaluation requires separate frozen M2-vs-M1 and M2-vs-M0 margins."
            )
        overall_differences = _target_balanced_bootstrap_differences(
            predictions["v4"][population_masks[primary_population]],
            predictions["global_terminal"][population_masks[primary_population]],
            terminal[population_masks[primary_population]],
            target_indices[population_masks[primary_population]],
            seed=int(frozen_plan["bootstrap_seed"]),
            draws=int(frozen_plan["bootstrap_draws"]),
        )
        overall_interval = [
            float(value) for value in np.quantile(overall_differences, [0.025, 0.975])
        ]
        interaction_threshold = (
            float(frozen_plan["interaction_scientific_minimum_improvement"]) + numerical_tolerance
        )
        overall_threshold = (
            float(frozen_plan["overall_scientific_minimum_improvement"]) + numerical_tolerance
        )
        metrics["interaction_outer_gate"] = {
            "m2_minus_m1_interval_95": interval,
            "m2_minus_m0_interval_95": overall_interval,
            "m1_alpha": shrunk_target_alpha,
            "m0_global_null_rmse": primary_metrics[f"global_terminal_{primary_metric}"],
            "m1_shrunk_target_rmse": primary_metrics[f"shrunk_target_only_{primary_metric}"],
            "m2_interaction_rmse": primary_metrics[model_key],
            "m1_minus_m0": (
                primary_metrics[f"shrunk_target_only_{primary_metric}"]
                - primary_metrics[f"global_terminal_{primary_metric}"]
            ),
            "interaction_threshold_including_numerical_tolerance": interaction_threshold,
            "overall_threshold_including_numerical_tolerance": overall_threshold,
            "aggregation": "target_balanced_RMSE",
        }
        scientific_gate_pass = bool(
            numerical_pass
            and _interaction_advancement_pass(
                selected_family=run.manifest.selected_family,
                interaction_bootstrap_upper=interval[1],
                overall_bootstrap_upper=overall_interval[1],
                required_interaction_improvement=interaction_threshold,
                required_overall_improvement=overall_threshold,
                interaction_displacement_rms=targeting_interaction_rms,
                minimum_interaction_displacement_rms=minimum_interaction_rms,
            )
        )
    else:
        scientific_gate_pass = bool(numerical_pass and interval[1] < -scientific_threshold)
    audit = {
        "schema_version": 1,
        "status": "engineering_complete" if numerical_pass else "numerical_failure",
        "qualified": False,
        "numerical_pass": numerical_pass,
        "scientific_gate_pass": scientific_gate_pass,
        "selected_family": run.manifest.selected_family,
        "interaction_family_eligible": family_eligible,
        "interaction_displacement_rms": targeting_interaction_rms,
        "minimum_interaction_displacement_rms": minimum_interaction_rms,
        "scientific_gate": (
            "nested_M2_beats_M1_and_M0_with_target_bootstrap"
            if interaction_pilot
            else "conditional_target_bootstrap_upper_below_negative_minimum"
        ),
        "interaction_vs_target_only_interval_95": interval if interaction_pilot else None,
        "interaction_vs_global_null_interval_95": overall_interval,
        "shrunk_target_main_weight": shrunk_target_alpha,
        "scientific_minimum_improvement": scientific_minimum,
        "scientific_improvement_threshold_including_numerical_tolerance": scientific_threshold,
        "outer_evaluation_access": (
            "current_lifecycle_bound_before_compile_but_historically_exposed"
            if frozen_plan.get("historically_exposed", False)
            else "one_shot_bound_before_compile"
        ),
        "checkpoint_selected_before_evaluation": True,
        "endpoint_used_for_eligibility": bool(
            frozen_plan.get("endpoint_used_for_eligibility", False)
        ),
        "eligibility_rule": frozen_plan.get("eligibility_rule"),
        "claim_status": "engineering_only_historical_endpoint_exposure",
        "batch_identifiability": "failed_timepoint_and_WTA_library_are_perfectly_confounded",
        "biological_replication": "unavailable",
        "evaluator_sha256": evaluator_hash,
        "device": str(device),
    }
    return metrics, plan, audit, frame


def evaluate_run(config_path: Path, *, device: str = "cpu") -> Path:
    import yaml

    root = config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text())
    workspace = (root / raw["workspace"]).resolve()
    run = open_inference_run(workspace / "inference", device=device, verify="full")
    destination = workspace / "evaluation"
    if (workspace / "input" / "outer-evaluation.json").is_file():
        metrics, plan, audit, frame = _evaluate_bound_outer(workspace, run, destination)
        predicted_mass = np.full(len(frame), np.nan)
        observed_counts = np.full(len(frame), np.nan)
        evaluable = np.ones(len(frame), dtype=bool)
    else:
        predicted_state, predicted_mass, _ = run.terminal()
        target_state = run.arrays["terminal_z"]
        evaluable = np.isfinite(target_state).all(axis=1)
        state_rmse = float(
            np.sqrt(np.mean((predicted_state[evaluable] - target_state[evaluable]) ** 2))
        )
        persistence_rmse = float(
            np.sqrt(np.mean((run.arrays["source_z"][evaluable] - target_state[evaluable]) ** 2))
        )
        observed_counts = run.arrays["terminal_counts"].astype(float)
        predicted_abundance = np.log(predicted_mass + 1e-12)
        observed_abundance = np.log(observed_counts + 0.5) - np.log(
            run.arrays["source_counts"] + 0.5
        )
        abundance_rho = (
            float(spearmanr(predicted_abundance, observed_abundance).statistic)
            if run.capabilities.predict_relative_mass and len(observed_counts) > 1
            else None
        )
        if abundance_rho is not None and not np.isfinite(abundance_rho):
            abundance_rho = None
        metrics = {
            "schema_version": 1,
            "state_rmse": state_rmse,
            "persistence_rmse": persistence_rmse,
            "state_delta_vs_persistence": state_rmse - persistence_rmse,
            "relative_abundance_spearman": abundance_rho,
            "evaluable_series": int(evaluable.sum()),
            "total_series": int(len(evaluable)),
        }
        plan = {
            "schema_version": 1,
            "run_id": run.manifest.run_id,
            "information_set": run.contract.information_set_hash,
            "one_shot": True,
            "baseline_registry_hash": run.contract.baseline_registry_hash,
            "multiplicity_plan_hash": run.contract.multiplicity_plan_hash,
            "selected_family": run.manifest.selected_family,
        }
        audit = {
            "schema_version": 1,
            "outer_evaluation_access": "one_shot",
            "checkpoint_selected_before_evaluation": True,
            "endpoint_used_for_eligibility": False,
            "claim_status": "engineering_only",
            "selected_family": run.manifest.selected_family,
            "interaction_family_eligible": False,
            "scientific_gate_pass": False,
        }
        frame = pd.DataFrame(
            {
                "series_id": run.arrays["series_ids"].astype(str),
                "target_index": run.arrays["target_index"],
                "pool_index": run.arrays["pool_index"],
                "state_evaluable": evaluable,
                "predicted_relative_mass": predicted_mass,
                "observed_terminal_count": observed_counts,
            }
        )

    def writer(temp: Path) -> None:
        (temp / "plan.json").write_bytes(canonical_json_bytes(plan) + b"\n")
        (temp / "metrics.json").write_bytes(canonical_json_bytes(metrics) + b"\n")
        (temp / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
        frame.to_parquet(temp / "predictions.parquet", index=False)
        final = destination
        payload = {
            "schema_version": 1,
            "evaluation_id": "pending",
            "run_id": run.manifest.run_id,
            "plan": _future_ref(
                workspace,
                final / "plan.json",
                temp / "plan.json",
                "credo.evaluation_plan",
                "application/json",
            ).model_dump(mode="json"),
            "metrics": _future_ref(
                workspace,
                final / "metrics.json",
                temp / "metrics.json",
                "credo.metrics",
                "application/json",
            ).model_dump(mode="json"),
            "predictions": _future_ref(
                workspace,
                final / "predictions.parquet",
                temp / "predictions.parquet",
                "credo.predictions",
                "application/x-parquet",
            ).model_dump(mode="json"),
            "audit": _future_ref(
                workspace,
                final / "audit.json",
                temp / "audit.json",
                "credo.evaluation_audit",
                "application/json",
            ).model_dump(mode="json"),
        }
        payload["evaluation_id"] = contract_id(payload, id_field="evaluation_id")
        manifest = EvaluationBundleManifest.model_validate(payload)
        (temp / "evaluation.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(destination, writer)
    return destination


def seal_run(config_path: Path) -> Path:
    import yaml

    root = config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text())
    workspace = (root / raw["workspace"]).resolve()
    inference_root = workspace / "inference"
    evaluation_root = workspace / "evaluation"
    verify_directory(inference_root)
    verify_directory(evaluation_root)
    inference = InferenceBundleManifest.model_validate_json(
        (inference_root / "inference.json").read_text()
    )
    evaluation = EvaluationBundleManifest.model_validate_json(
        (evaluation_root / "evaluation.json").read_text()
    )
    if evaluation.run_id != inference.run_id:
        raise ValueError("Evaluation does not belong to inference bundle.")
    destination = workspace / "sealed"

    def writer(temp: Path) -> None:
        claim = json.loads((evaluation_root / "audit.json").read_text())
        (temp / "claim-audit.json").write_bytes(canonical_json_bytes(claim) + b"\n")
        inference_ref = ArtifactRef(
            schema_id="credo.inference_bundle",
            schema_version=1,
            sha256=sha256_file(inference_root / "inference.json"),
            size_bytes=(inference_root / "inference.json").stat().st_size,
            media_type="application/json",
            relative_uri="../inference/inference.json".replace("../", "inference/"),
        )
        # Sealed refs are logical bundle IDs rather than navigable filesystem
        # paths; verification resolves parents from the sibling workspace.
        inference_ref = inference_ref.model_copy(
            update={"relative_uri": "inference/inference.json"}
        )
        evaluation_ref = ArtifactRef(
            schema_id="credo.evaluation_bundle",
            schema_version=1,
            sha256=sha256_file(evaluation_root / "evaluation.json"),
            size_bytes=(evaluation_root / "evaluation.json").stat().st_size,
            media_type="application/json",
            relative_uri="evaluation/evaluation.json",
        )
        claim_ref = _future_ref(
            workspace,
            destination / "claim-audit.json",
            temp / "claim-audit.json",
            "credo.claim_audit",
            "application/json",
        )
        payload = {
            "schema_version": 1,
            "sealed_id": "pending",
            "inference": inference_ref.model_dump(mode="json"),
            "evaluations": [evaluation_ref.model_dump(mode="json")],
            "claim_audit": claim_ref.model_dump(mode="json"),
        }
        payload["sealed_id"] = contract_id(payload, id_field="sealed_id")
        manifest = SealedRunManifest.model_validate(payload)
        (temp / "sealed.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(destination, writer)
    return destination
