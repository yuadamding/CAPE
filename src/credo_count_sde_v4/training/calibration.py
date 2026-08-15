"""Evidence-bearing null calibration for nested state-family selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..canonical import atomic_json, canonical_json_bytes, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    SemanticStudySnapshot,
    StateSelectionCalibration,
    StateSelectionCalibrationResults,
    StateSelectionCalibrationRow,
)
from ..prepare.pipeline import load_config, load_prepared_arrays
from ..runtime_identity import implementation_tree_hash
from .trainer import (
    _fit_shrunk_target_main_weight,
    _initialize_state_channels,
    _loss,
    _new_model_optimizer,
    _state_split,
    _tensor_problem,
    _training_diagnostics,
)


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def calibration_population_hashes(snapshot: SemanticStudySnapshot) -> tuple[str, str]:
    """Bind target multiplicity and the exact source/terminal support structure."""

    target_values = sorted(
        {series.target_index for series in snapshot.series if not series.is_control}
    )
    guide_hash = _hash(
        {
            "guide_counts_per_target": sorted(
                sum(
                    int(series.target_index == target and not series.is_control)
                    for series in snapshot.series
                )
                for target in target_values
            )
        }
    )
    support_hash = _hash(
        {
            "rows": [
                {
                    "series_id": series.series_id,
                    "target_index": series.target_index,
                    "is_control": series.is_control,
                    "source_count": series.source_count,
                    "terminal_count": series.terminal_count,
                }
                for series in sorted(snapshot.series, key=lambda item: item.series_id)
            ]
        }
    )
    return guide_hash, support_hash


def calibration_protocol_payload(results: StateSelectionCalibrationResults) -> dict[str, Any]:
    """Return the execution surface whose hash is bound by the summary receipt."""

    return results.model_dump(mode="json", exclude={"rows"})


def _prepared_problem(
    config_path: Path,
) -> tuple[Any, Any, SemanticStudySnapshot, dict[str, np.ndarray]]:
    from ..compile.compiler import _lookup_means

    config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / config.workspace).resolve()
    prepared, row_ids, latents = load_prepared_arrays(workspace)
    snapshot = SemanticStudySnapshot.model_validate_json(
        (root / config.semantic_snapshot).resolve().read_text()
    )
    source_z, terminal_z = _lookup_means(snapshot, row_ids, latents)
    duration = np.asarray([series.duration for series in snapshot.series], dtype=np.float32)
    grid_steps = np.ceil(duration / config.evaluation.max_step_duration).astype(np.int64)
    arrays = {
        "source_z": source_z,
        "terminal_z": terminal_z,
        "target_index": np.asarray(
            [series.target_index for series in snapshot.series], dtype=np.int64
        ),
        "pool_index": np.asarray([series.pool_index for series in snapshot.series], dtype=np.int64),
        "is_control": np.asarray([series.is_control for series in snapshot.series], dtype=np.uint8),
        "duration": duration,
        "grid_steps": grid_steps,
        "grid_step_size": duration / grid_steps,
        "source_counts": np.asarray(
            [series.source_count for series in snapshot.series], dtype=np.int64
        ),
        "terminal_counts": np.asarray(
            [series.terminal_count for series in snapshot.series], dtype=np.int64
        ),
        "series_ids": np.asarray([series.series_id for series in snapshot.series]),
    }
    return config, prepared, snapshot, arrays


def _conditional_source_permutation(
    arrays: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    """Destroy guide-level source association while preserving target/support structure."""

    result = {name: value.copy() for name, value in arrays.items()}
    generator = np.random.default_rng(seed)
    target = arrays["target_index"].astype(np.int64, copy=False)
    control = arrays["is_control"].astype(bool, copy=False)
    for target_value, is_control in sorted(
        set(zip(target.tolist(), control.tolist(), strict=True))
    ):
        local = np.where((target == target_value) & (control == is_control))[0]
        if len(local) > 1:
            result["source_z"][local] = arrays["source_z"][generator.permutation(local)]
    return result


def _fit_one_null(
    arrays: dict[str, np.ndarray], config: Any, seed: int, device: torch.device
) -> StateSelectionCalibrationRow:
    replicate_config = config.model_copy(
        update={"training": config.training.model_copy(update={"seed": seed})}
    )
    null_arrays = _conditional_source_permutation(arrays, seed)
    model, optimizer = _new_model_optimizer(replicate_config, device)
    problem = _tensor_problem(null_arrays, device)
    split = _state_split(null_arrays, replicate_config, device)
    _initialize_state_channels(model, problem, split.fit_indices, replicate_config)
    null_diagnostics = _training_diagnostics(model, problem, split, replicate_config)
    null_score = float(null_diagnostics["state_validation_global_null_rmse"])
    _fit_shrunk_target_main_weight(
        model, problem, split.fit_indices, replicate_config, materialize=True
    )
    candidates: list[tuple[float, float, int]] = []
    maximum_update = max(replicate_config.training.state_checkpoint_updates)
    checkpoint_set = set(replicate_config.training.state_checkpoint_updates)
    for update in range(1, maximum_update + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, problem, replicate_config, update, None, split)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite calibration loss at seed {seed}, update {update}."
            )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if update in checkpoint_set:
            diagnostics = _training_diagnostics(model, problem, split, replicate_config)
            candidates.append(
                (
                    float(diagnostics["state_validation_full_interaction_rmse"]),
                    float(diagnostics["state_validation_shrunk_target_only_rmse"]),
                    update,
                )
            )
    interaction_score, target_score, selected_update = min(candidates)
    interaction_gain = target_score - interaction_score
    overall_gain = null_score - interaction_score
    target_gain = null_score - target_score
    if (
        interaction_gain
        >= replicate_config.training.state_validation_interaction_minimum_improvement
        and overall_gain >= replicate_config.training.state_validation_target_minimum_improvement
    ):
        family: Literal[
            "global_terminal_null",
            "shrunk_sister_guide_target_terminal",
            "target_plus_source_target_interaction",
        ] = "target_plus_source_target_interaction"
    elif target_gain >= replicate_config.training.state_validation_target_minimum_improvement:
        family = "shrunk_sister_guide_target_terminal"
        selected_update = 0
    else:
        family = "global_terminal_null"
        selected_update = 0
    return StateSelectionCalibrationRow(
        replicate_index=0,
        seed=seed,
        selected_update=selected_update,
        selected_family=family,
        global_null_score=null_score,
        shrunk_target_only_score=target_score,
        interaction_score=interaction_score,
        false_interaction_selected=family == "target_plus_source_target_interaction",
    )


def run_state_selection_calibration(
    config_path: Path,
    output_root: Path,
    *,
    seeds: tuple[int, ...],
    device: str = "cpu",
    method: Literal[
        "target_label_permutation",
        "conditional_source_permutation",
        "synthetic_null_repeats",
    ] = "conditional_source_permutation",
) -> tuple[Path, Path]:
    """Run genuine null fits and publish row-level results plus a strict receipt."""

    if len(seeds) < 59 or len(set(seeds)) != len(seeds):
        raise ValueError("Calibration requires at least 59 unique seeds.")
    if method != "conditional_source_permutation":
        raise NotImplementedError(
            "Only conditional_source_permutation has an implemented genuine-fit generator."
        )
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    config, prepared, snapshot, arrays = _prepared_problem(config_path)
    if not config.model.source_target_interaction_rank:
        raise ValueError("State-selection calibration requires an interaction pilot config.")
    selected_device = torch.device(device)
    rows = []
    for index, seed in enumerate(seeds):
        row = _fit_one_null(arrays, config, seed, selected_device)
        rows.append(row.model_copy(update={"replicate_index": index}))
    from ..compile.compiler import _problem_hash

    root = config_path.parent.resolve()
    guide_hash, support_hash = calibration_population_hashes(snapshot)
    calibration_id = _hash(
        {
            "config": config.model_dump(mode="json"),
            "prepared_id": prepared.prepared_id,
            "split_hash": sha256_file((root / config.split_contract).resolve()),
            "seeds": seeds,
            "method": method,
        }
    )
    results = StateSelectionCalibrationResults(
        calibration_id=calibration_id,
        method=method,
        implementation_tree_hash=implementation_tree_hash(),
        calibration_code_hash=sha256_file(Path(__file__)),
        representation_id=prepared.prepared_id,
        split_manifest_hash=sha256_file((root / config.split_contract).resolve()),
        compiled_problem_hash=_problem_hash(arrays),
        interaction_rank=config.model.source_target_interaction_rank,
        interaction_scale=config.model.source_target_interaction_scale,
        learning_rate=config.training.learning_rate,
        state_batch_size=config.training.state_batch_size,
        source_target_main_penalty=config.training.source_target_main_penalty,
        source_target_interaction_penalty=config.training.source_target_interaction_penalty,
        target_minimum_improvement=(config.training.state_validation_target_minimum_improvement),
        interaction_minimum_improvement=(
            config.training.state_validation_interaction_minimum_improvement
        ),
        checkpoint_updates=config.training.state_checkpoint_updates,
        guide_per_target_distribution_hash=guide_hash,
        support_distribution_hash=support_hash,
        seeds=seeds,
        rows=tuple(rows),
    )
    results_path = output_root / "state-selection-calibration-results.json"
    atomic_json(results_path, results.model_dump(mode="json"))
    false_count = sum(row.false_interaction_selected for row in rows)
    if false_count:
        raise RuntimeError(
            f"Null calibration selected the interaction family {false_count}/{len(rows)} times."
        )
    upper = 1.0 - 0.05 ** (1.0 / len(rows))
    artifact = ArtifactRef(
        schema_id="credo.state_selection_calibration_results",
        schema_version=1,
        sha256=sha256_file(results_path),
        size_bytes=results_path.stat().st_size,
        media_type="application/json",
        relative_uri=results_path.name,
    )
    receipt = StateSelectionCalibration(
        calibration_id=calibration_id,
        method=method,
        repeated_seeds=len(rows),
        false_interaction_count=0,
        false_interaction_rate_upper_bound=upper,
        target_minimum_improvement=(config.training.state_validation_target_minimum_improvement),
        interaction_minimum_improvement=(
            config.training.state_validation_interaction_minimum_improvement
        ),
        checkpoint_updates=config.training.state_checkpoint_updates,
        results_artifact=artifact,
        calibration_protocol_hash=_hash(calibration_protocol_payload(results)),
    )
    receipt_path = output_root / "state-selection-calibration.json"
    atomic_json(receipt_path, receipt.model_dump(mode="json"))
    return results_path, receipt_path
