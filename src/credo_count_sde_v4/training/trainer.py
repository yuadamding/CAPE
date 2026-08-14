"""Update-based trainer with immutable checkpoint generations."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import sparse

from ..canonical import atomic_json, canonical_json_bytes, contract_id, sha256_bytes
from ..compile import load_compiled_problem
from ..contracts import (
    ArtifactRef,
    CheckpointManifest,
    CompiledRunContract,
    ResolvedConfig,
    RunIntent,
    SemanticStudySnapshot,
)
from ..errors import ResumeMismatchError
from ..model import CountSDEModel, DynamicPoolBank
from ..objectives import exact_count_loss
from ..persistence import (
    load_tensor_file,
    publish_directory,
    save_tensor_file,
    verify_directory,
)
from ..runtime_identity import environment_lock_hash, implementation_tree_hash
from ..store import CountStore


@dataclass
class _GeneDecoderData:
    row_ids: np.ndarray
    latents: np.ndarray
    counts: sparse.csr_matrix
    library_sizes: np.ndarray
    validation_row_ids: np.ndarray
    validation_latents: np.ndarray
    validation_counts: sparse.csr_matrix
    validation_library_sizes: np.ndarray


@dataclass(frozen=True)
class _StateSplit:
    fit_indices: torch.Tensor
    validation_indices: torch.Tensor
    fit_series_hash: str
    validation_series_hash: str


def _state_split(
    arrays: dict[str, np.ndarray], config: ResolvedConfig, device: torch.device
) -> _StateSplit:
    """Create a deterministic target-stratified training-only state split."""

    terminal = arrays["terminal_z"]
    evaluable = np.isfinite(terminal).all(axis=1)
    target = arrays["target_index"].astype(np.int64, copy=False)
    series = arrays["series_ids"].astype(str, copy=False)
    validation: list[int] = []
    if config.training.state_validation_fraction:
        for target_value in sorted(set(map(int, target[evaluable]))):
            positions = np.where(evaluable & (target == target_value))[0].tolist()
            if len(positions) < 2:
                continue
            requested = max(
                1,
                int(round(len(positions) * config.training.state_validation_fraction)),
            )
            count = min(
                len(positions) - 1,
                config.training.state_validation_max_per_target,
                requested,
            )
            ranked = sorted(
                positions,
                key=lambda index: hashlib.sha256(
                    f"{config.training.seed}:{target_value}:{series[index]}".encode()
                ).digest(),
            )
            validation.extend(ranked[:count])
    validation_array = np.asarray(sorted(validation), dtype=np.int64)
    validation_set = set(map(int, validation_array))
    fit_array = np.asarray(
        [index for index in np.where(evaluable)[0] if int(index) not in validation_set],
        dtype=np.int64,
    )
    if not len(fit_array):
        raise ValueError("The state training split is empty.")
    if config.training.state_validation_fraction and not len(validation_array):
        raise ValueError("The configured state validation split is empty.")

    def series_hash(indices: np.ndarray) -> str:
        payload = "\n".join(series[indices].tolist()).encode() + b"\n"
        return sha256_bytes(payload)

    return _StateSplit(
        fit_indices=torch.from_numpy(fit_array).to(device),
        validation_indices=torch.from_numpy(validation_array).to(device),
        fit_series_hash=series_hash(fit_array),
        validation_series_hash=series_hash(validation_array),
    )


def _gene_decoder_data(
    root: Path, workspace: Path, config: ResolvedConfig
) -> _GeneDecoderData | None:
    if not config.training.gene_decoder_batch_size:
        return None
    snapshot = SemanticStudySnapshot.model_validate_json(
        (workspace / "compiled/snapshot.json").read_text()
    )
    row_ids = np.asarray(
        [row for series in snapshot.series for row in (*series.source_rows, *series.terminal_rows)],
        dtype=np.int64,
    )
    with h5py.File(workspace / "prepared/latents.h5", "r") as handle:
        available = handle["row_ids"][:]
        z = handle["z"][:]
    lookup = {int(row): index for index, row in enumerate(available)}
    try:
        positions = np.asarray([lookup[int(row)] for row in row_ids], dtype=np.int64)
    except KeyError as error:
        raise ResumeMismatchError("Gene decoder row is absent from the latent cache.") from error
    latents = np.asarray(z[positions], dtype=np.float32)
    validation_count = min(
        config.training.gene_decoder_validation_max_rows,
        int(round(len(row_ids) * config.training.gene_decoder_validation_fraction)),
    )
    if config.training.gene_decoder_validation_fraction and validation_count == 0:
        validation_count = 1
    generator = np.random.default_rng(config.training.seed + 4_117_919)
    permutation = generator.permutation(len(row_ids))
    validation_positions = np.sort(permutation[:validation_count])
    training_positions = np.sort(permutation[validation_count:])
    # Materialize the immutable decoder row universe once.  Random HDF5 CSR
    # reads otherwise rescan every source-row window at every update, which is
    # correct but prohibitively I/O-bound for multi-thousand-update training.
    # The run contract already bounds this cohort-local cache through the Pod
    # memory request; gene counts remain sparse and are never moved wholesale
    # to the GPU.
    selected = CountStore((root / config.count_store).resolve()).rows(row_ids)
    counts = selected.matrix.tocsr()
    library_sizes = selected.library_sizes.astype(np.float32, copy=False)
    return _GeneDecoderData(
        row_ids=row_ids[training_positions],
        latents=latents[training_positions],
        counts=counts[training_positions].tocsr(),
        library_sizes=library_sizes[training_positions],
        validation_row_ids=row_ids[validation_positions],
        validation_latents=latents[validation_positions],
        validation_counts=counts[validation_positions].tocsr(),
        validation_library_sizes=library_sizes[validation_positions],
    )


def _gene_decoder_loss(
    model: CountSDEModel,
    data: _GeneDecoderData,
    config: ResolvedConfig,
    update: int,
    device: torch.device,
) -> torch.Tensor:
    generator = np.random.default_rng(config.training.seed + update * 1_000_003)
    size = min(config.training.gene_decoder_batch_size, len(data.row_ids))
    positions = generator.choice(len(data.row_ids), size=size, replace=False)
    matrix = data.counts[positions].tocsr()
    state = torch.from_numpy(data.latents[positions]).to(device)
    logits = model.decode_logits(state)
    selected = _csr_selected_logit_sum(logits, matrix)
    totals = torch.from_numpy(data.library_sizes[positions]).to(device)
    return torch.mean(torch.logsumexp(logits, dim=1) - selected / totals.clamp_min(1.0))


def _csr_selected_logit_sum(logits: torch.Tensor, matrix: sparse.csr_matrix) -> torch.Tensor:
    """Deterministically reduce observed-count logits by CSR row.

    CUDA ``scatter_add_`` uses atomic updates for repeated row indices.  A
    length-described segment reduction preserves CSR order and permits the
    deterministic-algorithm runtime gate used by release training.
    """

    matrix = matrix.tocsr()
    lengths = torch.from_numpy(np.diff(matrix.indptr).astype(np.int64, copy=False)).to(
        logits.device
    )
    columns = torch.from_numpy(matrix.indices.astype(np.int64, copy=False)).to(logits.device)
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False)).to(logits.device)
    rows = torch.repeat_interleave(torch.arange(matrix.shape[0], device=logits.device), lengths)
    contributions = logits[rows, columns] * values
    return torch.segment_reduce(contributions, reduce="sum", lengths=lengths)


@torch.no_grad()
def _gene_decoder_validation_diagnostics(
    model: CountSDEModel,
    data: _GeneDecoderData | None,
    device: torch.device,
) -> dict[str, float | str]:
    if data is None or not len(data.validation_row_ids):
        return {}
    total_loss = 0.0
    total_rows = 0
    for start in range(0, len(data.validation_row_ids), 1_024):
        stop = min(start + 1_024, len(data.validation_row_ids))
        matrix = data.validation_counts[start:stop].tocsr()
        state = torch.from_numpy(data.validation_latents[start:stop]).to(device)
        logits = model.decode_logits(state)
        selected = _csr_selected_logit_sum(logits, matrix)
        totals = torch.from_numpy(data.validation_library_sizes[start:stop]).to(device)
        losses = torch.logsumexp(logits, dim=1) - selected / totals.clamp_min(1.0)
        total_loss += float(losses.sum().cpu())
        total_rows += len(losses)
    return {
        "gene_decoder_validation_cross_entropy": total_loss / total_rows,
        "gene_decoder_validation_rows": float(total_rows),
        "gene_decoder_validation_rows_hash": sha256_bytes(
            np.asarray(data.validation_row_ids, dtype="<i8").tobytes()
        ),
    }


def _device(value: str | torch.device | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return result


def _tensor_problem(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in arrays.items():
        if value.dtype.kind in "USO":
            continue
        tensor = torch.from_numpy(value)
        if value.dtype.kind == "f":
            tensor = tensor.to(torch.float32)
        result[name] = tensor.to(device)
    return result


def _named_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    parameter_names: dict[torch.Tensor, str] = {
        parameter: name for name, parameter in model.named_parameters()
    }
    for parameter, state in optimizer.state.items():
        name = parameter_names[parameter]
        for key, value in state.items():
            if torch.is_tensor(value):
                tensors[f"state::{name}::{key}"] = value
            elif isinstance(value, (int, float, bool)):
                tensors[f"state::{name}::{key}"] = torch.tensor(value)
            else:
                raise TypeError(f"Unsupported optimizer state {name}.{key}: {type(value)}")
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        row = {key: value for key, value in group.items() if key != "params"}
        row["params"] = [parameter_names[parameter] for parameter in group["params"]]
        groups.append(row)
    return tensors, {"schema_version": 1, "optimizer": type(optimizer).__name__, "groups": groups}


def _restore_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    tree: dict[str, Any],
) -> None:
    named = dict(model.named_parameters())
    if tree.get("optimizer") != type(optimizer).__name__:
        raise ResumeMismatchError("Optimizer type changed across resume.")
    for key, value in tensors.items():
        _, parameter_name, state_name = key.split("::", 2)
        if parameter_name not in named:
            raise ResumeMismatchError(f"Optimizer parameter missing on resume: {parameter_name}.")
        optimizer.state[named[parameter_name]][state_name] = value.to(named[parameter_name].device)


def _checkpoint(
    workspace: Path,
    training_root: Path,
    contract: CompiledRunContract,
    model: CountSDEModel,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    loss: float,
    diagnostics: dict[str, float | str] | None,
    parent_checkpoint_id: str | None,
) -> CheckpointManifest:
    generation_root = training_root / "checkpoints" / f"generation-{update:09d}"
    manifest_holder: dict[str, CheckpointManifest] = {}

    def writer(temp: Path) -> None:
        save_tensor_file(temp / "model.safetensors", model.state_dict())
        optimizer_tensors, optimizer_tree = _named_optimizer_state(model, optimizer)
        save_tensor_file(temp / "optimizer.safetensors", optimizer_tensors)
        (temp / "optimizer-tree.json").write_bytes(canonical_json_bytes(optimizer_tree) + b"\n")
        rng_tensors: dict[str, torch.Tensor] = {"torch_cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            for index, cuda_state in enumerate(torch.cuda.get_rng_state_all()):
                rng_tensors[f"torch_cuda_{index}"] = cuda_state
        save_tensor_file(temp / "rng.safetensors", rng_tensors)
        sampler = {
            "schema_version": 1,
            "next_update": update,
            "random_policy": "counter_derived",
            "prefetch": "quiesced",
        }
        (temp / "sampler.json").write_bytes(canonical_json_bytes(sampler) + b"\n")
        training_state = {
            "schema_version": 1,
            "update": update,
            "stage": "joint",
            "loss": loss,
            "diagnostics": diagnostics or {},
        }
        (temp / "training-state.json").write_bytes(canonical_json_bytes(training_state) + b"\n")
        final = generation_root

        def future(name: str, schema: str, media: str) -> ArtifactRef:
            path = temp / name
            return ArtifactRef(
                schema_id=schema,
                schema_version=1,
                sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
                media_type=media,
                relative_uri=(final / name).relative_to(workspace).as_posix(),
            )

        payload = {
            "schema_version": 1,
            "checkpoint_id": "pending",
            "compiled_run_id": contract.compiled_run_id,
            "generation": update,
            "update": update,
            "stage": "joint",
            "parent_checkpoint_id": parent_checkpoint_id,
            "model": future(
                "model.safetensors", "credo.model_state", "application/x-safetensors"
            ).model_dump(mode="json"),
            "optimizer": future(
                "optimizer.safetensors", "credo.optimizer_state", "application/x-safetensors"
            ).model_dump(mode="json"),
            "optimizer_tree": future(
                "optimizer-tree.json", "credo.optimizer_tree", "application/json"
            ).model_dump(mode="json"),
            "rng": future(
                "rng.safetensors", "credo.rng_state", "application/x-safetensors"
            ).model_dump(mode="json"),
            "sampler": future("sampler.json", "credo.sampler_state", "application/json").model_dump(
                mode="json"
            ),
            "training_state": future(
                "training-state.json", "credo.training_state", "application/json"
            ).model_dump(mode="json"),
        }
        payload["checkpoint_id"] = contract_id(payload, id_field="checkpoint_id")
        manifest = CheckpointManifest.model_validate(payload)
        manifest_holder["value"] = manifest
        (temp / "checkpoint.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(generation_root, writer)
    latest = {
        "schema_version": 1,
        "generation": update,
        "checkpoint_id": manifest_holder["value"].checkpoint_id,
        "relative_uri": generation_root.relative_to(training_root).as_posix(),
    }
    atomic_json(training_root / "checkpoints" / "latest.json", latest)
    return manifest_holder["value"]


def _deterministic_prediction(
    model: CountSDEModel,
    source: torch.Tensor,
    duration: torch.Tensor,
    target: torch.Tensor,
    pool: torch.Tensor,
    control: torch.Tensor,
    grid_steps: torch.Tensor,
) -> torch.Tensor:
    """Use the same physical Euler grid as deterministic inference."""

    prediction = torch.empty_like(source)
    for steps_value in torch.unique(grid_steps).tolist():
        local = torch.where(grid_steps == steps_value)[0]
        z = source[local]
        dt = duration[local] / int(steps_value)
        for _ in range(int(steps_value)):
            z = model.state_step(
                z,
                dt,
                target[local],
                pool[local],
                control[local],
                total_steps=int(steps_value),
                source_z=source[local],
            )
        prediction[local] = z
    return prediction


@torch.no_grad()
def _initialize_state_channels(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    fit_indices: torch.Tensor,
    config: ResolvedConfig,
) -> dict[str, float]:
    """Initialize the nested constant-drift submodel at its closed-form optimum."""

    evaluable = torch.zeros(
        len(problem["terminal_z"]), dtype=torch.bool, device=problem["terminal_z"].device
    )
    evaluable[fit_indices] = True
    evaluable &= torch.isfinite(problem["terminal_z"]).all(dim=1)
    if model.config.terminal_anchor_drift:
        terminal = problem["terminal_z"][evaluable]
        target = problem["target_index"][evaluable].long()
        control = problem["is_control"][evaluable].bool()
        model.terminal_anchor.copy_(terminal.mean(dim=0))
        model.target_anchor_offset.zero_()
        model.target_anchor_gate.zero_()
        weight = model.config.target_anchor_weight
        if weight:
            for target_value in torch.unique(target[~control]).tolist():
                local = (target == target_value) & ~control
                offset = terminal[local].mean(dim=0) - model.terminal_anchor
                model.target_anchor_offset[int(target_value)].copy_(weight * offset)
                model.target_anchor_gate[int(target_value)] = weight
        if model.config.adaptive_target_anchor:
            source_support = problem["source_counts"][evaluable].float()
            terminal_support = problem["terminal_counts"][evaluable].float()
            reliability = torch.minimum(source_support, terminal_support)
            reliability = reliability.clamp(
                min=1.0, max=float(config.training.support_weight_cap)
            ).pow(config.training.support_weight_power)
            for target_value in torch.unique(target[~control]).tolist():
                local = (target == target_value) & ~control
                local_terminal = terminal[local]
                if len(local_terminal) < 2:
                    continue
                local_reliability = reliability[local]
                total_weight = local_reliability.sum()
                other_weight = (total_weight - local_reliability).clamp_min(1e-12)
                weighted_sum = (local_reliability[:, None] * local_terminal).sum(dim=0)
                other_mean = (
                    weighted_sum - local_reliability[:, None] * local_terminal
                ) / other_weight[:, None]
                # The expanded expression above is deliberately explicit: each
                # guide is predicted only from the other training guides for its
                # target.  No outer guide endpoint enters this gate.
                residual = local_terminal - model.terminal_anchor
                candidate = other_mean - model.terminal_anchor
                row_weight = local_reliability / local_reliability.sum().clamp_min(1e-12)
                denominator = (row_weight[:, None] * candidate.square()).sum()
                if denominator <= 0:
                    continue
                numerator = (row_weight[:, None] * residual * candidate).sum()
                gate = torch.clamp(
                    numerator / denominator,
                    min=0.0,
                    max=model.config.adaptive_target_anchor_max_weight,
                )
                baseline_error = residual.square().mean(dim=1)
                candidate_error = (residual - gate * candidate).square().mean(dim=1)
                guide_gain = baseline_error - candidate_error
                if torch.any(guide_gain < model.config.adaptive_target_anchor_min_guide_gain):
                    continue
                target_mean = (local_reliability[:, None] * local_terminal).sum(
                    dim=0
                ) / local_reliability.sum().clamp_min(1e-12)
                model.target_anchor_gate[int(target_value)] = gate
                model.target_anchor_offset[int(target_value)].copy_(
                    gate * (target_mean - model.terminal_anchor)
                )
        prediction = _deterministic_prediction(
            model,
            problem["source_z"][evaluable],
            problem["duration"][evaluable],
            target,
            problem["pool_index"][evaluable].long(),
            control,
            problem["grid_steps"][evaluable].long(),
        )
        rmse = torch.sqrt(torch.mean((prediction - terminal) ** 2))
        return {
            "analytic_terminal_anchor_rmse": float(rmse.cpu()),
            "source_carryover_alpha": model.config.source_carryover_alpha,
            "target_anchor_weight": model.config.target_anchor_weight,
            "adaptive_target_anchor_active_targets": float(
                torch.count_nonzero(model.target_anchor_gate).cpu()
            ),
            "adaptive_target_anchor_mean_gate": float(
                model.target_anchor_gate[model.target_anchor_gate > 0].mean().cpu()
            )
            if torch.any(model.target_anchor_gate > 0)
            else 0.0,
        }
    rate = problem["terminal_z"][evaluable] - problem["source_z"][evaluable]
    rate = rate / problem["duration"][evaluable, None]
    target = problem["target_index"][evaluable].long()
    control = problem["is_control"][evaluable].bool()
    base_rows = rate[control] if torch.any(control) else rate
    model.base_drift.copy_(base_rows.mean(dim=0))
    model.target_drift.zero_()
    for target_value in torch.unique(target[~control]).tolist():
        local = (target == target_value) & ~control
        model.target_drift[int(target_value)].copy_(rate[local].mean(dim=0) - model.base_drift)
    prediction = problem["source_z"][evaluable] + problem["duration"][evaluable, None] * (
        model.base_drift[None, :] + model.target_drift[target] * (~control).float()[:, None]
    )
    rmse = torch.sqrt(torch.mean((prediction - problem["terminal_z"][evaluable]) ** 2))
    return {"closed_form_initial_rmse": float(rmse.cpu())}


def _target_balanced_mse(
    error: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average series MSE within targets, then weight every target equally."""

    series_mse = error.square().mean(dim=1)
    target_losses: list[torch.Tensor] = []
    for target_value in torch.unique(target).tolist():
        local = target == target_value
        values = series_mse[local]
        if reliability is None:
            target_losses.append(values.mean())
        else:
            weights = reliability[local]
            weights = weights / weights.sum().clamp_min(1e-12)
            target_losses.append((weights * values).sum())
    if not target_losses:
        raise ValueError("Target-balanced state loss has no evaluable targets.")
    return torch.stack(target_losses).mean()


def _support_reliability(
    problem: dict[str, torch.Tensor], indices: torch.Tensor, config: ResolvedConfig
) -> torch.Tensor | None:
    if not config.training.support_weight_power:
        return None
    support = torch.minimum(problem["source_counts"][indices], problem["terminal_counts"][indices])
    support = support.float().clamp(min=1.0, max=float(config.training.support_weight_cap))
    return support.pow(config.training.support_weight_power)


@torch.no_grad()
def _training_diagnostics(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    state_split: _StateSplit,
    decoder_data: _GeneDecoderData | None = None,
) -> dict[str, float | str]:
    evaluable = torch.isfinite(problem["terminal_z"]).all(dim=1)
    indices = torch.where(evaluable)[0]
    prediction = _deterministic_prediction(
        model,
        problem["source_z"][indices],
        problem["duration"][indices],
        problem["target_index"][indices].long(),
        problem["pool_index"][indices].long(),
        problem["is_control"][indices].bool(),
        problem["grid_steps"][indices].long(),
    )
    error = prediction - problem["terminal_z"][indices]
    diagnostics: dict[str, float | str] = {
        "full_state_rmse": float(torch.sqrt(torch.mean(error.square())).cpu()),
        "finite_prediction_fraction": float(torch.isfinite(prediction).float().mean().cpu()),
        "state_fit_series_hash": state_split.fit_series_hash,
        "state_validation_series_hash": state_split.validation_series_hash,
        **_gene_decoder_validation_diagnostics(model, decoder_data, problem["source_z"].device),
    }
    for label, local_indices in (
        ("fit", state_split.fit_indices),
        ("validation", state_split.validation_indices),
    ):
        if not len(local_indices):
            continue
        local_prediction = _deterministic_prediction(
            model,
            problem["source_z"][local_indices],
            problem["duration"][local_indices],
            problem["target_index"][local_indices].long(),
            problem["pool_index"][local_indices].long(),
            problem["is_control"][local_indices].bool(),
            problem["grid_steps"][local_indices].long(),
        )
        local_error = local_prediction - problem["terminal_z"][local_indices]
        local_mse = _target_balanced_mse(
            local_error,
            problem["target_index"][local_indices].long(),
        )
        diagnostics[f"state_{label}_target_balanced_rmse"] = float(torch.sqrt(local_mse).cpu())
        diagnostics[f"state_{label}_series"] = float(len(local_indices))
    return diagnostics


def _write_selection(
    training_root: Path,
    contract: CompiledRunContract,
    config: ResolvedConfig,
    checkpoints: list[CheckpointManifest],
) -> None:
    eligible_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.update % config.training.checkpoint_every == 0
        or checkpoint.update == config.training.max_updates
    ]
    if config.training.checkpoint_selection in {
        "minimum_gene_decoder_validation",
        "minimum_state_validation",
    }:
        diagnostic_key = (
            "gene_decoder_validation_cross_entropy"
            if config.training.checkpoint_selection == "minimum_gene_decoder_validation"
            else "state_validation_target_balanced_rmse"
        )
        scored = []
        for checkpoint in eligible_checkpoints:
            generation = training_root / "checkpoints" / f"generation-{checkpoint.update:09d}"
            state = json.loads((generation / "training-state.json").read_text())
            score = state.get("diagnostics", {}).get(diagnostic_key)
            if score is not None:
                scored.append((float(score), checkpoint.update, checkpoint.checkpoint_id))
        if not scored:
            raise RuntimeError("No validation-scored checkpoint is available for selection.")
        score, update, checkpoint_id = min(scored)
        metric = diagnostic_key
    else:
        selected_update = config.training.selected_update or config.training.max_updates
        matches = [item for item in checkpoints if item.update == selected_update]
        if not matches:
            raise RuntimeError("The configured selected checkpoint was not persisted.")
        update, checkpoint_id = matches[0].update, matches[0].checkpoint_id
        score, metric = None, "configured_final_update"
    atomic_json(
        training_root / "selection.json",
        {
            "schema_version": 1,
            "compiled_run_id": contract.compiled_run_id,
            "policy": config.training.checkpoint_selection,
            "metric": metric,
            "score": score,
            "selected_update": update,
            "selected_checkpoint_id": checkpoint_id,
            "candidate_updates": [item.update for item in eligible_checkpoints],
        },
    )


def _loss(
    model: CountSDEModel,
    problem: dict[str, torch.Tensor],
    config: ResolvedConfig,
    update: int,
    decoder_data: _GeneDecoderData | None,
    state_split: _StateSplit,
) -> torch.Tensor:
    phase = (
        "state"
        if config.intent is RunIntent.COUNT_STATE or update % 2 == 1
        else "complete_count_block"
    )
    candidates = state_split.fit_indices
    generator = torch.Generator(device=problem["source_z"].device)
    generator.manual_seed(config.training.seed + update)
    order = candidates[
        torch.randperm(len(candidates), generator=generator, device=candidates.device)
    ]
    indices = order[: config.training.state_batch_size]
    source = problem["source_z"][indices]
    duration = problem["duration"][indices]
    target = problem["target_index"][indices]
    pool = problem["pool_index"][indices]
    control = problem["is_control"][indices].bool()
    context_mode = "pool_dynamic" if config.intent is RunIntent.COUNT_CONTEXT else "source_fixed"
    pool_state_mean: torch.Tensor | None = None
    pool_log_mass: torch.Tensor | None = None
    if config.intent is RunIntent.COUNT_CONTEXT:
        bank = DynamicPoolBank.from_series(
            problem["source_z"],
            problem["source_counts"].float() + config.source_smoothing,
            problem["pool_index"],
            pool_count=config.model.pool_count,
        )
        pool_state_mean, pool_log_mass = bank.state_mean, bank.log_mass
    if config.intent is RunIntent.COUNT_CONTEXT:
        prediction = source + duration[:, None] * model.drift(
            target,
            pool,
            control,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
            z=source,
        )
    else:
        prediction = _deterministic_prediction(
            model, source, duration, target, pool, control, problem["grid_steps"][indices]
        )
    state_error = prediction - problem["terminal_z"][indices]
    state_loss = _target_balanced_mse(
        state_error,
        target.long(),
        _support_reliability(problem, indices, config),
    )
    if phase == "state":
        train_state = decoder_data is None or config.training.train_state_with_gene_decoder
        state_parameters = [model.base_drift, model.target_drift, model.target_selection]
        if config.model.source_conditioned_anchor and train_state:
            assert model.source_anchor_hidden is not None
            assert model.source_anchor_output is not None
            state_parameters.extend(
                [
                    model.source_anchor_hidden.weight,
                    model.source_anchor_hidden.bias,
                    model.source_anchor_output.weight,
                ]
            )
        if config.model.state_dependent_drift:
            state_parameters.extend(
                [model.state_drift_hidden.weight, model.state_drift_output.weight]
            )
        if config.model.shared_diffusion:
            state_parameters.append(model.log_diffusion)
        if config.intent is RunIntent.COUNT_CONTEXT:
            state_parameters.extend(
                [
                    model.state_pool_projection,
                    model.state_mass_context,
                    model.context_to_state,
                ]
            )
        state_objective = (
            state_loss.detach()
            if (config.model.terminal_anchor_drift and not train_state)
            else state_loss
        )
        loss = state_objective + 1e-4 * sum(
            parameter.square().mean() for parameter in state_parameters
        )
        if config.training.target_drift_penalty:
            noncontrol_targets = torch.unique(target[~control].long())
            if len(noncontrol_targets):
                target_displacement = duration.mean() * model.target_drift[noncontrol_targets]
                loss = loss + (
                    config.training.target_drift_penalty * target_displacement.square().mean()
                )
        if config.training.source_drift_penalty and config.model.state_dependent_drift:
            source_displacement = duration[:, None] * model.state_drift_output(
                torch.tanh(model.state_drift_hidden(source))
            )
            loss = loss + config.training.source_drift_penalty * source_displacement.square().mean()
        if (
            config.training.source_drift_penalty
            and config.model.source_conditioned_anchor
            and train_state
        ):
            assert model.source_anchor_hidden is not None
            assert model.source_anchor_output is not None
            source_displacement = config.model.source_anchor_residual_scale * torch.tanh(
                model.source_anchor_output(torch.tanh(model.source_anchor_hidden(source)))
            )
            loss = loss + config.training.source_drift_penalty * source_displacement.square().mean()
        if decoder_data is not None:
            decoder_loss = _gene_decoder_loss(
                model, decoder_data, config, update, problem["source_z"].device
            )
            loss = loss + config.training.gene_decoder_loss_weight * decoder_loss
    else:
        # Exact count events use complete pool blocks and connect only to the
        # scalar-fitness channel. State-dynamics tensors are structurally absent
        # from this graph, so a count update cannot reopen drift or diffusion.
        raw = model.raw_fitness(
            problem["target_index"],
            problem["pool_index"],
            problem["is_control"].bool(),
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
        )
        count_loss = exact_count_loss(
            problem["terminal_counts"],
            problem["source_counts"],
            raw,
            problem["duration"],
            problem["pool_index"],
            model.log_concentration,
        )
        count_parameters = [
            model.pool_reference_fitness,
            model.target_fitness,
            model.log_concentration,
        ]
        if config.model.source_efficacy_sensitivity:
            count_parameters.append(model.guide_efficacy_logit)
        if config.intent is RunIntent.COUNT_CONTEXT:
            count_parameters.extend(
                [
                    model.fitness_pool_projection,
                    model.fitness_mass_context,
                    model.target_fitness_context,
                ]
            )
        loss = count_loss / max(float(problem["terminal_counts"].sum()), 1.0)
        loss = loss + 1e-4 * sum(parameter.square().mean() for parameter in count_parameters)
    return loss


def _new_model_optimizer(
    config: ResolvedConfig, device: torch.device
) -> tuple[CountSDEModel, torch.optim.Optimizer]:
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    random.seed(config.training.seed)
    torch.use_deterministic_algorithms(config.training.deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = config.training.deterministic
        torch.backends.cudnn.benchmark = not config.training.deterministic
    model = CountSDEModel(config.model, config.intent).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
    )
    return model, optimizer


def train_model(
    config_path: Path,
    *,
    device: str | torch.device | None = None,
    stop_after: int | None = None,
    initial_checkpoint: Path | None = None,
) -> Path:
    from ..prepare.pipeline import load_config

    requested_config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / requested_config.workspace).resolve()
    verify_directory(workspace / "compiled")
    contract, arrays = load_compiled_problem(workspace)
    config = ResolvedConfig.model_validate_json(
        (workspace / "compiled" / "config.json").read_text()
    )
    if config.model_dump(mode="json") != requested_config.model_dump(mode="json"):
        raise ResumeMismatchError("External configuration changed after compilation.")
    if contract.implementation_tree_hash != implementation_tree_hash():
        raise ResumeMismatchError("Implementation changed after compilation.")
    if contract.environment_lock_hash != environment_lock_hash():
        raise ResumeMismatchError("Environment changed after compilation.")
    training_root = workspace / "training"
    if training_root.exists():
        raise FileExistsError("Training attempt already exists; use resume.")
    (training_root / "checkpoints").mkdir(parents=True)
    (training_root / "intent.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "compiled_run_id": contract.compiled_run_id,
                "config_hash": contract.resolved_config_hash,
            }
        )
        + b"\n"
    )
    selected_device = _device(device)
    model, optimizer = _new_model_optimizer(config, selected_device)
    fork_parent: str | None = None
    if initial_checkpoint is not None:
        verify_directory(initial_checkpoint)
        parent = CheckpointManifest.model_validate_json(
            (initial_checkpoint / "checkpoint.json").read_text()
        )
        state = load_tensor_file(initial_checkpoint / "model.safetensors", device=selected_device)
        model.load_state_dict(state, strict=True)
        fork_parent = parent.checkpoint_id
    problem = _tensor_problem(arrays, selected_device)
    state_split = _state_split(arrays, config, selected_device)
    decoder_data = _gene_decoder_data(root, workspace, config)
    initialization = (
        _initialize_state_channels(model, problem, state_split.fit_indices, config)
        if initial_checkpoint is None
        else {}
    )
    end = min(stop_after or config.training.max_updates, config.training.max_updates)
    checkpoint: CheckpointManifest | None = None
    loss_value = float("nan")
    if config.training.analytic_fit:
        diagnostics = {
            **initialization,
            **_training_diagnostics(model, problem, state_split, decoder_data),
        }
        _checkpoint(
            workspace,
            training_root,
            contract,
            model,
            optimizer,
            update=1,
            loss=diagnostics["full_state_rmse"] ** 2,
            diagnostics=diagnostics,
            parent_checkpoint_id=fork_parent,
        )
        return training_root
    checkpoints: list[CheckpointManifest] = []
    for update in range(1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, problem, config, update, decoder_data, state_split)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {update}.")
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if update % config.training.checkpoint_every == 0 or update == end:
            diagnostics = {
                **initialization,
                **_training_diagnostics(model, problem, state_split, decoder_data),
            }
            checkpoint = _checkpoint(
                workspace,
                training_root,
                contract,
                model,
                optimizer,
                update=update,
                loss=loss_value,
                diagnostics=diagnostics,
                parent_checkpoint_id=checkpoint.checkpoint_id if checkpoint else fork_parent,
            )
            checkpoints.append(checkpoint)
    if end == config.training.max_updates:
        _write_selection(training_root, contract, config, checkpoints)
    return training_root


def _load_latest(
    workspace: Path,
    config: ResolvedConfig,
    device: torch.device,
) -> tuple[
    CompiledRunContract,
    dict[str, np.ndarray],
    CountSDEModel,
    torch.optim.Optimizer,
    CheckpointManifest,
]:
    contract, arrays = load_compiled_problem(workspace)
    training_root = workspace / "training"
    intent = json.loads((training_root / "intent.json").read_text())
    if (
        intent["compiled_run_id"] != contract.compiled_run_id
        or intent["config_hash"] != contract.resolved_config_hash
    ):
        raise ResumeMismatchError("Compiled contract changed across resume.")
    latest = json.loads((training_root / "checkpoints" / "latest.json").read_text())
    generation = training_root / latest["relative_uri"]
    manifest = CheckpointManifest.model_validate_json((generation / "checkpoint.json").read_text())
    model, optimizer = _new_model_optimizer(config, device)
    state = load_tensor_file(generation / "model.safetensors", device=device)
    model.load_state_dict(state, strict=True)
    opt_tensors = load_tensor_file(generation / "optimizer.safetensors", device=device)
    opt_tree = json.loads((generation / "optimizer-tree.json").read_text())
    _restore_optimizer(model, optimizer, opt_tensors, opt_tree)
    rng = load_tensor_file(generation / "rng.safetensors")
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if device.type == "cuda":
        states = [rng[key].cpu() for key in sorted(rng) if key.startswith("torch_cuda_")]
        if states:
            torch.cuda.set_rng_state_all(states)
    return contract, arrays, model, optimizer, manifest


def resume_training(config_path: Path, *, device: str | torch.device | None = None) -> Path:
    from ..prepare.pipeline import load_config

    requested_config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / requested_config.workspace).resolve()
    verify_directory(workspace / "compiled")
    compiled_config = ResolvedConfig.model_validate_json(
        (workspace / "compiled" / "config.json").read_text()
    )
    if compiled_config.model_dump(mode="json") != requested_config.model_dump(mode="json"):
        raise ResumeMismatchError("External configuration changed after compilation.")
    contract, _ = load_compiled_problem(workspace)
    if contract.implementation_tree_hash != implementation_tree_hash():
        raise ResumeMismatchError("Implementation changed after compilation.")
    if contract.environment_lock_hash != environment_lock_hash():
        raise ResumeMismatchError("Environment changed after compilation.")
    config = compiled_config
    selected_device = _device(device)
    contract, arrays, model, optimizer, previous = _load_latest(workspace, config, selected_device)
    if previous.compiled_run_id != contract.compiled_run_id:
        raise ResumeMismatchError("Checkpoint belongs to another compiled run.")
    if previous.update >= config.training.max_updates:
        raise ResumeMismatchError("Training is already complete under the compiled budget.")
    problem = _tensor_problem(arrays, selected_device)
    state_split = _state_split(arrays, config, selected_device)
    decoder_data = _gene_decoder_data(root, workspace, config)
    training_root = workspace / "training"
    checkpoint = previous
    checkpoints: list[CheckpointManifest] = []
    for generation in sorted((training_root / "checkpoints").glob("generation-*")):
        checkpoints.append(
            CheckpointManifest.model_validate_json((generation / "checkpoint.json").read_text())
        )
    for update in range(previous.update + 1, config.training.max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, problem, config, update, decoder_data, state_split)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {update}.")
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if update % config.training.checkpoint_every == 0 or update == config.training.max_updates:
            diagnostics = _training_diagnostics(model, problem, state_split, decoder_data)
            checkpoint = _checkpoint(
                workspace,
                training_root,
                contract,
                model,
                optimizer,
                update=update,
                loss=float(loss.detach().cpu()),
                diagnostics=diagnostics,
                parent_checkpoint_id=checkpoint.checkpoint_id,
            )
            checkpoints.append(checkpoint)
    _write_selection(training_root, contract, config, checkpoints)
    return training_root


def load_training_state(
    config_path: Path, *, device: str | torch.device = "cpu"
) -> tuple[CompiledRunContract, dict[str, np.ndarray], CountSDEModel, CheckpointManifest]:
    from ..prepare.pipeline import load_config

    config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / config.workspace).resolve()
    contract, arrays, model, _, checkpoint = _load_latest(workspace, config, _device(device))
    if config.training.checkpoint_selection in {
        "minimum_gene_decoder_validation",
        "minimum_state_validation",
    }:
        selection = json.loads((workspace / "training/selection.json").read_text())
        if selection.get("compiled_run_id") != contract.compiled_run_id:
            raise ResumeMismatchError("Checkpoint selection belongs to another compiled run.")
        selected_update = int(selection["selected_update"])
    else:
        selected_update = config.training.selected_update or config.training.max_updates
    if checkpoint.update != selected_update:
        generation = workspace / "training/checkpoints" / f"generation-{selected_update:09d}"
        verify_directory(generation)
        checkpoint = CheckpointManifest.model_validate_json(
            (generation / "checkpoint.json").read_text()
        )
        if checkpoint.compiled_run_id != contract.compiled_run_id:
            raise ResumeMismatchError("Selected checkpoint belongs to another compiled run.")
        state = load_tensor_file(generation / "model.safetensors", device=device)
        model.load_state_dict(state, strict=True)
    return contract, arrays, model, checkpoint
