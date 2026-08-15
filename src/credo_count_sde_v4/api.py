"""Canonical Python lifecycle API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .canonical import sha256_file
from .compat.credo3 import verify_frozen_credo
from .compile import compile_problem
from .contracts import (
    CompiledRunContract,
    CountStoreManifest,
    EvaluationBundleManifest,
    InferenceBundleManifest,
    LifecycleState,
    PreparedRepresentation,
    ResolvedConfig,
    SealedRunManifest,
    SelectionManifest,
    SemanticStudySnapshot,
    StateSelectionCalibration,
    VerifyLevel,
)
from .errors import IntegrityError
from .evaluation import evaluate_run, seal_run
from .inference import V4Run, finalize_inference, open_inference_run
from .persistence import LifecycleLedger, verify_directory
from .prepare import prepare_representation
from .store import CountStore
from .training import resume_training, train_model


def _supported_preflight() -> None:
    """Bind every supported lifecycle call to the exact frozen dependency."""

    verify_frozen_credo()


def resolve_config(config_path: Path) -> ResolvedConfig:
    """Parse and fully default the strict run configuration."""

    _supported_preflight()
    from .prepare.pipeline import load_config

    return load_config(config_path)


def validate_contract(path: Path) -> dict[str, Any]:
    """Validate canonical JSON syntax and reject non-object contracts."""

    _supported_preflight()
    from .canonical import canonical_json_bytes, sha256_file

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError("A contract must be a JSON object with schema_version.")
    canonical_json_bytes(payload)
    discriminators = (
        ("compiled_run_id", CompiledRunContract),
        ("prepared_id", PreparedRepresentation),
        ("evaluation_id", EvaluationBundleManifest),
        ("run_id", InferenceBundleManifest),
        ("sealed_id", SealedRunManifest),
        ("selection_id", SelectionManifest),
        ("calibration_id", StateSelectionCalibration),
        ("store_id", CountStoreManifest),
        ("study_id", SemanticStudySnapshot),
    )
    selected = next((model for field, model in discriminators if field in payload), None)
    if selected is None:
        raise ValueError("Unknown contract discriminator.")
    selected.model_validate(payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload["schema_version"],
        "contract_type": selected.__name__,
    }


def estimate(config_path: Path) -> dict[str, Any]:
    """Return a conservative plan-size receipt without reading endpoints."""

    config = resolve_config(config_path)
    model = config.model
    state_parameters = model.state_dim * (1 + model.target_count)
    measure_parameters = model.pool_count + model.target_count + model.pool_count
    context_parameters = (
        model.context_rank * (2 * model.pool_count + model.target_count + model.state_dim)
        if config.intent.value == "count_context"
        else 0
    )
    return {
        "schema_version": 1,
        "intent": config.intent.value,
        "approximate_trainable_parameters": state_parameters
        + measure_parameters
        + context_parameters,
        "training_updates": config.training.max_updates,
        "evaluation_particle_steps_per_series": config.evaluation.particles
        * config.evaluation.steps,
        "hard_output_bytes": config.evaluation.output_bytes_limit,
    }


def _workspace(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text())
    return (config_path.parent.resolve() / payload["workspace"]).resolve()


def _ledger(config_path: Path) -> LifecycleLedger:
    return LifecycleLedger(_workspace(config_path) / "ledger" / "events.jsonl")


def _trained_artifact_id(training_root: Path) -> str:
    selection = training_root / "selection.json"
    if selection.exists():
        return str(json.loads(selection.read_text())["selected_checkpoint_id"])
    latest = json.loads((training_root / "checkpoints" / "latest.json").read_text())
    return str(latest["checkpoint_id"])


def prepare(config_path: Path) -> Path:
    _supported_preflight()
    result = prepare_representation(config_path)
    manifest = json.loads((result / "prepared.json").read_text())
    _ledger(config_path).transition(LifecycleState.PREPARED, artifact_id=manifest["prepared_id"])
    return result


def compile_run(config_path: Path) -> Path:
    _supported_preflight()
    result = compile_problem(config_path)
    manifest = json.loads((result / "contract.json").read_text())
    _ledger(config_path).transition(
        LifecycleState.COMPILED, artifact_id=manifest["compiled_run_id"]
    )
    return result


def train(config_path: Path, *, device: str | None = None) -> Path:
    _supported_preflight()
    result = train_model(config_path, device=device)
    _ledger(config_path).transition(
        LifecycleState.TRAINED, artifact_id=_trained_artifact_id(result)
    )
    return result


def fork(config_path: Path, *, from_checkpoint: Path, device: str | None = None) -> Path:
    """Start a new compiled attempt from compatible inference weights only."""

    _supported_preflight()
    result = train_model(config_path, device=device, initial_checkpoint=from_checkpoint)
    _ledger(config_path).transition(
        LifecycleState.TRAINED,
        artifact_id=_trained_artifact_id(result),
        details={"fork_parent": str(from_checkpoint)},
    )
    return result


def resume(config_path: Path, *, device: str | None = None) -> Path:
    _supported_preflight()
    result = resume_training(config_path, device=device)
    if _ledger(config_path).state() is LifecycleState.COMPILED:
        _ledger(config_path).transition(
            LifecycleState.TRAINED, artifact_id=_trained_artifact_id(result)
        )
    return result


def finalize(config_path: Path) -> Path:
    _supported_preflight()
    result = finalize_inference(config_path)
    manifest = json.loads((result / "inference.json").read_text())
    _ledger(config_path).transition(LifecycleState.FINALIZED, artifact_id=manifest["run_id"])
    return result


def open_run(path: Path, *, device: str = "cpu", verify: str = "full") -> V4Run:
    _supported_preflight()
    return open_inference_run(path, device=device, verify=verify)


def evaluate(config_path: Path, *, device: str = "cpu") -> Path:
    _supported_preflight()
    result = evaluate_run(config_path, device=device)
    manifest = json.loads((result / "evaluation.json").read_text())
    _ledger(config_path).transition(LifecycleState.EVALUATED, artifact_id=manifest["evaluation_id"])
    return result


def seal(config_path: Path) -> Path:
    _supported_preflight()
    result = seal_run(config_path)
    manifest = json.loads((result / "sealed.json").read_text())
    _ledger(config_path).transition(LifecycleState.SEALED, artifact_id=manifest["sealed_id"])
    return result


def verify(path: Path, *, level: str | VerifyLevel = VerifyLevel.CONTENT) -> dict[str, Any]:
    _supported_preflight()
    selected = VerifyLevel(level)
    result: dict[str, Any] = {"root": str(path), "level": selected.value}
    result["manifest"] = verify_directory(path)
    workspace = path.parent
    if (path / "sealed.json").exists():
        sealed = SealedRunManifest.model_validate_json((path / "sealed.json").read_text())
        inference_root = workspace / "inference"
        evaluation_root = workspace / "evaluation"
        verify_directory(inference_root)
        verify_directory(evaluation_root)
        if sha256_file(inference_root / "inference.json") != sealed.inference.sha256:
            raise IntegrityError("Sealed inference reference does not match sibling bundle.")
        if len(sealed.evaluations) != 1 or (
            sha256_file(evaluation_root / "evaluation.json") != sealed.evaluations[0].sha256
        ):
            raise IntegrityError("Sealed evaluation reference does not match sibling bundle.")
        if sha256_file(path / "claim-audit.json") != sealed.claim_audit.sha256:
            raise IntegrityError("Sealed claim-audit reference mismatch.")
    if selected in {VerifyLevel.CONTENT, VerifyLevel.RELOAD, VerifyLevel.RESUME, VerifyLevel.FULL}:
        if (workspace / "compiled" / "config.json").exists():
            config = json.loads((workspace / "compiled" / "config.json").read_text())
            config_root = Path(config.get("_config_root", workspace.parent))
            count_path = config_root / config["count_store"]
            if not count_path.exists():
                # Normal resolved configs use a path relative to the original
                # config root, which is the workspace parent in synthetic runs.
                count_path = workspace.parent / config["count_store"]
            if count_path.exists():
                result["count_store"] = (
                    CountStore(count_path).verify(full=True).model_dump(mode="json")
                )
    if selected in {VerifyLevel.RELOAD, VerifyLevel.FULL} and (workspace / "inference").exists():
        run = open_inference_run(workspace / "inference", verify="full")
        mean, mass, _ = run.terminal(particles=8, steps=2)
        result["reload"] = {"series": len(mean), "finite_mass": bool((mass > 0).all())}
    return result
