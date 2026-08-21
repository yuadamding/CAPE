from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError
from scipy import sparse

from credo_count_sde_v4.canonical import atomic_json, contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    G00CImplementationAuthorityV2,
    G00CImplementationBindingV2,
    G00CMaterializationReceiptV3,
    G00CPublicationArtifactV3,
    G00CPublicationFreezeV1,
    G00CPublicationManifestV3,
    G00CRefitReplayReceiptV3,
    G00CSamplerEvidenceV3,
    G00CSamplerPlanEntryV3,
    G00CSamplerPlanV3,
    G00CSupportAuditContractV2,
    StrictModel,
)
from credo_count_sde_v4.store import (
    derive_refit_seed_schedule,
    g00c_publication_v3,
    g00c_refit_replay_v3,
    g00c_sampler_v3,
    g00c_v4,
)
from credo_count_sde_v4.store.g00c_materialization_v3 import (
    verify_g00c_materialization_v3,
)
from credo_count_sde_v4.store.g00c_publication_v3 import (
    expected_publication_payload_v3,
    publish_g00c_v3,
    verify_g00c_publication_v3,
)
from credo_count_sde_v4.store.g00c_refit_replay_v3 import (
    recompute_refit_rows_v3,
    verify_g00c_refit_replay_v3,
)
from credo_count_sde_v4.store.g00c_sampler_v3 import (
    derive_support_table_v3,
    replay_sampler_plan_v3,
    verify_g00c_sampler_v3,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "0" * 64
    normalized = model.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in model.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _artifact(name: str, digest: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.dev36",
        schema_version=1,
        sha256=digest * 64,
        size_bytes=1,
        media_type="application/octet-stream",
        relative_uri=name,
    )


def _file_artifact(root: Path, name: str, content: bytes) -> ArtifactRef:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef(
        schema_id="test.dev36",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type="application/octet-stream",
        relative_uri=name,
    )


def _json_artifact(root: Path, name: str, payload: Any) -> ArtifactRef:
    content = (
        payload.model_dump_json().encode()
        if isinstance(payload, StrictModel)
        else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    return _file_artifact(root, name, content).model_copy(update={"media_type": "application/json"})


def _parquet_artifact(root: Path, name: str, table: pd.DataFrame) -> ArtifactRef:
    path = root / name
    table.to_parquet(path, index=False)
    return ArtifactRef(
        schema_id="test.dev36",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type="application/vnd.apache.parquet",
        relative_uri=name,
    )


def test_dev36_implementation_authority_requires_distinct_bound_materializer() -> None:
    roles = (
        "feature_ranking",
        "residual_frequency",
        "refit",
        "sampler",
        "support_auditor",
        "monitor",
        "materialization_verifier",
        "refit_replay_verifier",
        "publication_verifier",
        "execution_verifier",
        "decision_verifier",
    )
    bindings = tuple(
        G00CImplementationBindingV2(
            role=role,  # type: ignore[arg-type]
            artifact=_artifact(f"{role}.py", "a" if role == "materialization_verifier" else "b"),
        )
        for role in roles
    )
    with pytest.raises(ValidationError, match="distinct source artifact"):
        G00CImplementationAuthorityV2(
            dev36_code_commit="1" * 40,
            wheel=_artifact("dev36.whl", "c"),
            normalized_sdist=_artifact("dev36.tar.gz", "d"),
            implementation_tree_sha256="e" * 64,
            environment_lock=_artifact("environment.json", "f"),
            environment_kind="exact_local_lock",
            execution_environment_digest=f"sha256:{'0' * 64}",
            implementations=tuple(
                binding.model_copy(update={"artifact": _artifact(f"{binding.role}.py", "b")})
                for binding in bindings
            ),
        )


def test_dev36_sampler_exact_resume_and_derived_zero_support() -> None:
    schedule = derive_refit_seed_schedule("1" * 64)
    plan = _identified(
        G00CSamplerPlanV3,
        {
            "execution_authority_id": "2" * 64,
            "seed_schedule_id": schedule.schedule_id,
            "entries": (
                G00CSamplerPlanEntryV3(
                    candidate_kind="training_cells",
                    candidate_value=4,
                    refit_draw_id=0,
                    macro_updates=2,
                ),
            ),
            "resume_after_macro_update": 1,
        },
        "plan_id",
    )
    hierarchy = pd.DataFrame(
        {
            "row_id": (10, 11, 12, 13, 14),
            "source_index": (0, 0, 1, 1, 1),
            "target_code": (0, 1, 0, 1, 2),
            "guide_code": (0, 1, 2, 3, 4),
            "is_control": (True, False, False, False, False),
        }
    )
    order = np.asarray((10, 11, 12, 13, 14), dtype=np.int64)
    uninterrupted, state = replay_sampler_plan_v3(plan, hierarchy, order, schedule, resumed=False)
    resumed, resumed_state = replay_sampler_plan_v3(plan, hierarchy, order, schedule, resumed=True)
    assert uninterrupted.equals(resumed)
    assert state.equals(resumed_state)
    # Frozen Dev37 PCG64DXSM characterization: the optimized prefix cache and
    # typed trace construction must not change the historical stream.
    assert uninterrupted["row_id"].iloc[:16].tolist() == [
        13,
        13,
        13,
        13,
        13,
        13,
        12,
        10,
        11,
        12,
        13,
        10,
        12,
        10,
        10,
        10,
    ]
    assert uninterrupted["thinning_draw"].iloc[:8].tolist() == [
        6290477717619121792,
        13064860109156429178,
        5057985643900009592,
        7633956249123397418,
        6611851478989297693,
        1200645861071004140,
        12283433943111019792,
        1443012210241635411,
    ]
    assert state["sampler_state_sha256"].tolist() == [
        "0d4642ee0574757215e0e72414deb37ec4f187860ae01d2aab70ed791d38f93c",
        "ea501171180b4f08f969f545ed30599a310aae440a856ca483669ad8b715c6db",
    ]
    assert tuple(uninterrupted["microbatch"].iloc[:4096].unique()) == tuple(range(8))
    assert np.all(uninterrupted["inverse_probability_weight"].to_numpy() > 0)
    contract = _identified(
        G00CSupportAuditContractV2,
        {
            "stage": "extension",
            "feature_candidate_counts": (),
            "cell_candidate_counts": (2_000_000,),
        },
        "contract_id",
    )
    # A model-constructed extension entry lets the small fixture exercise the
    # same support derivation without pretending to be a real frozen grid.
    small_entry = G00CSamplerPlanEntryV3(
        candidate_kind="training_cells",
        candidate_value=4,
        refit_draw_id=0,
        macro_updates=2,
    )
    small_plan = plan.model_copy(update={"entries": (small_entry,)})
    small_contract = contract.model_copy(update={"cell_candidate_counts": (4,)})
    support = derive_support_table_v3(small_contract, small_plan, hierarchy, order, uninterrupted)
    assert not support["zero_support"].any()
    corrupted = uninterrupted.copy()
    corrupted.loc[0, "inverse_probability_weight"] *= 2
    changed = derive_support_table_v3(small_contract, small_plan, hierarchy, order, corrupted)
    assert not changed.equals(support)
    no_target_one = uninterrupted.loc[uninterrupted["target_code"] != 1]
    zero_support = derive_support_table_v3(
        small_contract, small_plan, hierarchy, order, no_target_one
    )
    missing = zero_support.loc[
        (zero_support["dimension"] == "target") & (zero_support["stratum_id"] == "1")
    ]
    assert missing["zero_support"].item()
    assert not missing["selection_eligible"].item()
    with pytest.raises(Exception, match="omits a support-audit candidate"):
        derive_support_table_v3(
            small_contract.model_copy(update={"cell_candidate_counts": (5,)}),
            small_plan,
            hierarchy,
            order,
            uninterrupted,
        )
    with pytest.raises(Exception, match="no support observations"):
        derive_support_table_v3(
            small_contract,
            small_plan,
            hierarchy,
            order,
            uninterrupted.iloc[0:0],
        )
    with pytest.raises(Exception, match="exceeds the frozen nested row order"):
        g00c_sampler_v3._entry_rows(
            small_entry.model_copy(update={"candidate_value": 6}),
            order,
        )


def test_dev36_sampler_evidence_is_replayed_from_artifacts(tmp_path: Path) -> None:
    schedule = derive_refit_seed_schedule("3" * 64)
    schedule_ref = _json_artifact(tmp_path, "schedule.json", schedule)
    hierarchy = pd.DataFrame(
        {
            "row_id": (20, 21, 22, 23),
            "source_index": (0, 0, 1, 1),
            "target_code": (0, 1, 0, 1),
            "guide_code": (0, 1, 2, 3),
            "is_control": (True, False, False, False),
        }
    )
    hierarchy_ref = _parquet_artifact(tmp_path, "hierarchy.parquet", hierarchy)
    order = pd.DataFrame({"rank": (1, 2, 3, 4), "row_id": (20, 21, 22, 23)})
    order_ref = _parquet_artifact(tmp_path, "order.parquet", order)
    plan = _identified(
        G00CSamplerPlanV3,
        {
            "execution_authority_id": "4" * 64,
            "seed_schedule_id": schedule.schedule_id,
            "entries": (
                G00CSamplerPlanEntryV3(
                    candidate_kind="training_cells",
                    candidate_value=4,
                    refit_draw_id=0,
                    macro_updates=2,
                ),
            ),
            "resume_after_macro_update": 1,
        },
        "plan_id",
    )
    plan_ref = _json_artifact(tmp_path, "plan.json", plan)
    uninterrupted, states = replay_sampler_plan_v3(
        plan, hierarchy, order["row_id"].to_numpy(), schedule, resumed=False
    )
    resumed, resumed_states = replay_sampler_plan_v3(
        plan, hierarchy, order["row_id"].to_numpy(), schedule, resumed=True
    )
    uninterrupted_ref = _parquet_artifact(tmp_path, "draws.parquet", uninterrupted)
    resumed_ref = _parquet_artifact(tmp_path, "resumed-draws.parquet", resumed)
    state_ref = _parquet_artifact(tmp_path, "states.parquet", states)
    resumed_state_ref = _parquet_artifact(tmp_path, "resumed-states.parquet", resumed_states)
    implementation_ref = _file_artifact(tmp_path, "sampler.py", b"# exact sampler\n")
    authority = SimpleNamespace(
        authority_id="4" * 64,
        sampler_row_hierarchy=hierarchy_ref,
        sampler_hierarchy_rows=4,
        nested_training_row_order=order_ref,
        seed_schedule=schedule_ref,
        implementation=SimpleNamespace(
            implementations=(SimpleNamespace(role="sampler", artifact=implementation_ref),)
        ),
    )
    evidence = _identified(
        G00CSamplerEvidenceV3,
        {
            "execution_authority_id": authority.authority_id,
            "sampler_plan": plan_ref,
            "sampler_plan_id": plan.plan_id,
            "row_hierarchy": hierarchy_ref,
            "nested_training_row_order": order_ref,
            "uninterrupted_draw_trace": uninterrupted_ref,
            "resumed_draw_trace": resumed_ref,
            "uninterrupted_state_trace": state_ref,
            "resumed_state_trace": resumed_state_ref,
            "implementation_sha256": implementation_ref.sha256,
        },
        "evidence_id",
    )
    verified_plan, _, _, verified_trace = verify_g00c_sampler_v3(
        tmp_path,
        authority,
        evidence,  # type: ignore[arg-type]
    )
    assert verified_plan == plan
    assert verified_trace.equals(uninterrupted)
    with pytest.raises(Exception, match="exact frozen grids"):
        verify_g00c_sampler_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            evidence,
            expected_candidate_values={"training_cells": (4,)},
        )
    corrupted = uninterrupted.copy()
    corrupted.loc[0, "row_id"] = 999
    bad_ref = _parquet_artifact(tmp_path, "bad-draws.parquet", corrupted)
    with pytest.raises(Exception, match="differs from executable replay"):
        verify_g00c_sampler_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            evidence.model_copy(update={"uninterrupted_draw_trace": bad_ref}),
        )


def test_dev36_refit_replay_recomputes_and_rejects_coherent_false_surface(
    tmp_path: Path,
) -> None:
    records = pd.DataFrame(
        {
            "candidate_kind": ("feature_count", "feature_count"),
            "draw_id": (0, 0),
            "candidate_value": (256, 4096),
            "validation_total_count": (7, 7),
            "validation_nll_sum": (1.0, 1.0),
            "validation_nll_per_count": (1 / 7, 1 / 7),
            "final_state_hash": ("0" * 64, "1" * 64),
        }
    )
    training = np.zeros((2, 4096), dtype=np.int64)
    validation = np.zeros((2, 4096), dtype=np.int64)
    training[:, 0] = (20, 40)
    validation[:, :2] = ((3, 4), (2, 5))
    statistics = tmp_path / "statistics.npz"
    np.savez(
        statistics,
        candidate_value=np.asarray((256, 4096)),
        draw_id=np.asarray((0, 0)),
        training_counts=training,
        validation_counts=validation,
    )
    receipt = G00CRefitReplayReceiptV3.model_construct(
        candidate_kind="feature_count",
        selected_candidate=256,
        reference_candidate=4096,
        modeled_feature_count=4096,
        preregistered_audit_draw_ids=(0, 6, 12, 18, 24, 30, 36, 42, 48, 58),
    )
    recomputed = recompute_refit_rows_v3(records, receipt, statistics)
    assert len(recomputed) == 2
    assert not np.array_equal(
        recomputed["validation_nll_sum"].to_numpy(),
        records["validation_nll_sum"].to_numpy(),
    )
    assert recomputed["final_state_hash"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_dev36_refit_sufficient_statistics_fail_closed(tmp_path: Path) -> None:
    records = pd.DataFrame(
        {
            "candidate_kind": ("feature_count",),
            "draw_id": (0,),
            "candidate_value": (256,),
        }
    )
    receipt = SimpleNamespace(
        candidate_kind="feature_count",
        selected_candidate=256,
        reference_candidate=4096,
        modeled_feature_count=4096,
        preregistered_audit_draw_ids=(),
    )

    def save(name: str, **values: np.ndarray) -> Path:
        path = tmp_path / name
        np.savez(path, **values)
        return path

    with pytest.raises(Exception, match="another schema"):
        recompute_refit_rows_v3(
            records, receipt, save("schema.npz", candidate_value=np.array([256]))
        )
    base = {
        "candidate_value": np.asarray((256,), dtype=np.int64),
        "draw_id": np.asarray((0,), dtype=np.int64),
        "training_counts": np.zeros((1, 4096), dtype=np.int64),
        "validation_counts": np.zeros((1, 4096), dtype=np.int64),
    }
    malformed = dict(base)
    malformed["training_counts"] = np.zeros((1, 8), dtype=np.int64)
    with pytest.raises(Exception, match="malformed"):
        recompute_refit_rows_v3(records, receipt, save("shape.npz", **malformed))
    duplicate = {key: np.repeat(value, 2, axis=0) for key, value in base.items()}
    with pytest.raises(Exception, match="duplicate identities"):
        recompute_refit_rows_v3(records, receipt, save("duplicate.npz", **duplicate))
    missing = dict(base)
    missing["candidate_value"] = np.asarray((512,), dtype=np.int64)
    with pytest.raises(Exception, match="lacks required"):
        recompute_refit_rows_v3(records, receipt, save("missing.npz", **missing))
    with pytest.raises(Exception, match="denominator must be positive"):
        recompute_refit_rows_v3(records, receipt, save("zero.npz", **base))
    unreadable = tmp_path / "unreadable.npz"
    unreadable.write_bytes(b"not npz")
    with pytest.raises(Exception, match="cannot be read"):
        recompute_refit_rows_v3(records, receipt, unreadable)
    with pytest.raises(Exception, match="no unique refit_replay_verifier"):
        g00c_refit_replay_v3._implementation_hash(
            SimpleNamespace(implementation=SimpleNamespace(implementations=())),  # type: ignore[arg-type]
            "refit_replay_verifier",
        )


def test_dev36_refit_receipt_executes_all_selected_and_reference_draws(
    tmp_path: Path,
) -> None:
    schedule = derive_refit_seed_schedule("5" * 64)
    schedule_ref = _json_artifact(tmp_path, "refit-schedule.json", schedule)
    rows: list[dict[str, object]] = []
    for seed in schedule.records:
        for candidate in (256, 4096):
            rows.append(
                {
                    "candidate_kind": "feature_count",
                    "draw_id": seed.draw_id,
                    "candidate_value": candidate,
                    "initialization": seed.initialization,
                    "training_sampler": seed.training_sampler,
                    "thinning": seed.thinning,
                    "validation_evaluation": seed.validation_evaluation,
                    "stochastic_optimizer_or_augmentation": (
                        seed.stochastic_optimizer_or_augmentation
                    ),
                    "restart_interruption_point": seed.restart_interruption_point,
                    "fit_row_hash": "1" * 64,
                    "validation_row_hash": "2" * 64,
                    "model_config_hash": "3" * 64,
                    "final_state_hash": "0" * 64,
                    "validation_total_count": 0,
                    "validation_nll_sum": 0.0,
                    "validation_nll_per_count": 0.0,
                    "fit_status": "pass",
                }
            )
    records = pd.DataFrame(rows)
    training = np.zeros((len(records), 4096), dtype=np.int64)
    validation = np.zeros_like(training)
    training[:, 0] = np.arange(1, len(records) + 1)
    validation[:, 0] = 3
    validation[:, 1] = 4
    statistics_path = tmp_path / "refit-statistics.npz"
    np.savez(
        statistics_path,
        candidate_value=records["candidate_value"].to_numpy(),
        draw_id=records["draw_id"].to_numpy(),
        training_counts=training,
        validation_counts=validation,
    )
    statistics_ref = ArtifactRef(
        schema_id="test.dev36",
        schema_version=1,
        sha256=sha256_file(statistics_path),
        size_bytes=statistics_path.stat().st_size,
        media_type="application/x-npz",
        relative_uri=statistics_path.name,
    )
    provisional = G00CRefitReplayReceiptV3.model_construct(
        candidate_kind="feature_count",
        selected_candidate=256,
        reference_candidate=4096,
        modeled_feature_count=4096,
        preregistered_audit_draw_ids=(0, 6, 12, 18, 24, 30, 36, 42, 48, 58),
    )
    replayed = recompute_refit_rows_v3(records, provisional, statistics_path)
    key = records.set_index(["draw_id", "candidate_value"])
    for row in replayed.itertuples(index=False):
        key.loc[(row.draw_id, row.candidate_value), "final_state_hash"] = row.final_state_hash
        key.loc[(row.draw_id, row.candidate_value), "validation_total_count"] = (
            row.validation_total_count
        )
        key.loc[(row.draw_id, row.candidate_value), "validation_nll_sum"] = row.validation_nll_sum
        key.loc[(row.draw_id, row.candidate_value), "validation_nll_per_count"] = (
            row.validation_nll_per_count
        )
    records = key.reset_index()[list(rows[0])]
    records_ref = _parquet_artifact(tmp_path, "refit-records.parquet", records)
    replayed_ref = _parquet_artifact(tmp_path, "replayed.parquet", replayed)
    replay_impl = _file_artifact(tmp_path, "refit-replay.py", b"# replay\n")
    authority = SimpleNamespace(
        authority_id="6" * 64,
        selection_freeze_id="7" * 64,
        seed_schedule_id=schedule.schedule_id,
        seed_schedule=schedule_ref,
        implementation=SimpleNamespace(
            implementations=(SimpleNamespace(role="refit_replay_verifier", artifact=replay_impl),)
        ),
    )
    receipt = _identified(
        G00CRefitReplayReceiptV3,
        {
            "execution_authority_id": authority.authority_id,
            "selection_freeze_id": authority.selection_freeze_id,
            "seed_schedule_id": schedule.schedule_id,
            "candidate_kind": "feature_count",
            "selected_candidate": 256,
            "reference_candidate": 4096,
            "modeled_feature_count": 4096,
            "refit_records": records_ref,
            "sufficient_statistics": statistics_ref,
            "replayed_rows": replayed_ref,
            "replay_implementation_sha256": replay_impl.sha256,
        },
        "receipt_id",
    )
    observed = verify_g00c_refit_replay_v3(
        tmp_path,
        authority,
        receipt,  # type: ignore[arg-type]
    )
    assert observed.equals(replayed)
    false = replayed.copy()
    false.loc[0, "validation_nll_sum"] += 0.25
    false_ref = _parquet_artifact(tmp_path, "false-replayed.parquet", false)
    with pytest.raises(Exception, match="differs from executable refit"):
        verify_g00c_refit_replay_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            receipt.model_copy(update={"replayed_rows": false_ref}),
        )


def test_dev36_publication_inventory_is_status_exact() -> None:
    freeze = G00CPublicationFreezeV1(
        always_required_artifacts=(
            "G00C_REFIT_SEED_SCHEDULE.json",
            "ROW_HASHES.json",
            "FEATURE_SELECTION_CURVE.parquet",
            "CELL_SELECTION_CURVE.parquet",
            "REFIT_PROVENANCE.parquet",
            "REPLAY_AUDIT.json",
            "SUPPORT_AUDIT.parquet",
            "SAMPLER_RESTART.json",
            "DECISION_RECEIPT.json",
            "artifacts.json",
            "COMMITTED",
            "SHA256SUMS",
        ),
        pass_only_required_artifacts=(
            "SELECTED_FEATURES.parquet",
            "SELECTED_ROWS.parquet",
            "compact.h5",
            "puroR-sidecar.h5",
            "PHYSICAL_RUNS.parquet",
            "COMPACT_VERIFICATION.json",
            "WRITER_RESTART.json",
        ),
        extension_required_artifacts=("BASE_GRID_STOP_RECEIPT.json",),
        fail_no_saturation_required_artifacts=(
            "BASE_GRID_STOP_RECEIPT.json",
            "EXTENSION_RESULT.json",
        ),
    )
    always_payload = set(freeze.always_required_artifacts) - {
        "artifacts.json",
        "COMMITTED",
        "SHA256SUMS",
    }
    assert expected_publication_payload_v3(freeze, "pass") == always_payload | set(
        freeze.pass_only_required_artifacts
    )
    assert expected_publication_payload_v3(freeze, "extension_required") == always_payload | {
        "BASE_GRID_STOP_RECEIPT.json"
    }
    assert "compact.h5" not in expected_publication_payload_v3(freeze, "fail_no_saturation")


def test_dev36_publication_verifier_reads_exact_inventory(tmp_path: Path) -> None:
    freeze = G00CPublicationFreezeV1(
        always_required_artifacts=(
            "G00C_REFIT_SEED_SCHEDULE.json",
            "ROW_HASHES.json",
            "FEATURE_SELECTION_CURVE.parquet",
            "CELL_SELECTION_CURVE.parquet",
            "REFIT_PROVENANCE.parquet",
            "REPLAY_AUDIT.json",
            "SUPPORT_AUDIT.parquet",
            "SAMPLER_RESTART.json",
            "DECISION_RECEIPT.json",
            "artifacts.json",
            "COMMITTED",
            "SHA256SUMS",
        ),
        pass_only_required_artifacts=(
            "SELECTED_FEATURES.parquet",
            "SELECTED_ROWS.parquet",
            "compact.h5",
            "puroR-sidecar.h5",
            "PHYSICAL_RUNS.parquet",
            "COMPACT_VERIFICATION.json",
            "WRITER_RESTART.json",
        ),
        extension_required_artifacts=("BASE_GRID_STOP_RECEIPT.json",),
        fail_no_saturation_required_artifacts=(
            "BASE_GRID_STOP_RECEIPT.json",
            "EXTENSION_RESULT.json",
        ),
    )
    expected = expected_publication_payload_v3(freeze, "extension_required")
    entries = []
    for index, name in enumerate(sorted(expected)):
        artifact = _file_artifact(tmp_path, name, f"payload-{index}".encode())
        entries.append(
            G00CPublicationArtifactV3(
                name=name, sha256=artifact.sha256, size_bytes=artifact.size_bytes
            )
        )
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("".join(f"{entry.sha256}  {entry.name}\n" for entry in entries))
    sums_ref = _file_artifact(tmp_path, "SHA256SUMS", sums.read_bytes())
    (tmp_path / "COMMITTED").write_text("extension_required\n")
    committed_ref = _file_artifact(tmp_path, "COMMITTED", (tmp_path / "COMMITTED").read_bytes())
    publisher = _artifact("publisher.py", "c")
    event = {
        "destination_preexisted": False,
        "directory_fsync_completed": True,
        "manifest_written_last": True,
        "no_clobber": True,
        "publisher_implementation_sha256": publisher.sha256,
        "status": "pass",
    }
    event_ref = _json_artifact(tmp_path, "PUBLICATION_EVENT.json", event)
    authority = SimpleNamespace(
        authority_id="8" * 64,
        implementation=SimpleNamespace(
            implementations=(SimpleNamespace(role="publication_verifier", artifact=publisher),)
        ),
    )
    bundle = SimpleNamespace(
        selection_freeze_id="9" * 64,
        feature_selection_result_id="a" * 64,
        sample_size_selection_result_id="b" * 64,
        terminal_status="extension_required",
    )
    manifest = _identified(
        G00CPublicationManifestV3,
        {
            "execution_authority_id": authority.authority_id,
            "selection_freeze_id": bundle.selection_freeze_id,
            "feature_selection_result_id": bundle.feature_selection_result_id,
            "sample_size_selection_result_id": bundle.sample_size_selection_result_id,
            "terminal_status": bundle.terminal_status,
            "artifacts": tuple(entries),
            "sha256sums": sums_ref,
            "committed": committed_ref,
            "publication_event_receipt": event_ref,
            "publisher_implementation_sha256": publisher.sha256,
        },
        "manifest_id",
    )
    _json_artifact(tmp_path, "artifacts.json", manifest)
    assert (
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest,
        )
        == expected
    )
    with pytest.raises(Exception, match="cross-wired"):
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest.model_copy(update={"execution_authority_id": "0" * 64}),
        )
    with pytest.raises(Exception, match="inventories differ"):
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest.model_copy(update={"artifacts": manifest.artifacts[:-1]}),
        )
    payload = tmp_path / entries[0].name
    original_payload = payload.read_bytes()
    payload.write_bytes(b"changed")
    with pytest.raises(Exception, match="payload failed verification"):
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest,
        )
    payload.write_bytes(original_payload)
    (tmp_path / "COMMITTED").write_text("wrong\n")
    wrong_committed = ArtifactRef(
        schema_id="test.dev36",
        schema_version=1,
        sha256=sha256_file(tmp_path / "COMMITTED"),
        size_bytes=(tmp_path / "COMMITTED").stat().st_size,
        media_type="application/octet-stream",
        relative_uri="COMMITTED",
    )
    with pytest.raises(Exception, match="COMMITTED marker"):
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest.model_copy(update={"committed": wrong_committed}),
        )
    (tmp_path / "COMMITTED").write_text("extension_required\n")
    atomic_json(tmp_path / "PUBLICATION_EVENT.json", {"status": "wrong"})
    wrong_event = ArtifactRef(
        schema_id="test.dev36",
        schema_version=1,
        sha256=sha256_file(tmp_path / "PUBLICATION_EVENT.json"),
        size_bytes=(tmp_path / "PUBLICATION_EVENT.json").stat().st_size,
        media_type="application/json",
        relative_uri="PUBLICATION_EVENT.json",
    )
    with pytest.raises(Exception, match="does not prove"):
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest.model_copy(update={"publication_event_receipt": wrong_event}),
        )
    (tmp_path / "PUBLICATION_EVENT.json").write_bytes(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    )
    (tmp_path / "UNDECLARED.bin").write_bytes(b"extra")
    with pytest.raises(Exception, match="missing or extra"):
        verify_g00c_publication_v3(
            tmp_path,
            authority,  # type: ignore[arg-type]
            freeze,
            bundle,  # type: ignore[arg-type]
            manifest,
        )
    with pytest.raises(Exception, match="unknown terminal status"):
        expected_publication_payload_v3(freeze, "unknown")  # type: ignore[arg-type]
    malformed = tmp_path / "malformed-SHA256SUMS"
    malformed.write_text("not-a-valid-line\n")
    with pytest.raises(Exception, match="malformed"):
        g00c_publication_v3._parse_sha256sums(malformed)


def test_dev36_publisher_commits_manifest_last_and_refuses_clobber(tmp_path: Path) -> None:
    freeze = G00CPublicationFreezeV1(
        always_required_artifacts=(
            "G00C_REFIT_SEED_SCHEDULE.json",
            "ROW_HASHES.json",
            "FEATURE_SELECTION_CURVE.parquet",
            "CELL_SELECTION_CURVE.parquet",
            "REFIT_PROVENANCE.parquet",
            "REPLAY_AUDIT.json",
            "SUPPORT_AUDIT.parquet",
            "SAMPLER_RESTART.json",
            "DECISION_RECEIPT.json",
            "artifacts.json",
            "COMMITTED",
            "SHA256SUMS",
        ),
        pass_only_required_artifacts=(
            "SELECTED_FEATURES.parquet",
            "SELECTED_ROWS.parquet",
            "compact.h5",
            "puroR-sidecar.h5",
            "PHYSICAL_RUNS.parquet",
            "COMPACT_VERIFICATION.json",
            "WRITER_RESTART.json",
        ),
        extension_required_artifacts=("BASE_GRID_STOP_RECEIPT.json",),
        fail_no_saturation_required_artifacts=(
            "BASE_GRID_STOP_RECEIPT.json",
            "EXTENSION_RESULT.json",
        ),
    )
    destination = tmp_path / "published"
    publisher = _artifact("publication-verifier.py", "e")
    authority = SimpleNamespace(
        authority_id="1" * 64,
        implementation=SimpleNamespace(
            implementations=(SimpleNamespace(role="publication_verifier", artifact=publisher),)
        ),
    )
    bundle = SimpleNamespace(
        selection_freeze_id="2" * 64,
        feature_selection_result_id="3" * 64,
        sample_size_selection_result_id="4" * 64,
        terminal_status="failed_integrity",
    )

    def writer(root: Path) -> G00CPublicationManifestV3:
        entries = []
        for index, name in enumerate(
            sorted(expected_publication_payload_v3(freeze, bundle.terminal_status))
        ):
            artifact = _file_artifact(root, name, f"payload-{index}".encode())
            entries.append(
                G00CPublicationArtifactV3(
                    name=name,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                )
            )
        sums = root / "SHA256SUMS"
        sums.write_text("".join(f"{entry.sha256}  {entry.name}\n" for entry in entries))
        sums_ref = _file_artifact(root, "SHA256SUMS", sums.read_bytes())
        committed_ref = _file_artifact(root, "COMMITTED", b"failed_integrity\n")
        event_payload = {
            "destination_preexisted": False,
            "directory_fsync_completed": True,
            "manifest_written_last": True,
            "no_clobber": True,
            "publisher_implementation_sha256": publisher.sha256,
            "status": "pass",
        }
        atomic_json(root / "PUBLICATION_EVENT.json", event_payload)
        event_ref = ArtifactRef(
            schema_id="test.dev36",
            schema_version=1,
            sha256=sha256_file(root / "PUBLICATION_EVENT.json"),
            size_bytes=(root / "PUBLICATION_EVENT.json").stat().st_size,
            media_type="application/json",
            relative_uri="PUBLICATION_EVENT.json",
        )
        return _identified(
            G00CPublicationManifestV3,
            {
                "execution_authority_id": authority.authority_id,
                "selection_freeze_id": bundle.selection_freeze_id,
                "feature_selection_result_id": bundle.feature_selection_result_id,
                "sample_size_selection_result_id": bundle.sample_size_selection_result_id,
                "terminal_status": bundle.terminal_status,
                "artifacts": tuple(entries),
                "sha256sums": sums_ref,
                "committed": committed_ref,
                "publication_event_receipt": event_ref,
                "publisher_implementation_sha256": publisher.sha256,
            },
            "manifest_id",
        )

    publish_g00c_v3(
        destination,
        terminal_status=bundle.terminal_status,
        publisher_implementation_sha256=publisher.sha256,
        writer=writer,
    )
    manifest = G00CPublicationManifestV3.model_validate_json(
        (destination / "artifacts.json").read_text()
    )
    verify_g00c_publication_v3(
        destination,
        authority,  # type: ignore[arg-type]
        freeze,
        bundle,  # type: ignore[arg-type]
        manifest,
    )
    with pytest.raises(FileExistsError):
        publish_g00c_v3(
            destination,
            terminal_status=bundle.terminal_status,
            publisher_implementation_sha256=publisher.sha256,
            writer=writer,
        )
    rejected = tmp_path / "rejected"
    with pytest.raises(Exception, match="cross-wired manifest"):
        publish_g00c_v3(
            rejected,
            terminal_status="pass",
            publisher_implementation_sha256=publisher.sha256,
            writer=writer,
        )
    assert not rejected.exists()


def test_dev36_materialization_api_has_no_callback_escape_hatch() -> None:
    parameters = inspect.signature(verify_g00c_materialization_v3).parameters
    assert "materialized_pass_verifier" not in parameters
    assert "callback" not in parameters
    assert tuple(parameters) == (
        "root",
        "store",
        "authority",
        "receipt",
        "expected_selected_feature_ids",
        "expected_selected_rows",
    )


def test_dev36_materialization_receipt_cannot_self_declare_without_full_chain() -> None:
    with pytest.raises(ValidationError):
        G00CMaterializationReceiptV3.model_validate(
            {
                "receipt_id": "0" * 64,
                "execution_authority_id": "1" * 64,
                "verifier_implementation_sha256": "2" * 64,
                "status": "pass",
            }
        )


def test_dev36_materializer_compares_primary_and_sidecar_to_source_bytes(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    store_root.mkdir()
    locator = store_root / "locator.h5"
    with h5py.File(locator, "w") as handle:
        handle.create_dataset("row_ids_sorted", data=np.asarray((1, 2, 3, 4), dtype=np.int64))
        handle.create_dataset(
            "source_indices_sorted", data=np.asarray((0, 0, 1, 1), dtype=np.int16)
        )
        handle.create_dataset("source_rows_sorted", data=np.asarray((0, 1, 0, 1), dtype=np.int64))
        handle.create_dataset("guide_codes_sorted", data=np.asarray((0, 1, 2, 3), dtype=np.int32))
        handle.create_dataset("target_codes_sorted", data=np.asarray((0, 1, 2, 3), dtype=np.int32))
        handle.create_dataset("guide_ids", data=np.asarray(("ctrl", "g1", "g2", "g3"), dtype="S8"))
        handle.create_dataset("target_ids", data=np.asarray(("CTRL", "T1", "T2", "T3"), dtype="S8"))
    locator_ref = ArtifactRef(
        schema_id="test.locator",
        schema_version=1,
        sha256=sha256_file(locator),
        size_bytes=locator.stat().st_size,
        media_type="application/x-hdf5",
        relative_uri=locator.name,
    )
    matrix = sparse.csr_matrix(
        np.asarray(((2, 0, 1), (3, 1, 0), (4, 0, 2), (5, 0, 3)), dtype=np.int32)
    )

    class TinyStore:
        path = store_root
        manifest = SimpleNamespace(
            row_locator=locator_ref,
            features=3,
            sources=(
                SimpleNamespace(donor_id="D2", physical_time_hours=0.0, checkpoint="Rest"),
                SimpleNamespace(donor_id="D3", physical_time_hours=0.0, checkpoint="Rest"),
            ),
        )

        def _locator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.asarray((1, 2, 3, 4), dtype=np.int64),
                np.asarray((0, 0, 1, 1), dtype=np.int16),
                np.asarray((0, 1, 0, 1), dtype=np.int64),
            )

        def rows(self, row_ids: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(matrix=matrix[np.asarray(row_ids, dtype=np.int64) - 1])

    def write_compact(path: Path, feature_id: str, values: sparse.csr_matrix) -> ArtifactRef:
        values = values.tocsr()
        with h5py.File(path, "w") as handle:
            handle.create_dataset("row_ids", data=np.asarray((1, 2, 3), dtype=np.int64))
            handle.create_dataset("feature_ids", data=np.asarray((feature_id,), dtype="S32"))
            handle.create_dataset("indptr", data=values.indptr.astype(np.int64))
            handle.create_dataset("indices", data=values.indices.astype(np.uint16))
            handle.create_dataset("data", data=values.data.astype(np.uint16))
        return ArtifactRef(
            schema_id="test.compact",
            schema_version=1,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            media_type="application/x-hdf5",
            relative_uri=path.name,
        )

    compact_ref = write_compact(tmp_path / "compact.h5", "gene0", matrix[:3, [0]])
    sidecar_ref = write_compact(tmp_path / "puroR-sidecar.h5", "CUSTOM001_PuroR", matrix[:3, [2]])
    features_ref = _parquet_artifact(
        tmp_path,
        "SELECTED_FEATURES.parquet",
        pd.DataFrame({"rank": (1,), "canonical_index": (0,), "feature_id": ("gene0",)}),
    )
    rows_ref = _parquet_artifact(tmp_path, "SELECTED_ROWS.parquet", pd.DataFrame({"row_id": (1,)}))
    physical_ref = _parquet_artifact(
        tmp_path,
        "PHYSICAL_RUNS.parquet",
        pd.DataFrame(
            {
                "block_index": (0, 1, 2),
                "start_row_offset": (0, 1, 2),
                "stop_row_offset": (1, 2, 3),
                "donor_id": ("D2", "D2", "D3"),
                "checkpoint": ("Rest", "Rest", "Rest"),
                "target_id": ("CTRL", "T1", "T2"),
                "guide_id": ("ctrl", "g1", "g2"),
            }
        ),
    )
    roles_ref = _parquet_artifact(
        tmp_path,
        "roles.parquet",
        pd.DataFrame(
            {
                "row_id": (1, 2, 3, 4),
                "role": (
                    "training_fit",
                    "training_validation",
                    "heldout_source_query",
                    "protected_heldout_stimulated",
                ),
            }
        ),
    )
    protected_hash = hashlib.sha256(np.asarray((4,), dtype="<i8").tobytes()).hexdigest()
    selected_hash = hashlib.sha256(np.asarray((1,), dtype="<i8").tobytes()).hexdigest()
    ordered_hash = hashlib.sha256(np.asarray((1, 2, 3), dtype="<i8").tobytes()).hexdigest()
    access_ref = _json_artifact(
        tmp_path,
        "protected-access.json",
        {
            "compact_rows_read": 3,
            "protected_expression_reads": 0,
            "protected_row_ids_sha256": protected_hash,
            "schema_version": 1,
            "status": "pass",
        },
    )
    reload_ref = _json_artifact(
        tmp_path,
        "reload.json",
        {
            "compact_payload_sha256": compact_ref.sha256,
            "primary_features": 1,
            "puro_r_sidecar_sha256": sidecar_ref.sha256,
            "rows": 3,
            "schema_version": 1,
            "sidecar_features": 1,
            "status": "pass",
        },
    )
    implementation = _artifact("materialization-verifier.py", "d")
    authority = SimpleNamespace(
        authority_id="e" * 64,
        row_role_freeze=roles_ref,
        implementation=SimpleNamespace(
            implementations=(
                SimpleNamespace(role="materialization_verifier", artifact=implementation),
            )
        ),
    )
    receipt = _identified(
        G00CMaterializationReceiptV3,
        {
            "execution_authority_id": authority.authority_id,
            "compact_payload": compact_ref,
            "puro_r_sidecar": sidecar_ref,
            "selected_features": features_ref,
            "selected_rows": rows_ref,
            "physical_runs": physical_ref,
            "selected_feature_count": 1,
            "compact_row_count": 3,
            "puro_r_canonical_index": 2,
            "compact_verification_block_rows": 2,
            "compact_data_dtype": "uint16",
            "compact_index_dtype": "uint16",
            "maximum_observed_count": 5,
            "selected_row_set_sha256": selected_hash,
            "compact_ordered_row_ids_sha256": ordered_hash,
            "protected_row_ids_sha256": protected_hash,
            "uninterrupted_compact_payload": compact_ref,
            "resumed_compact_payload": compact_ref,
            "uninterrupted_puro_r_sidecar": sidecar_ref,
            "resumed_puro_r_sidecar": sidecar_ref,
            "protected_access_receipt": access_ref,
            "reload_receipt": reload_ref,
            "verifier_implementation_sha256": implementation.sha256,
        },
        "receipt_id",
    )
    verify_g00c_materialization_v3(
        tmp_path,
        TinyStore(),  # type: ignore[arg-type]
        authority,  # type: ignore[arg-type]
        receipt,
        expected_selected_feature_ids=("gene0",),
        expected_selected_rows=np.asarray((1,), dtype=np.int64),
    )
    with pytest.raises(Exception, match="selection differs"):
        verify_g00c_materialization_v3(
            tmp_path,
            TinyStore(),  # type: ignore[arg-type]
            authority,  # type: ignore[arg-type]
            receipt,
            expected_selected_feature_ids=("wrong",),
            expected_selected_rows=np.asarray((1,), dtype=np.int64),
        )


def test_dev36_complete_execution_and_decision_seal_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    refs = {
        name: _artifact(f"{name}.json", character)
        for name, character in zip(
            (
                "authority",
                "freeze",
                "schedule",
                "feature",
                "sample",
                "sampler",
                "support",
                "feature_replay",
                "sample_replay",
                "materialization",
                "publication",
            ),
            "0123456789a",
            strict=True,
        )
    }
    decision_implementation = _artifact("decision-verifier.py", "f")
    authority = SimpleNamespace(
        authority_id="1" * 64,
        selection_freeze=refs["freeze"],
        seed_schedule=refs["schedule"],
        seed_schedule_id="2" * 64,
        row_role_freeze=_artifact("row-roles.parquet", "1"),
        base_support_audit_contract=_artifact("support-contract.json", "2"),
        implementation=SimpleNamespace(
            implementations=(
                SimpleNamespace(role="decision_verifier", artifact=decision_implementation),
            )
        ),
    )
    freeze = SimpleNamespace(
        freeze_id="3" * 64,
        base_cell_grid=(2, 4),
        extension_additional_cell_grid=(),
        feature_ranking=SimpleNamespace(candidate_feature_counts=(256, 4096)),
        publication=SimpleNamespace(),
        dev33_canary=SimpleNamespace(authority_archive_sha256="4" * 64),
    )
    schedule = SimpleNamespace(schedule_id=authority.seed_schedule_id)
    refit_records = tuple(range(59))
    feature = SimpleNamespace(
        result_id="5" * 64,
        refit_records=refit_records,
        selected_feature_count=256,
    )
    sample = SimpleNamespace(
        result_id="6" * 64,
        refit_records=refit_records,
        grid_stage="base",
        selection_status="selected",
        selected_training_cells=2,
    )
    sampler = SimpleNamespace(evidence_id="7" * 64)
    support = SimpleNamespace(
        receipt_id="8" * 64,
        support_contract=authority.base_support_audit_contract,
        sampler_evidence=refs["sampler"],
    )
    feature_replay = SimpleNamespace(
        receipt_id="9" * 64,
        candidate_kind="feature_count",
        refit_records=refit_records,
        selected_candidate=256,
        reference_candidate=4096,
        modeled_feature_count=4096,
    )
    sample_replay = SimpleNamespace(
        receipt_id="a" * 64,
        candidate_kind="training_cells",
        refit_records=refit_records,
        selected_candidate=2,
        reference_candidate=4,
        modeled_feature_count=256,
    )
    materialization = SimpleNamespace(receipt_id="b" * 64)
    publication = SimpleNamespace(
        manifest_id="c" * 64,
        artifacts=(SimpleNamespace(sha256=decision_implementation.sha256),),
    )
    bundle = SimpleNamespace(
        bundle_id="d" * 64,
        execution_authority=refs["authority"],
        execution_authority_id=authority.authority_id,
        selection_freeze=refs["freeze"],
        selection_freeze_id=freeze.freeze_id,
        seed_schedule=refs["schedule"],
        seed_schedule_id=schedule.schedule_id,
        row_roles=authority.row_role_freeze,
        base_support_audit_contract=authority.base_support_audit_contract,
        extension_support_audit_contract=None,
        feature_selection_result=refs["feature"],
        feature_selection_result_id=feature.result_id,
        sample_size_selection_result=refs["sample"],
        sample_size_selection_result_id=sample.result_id,
        sampler_evidence=refs["sampler"],
        base_support_audit_receipt=refs["support"],
        extension_support_audit_receipt=None,
        feature_refit_replay_receipt=refs["feature_replay"],
        sample_refit_replay_receipt=refs["sample_replay"],
        materialization_receipt=refs["materialization"],
        failure_receipt=None,
        publication_manifest=refs["publication"],
        terminal_status="pass",
    )
    by_uri = {
        refs["authority"].relative_uri: authority,
        refs["schedule"].relative_uri: schedule,
        refs["feature"].relative_uri: feature,
        refs["sample"].relative_uri: sample,
        refs["sampler"].relative_uri: sampler,
        refs["support"].relative_uri: support,
        refs["feature_replay"].relative_uri: feature_replay,
        refs["sample_replay"].relative_uri: sample_replay,
        refs["materialization"].relative_uri: materialization,
        refs["publication"].relative_uri: publication,
    }
    monkeypatch.setattr(
        g00c_v4, "_read_model", lambda _root, artifact, _model: by_uri[artifact.relative_uri]
    )
    monkeypatch.setattr(g00c_v4, "verify_g00c_d1_freeze_v1", lambda _root, _authority: freeze)
    monkeypatch.setattr(
        g00c_v4,
        "verify_g00c_sampler_v3",
        lambda *_args, **_kwargs: (
            SimpleNamespace(),
            pd.DataFrame(),
            np.asarray((), dtype=np.int64),
            pd.DataFrame(),
        ),
    )
    support_rows = []
    for kind, candidates in (("feature_count", (256, 4096)), ("training_cells", (2, 4))):
        for candidate in candidates:
            for dimension in (
                "donor_checkpoint",
                "target",
                "guide",
                "control_vs_targeting",
                "sampler_stratum",
            ):
                support_rows.append(
                    {
                        "candidate_kind": kind,
                        "candidate_value": candidate,
                        "dimension": dimension,
                        "stratum_id": f"{dimension}-1",
                        "cells": 8,
                        "weighted_effective_sample_size": 8.0,
                        "maximum_to_median_weight_ratio": 1.0,
                        "selection_eligible": True,
                        "zero_support": False,
                    }
                )
    support_table = pd.DataFrame(support_rows)
    support_by_receipt_id = {support.receipt_id: support_table}

    def verify_support(
        _root: Path, _authority: Any, _sampler: Any, receipt: Any, **_kwargs: Any
    ) -> pd.DataFrame:
        return support_by_receipt_id[receipt.receipt_id]

    monkeypatch.setattr(g00c_v4, "verify_g00c_support_v3", verify_support)
    monkeypatch.setattr(
        g00c_v4,
        "_verify_feature_selection_v3",
        lambda *_args, **_kwargs: (("gene0",), None),
    )
    monkeypatch.setattr(
        g00c_v4,
        "_verify_sample_size_selection_v3",
        lambda *_args, **_kwargs: np.asarray((1, 2), dtype=np.int64),
    )
    monkeypatch.setattr(g00c_v4, "verify_g00c_refit_replay_v3", lambda *_args: None)
    monkeypatch.setattr(g00c_v4, "verify_g00c_materialization_v3", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(g00c_v4, "verify_g00c_publication_v3", lambda *_args: None)

    verified = g00c_v4.verify_g00c_execution_v4(
        tmp_path,
        Path("/unused-publication"),
        bundle,  # type: ignore[arg-type]
        source_store=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert verified.terminal_status == "pass"
    assert verified.materialization_receipt_id == materialization.receipt_id
    receipt = g00c_v4.build_g00c_decision_receipt_v4(verified)
    assert receipt.may_parent_g00d
    g00c_v4.verify_g00c_decision_v4(
        tmp_path,
        Path("/unused-publication"),
        bundle,  # type: ignore[arg-type]
        receipt,
        source_store=SimpleNamespace(),  # type: ignore[arg-type]
    )
    with pytest.raises(Exception, match="differs from recomputed"):
        g00c_v4.verify_g00c_decision_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            receipt.model_copy(update={"may_parent_g00d": False}),
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    publication.artifacts = ()
    with pytest.raises(Exception, match="decision verifier bytes"):
        g00c_v4.verify_g00c_decision_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            receipt,
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    publication.artifacts = (SimpleNamespace(sha256=decision_implementation.sha256),)

    original_authority_id = bundle.execution_authority_id
    bundle.execution_authority_id = "0" * 64
    with pytest.raises(Exception, match="cross-wired"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    bundle.execution_authority_id = original_authority_id
    original_feature_id = bundle.feature_selection_result_id
    bundle.feature_selection_result_id = "0" * 64
    with pytest.raises(Exception, match="result identities"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    bundle.feature_selection_result_id = original_feature_id
    original_support_contract = support.support_contract
    support.support_contract = _artifact("wrong-support-contract.json", "e")
    with pytest.raises(Exception, match="base support receipt is cross-wired"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    support.support_contract = original_support_contract
    with pytest.raises(Exception, match="direct source-backed"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
        )
    bundle.terminal_status = "extension_required"
    with pytest.raises(Exception, match="differs from recomputed selection"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    bundle.terminal_status = "pass"

    freeze.extension_additional_cell_grid = (8,)
    sample.grid_stage = "extension"
    with pytest.raises(Exception, match="lacks derived support evidence"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
            source_store=SimpleNamespace(),  # type: ignore[arg-type]
        )
    sample.grid_stage = "base"
    freeze.extension_additional_cell_grid = ()

    extension_contract_ref = _artifact("extension-contract.json", "c")
    extension_receipt_ref = _artifact("extension-support.json", "d")
    extension_support = SimpleNamespace(
        receipt_id="e" * 64,
        support_contract=extension_contract_ref,
        sampler_evidence=refs["sampler"],
    )
    by_uri[extension_receipt_ref.relative_uri] = extension_support
    extension_rows = []
    for dimension in (
        "donor_checkpoint",
        "target",
        "guide",
        "control_vs_targeting",
        "sampler_stratum",
    ):
        extension_rows.append(
            {
                "candidate_kind": "training_cells",
                "candidate_value": 8,
                "dimension": dimension,
                "stratum_id": f"{dimension}-1",
                "cells": 8,
                "weighted_effective_sample_size": 8.0,
                "maximum_to_median_weight_ratio": 1.0,
                "selection_eligible": True,
                "zero_support": False,
            }
        )
    support_by_receipt_id[extension_support.receipt_id] = pd.DataFrame(extension_rows)
    freeze.extension_additional_cell_grid = (8,)
    sample.grid_stage = "extension"
    sample.selected_training_cells = 8
    sample_replay.selected_candidate = 8
    sample_replay.reference_candidate = 8
    bundle.extension_support_audit_contract = extension_contract_ref
    bundle.extension_support_audit_receipt = extension_receipt_ref
    extended = g00c_v4.verify_g00c_execution_v4(
        tmp_path,
        Path("/unused-publication"),
        bundle,  # type: ignore[arg-type]
        source_store=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert extended.support_audit_receipt_ids == (support.receipt_id, extension_support.receipt_id)

    freeze.extension_additional_cell_grid = ()
    sample.grid_stage = "base"
    sample.selected_training_cells = 2
    sample.selection_status = "failed_integrity"
    sample_replay.selected_candidate = 2
    sample_replay.reference_candidate = 4
    bundle.extension_support_audit_contract = None
    bundle.extension_support_audit_receipt = None
    bundle.materialization_receipt = None
    bundle.terminal_status = "failed_integrity"
    bundle.failure_receipt = None
    with pytest.raises(Exception, match="lacks a failure receipt"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
        )
    failure_payload = {
        "biological_claims": False,
        "execution_authority_id": authority.authority_id,
        "schema_version": 1,
        "selection_freeze_id": freeze.freeze_id,
        "status": "failed_integrity",
    }
    bundle.failure_receipt = _json_artifact(tmp_path, "failure.json", failure_payload)
    failed = g00c_v4.verify_g00c_execution_v4(
        tmp_path,
        Path("/unused-publication"),
        bundle,  # type: ignore[arg-type]
    )
    assert failed.materialization_receipt_id is None
    wrong_failure = _json_artifact(
        tmp_path, "wrong-failure.json", {**failure_payload, "biological_claims": True}
    )
    bundle.failure_receipt = wrong_failure
    with pytest.raises(Exception, match="not exact"):
        g00c_v4.verify_g00c_execution_v4(
            tmp_path,
            Path("/unused-publication"),
            bundle,  # type: ignore[arg-type]
        )


def test_dev36_execution_helper_surfaces_fail_closed() -> None:
    implementation = _artifact("decision.py", "a")
    authority = SimpleNamespace(
        implementation=SimpleNamespace(
            implementations=(SimpleNamespace(role="decision_verifier", artifact=implementation),)
        )
    )
    assert g00c_v4._implementation_hash(authority, "decision_verifier") == implementation.sha256
    with pytest.raises(Exception, match="no unique missing"):
        g00c_v4._implementation_hash(authority, "missing")

    class NestedArtifacts(StrictModel):
        direct: ArtifactRef
        sequence: tuple[ArtifactRef, ...]
        mapping: dict[str, ArtifactRef]

    nested = NestedArtifacts(
        direct=_artifact("direct", "1"),
        sequence=(_artifact("sequence", "2"),),
        mapping={"item": _artifact("mapping", "3")},
    )
    assert len(g00c_v4._artifact_refs(nested)) == 3
    assert g00c_v4._artifact_refs([nested.direct, {"nested": nested.sequence}])

    support = pd.DataFrame(
        {
            "candidate_kind": ("feature_count",),
            "candidate_value": (256,),
            "dimension": ("target",),
            "stratum_id": ("target-1",),
            "cells": (1,),
            "weighted_effective_sample_size": (1.0,),
            "maximum_to_median_weight_ratio": (1.0,),
            "selection_eligible": (True,),
            "zero_support": (False,),
        }
    )
    with pytest.raises(Exception, match="incomplete or nonfinite"):
        g00c_v4._derived_eligibility((support,), kind="feature_count", candidates=(256,))
    with pytest.raises(Exception, match="omits a frozen candidate"):
        g00c_v4._derived_eligibility((support,), kind="training_cells", candidates=(2,))
    result = SimpleNamespace(refit_records=(1, 2))
    receipt = SimpleNamespace(
        candidate_kind="feature_count",
        refit_records=(1, 2),
        selected_candidate=512,
        reference_candidate=4096,
        modeled_feature_count=4096,
    )
    with pytest.raises(Exception, match="another selection surface"):
        g00c_v4._verify_refit_receipt_binding(
            receipt,
            result=result,
            kind="feature_count",
            selected=256,
            reference=4096,
            modeled_features=4096,
        )
