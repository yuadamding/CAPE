from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter
from scipy import sparse

from credo_count_sde_v4.canonical import contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    CompactSamplerContract,
    FoldNativeCompactViewContractV3,
    FoldRowRoleRecord,
    G00CCompactVerificationReceiptV2,
    G00CDecisionReceiptV2,
    G00CExecutionBundleV2,
    G00CFeatureSelectionResultV2,
    G00CSamplerEvidence,
    G00CSampleSizeSelectionResultV2,
    RefitReplayPolicyV1,
    StrictModel,
    TrainingOnlyFeatureSelectionContractV2,
    TrainingOnlySampleSizeSelectionContractV2,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import RefitReplayObservation, sample_size_decision_v2
from credo_count_sde_v4.store.g00c_v2 import (
    REFIT_COLUMNS,
    _expected_physical_order,
    _load_refits,
    _ordered_row_hash,
    _replay_required_refits,
    _row_set_hash,
    _rss_bytes,
    _verify_compact_payload_streaming,
    _verify_feature_selection,
    _verify_row_roles,
    validate_g00c_execution_v2,
    verify_sample_size_selection_v2,
)

ModelT = TypeVar("ModelT", bound=StrictModel)
ROOT = Path(__file__).resolve().parents[2]


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized = model.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in model.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _artifact(path: Path, root: Path, media_type: str = "application/json") -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.artifact",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.relative_to(root).as_posix(),
    )


def _refit_records(kind: str, candidates: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for draw in range(59):
        for candidate in candidates:
            total = 10_000 + draw
            nll = float(total) * (1.0 + candidate / 10_000_000)
            rows.append(
                (
                    kind,
                    draw,
                    1000 + draw,
                    candidate,
                    hashlib.sha256(f"fit-{candidate}".encode()).hexdigest(),
                    "2" * 64,
                    hashlib.sha256(f"thin-{draw}-{candidate}".encode()).hexdigest(),
                    "3" * 64,
                    hashlib.sha256(f"initial-{draw}-{candidate}".encode()).hexdigest(),
                    hashlib.sha256(f"final-{draw}-{candidate}".encode()).hexdigest(),
                    total,
                    nll,
                    nll / total,
                    "pass",
                )
            )
    return pd.DataFrame(rows, columns=REFIT_COLUMNS)


class _ExactReplay:
    implementation_sha256 = "1" * 64

    def __call__(self, _kind: str, row: dict[str, Any]) -> RefitReplayObservation:
        return RefitReplayObservation(
            final_state_hash=str(row["final_state_hash"]),
            validation_total_count=int(row["validation_total_count"]),
            validation_nll_sum=float(row["validation_nll_sum"]),
            validation_nll_per_count=float(row["validation_nll_per_count"]),
        )


def test_base_grid_maximum_is_extension_required_not_a_self_reference_pass() -> None:
    base = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    status, selected = sample_size_decision_v2(
        grid_stage="base",
        candidates=base,
        p95=np.asarray([0.5, 0.4, 0.3, 0.2, 0.0]),
        epsilon=0.1,
    )
    assert (status, selected) == ("extension_required", None)

    status, selected = sample_size_decision_v2(
        grid_stage="base",
        candidates=base,
        p95=np.asarray([0.5, 0.4, 0.3, 0.05, 0.0]),
        epsilon=0.1,
    )
    assert (status, selected) == ("selected", 500_000)

    extension = (*base, 2_000_000)
    status, selected = sample_size_decision_v2(
        grid_stage="extension",
        candidates=extension,
        p95=np.asarray([0.5, 0.4, 0.3, 0.2, 0.15, 0.0]),
        epsilon=0.1,
    )
    assert (status, selected) == ("selected", 2_000_000)


def test_dev33_contracts_reject_invalid_replay_and_extension_states(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique, sorted"):
        RefitReplayPolicyV1(
            implementation_sha256="1" * 64,
            preregistered_audit_draw_ids=(2, 2),
        )
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}\n")
    artifact = _artifact(dummy, tmp_path)
    replay = RefitReplayPolicyV1(
        implementation_sha256="1" * 64,
        preregistered_audit_draw_ids=(0,),
    )
    with pytest.raises(ValueError, match="base stage cannot bind"):
        TrainingOnlySampleSizeSelectionContractV2(
            equivalence_epsilon=0.01,
            candidate_cells=(50_000, 100_000, 250_000, 500_000, 1_000_000),
            base_grid_extension_required_receipt=artifact,
            refit_replay=replay,
        )
    with pytest.raises(ValueError, match="requires the passed base-grid stop"):
        TrainingOnlySampleSizeSelectionContractV2(
            equivalence_epsilon=0.01,
            candidate_cells=(50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000),
            two_million_extension_triggered_by_no_saturation=True,
            grid_stage="extension",
            refit_replay=replay,
        )
    with pytest.raises(ValueError, match="requires its exact row artifact"):
        G00CSampleSizeSelectionResultV2(
            curve=artifact,
            refit_records=artifact,
            training_scale_row_order=artifact,
            selection_status="selected",
            selected_training_cells=50_000,
        )


def test_b0_eligibility_interpretation_preserves_literal_rule_and_controls() -> None:
    path = ROOT / "provenance/g00/dev32-b0-a2/ELIGIBILITY_INTERPRETATION_AMENDMENT.json"
    payload = json.loads(path.read_text())
    assert payload["serialized_eligibility_rule"] == (
        "guide_group == targeting single sgRNA AND low_quality == false"
    )
    assert payload["operational_predicate"] == {
        "guide_group_raw_category_equals": "targeting single sgRNA",
        "low_quality_equals": False,
    }
    assert payload["crosswalk_targeting_guides"] == 24_972
    assert payload["crosswalk_control_guides"] == 984
    assert payload["sealed_b0_bundle_mutated"] is False


def test_refit_records_require_full_provenance_and_executable_replay(tmp_path: Path) -> None:
    candidates = (256, 512, 1024, 2048, 4096)
    records = _refit_records("feature_count", candidates)
    path = tmp_path / "refits.parquet"
    records.to_parquet(path, index=False)
    loaded = _load_refits(
        tmp_path,
        _artifact(path, tmp_path, "application/x-parquet"),
        kind="feature_count",
        candidates=candidates,
        validation_row_hash="2" * 64,
    )
    policy = RefitReplayPolicyV1(
        implementation_sha256="4" * 64,
        preregistered_audit_draw_ids=(2, 7),
    )
    with pytest.raises(IntegrityError, match="executor is required"):
        _replay_required_refits(
            loaded,
            kind="feature_count",
            selected=256,
            reference=4096,
            policy=policy,
            replay_refit=None,
        )

    replayed: list[tuple[int, int]] = []

    class Replay:
        implementation_sha256 = policy.implementation_sha256

        def __call__(self, _kind: str, row: dict[str, Any]) -> RefitReplayObservation:
            replayed.append((int(row["draw_id"]), int(row["candidate_value"])))
            return RefitReplayObservation(
                final_state_hash=str(row["final_state_hash"]),
                validation_total_count=int(row["validation_total_count"]),
                validation_nll_sum=float(row["validation_nll_sum"]),
                validation_nll_per_count=float(row["validation_nll_per_count"]),
            )

    replay = Replay()

    replay.implementation_sha256 = "f" * 64
    with pytest.raises(IntegrityError, match="implementation differs"):
        _replay_required_refits(
            loaded,
            kind="feature_count",
            selected=256,
            reference=4096,
            policy=policy,
            replay_refit=replay,
        )
    replay.implementation_sha256 = policy.implementation_sha256

    _replay_required_refits(
        loaded,
        kind="feature_count",
        selected=256,
        reference=4096,
        policy=policy,
        replay_refit=replay,
    )
    assert len(replayed) == 124  # 59 selected + 59 reference + 2 draws x 3 others.

    class CorruptReplay:
        implementation_sha256 = policy.implementation_sha256

        def __call__(self, _kind: str, row: dict[str, Any]) -> RefitReplayObservation:
            observed = replay(_kind, row)
            return RefitReplayObservation(
                final_state_hash="f" * 64,
                validation_total_count=observed.validation_total_count,
                validation_nll_sum=observed.validation_nll_sum,
                validation_nll_per_count=observed.validation_nll_per_count,
            )

    with pytest.raises(IntegrityError, match="replay differs"):
        _replay_required_refits(
            loaded,
            kind="feature_count",
            selected=256,
            reference=4096,
            policy=policy,
            replay_refit=CorruptReplay(),
        )

    summary_only = records[["draw_id", "candidate_value", "validation_nll_per_count"]]
    summary_path = tmp_path / "summary-only.parquet"
    summary_only.to_parquet(summary_path, index=False)
    with pytest.raises(IntegrityError, match="invalid schema"):
        _load_refits(
            tmp_path,
            _artifact(summary_path, tmp_path, "application/x-parquet"),
            kind="feature_count",
            candidates=candidates,
            validation_row_hash="2" * 64,
        )


def test_feature_selection_is_recomputed_from_all_refits_and_exact_order(tmp_path: Path) -> None:
    candidates = (256, 512, 1024, 2048, 4096)
    fit_hash = "a" * 64
    records = _refit_records("feature_count", candidates)
    records["fit_row_hash"] = fit_hash
    refits_path = tmp_path / "feature-refits.parquet"
    records.to_parquet(refits_path, index=False)
    pivot = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_nll_per_count"
    )
    p95 = np.quantile(np.abs(pivot.to_numpy() - pivot[4096].to_numpy()[:, None]), 0.95, axis=0)
    curve = pd.DataFrame(
        {
            "feature_count": candidates,
            "mean_validation_nll": pivot.mean(axis=0).to_numpy(),
            "p95_absolute_difference_to_4096": p95,
        }
    )
    curve_path = tmp_path / "feature-curve.parquet"
    curve.to_parquet(curve_path, index=False)
    ordered = pd.DataFrame(
        {
            "rank": [*range(1, 4097), 4097],
            "canonical_index": [*range(4096), 4096],
            "feature_id": [*(f"gene-{index}" for index in range(4096)), "CUSTOM001_PuroR"],
            "in_primary_metric": [*[True] * 4096, False],
            "is_sidecar": [*[False] * 4096, True],
        }
    )
    ordered_path = tmp_path / "ordered-features.parquet"
    ordered.to_parquet(ordered_path, index=False)
    policy = RefitReplayPolicyV1(
        implementation_sha256=_ExactReplay.implementation_sha256,
        preregistered_audit_draw_ids=(0, 58),
    )
    protocol = TrainingOnlyFeatureSelectionContractV2(
        implementation_sha256="b" * 64,
        fit_rows_hash=fit_hash,
        validation_rows_hash="2" * 64,
        minimum_improvement_margin=1.0,
        ordered_feature_table=_artifact(ordered_path, tmp_path, "application/x-parquet"),
        refit_replay=policy,
    )
    selected_ids = ordered["feature_id"].iloc[:256].tolist()
    feature_hash = hashlib.sha256(
        json.dumps(selected_ids, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    result = G00CFeatureSelectionResultV2(
        curve=_artifact(curve_path, tmp_path, "application/x-parquet"),
        refit_records=_artifact(refits_path, tmp_path, "application/x-parquet"),
        ordered_features=_artifact(ordered_path, tmp_path, "application/x-parquet"),
        fit_rows_hash=fit_hash,
        validation_rows_hash="2" * 64,
        selected_feature_count=256,
    )
    contract = SimpleNamespace(feature_selection=protocol)
    bundle = SimpleNamespace(
        feature_selection=result,
        compact_payload_feature_order_hash=feature_hash,
        compact_payload_features=256,
    )
    feature_ids, indices = _verify_feature_selection(
        tmp_path,
        contract,
        bundle,
        store_feature_width=4097,
        replay_refit=_ExactReplay(),
    )
    assert feature_ids == selected_ids
    assert np.array_equal(indices, np.arange(256))

    bad_curve = curve.copy()
    bad_curve.loc[0, "mean_validation_nll"] += 0.1
    bad_curve_path = tmp_path / "bad-feature-curve.parquet"
    bad_curve.to_parquet(bad_curve_path, index=False)
    bad_result = result.model_copy(
        update={"curve": _artifact(bad_curve_path, tmp_path, "application/x-parquet")}
    )
    with pytest.raises(IntegrityError, match="not derived from full refit records"):
        _verify_feature_selection(
            tmp_path,
            contract,
            SimpleNamespace(
                feature_selection=bad_result,
                compact_payload_feature_order_hash=feature_hash,
                compact_payload_features=256,
            ),
            store_feature_width=4097,
            replay_refit=_ExactReplay(),
        )


def test_sample_size_base_stop_is_verified_from_nested_prefix_refits(tmp_path: Path) -> None:
    candidates = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    ordered_rows = np.arange(1_000_000, dtype=np.int64)
    records = _refit_records("training_cells", candidates)
    for candidate in candidates:
        records.loc[records["candidate_value"] == candidate, "fit_row_hash"] = _ordered_row_hash(
            ordered_rows[:candidate]
        )
    refits_path = tmp_path / "sample-refits.parquet"
    records.to_parquet(refits_path, index=False)
    pivot = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_nll_per_count"
    )
    p95 = np.quantile(np.abs(pivot.to_numpy() - pivot[1_000_000].to_numpy()[:, None]), 0.95, axis=0)
    curve = pd.DataFrame(
        {
            "training_cells": candidates,
            "mean_validation_nll": pivot.mean(axis=0).to_numpy(),
            "p95_absolute_difference_to_nmax": p95,
        }
    )
    curve_path = tmp_path / "sample-curve.parquet"
    curve.to_parquet(curve_path, index=False)
    order_path = tmp_path / "training-order.parquet"
    pd.DataFrame({"rank": np.arange(1, 1_000_001), "row_id": ordered_rows}).to_parquet(
        order_path, index=False
    )
    policy = RefitReplayPolicyV1(
        implementation_sha256=_ExactReplay.implementation_sha256,
        preregistered_audit_draw_ids=(1,),
    )
    protocol = SimpleNamespace(
        candidate_cells=candidates,
        grid_stage="base",
        equivalence_epsilon=0.0,
        refit_replay=policy,
    )
    result = G00CSampleSizeSelectionResultV2(
        curve=_artifact(curve_path, tmp_path, "application/x-parquet"),
        refit_records=_artifact(refits_path, tmp_path, "application/x-parquet"),
        training_scale_row_order=_artifact(order_path, tmp_path, "application/x-parquet"),
        selection_status="extension_required",
    )
    selected = verify_sample_size_selection_v2(
        tmp_path,
        SimpleNamespace(
            sample_size_selection=protocol,
            feature_selection=SimpleNamespace(validation_rows_hash="2" * 64),
        ),
        SimpleNamespace(sample_size_selection=result),
        {"training_fit": ordered_rows},
        replay_refit=_ExactReplay(),
    )
    assert selected is None

    selected_rows_path = tmp_path / "selected-training-rows.parquet"
    pd.DataFrame({"row_id": ordered_rows[:50_000]}).to_parquet(selected_rows_path, index=False)
    selected_result = G00CSampleSizeSelectionResultV2(
        curve=result.curve,
        refit_records=result.refit_records,
        training_scale_row_order=result.training_scale_row_order,
        selected_training_rows=_artifact(selected_rows_path, tmp_path, "application/x-parquet"),
        selection_status="selected",
        selected_training_cells=50_000,
    )
    protocol.equivalence_epsilon = 1.0
    selected = verify_sample_size_selection_v2(
        tmp_path,
        SimpleNamespace(
            sample_size_selection=protocol,
            feature_selection=SimpleNamespace(validation_rows_hash="2" * 64),
        ),
        SimpleNamespace(sample_size_selection=selected_result),
        {"training_fit": ordered_rows},
        replay_refit=_ExactReplay(),
    )
    assert np.array_equal(selected, ordered_rows[:50_000])

    wrong_rows_path = tmp_path / "selected-training-rows-wrong.parquet"
    pd.DataFrame({"wrong": ordered_rows[:50_000]}).to_parquet(wrong_rows_path, index=False)
    with pytest.raises(IntegrityError, match="invalid schema"):
        verify_sample_size_selection_v2(
            tmp_path,
            SimpleNamespace(
                sample_size_selection=protocol,
                feature_selection=SimpleNamespace(validation_rows_hash="2" * 64),
            ),
            SimpleNamespace(
                sample_size_selection=selected_result.model_copy(
                    update={
                        "selected_training_rows": _artifact(
                            wrong_rows_path, tmp_path, "application/x-parquet"
                        )
                    }
                )
            ),
            {"training_fit": ordered_rows},
            replay_refit=_ExactReplay(),
        )

    with pytest.raises(IntegrityError, match="no qualifying sample size"):
        sample_size_decision_v2(
            grid_stage="extension",
            candidates=candidates,
            p95=np.ones(len(candidates)),
            epsilon=0.1,
        )
    with pytest.raises(IntegrityError, match="grid stage is invalid"):
        sample_size_decision_v2(
            grid_stage="unknown",
            candidates=candidates,
            p95=np.zeros(len(candidates)),
            epsilon=0.1,
        )


def _streaming_fixture(tmp_path: Path, *, wrong_order: bool = False) -> tuple[Any, ...]:
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}\n")
    dummy_ref = _artifact(dummy, tmp_path)
    locator_root = tmp_path / "virtual"
    locator_root.mkdir()
    locator_path = locator_root / "locator.h5"
    row_ids = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    source_indices = np.asarray([0, 0, 1, 2, 3], dtype=np.int16)
    source_rows = np.asarray([1, 0, 0, 0, 0], dtype=np.int64)
    with h5py.File(locator_path, "x") as handle:
        handle.create_dataset("row_ids_sorted", data=row_ids)
        handle.create_dataset("source_indices_sorted", data=source_indices)
        handle.create_dataset("source_rows_sorted", data=source_rows)
        handle.create_dataset("guide_codes_sorted", data=np.asarray([0, 0, 1, 2, 3]))
        handle.create_dataset("target_codes_sorted", data=np.asarray([0, 0, 1, 2, 3]))
        handle.create_dataset(
            "guide_ids",
            data=np.asarray(["g0", "g1", "g2", "g3"], dtype=h5py.string_dtype()),
        )
        handle.create_dataset(
            "target_ids",
            data=np.asarray(["t0", "t1", "t2", "t3"], dtype=h5py.string_dtype()),
        )
    sources = (
        SimpleNamespace(donor_id="D1", checkpoint="Rest", physical_time_hours=0.0),
        SimpleNamespace(donor_id="D1", checkpoint="Stim8hr", physical_time_hours=8.0),
        SimpleNamespace(donor_id="D2", checkpoint="Rest", physical_time_hours=0.0),
        SimpleNamespace(donor_id="D2", checkpoint="Stim8hr", physical_time_hours=8.0),
    )
    row_calls: list[int] = []

    def rows(requested: np.ndarray) -> SimpleNamespace:
        values = np.asarray(requested, dtype=np.int64) % 5 + 1
        row_calls.append(len(values))
        matrix = sparse.csr_matrix(
            (
                values.astype(np.int32),
                np.zeros(len(values), dtype=np.int32),
                np.arange(len(values) + 1, dtype=np.int64),
            ),
            shape=(len(values), 2),
        )
        return SimpleNamespace(matrix=matrix)

    store = SimpleNamespace(
        path=locator_root,
        manifest=SimpleNamespace(
            features=2,
            row_locator=SimpleNamespace(relative_uri="locator.h5"),
            sources=sources,
        ),
        _locator=lambda: (row_ids, source_indices, source_rows),
        rows=rows,
    )
    replay = RefitReplayPolicyV1(
        implementation_sha256="1" * 64,
        preregistered_audit_draw_ids=(0,),
    )
    feature = TrainingOnlyFeatureSelectionContractV2(
        implementation_sha256="2" * 64,
        fit_rows_hash=_row_set_hash(np.asarray([10, 11])),
        validation_rows_hash=_row_set_hash(np.asarray([12])),
        minimum_improvement_margin=0.01,
        ordered_feature_table=dummy_ref,
        refit_replay=replay,
    )
    sample = TrainingOnlySampleSizeSelectionContractV2(
        equivalence_epsilon=0.01,
        candidate_cells=(50_000, 100_000, 250_000, 500_000, 1_000_000),
        refit_replay=replay,
    )
    sampler = CompactSamplerContract(
        implementation_sha256="3" * 64,
        rng_algorithm="PCG64DXSM",
        rng_seed=7,
        thinning_rule="without_replacement",
        resume_cursor_schema="cursor-v1",
        resume_cursor_initial_hash="4" * 64,
    )
    role_ids = {
        "training_fit": np.asarray([10, 11]),
        "training_validation": np.asarray([12]),
        "heldout_source_query": np.asarray([13]),
        "protected_heldout_stimulated": np.asarray([14]),
    }
    contract = _identified(
        FoldNativeCompactViewContractV3,
        {
            "schema_version": 3,
            "parent_source_authority_id": "authority",
            "parent_virtual_store_id": "store",
            "source_plane_amendment_id": "amendment",
            "source_plane_amendment": dummy_ref.model_dump(mode="json"),
            "source_plane_amendment_receipt_id": "receipt",
            "source_plane_amendment_receipt": dummy_ref.model_dump(mode="json"),
            "parent_eligible_row_ids_hash": _row_set_hash(row_ids),
            "parent_guide_target_crosswalk_hash": "5" * 64,
            "outer_split_id": "lodo-D2",
            "training_donor_ids": ["D1"],
            "heldout_donor_id": "D2",
            "row_roles": [
                FoldRowRoleRecord(
                    role=role, rows=len(ids), row_ids_hash=_row_set_hash(ids)
                ).model_dump(mode="json")
                for role, ids in role_ids.items()
            ],
            "row_role_audit": dummy_ref.model_dump(mode="json"),
            "row_role_assignment_implementation_sha256": "6" * 64,
            "feature_selection": feature.model_dump(mode="json"),
            "sample_size_selection": sample.model_dump(mode="json"),
            "sampler": sampler.model_dump(mode="json"),
            "count_dtype": "uint16",
            "index_dtype": "uint16",
            "maximum_observed_count": 10_569,
            "physical_layout": "donor_checkpoint_target_guide_row",
            "compact_verification_block_rows": 2,
            "compact_verifier_implementation_sha256": "7" * 64,
            "maximum_verifier_rss_bytes": 1_000_000_000_000,
            "maximum_source_handles": 1,
        },
        "fold_view_id",
    )
    expected_set = np.asarray([10, 11, 12, 13], dtype=np.int64)
    expected_order, blocks = _expected_physical_order(store, expected_set)
    payload_order = expected_order.copy()
    if wrong_order:
        payload_order[:2] = payload_order[1::-1]
    compact_path = tmp_path / "compact.h5"
    payload_values = (payload_order % 5 + 1).astype(np.uint16)
    with h5py.File(compact_path, "x") as handle:
        handle.create_dataset("row_ids", data=payload_order)
        handle.create_dataset("feature_ids", data=np.asarray(["g0"], dtype=h5py.string_dtype()))
        handle.create_dataset("indptr", data=np.arange(len(payload_order) + 1, dtype=np.int64))
        handle.create_dataset("indices", data=np.zeros(len(payload_order), dtype=np.uint16))
        handle.create_dataset("data", data=payload_values)
    compact_ref = _artifact(compact_path, tmp_path, "application/x-hdf5")
    compact_receipt = _identified(
        G00CCompactVerificationReceiptV2,
        {
            "schema_version": 2,
            "fold_view_id": contract.fold_view_id,
            "compact_payload_sha256": compact_ref.sha256,
            "verifier_implementation_sha256": contract.compact_verifier_implementation_sha256,
            "verifier_block_rows": 2,
            "maximum_loaded_nonzeros": 2,
            "total_rows_checked": 4,
            "total_nonzeros_checked": 4,
            "total_count_sum": int(payload_values.sum(dtype=np.uint64)),
            "peak_process_rss_bytes": _rss_bytes(),
            "maximum_open_source_handles": 1,
            "compact_row_set_sha256": _row_set_hash(expected_order),
            "compact_ordered_row_ids_sha256": _ordered_row_hash(expected_order),
            "contiguous_physical_blocks": blocks,
            "status": "pass",
        },
        "receipt_id",
    )
    compact_receipt_path = tmp_path / "compact-verification.json"
    compact_receipt_path.write_text(compact_receipt.model_dump_json() + "\n")
    sample_rows = tmp_path / "selected.parquet"
    pd.DataFrame({"row_id": [10, 11]}).to_parquet(sample_rows, index=False)
    feature_result = G00CFeatureSelectionResultV2(
        curve=dummy_ref,
        refit_records=dummy_ref,
        ordered_features=dummy_ref,
        fit_rows_hash=feature.fit_rows_hash,
        validation_rows_hash=feature.validation_rows_hash,
        selected_feature_count=1,
    )
    sample_result = G00CSampleSizeSelectionResultV2(
        curve=dummy_ref,
        refit_records=dummy_ref,
        selected_training_rows=_artifact(sample_rows, tmp_path, "application/x-parquet"),
        training_scale_row_order=dummy_ref,
        selection_status="selected",
        selected_training_cells=2,
    )
    sampler_evidence = G00CSamplerEvidence(
        sampler_plan=dummy_ref,
        epoch_index=dummy_ref,
        uninterrupted_draw_trace=dummy_ref,
        resumed_draw_trace=dummy_ref,
        uninterrupted_state_trace=dummy_ref,
        resumed_state_trace=dummy_ref,
        resume_test=dummy_ref,
    )
    bundle = _identified(
        G00CExecutionBundleV2,
        {
            "schema_version": 2,
            "fold_view_id": contract.fold_view_id,
            "row_roles": dummy_ref.model_dump(mode="json"),
            "feature_selection": feature_result.model_dump(mode="json"),
            "sample_size_selection": sample_result.model_dump(mode="json"),
            "sampler": sampler_evidence.model_dump(mode="json"),
            "compact_payload": compact_ref.model_dump(mode="json"),
            "compact_verification_receipt": _artifact(compact_receipt_path, tmp_path).model_dump(
                mode="json"
            ),
            "publication_manifest": dummy_ref.model_dump(mode="json"),
            "reload_receipt": dummy_ref.model_dump(mode="json"),
            "compact_payload_rows": 4,
            "compact_payload_features": 1,
            "compact_row_set_sha256": _row_set_hash(expected_order),
            "compact_ordered_row_ids_sha256": _ordered_row_hash(expected_order),
            "compact_payload_feature_order_hash": "8" * 64,
            "compact_payload_counts_sha256": compact_ref.sha256,
            "maximum_observed_count": 10_569,
            "count_dtype": "uint16",
            "index_dtype": "uint16",
        },
        "bundle_id",
    )
    role_rows = {**role_ids, "__compact__": expected_set}
    return store, contract, bundle, role_rows, row_calls, expected_order


def test_compact_payload_rejects_set_equal_but_physically_reordered_rows(tmp_path: Path) -> None:
    store, contract, bundle, role_rows, _, expected_order = _streaming_fixture(
        tmp_path, wrong_order=True
    )
    derived_from_unsorted, _ = _expected_physical_order(
        store, np.asarray([13, 10, 12, 11], dtype=np.int64)
    )
    assert np.array_equal(derived_from_unsorted, expected_order)
    assert _row_set_hash(expected_order) == bundle.compact_row_set_sha256
    with pytest.raises(IntegrityError, match="exact physical row order"):
        _verify_compact_payload_streaming(
            tmp_path,
            store,
            contract,
            bundle,
            role_rows,
            np.asarray([10, 11]),
            ["g0"],
            np.asarray([0]),
        )


def test_compact_payload_verification_streams_bounded_csr_blocks(tmp_path: Path) -> None:
    store, contract, bundle, role_rows, row_calls, _ = _streaming_fixture(tmp_path)
    _verify_compact_payload_streaming(
        tmp_path,
        store,
        contract,
        bundle,
        role_rows,
        np.asarray([10, 11]),
        ["g0"],
        np.asarray([0]),
    )
    assert row_calls == [2, 2]


def test_row_roles_reconcile_exact_g00b_donor_checkpoint_permissions(tmp_path: Path) -> None:
    store, contract, bundle, _, _, _ = _streaming_fixture(tmp_path)
    row_roles = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13, 14],
            "role": [
                "training_fit",
                "training_fit",
                "training_validation",
                "heldout_source_query",
                "protected_heldout_stimulated",
            ],
            "donor_id": ["D1", "D1", "D1", "D2", "D2"],
            "checkpoint": ["Rest", "Rest", "Stim8hr", "Rest", "Stim8hr"],
            "in_compact_payload": [True, True, True, True, False],
        }
    )
    row_roles_path = tmp_path / "row-roles.parquet"
    row_roles.to_parquet(row_roles_path, index=False)
    bundle = bundle.model_copy(
        update={"row_roles": _artifact(row_roles_path, tmp_path, "application/x-parquet")}
    )
    roles = _verify_row_roles(tmp_path, store, contract, bundle)
    assert np.array_equal(roles["training_fit"], np.asarray([10, 11]))
    assert np.array_equal(roles["__compact__"], np.asarray([10, 11, 12, 13]))

    row_roles.loc[row_roles["row_id"] == 13, "donor_id"] = "D1"
    bad_path = tmp_path / "row-roles-bad-donor.parquet"
    row_roles.to_parquet(bad_path, index=False)
    with pytest.raises(IntegrityError, match="donor identities differ"):
        _verify_row_roles(
            tmp_path,
            store,
            contract,
            bundle.model_copy(
                update={"row_roles": _artifact(bad_path, tmp_path, "application/x-parquet")}
            ),
        )


def test_top_level_dev33_gate_requires_all_evidence_and_separately_stops_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, contract, bundle, role_rows, _, _ = _streaming_fixture(tmp_path)
    publication_path = tmp_path / "publication.json"
    publication_path.write_text(
        json.dumps({"status": "pass", "fold_view_id": contract.fold_view_id}) + "\n"
    )
    reload_path = tmp_path / "reload.json"
    reload_path.write_text(
        json.dumps({"status": "pass", "compact_payload_sha256": bundle.compact_payload.sha256})
        + "\n"
    )
    bundle = bundle.model_copy(
        update={
            "row_roles": contract.row_role_audit,
            "publication_manifest": _artifact(publication_path, tmp_path),
            "reload_receipt": _artifact(reload_path, tmp_path),
        }
    )
    receipt_payload = {
        "schema_version": 2,
        "fold_view_id": contract.fold_view_id,
        "execution_bundle_id": bundle.bundle_id,
        "row_roles_verified": True,
        "feature_selection_verified": True,
        "sample_size_selection_verified": True,
        "refit_replay_verified": True,
        "sampler_sequence_verified": True,
        "physical_order_verified": True,
        "streaming_verification_verified": True,
        "compact_counts_verified": True,
        "protected_rows_absent": True,
        "immutable_publication_verified": True,
        "full_reload_verified": True,
        "status": "pass",
    }
    receipt = _identified(G00CDecisionReceiptV2, receipt_payload, "receipt_id")
    monkeypatch.setattr(
        "credo_count_sde_v4.store.g00c_v2._verify_row_roles", lambda *_args: role_rows
    )
    monkeypatch.setattr(
        "credo_count_sde_v4.store.g00c_v2._verify_feature_selection",
        lambda *_args, **_kwargs: (["g0"], np.asarray([0])),
    )
    monkeypatch.setattr(
        "credo_count_sde_v4.store.g00c_v2.verify_sample_size_selection_v2",
        lambda *_args, **_kwargs: np.asarray([10, 11]),
    )
    monkeypatch.setattr("credo_count_sde_v4.store.g00c_v2._verify_sampler", lambda *_args: None)
    monkeypatch.setattr(
        "credo_count_sde_v4.store.g00c_v2._verify_compact_payload_streaming",
        lambda *_args: None,
    )
    validate_g00c_execution_v2(
        tmp_path,
        store,
        contract,
        bundle,
        receipt,
        replay_refit=_ExactReplay(),
    )

    monkeypatch.setattr(
        "credo_count_sde_v4.store.g00c_v2.verify_sample_size_selection_v2",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(IntegrityError, match="separately frozen two-million extension"):
        validate_g00c_execution_v2(
            tmp_path,
            store,
            contract,
            bundle,
            receipt,
            replay_refit=_ExactReplay(),
        )

    monkeypatch.setattr(
        "credo_count_sde_v4.store.g00c_v2.verify_sample_size_selection_v2",
        lambda *_args, **_kwargs: np.asarray([10, 11]),
    )
    bad_publication_path = tmp_path / "publication-fail.json"
    bad_publication_path.write_text(
        json.dumps({"status": "fail", "fold_view_id": contract.fold_view_id}) + "\n"
    )
    with pytest.raises(IntegrityError, match="immutable-publication"):
        validate_g00c_execution_v2(
            tmp_path,
            store,
            contract,
            bundle.model_copy(
                update={"publication_manifest": _artifact(bad_publication_path, tmp_path)}
            ),
            receipt,
            replay_refit=_ExactReplay(),
        )

    bad_reload_path = tmp_path / "reload-fail.json"
    bad_reload_path.write_text(json.dumps({"status": "fail"}) + "\n")
    with pytest.raises(IntegrityError, match="full-reload"):
        validate_g00c_execution_v2(
            tmp_path,
            store,
            contract,
            bundle.model_copy(update={"reload_receipt": _artifact(bad_reload_path, tmp_path)}),
            receipt,
            replay_refit=_ExactReplay(),
        )

    false_flags = receipt_payload | {"compact_counts_verified": False, "status": "fail"}
    false_receipt = _identified(G00CDecisionReceiptV2, false_flags, "receipt_id")
    with pytest.raises(IntegrityError, match="decision flags disagree"):
        validate_g00c_execution_v2(
            tmp_path,
            store,
            contract,
            bundle,
            false_receipt,
            replay_refit=_ExactReplay(),
        )
