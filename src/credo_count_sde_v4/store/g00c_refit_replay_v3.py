"""Executable closed-form refit replay for Dev36 G00C evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV2,
    G00CRefitReplayReceiptV3,
    G00CRefitSeedScheduleV1,
)
from ..errors import IntegrityError
from .g00c_selection import checkpoint_multinomial_refit_common_support
from .g00c_v3 import REFIT_V3_COLUMNS, _path, _read_model

REPLAY_COLUMNS = (
    "candidate_kind",
    "draw_id",
    "candidate_value",
    "validation_total_count",
    "validation_nll_sum",
    "validation_nll_per_count",
    "final_state_hash",
)


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV2, role: str) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev36 authority has no unique {role} implementation.")
    return matches[0]


def _replay_targets(
    records: pd.DataFrame,
    receipt: G00CRefitReplayReceiptV3,
) -> pd.DataFrame:
    audit = set(receipt.preregistered_audit_draw_ids)
    return records.loc[
        (records["candidate_value"].astype(int) == receipt.selected_candidate)
        | (records["candidate_value"].astype(int) == receipt.reference_candidate)
        | records["draw_id"].astype(int).isin(audit)
    ].sort_values(["draw_id", "candidate_value"], kind="stable")


def recompute_refit_rows_v3(
    records: pd.DataFrame,
    receipt: G00CRefitReplayReceiptV3,
    statistics_path: Path,
) -> pd.DataFrame:
    """Recompute probabilities, weighted NLL, per-count NLL, and state hashes."""

    try:
        with np.load(statistics_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "candidate_value",
                "draw_id",
                "training_counts",
                "validation_counts",
            }:
                raise IntegrityError("Dev36 refit sufficient statistics have another schema.")
            candidate_values = np.asarray(payload["candidate_value"], dtype=np.int64)
            draw_ids = np.asarray(payload["draw_id"], dtype=np.int64)
            training_counts = np.asarray(payload["training_counts"], dtype=np.int64)
            validation_counts = np.asarray(payload["validation_counts"], dtype=np.int64)
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError("Dev36 refit sufficient statistics cannot be read.") from exc
    if (
        candidate_values.ndim != 1
        or draw_ids.shape != candidate_values.shape
        or training_counts.shape != (len(candidate_values), 4096)
        or validation_counts.shape != training_counts.shape
        or np.any(training_counts < 0)
        or np.any(validation_counts < 0)
    ):
        raise IntegrityError("Dev36 refit sufficient statistics are malformed.")
    index = {
        (int(draw_id), int(candidate)): position
        for position, (draw_id, candidate) in enumerate(
            zip(draw_ids, candidate_values, strict=True)
        )
    }
    if len(index) != len(candidate_values):
        raise IntegrityError("Dev36 refit sufficient statistics contain duplicate identities.")
    rows: list[dict[str, object]] = []
    for record in _replay_targets(records, receipt).itertuples(index=False):
        key = (int(record.draw_id), int(record.candidate_value))
        if key not in index:
            raise IntegrityError("Dev36 refit replay lacks required sufficient statistics.")
        position = index[key]
        modeled_features = (
            int(record.candidate_value)
            if receipt.candidate_kind == "feature_count"
            else receipt.modeled_feature_count
        )
        fit = checkpoint_multinomial_refit_common_support(
            training_counts[position : position + 1],
            modeled_features=modeled_features,
            per_feature_pseudocount=0.5,
        )
        probabilities = fit.expanded_probabilities[0]
        validation = validation_counts[position]
        total = int(validation.sum())
        if total <= 0:
            raise IntegrityError("Dev36 replay validation denominator must be positive.")
        nll_sum = float(-np.dot(validation.astype(np.float64), np.log(probabilities)))
        rows.append(
            {
                "candidate_kind": receipt.candidate_kind,
                "draw_id": key[0],
                "candidate_value": key[1],
                "validation_total_count": total,
                "validation_nll_sum": nll_sum,
                "validation_nll_per_count": nll_sum / total,
                "final_state_hash": hashlib.sha256(
                    np.asarray(probabilities, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows, columns=REPLAY_COLUMNS)


def verify_g00c_refit_replay_v3(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV2,
    receipt: G00CRefitReplayReceiptV3,
) -> pd.DataFrame:
    """Reject coherent false curves by rerunning the frozen closed-form estimator."""

    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.selection_freeze_id != authority.selection_freeze_id
        or receipt.seed_schedule_id != authority.seed_schedule_id
        or receipt.replay_implementation_sha256
        != _implementation_hash(authority, "refit_replay_verifier")
    ):
        raise IntegrityError("Dev36 refit replay receipt is cross-wired.")
    schedule = _read_model(root, authority.seed_schedule, G00CRefitSeedScheduleV1)
    records = pd.read_parquet(_path(root, receipt.refit_records))
    if tuple(records.columns) != REFIT_V3_COLUMNS:
        raise IntegrityError("Dev36 replay refit table has an invalid schema.")
    if set(records["draw_id"].astype(int)) != set(range(59)):
        raise IntegrityError("Dev36 refit replay does not cover all 59 draws.")
    expected_seeds = {record.draw_id: record for record in schedule.records}
    for stream in (
        "initialization",
        "training_sampler",
        "thinning",
        "validation_evaluation",
        "stochastic_optimizer_or_augmentation",
        "restart_interruption_point",
    ):
        expected = records["draw_id"].map(
            {draw: int(getattr(record, stream)) for draw, record in expected_seeds.items()}
        )
        if not np.array_equal(records[stream].to_numpy(dtype=np.uint64), expected.to_numpy()):
            raise IntegrityError(f"Dev36 refit replay uses another {stream} seed stream.")
    recomputed = recompute_refit_rows_v3(
        records,
        receipt,
        _path(root, receipt.sufficient_statistics),
    )
    observed = pd.read_parquet(_path(root, receipt.replayed_rows))
    if tuple(observed.columns) != REPLAY_COLUMNS or len(observed) != len(recomputed):
        raise IntegrityError("Dev36 replay receipt has another row surface.")
    numeric = ("validation_nll_sum", "validation_nll_per_count")
    exact = tuple(column for column in REPLAY_COLUMNS if column not in numeric)
    if any(not observed[column].equals(recomputed[column]) for column in exact) or any(
        not np.array_equal(
            observed[column].to_numpy(dtype=np.float64),
            recomputed[column].to_numpy(dtype=np.float64),
        )
        for column in numeric
    ):
        raise IntegrityError("Dev36 replayed NLL or final state differs from executable refit.")
    indexed = records.set_index(["draw_id", "candidate_value"])
    for row in recomputed.itertuples(index=False):
        source = indexed.loc[(row.draw_id, row.candidate_value)]
        if (
            int(source["validation_total_count"]) != row.validation_total_count
            or float(source["validation_nll_sum"]) != row.validation_nll_sum
            or float(source["validation_nll_per_count"]) != row.validation_nll_per_count
            or str(source["final_state_hash"]) != row.final_state_hash
        ):
            raise IntegrityError("Dev36 authoritative refit table differs from executable replay.")
    return recomputed
