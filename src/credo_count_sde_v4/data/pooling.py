"""Explicit, cohort-neutral pooling for guide-level finite measures.

T00 deliberately performs no representation learning and exposes no model
channel.  It canonicalizes cell membership, freezes source-only eligibility,
and constructs pooled relative masses with a Jeffreys pseudocount.  Cohort
adapters remain outside this repository and supply the two small input tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..canonical import canonical_json_bytes, contract_id, path_manifest, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    ComponentTestContract,
    ComponentTestReceipt,
    PooledFiniteMeasureBundle,
)
from ..errors import ContractError, IntegrityError
from ..persistence import publish_directory, verify_directory

_CELL_COLUMNS = ("row_id", "cell_id", "sample_id", "guide_id", "checkpoint")
_CATALOG_COLUMNS = ("guide_id", "target_id", "is_control")


def _native_records(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for name, value in zip(columns, row, strict=True):
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            record[name] = value
        records.append(record)
    return records


def _frame_hash(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    return sha256_bytes(canonical_json_bytes(_native_records(frame, columns)))


def _id_hash(values: pd.Series) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(map(int, values.tolist()))))


def _artifact(path: Path, *, schema_id: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.name,
    )


def _normalize_bool(values: pd.Series, *, name: str) -> pd.Series:
    if any(value not in (True, False, 0, 1) for value in values.tolist()):
        raise ContractError(f"{name} must contain only boolean values.")
    return values.astype(bool)


def _validate_inputs(
    cells: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    source_checkpoint: str,
    terminal_checkpoint: str,
    feature_order_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    missing_cells = set(_CELL_COLUMNS) - set(cells.columns)
    missing_catalog = set(_CATALOG_COLUMNS) - set(catalog.columns)
    if missing_cells or missing_catalog:
        raise ContractError(
            f"Pooling input columns are incomplete: cells={sorted(missing_cells)}, "
            f"catalog={sorted(missing_catalog)}."
        )
    if source_checkpoint == terminal_checkpoint:
        raise ContractError("Source and terminal checkpoints must differ.")
    if set(feature_order_hashes) != {source_checkpoint, terminal_checkpoint}:
        raise ContractError("Feature-order hashes must cover the two checkpoints exactly.")
    feature_hashes = set(feature_order_hashes.values())
    if len(feature_hashes) != 1 or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in feature_hashes
    ):
        raise ContractError("Source and terminal feature-order hashes must be one valid SHA-256.")

    normalized_cells = cells.loc[:, _CELL_COLUMNS].copy()
    normalized_catalog = catalog.loc[:, _CATALOG_COLUMNS].copy()
    if normalized_cells.isna().any().any() or normalized_catalog.isna().any().any():
        raise ContractError("Pooled cell and catalog identifiers cannot be null.")
    normalized_cells["row_id"] = normalized_cells["row_id"].astype(np.int64)
    for column in ("cell_id", "sample_id", "guide_id", "checkpoint"):
        normalized_cells[column] = normalized_cells[column].astype(str)
        if (normalized_cells[column].str.len() == 0).any():
            raise ContractError(f"{column} cannot contain an empty identifier.")
    for column in ("guide_id", "target_id"):
        normalized_catalog[column] = normalized_catalog[column].astype(str)
        if (normalized_catalog[column].str.len() == 0).any():
            raise ContractError(f"{column} cannot contain an empty identifier.")
    normalized_catalog["is_control"] = _normalize_bool(
        normalized_catalog["is_control"], name="is_control"
    )
    if normalized_cells["row_id"].duplicated().any():
        raise ContractError("Every input row_id must identify exactly one retained cell.")
    if normalized_cells["cell_id"].duplicated().any():
        raise ContractError("A cell_id cannot switch guides, checkpoints, or samples.")
    if normalized_catalog["guide_id"].duplicated().any():
        raise ContractError("The guide catalog must contain one row per guide.")
    checkpoints = set(normalized_cells["checkpoint"])
    if not checkpoints <= {source_checkpoint, terminal_checkpoint}:
        raise ContractError("Pooling input contains an undeclared checkpoint.")
    cell_guides = set(normalized_cells["guide_id"])
    catalog_guides = set(normalized_catalog["guide_id"])
    if not cell_guides <= catalog_guides:
        raise ContractError("Every observed guide must occur in the frozen guide catalog.")
    if not normalized_catalog["is_control"].any() or normalized_catalog["is_control"].all():
        raise ContractError("The catalog must contain both control and targeting guides.")
    control_targets = normalized_catalog.loc[normalized_catalog.is_control, "target_id"]
    targeting_targets = normalized_catalog.loc[~normalized_catalog.is_control, "target_id"]
    if set(control_targets) & set(targeting_targets):
        raise ContractError("A target identity cannot be both control and targeting.")
    return normalized_cells, normalized_catalog, next(iter(feature_hashes))


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def build_pooled_finite_measures(
    destination: Path,
    *,
    cells: pd.DataFrame,
    guide_catalog: pd.DataFrame,
    source_checkpoint: str,
    terminal_checkpoint: str,
    feature_order_hashes: Mapping[str, str],
    minimum_source_cells: int = 1,
    mass_pseudocount: float = 0.5,
) -> Path:
    """Publish one immutable T00 pooled finite-measure qualification bundle."""

    if minimum_source_cells < 1:
        raise ContractError("minimum_source_cells must be positive.")
    if not np.isfinite(mass_pseudocount) or mass_pseudocount <= 0:
        raise ContractError("mass_pseudocount must be finite and positive.")
    normalized_cells, catalog, feature_hash = _validate_inputs(
        cells,
        guide_catalog,
        source_checkpoint=source_checkpoint,
        terminal_checkpoint=terminal_checkpoint,
        feature_order_hashes=feature_order_hashes,
    )
    input_order = normalized_cells.sort_values("row_id", kind="stable").reset_index(drop=True)
    input_cell_universe_hash = _frame_hash(input_order, _CELL_COLUMNS)
    counts = (
        normalized_cells.groupby(["guide_id", "checkpoint"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(catalog.guide_id, fill_value=0)
    )
    for checkpoint in (source_checkpoint, terminal_checkpoint):
        if checkpoint not in counts:
            counts[checkpoint] = 0
    eligibility = catalog.copy()
    eligibility["source_cells"] = counts[source_checkpoint].to_numpy(dtype=np.int64)
    eligibility["terminal_cells"] = counts[terminal_checkpoint].to_numpy(dtype=np.int64)
    eligibility["source_eligible"] = eligibility.source_cells >= minimum_source_cells
    eligibility["terminal_used_for_eligibility"] = False
    eligibility["eligibility_rule"] = f"source_cells>={minimum_source_cells}"
    eligibility = eligibility.sort_values("guide_id", kind="stable").reset_index(drop=True)
    retained_guides = set(eligibility.loc[eligibility.source_eligible, "guide_id"])
    if not retained_guides:
        raise ContractError("Source-only eligibility retained no guides.")
    missing_terminal = eligibility.loc[
        eligibility.source_eligible & (eligibility.terminal_cells == 0), "guide_id"
    ].tolist()
    if missing_terminal:
        raise ContractError(
            "A source-eligible guide lacks a terminal empirical law; fail instead of "
            f"using terminal counts for eligibility: {missing_terminal[:10]}."
        )
    retained_catalog = (
        catalog[catalog.guide_id.isin(retained_guides)]
        .sort_values("guide_id", kind="stable")
        .reset_index(drop=True)
    )
    if not retained_catalog.is_control.any() or retained_catalog.is_control.all():
        raise ContractError("Source-only eligibility must retain controls and perturbations.")
    excluded = normalized_cells[~normalized_cells.guide_id.isin(retained_guides)]
    retained = normalized_cells[normalized_cells.guide_id.isin(retained_guides)].copy()
    checkpoint_rank = {source_checkpoint: 0, terminal_checkpoint: 1}
    retained["_checkpoint_rank"] = retained.checkpoint.map(checkpoint_rank)
    retained = retained.sort_values(
        ["_checkpoint_rank", "guide_id", "cell_id", "row_id"], kind="stable"
    ).reset_index(drop=True)
    retained = retained.drop(columns="_checkpoint_rank")
    retained.insert(3, "source_sample_id", retained.pop("sample_id"))
    retained.insert(3, "sample_id", "pooled")
    retained.insert(0, "pooled_cell_offset", np.arange(len(retained), dtype=np.int64))

    measures: list[dict[str, Any]] = []
    for checkpoint in (source_checkpoint, terminal_checkpoint):
        local_counts = retained[retained.checkpoint == checkpoint].groupby("guide_id").size()
        denominator = float(
            sum(float(local_counts.get(guide, 0)) + mass_pseudocount for guide in retained_guides)
        )
        for row in retained_catalog.itertuples(index=False):
            local = retained[
                (retained.guide_id == row.guide_id) & (retained.checkpoint == checkpoint)
            ]
            count = len(local)
            if count == 0:  # guarded above, retained for defensive clarity
                raise ContractError(
                    "Every retained guide/checkpoint empirical law must be nonempty."
                )
            offsets = local.pooled_cell_offset.to_numpy(dtype=np.int64)
            if not np.array_equal(offsets, np.arange(offsets[0], offsets[-1] + 1)):
                raise AssertionError("Canonical pooling failed to create contiguous measures.")
            relative_mass = (count + mass_pseudocount) / denominator
            measures.append(
                {
                    "guide_id": row.guide_id,
                    "target_id": row.target_id,
                    "is_control": bool(row.is_control),
                    "checkpoint": checkpoint,
                    "cell_offset_start": int(offsets[0]),
                    "cell_offset_stop": int(offsets[-1] + 1),
                    "cell_count": count,
                    "relative_mass": relative_mass,
                    "atom_weight": relative_mass / count,
                }
            )
    measure_frame = (
        pd.DataFrame(measures)
        .sort_values(
            ["checkpoint", "guide_id"],
            key=lambda column: (
                column.map(checkpoint_rank) if column.name == "checkpoint" else column
            ),
            kind="stable",
        )
        .reset_index(drop=True)
    )
    per_guide = measure_frame.pivot(
        index=["guide_id", "target_id", "is_control"],
        columns="checkpoint",
        values=["cell_count", "relative_mass"],
    ).sort_index()
    per_guide.columns = [f"{metric}_{checkpoint}" for metric, checkpoint in per_guide.columns]
    per_guide = per_guide.reset_index()
    per_target = (
        per_guide.groupby(["target_id", "is_control"], as_index=False, observed=True)
        .agg(
            guide_count=("guide_id", "size"),
            source_cells=(f"cell_count_{source_checkpoint}", "sum"),
            terminal_cells=(f"cell_count_{terminal_checkpoint}", "sum"),
            source_relative_mass=(f"relative_mass_{source_checkpoint}", "sum"),
            terminal_relative_mass=(f"relative_mass_{terminal_checkpoint}", "sum"),
        )
        .sort_values(["is_control", "target_id"], kind="stable")
        .reset_index(drop=True)
    )
    retained_hash_frame = retained.sort_values("row_id", kind="stable").reset_index(drop=True)
    retained_hash_columns = (
        "row_id",
        "cell_id",
        "sample_id",
        "source_sample_id",
        "guide_id",
        "checkpoint",
    )
    retained_cell_universe_hash = _frame_hash(retained_hash_frame, retained_hash_columns)
    excluded_cell_universe_hash = _id_hash(excluded.row_id)
    input_hashes = {
        "input_cell_universe": input_cell_universe_hash,
        "retained_cell_universe": retained_cell_universe_hash,
        "excluded_cell_universe": excluded_cell_universe_hash,
        "feature_order": feature_hash,
        "guide_catalog": _frame_hash(
            catalog.sort_values("guide_id", kind="stable").reset_index(drop=True),
            _CATALOG_COLUMNS,
        ),
    }
    config = {
        "source_checkpoint": source_checkpoint,
        "terminal_checkpoint": terminal_checkpoint,
        "minimum_source_cells": minimum_source_cells,
        "mass_pseudocount": mass_pseudocount,
        "eligibility_uses_terminal_counts": False,
        "sample_id": "pooled",
    }

    def writer(temp: Path) -> None:
        retained.to_parquet(temp / "cells.parquet", index=False)
        retained_catalog.to_parquet(temp / "guide-catalog.parquet", index=False)
        eligibility.to_parquet(temp / "eligibility.parquet", index=False)
        measure_frame.to_parquet(temp / "finite-measures.parquet", index=False)
        per_guide.to_parquet(temp / "PER_GUIDE_METRICS.parquet", index=False)
        per_target.to_parquet(temp / "PER_TARGET_METRICS.parquet", index=False)
        contract_payload = {
            "schema_version": 1,
            "test_contract_id": "pending",
            "test_id": "T00_POOLED_DATA_CONTRACT",
            "component": "pooled_finite_measure_data",
            "primary_metric": "contract_invariant_failures",
            "primary_baseline": "exact_contract",
            "required_margin": 0.0,
            "drift": "off",
            "diffusion": "off",
            "reaction": "off",
            "ecology": "off",
            "decoder": "off",
            "update_zero_selectable": True,
            "post_selection_refit_required": False,
        }
        contract_payload["test_contract_id"] = contract_id(
            contract_payload, id_field="test_contract_id"
        )
        test_contract = ComponentTestContract.model_validate(contract_payload)
        _write_json(temp / "TEST_CONTRACT.json", test_contract.model_dump(mode="json"))
        (temp / "CONFIG.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
        _write_json(temp / "INPUTS.sha256", {"schema_version": 1, **input_hashes})
        _write_json(
            temp / "NULL_CALIBRATION.json",
            {"schema_version": 1, "test_id": test_contract.test_id, "status": "not_applicable"},
        )
        _write_json(
            temp / "BOOTSTRAP_RESULTS.json",
            {"schema_version": 1, "status": "not_applicable", "unit": "exact_contract"},
        )
        _write_json(
            temp / "CHANNEL_ACTIVITY.json",
            {
                "schema_version": 1,
                "drift": 0.0,
                "diffusion": 0.0,
                "reaction": 0.0,
                "ecology": 0.0,
                "decoder": 0.0,
            },
        )
        _write_json(
            temp / "SELECTED_MODEL.json",
            {"schema_version": 1, "selected_model": None, "selected_update": 0},
        )
        payload = {
            "schema_version": 1,
            "pooled_data_id": "pending",
            "sample_id": "pooled",
            "source_checkpoint": source_checkpoint,
            "terminal_checkpoint": terminal_checkpoint,
            "feature_order_hash": feature_hash,
            "input_cell_universe_hash": input_cell_universe_hash,
            "retained_cell_universe_hash": retained_cell_universe_hash,
            "excluded_cell_universe_hash": excluded_cell_universe_hash,
            "source_eligibility_min_cells": minimum_source_cells,
            "eligibility_uses_terminal_counts": False,
            "mass_pseudocount": mass_pseudocount,
            "retained_cells": len(retained),
            "retained_guides": len(retained_catalog),
            "targeting_guides": int((~retained_catalog.is_control).sum()),
            "control_guides": int(retained_catalog.is_control.sum()),
            "perturbation_targets": int(
                retained_catalog.loc[~retained_catalog.is_control, "target_id"].nunique()
            ),
            "cells": _artifact(
                temp / "cells.parquet",
                schema_id="credo.pooled_cells",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "guide_catalog": _artifact(
                temp / "guide-catalog.parquet",
                schema_id="credo.pooled_guide_catalog",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "eligibility": _artifact(
                temp / "eligibility.parquet",
                schema_id="credo.source_only_eligibility",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "finite_measures": _artifact(
                temp / "finite-measures.parquet",
                schema_id="credo.pooled_finite_measures",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "per_guide_metrics": _artifact(
                temp / "PER_GUIDE_METRICS.parquet",
                schema_id="credo.component_guide_metrics",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "per_target_metrics": _artifact(
                temp / "PER_TARGET_METRICS.parquet",
                schema_id="credo.component_target_metrics",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
        }
        payload["pooled_data_id"] = contract_id(payload, id_field="pooled_data_id")
        bundle = PooledFiniteMeasureBundle.model_validate(payload)
        _write_json(temp / "pooled-data.json", bundle.model_dump(mode="json"))
        receipt_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "test_id": test_contract.test_id,
            "status": "pass",
            "primary_metric": test_contract.primary_metric,
            "primary_baseline": test_contract.primary_baseline,
            "point_delta": 0.0,
            "bootstrap_interval": [0.0, 0.0],
            "required_margin": 0.0,
            "channel_activity": 0.0,
            "protected_metrics_pass": True,
            "selected_update": 0,
            "input_hashes": {**input_hashes, "pooled_data": bundle.pooled_data_id},
            "config_hash": sha256_bytes(canonical_json_bytes(config)),
            "implementation_hash": sha256_file(Path(__file__)),
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = ComponentTestReceipt.model_validate(receipt_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_pooled_finite_measures(destination)
    return destination


def verify_pooled_finite_measures(path: Path) -> PooledFiniteMeasureBundle:
    """Recompute all T00 conservation and coverage invariants."""

    verify_directory(path)
    bundle = PooledFiniteMeasureBundle.model_validate_json((path / "pooled-data.json").read_text())
    receipt = ComponentTestReceipt.model_validate_json((path / "TEST_RECEIPT.json").read_text())
    test_contract = ComponentTestContract.model_validate_json(
        (path / "TEST_CONTRACT.json").read_text()
    )
    if receipt.test_id != test_contract.test_id or receipt.status != "pass":
        raise IntegrityError("T00 receipt and contract are inconsistent.")
    for ref in (
        bundle.cells,
        bundle.guide_catalog,
        bundle.eligibility,
        bundle.finite_measures,
        bundle.per_guide_metrics,
        bundle.per_target_metrics,
    ):
        artifact = path / ref.relative_uri
        if artifact.stat().st_size != ref.size_bytes or sha256_file(artifact) != ref.sha256:
            raise IntegrityError(f"Pooled artifact differs from its reference: {ref.relative_uri}.")
    cells = pd.read_parquet(path / bundle.cells.relative_uri)
    catalog = pd.read_parquet(path / bundle.guide_catalog.relative_uri)
    eligibility = pd.read_parquet(path / bundle.eligibility.relative_uri)
    measures = pd.read_parquet(path / bundle.finite_measures.relative_uri)
    if len(cells) != bundle.retained_cells or len(catalog) != bundle.retained_guides:
        raise IntegrityError("Pooled cell or guide cardinality differs from its manifest.")
    if cells.row_id.duplicated().any() or cells.cell_id.duplicated().any():
        raise IntegrityError("Pooled cells are not covered uniquely.")
    if set(cells.sample_id) != {"pooled"}:
        raise IntegrityError("The explicit pooled adapter did not emit sample_id='pooled'.")
    if eligibility.terminal_used_for_eligibility.astype(bool).any():
        raise IntegrityError("Terminal counts were used by the source-only eligibility artifact.")
    if set(cells.guide_id) != set(catalog.guide_id):
        raise IntegrityError("Retained cell and guide catalogs are incomplete.")
    coverage = np.zeros(len(cells), dtype=np.int64)
    for row in measures.itertuples(index=False):
        start, stop = int(row.cell_offset_start), int(row.cell_offset_stop)
        if start < 0 or stop > len(cells) or stop <= start:
            raise IntegrityError("A finite-measure cell span is invalid.")
        local = cells.iloc[start:stop]
        if (
            len(local) != int(row.cell_count)
            or set(local.guide_id) != {row.guide_id}
            or set(local.checkpoint) != {row.checkpoint}
        ):
            raise IntegrityError("A finite-measure cell span does not match its guide/checkpoint.")
        if not np.isclose(float(row.atom_weight) * int(row.cell_count), row.relative_mass):
            raise IntegrityError("Finite-measure atom weights do not sum to declared mass.")
        coverage[start:stop] += 1
    if not np.all(coverage == 1):
        raise IntegrityError("Pooled cells are not covered exactly once by finite measures.")
    mass_sums = measures.groupby("checkpoint", observed=True).relative_mass.sum()
    wrong_checkpoints = set(mass_sums.index) != {
        bundle.source_checkpoint,
        bundle.terminal_checkpoint,
    }
    if wrong_checkpoints or not np.allclose(mass_sums.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise IntegrityError("Pooled relative masses do not sum to one at every checkpoint.")
    retained_hash_frame = cells.sort_values("row_id", kind="stable").reset_index(drop=True)
    retained_hash_columns = (
        "row_id",
        "cell_id",
        "sample_id",
        "source_sample_id",
        "guide_id",
        "checkpoint",
    )
    if (
        _frame_hash(retained_hash_frame, retained_hash_columns)
        != bundle.retained_cell_universe_hash
    ):
        raise IntegrityError("Retained cell-universe hash mismatch.")
    checksum_rows = [
        line.split(maxsplit=1) for line in (path / "SHA256SUMS").read_text().splitlines()
    ]
    for expected, relative in checksum_rows:
        relative = relative.strip()
        if sha256_file(path / relative) != expected:
            raise IntegrityError(f"T00 SHA256SUMS mismatch: {relative}.")
    return bundle
