from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credo_count_sde_v4.contracts import PooledFiniteMeasureBundle
from credo_count_sde_v4.data import (
    build_pooled_finite_measures,
    verify_pooled_finite_measures,
)
from credo_count_sde_v4.errors import ContractError


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    row_id = 0
    counts = {
        "ctrl": (2, 2),
        "g1": (2, 2),
        "g2": (2, 2),
        # Source-ineligible despite a large terminal count.
        "excluded": (1, 3),
    }
    for guide, (source, terminal) in counts.items():
        for checkpoint, count in (("P4", source), ("P60", terminal)):
            for replicate in range(count):
                rows.append(
                    {
                        "row_id": row_id,
                        "cell_id": f"{checkpoint}:{guide}:{replicate}",
                        "sample_id": f"technical-{replicate % 2}",
                        "guide_id": guide,
                        "checkpoint": checkpoint,
                    }
                )
                row_id += 1
    catalog = pd.DataFrame(
        [
            {"guide_id": "ctrl", "target_id": "__control__", "is_control": True},
            {"guide_id": "g1", "target_id": "T1", "is_control": False},
            {"guide_id": "g2", "target_id": "T1", "is_control": False},
            {"guide_id": "excluded", "target_id": "T2", "is_control": False},
        ]
    )
    return pd.DataFrame(rows), catalog


def _build(
    path: Path,
    cells: pd.DataFrame | None = None,
    catalog: pd.DataFrame | None = None,
) -> Path:
    default_cells, default_catalog = _inputs()
    return build_pooled_finite_measures(
        path,
        cells=default_cells if cells is None else cells,
        guide_catalog=default_catalog if catalog is None else catalog,
        source_checkpoint="P4",
        terminal_checkpoint="P60",
        feature_order_hashes={"P4": "a" * 64, "P60": "a" * 64},
        minimum_source_cells=2,
    )


def test_pooling_is_order_invariant(tmp_path: Path) -> None:
    cells, catalog = _inputs()
    first = _build(tmp_path / "first", cells, catalog)
    second = _build(
        tmp_path / "second",
        cells.sample(frac=1.0, random_state=7).reset_index(drop=True),
        catalog.sample(frac=1.0, random_state=9).reset_index(drop=True),
    )
    first_manifest = PooledFiniteMeasureBundle.model_validate_json(
        (first / "pooled-data.json").read_text()
    )
    second_manifest = PooledFiniteMeasureBundle.model_validate_json(
        (second / "pooled-data.json").read_text()
    )
    assert first_manifest == second_manifest


def test_pooled_cells_are_covered_exactly_once(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "bundle")
    manifest = verify_pooled_finite_measures(bundle)
    cells = pd.read_parquet(bundle / manifest.cells.relative_uri)
    measures = pd.read_parquet(bundle / manifest.finite_measures.relative_uri)
    covered = np.concatenate(
        [
            np.arange(row.cell_offset_start, row.cell_offset_stop)
            for row in measures.itertuples(index=False)
        ]
    )
    assert np.array_equal(np.sort(covered), np.arange(len(cells)))
    assert set(cells.sample_id) == {"pooled"}


def test_pooled_mass_sums_to_one_at_each_time(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "bundle")
    measures = pd.read_parquet(bundle / "finite-measures.parquet")
    assert np.allclose(
        measures.groupby("checkpoint", observed=True).relative_mass.sum().to_numpy(), 1.0
    )


def test_finite_measure_weights_sum_to_mass(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "bundle")
    measures = pd.read_parquet(bundle / "finite-measures.parquet")
    assert np.allclose(
        measures.atom_weight.to_numpy() * measures.cell_count.to_numpy(),
        measures.relative_mass.to_numpy(),
    )


def test_control_and_target_catalogs_are_complete(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "bundle")
    manifest = verify_pooled_finite_measures(bundle)
    catalog = pd.read_parquet(bundle / manifest.guide_catalog.relative_uri)
    assert set(catalog.guide_id) == {"ctrl", "g1", "g2"}
    assert catalog.is_control.sum() == 1
    assert set(catalog.loc[~catalog.is_control, "target_id"]) == {"T1"}

    cells, source_catalog = _inputs()
    cells.loc[0, "guide_id"] = "unknown"
    with pytest.raises(ContractError, match="frozen guide catalog"):
        _build(tmp_path / "unknown", cells, source_catalog)


def test_source_only_eligibility_does_not_use_terminal_counts(tmp_path: Path) -> None:
    bundle = _build(tmp_path / "bundle")
    eligibility = pd.read_parquet(bundle / "eligibility.parquet").set_index("guide_id")
    assert eligibility.loc["excluded", "source_cells"] == 1
    assert eligibility.loc["excluded", "terminal_cells"] == 3
    assert not eligibility.loc["excluded", "source_eligible"]
    assert not eligibility.terminal_used_for_eligibility.any()

    cells, catalog = _inputs()
    # Source eligibility is unchanged; a missing endpoint fails the component
    # rather than silently converting endpoint support into an eligibility rule.
    cells = cells[~((cells.guide_id == "g1") & (cells.checkpoint == "P60"))]
    with pytest.raises(ContractError, match="lacks a terminal empirical law"):
        _build(tmp_path / "missing-terminal", cells, catalog)


def test_feature_order_hash_is_identical_across_times(tmp_path: Path) -> None:
    cells, catalog = _inputs()
    with pytest.raises(ContractError, match="feature-order hashes"):
        build_pooled_finite_measures(
            tmp_path / "mismatch",
            cells=cells,
            guide_catalog=catalog,
            source_checkpoint="P4",
            terminal_checkpoint="P60",
            feature_order_hashes={"P4": "a" * 64, "P60": "b" * 64},
            minimum_source_cells=2,
        )


def test_no_guide_switching(tmp_path: Path) -> None:
    cells, catalog = _inputs()
    duplicate = cells.iloc[[0]].copy()
    duplicate["row_id"] = int(cells.row_id.max()) + 1
    duplicate["guide_id"] = "g1"
    cells = pd.concat([cells, duplicate], ignore_index=True)
    with pytest.raises(ContractError, match="cell_id cannot switch"):
        _build(tmp_path / "switched", cells, catalog)
