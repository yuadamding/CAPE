from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from credo_count_sde_v4.contracts import FeatureKey, RawCountMassNoiseBundle
from credo_count_sde_v4.data import build_pooled_finite_measures
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.noise import (
    qualify_raw_count_mass_noise,
    verify_raw_count_mass_noise,
)
from credo_count_sde_v4.store import build_count_store


def _fixture(root: Path) -> tuple[Path, Path]:
    catalog = pd.DataFrame(
        [
            {"guide_id": "ctrl1", "target_id": "__control__", "is_control": True},
            {"guide_id": "ctrl2", "target_id": "__control__", "is_control": True},
            {"guide_id": "g1", "target_id": "T1", "is_control": False},
            {"guide_id": "g2", "target_id": "T1", "is_control": False},
            {"guide_id": "g3", "target_id": "T2", "is_control": False},
        ]
    )
    counts = {
        "ctrl1": (8, 8),
        "ctrl2": (8, 7),
        "g1": (8, 5),
        "g2": (7, 10),
        "g3": (6, 12),
    }
    rows: list[dict[str, object]] = []
    matrix_rows: list[np.ndarray] = []
    row_id = 0
    for guide_index, (guide, pair) in enumerate(counts.items()):
        for checkpoint_index, (checkpoint, cell_count) in enumerate(
            zip(("P4", "P60"), pair, strict=True)
        ):
            for replicate in range(cell_count):
                rows.append(
                    {
                        "row_id": row_id,
                        "cell_id": f"{checkpoint}:{guide}:{replicate}",
                        "sample_id": f"technical-{replicate % 2}",
                        "guide_id": guide,
                        "checkpoint": checkpoint,
                    }
                )
                values = np.ones(12, dtype=np.int32)
                values[guide_index] += 3 + checkpoint_index
                values[(guide_index + replicate + 3) % 12] += 2
                matrix_rows.append(values)
                row_id += 1
    cells = pd.DataFrame(rows)
    features = tuple(
        FeatureKey(namespace="test", namespace_version="1", feature_id=f"gene-{index}")
        for index in range(12)
    )
    count_path = root / "counts.h5"
    manifest = build_count_store(
        count_path,
        sparse.csr_matrix(np.stack(matrix_rows)),
        row_ids=cells.row_id.to_numpy(dtype=np.int64),
        features=features,
    )
    pooled = root / "T00"
    build_pooled_finite_measures(
        pooled,
        cells=cells,
        guide_catalog=catalog,
        source_checkpoint="P4",
        terminal_checkpoint="P60",
        feature_order_hashes={
            "P4": manifest.feature_index_hash,
            "P60": manifest.feature_index_hash,
        },
        minimum_source_cells=2,
    )
    return pooled, count_path


def _qualify(root: Path, name: str) -> Path:
    pooled, counts = _fixture(root)
    return qualify_raw_count_mass_noise(
        root / name,
        pooled_bundle=pooled,
        count_store=counts,
        split_repeats=100,
        mass_bootstrap_repeats=100,
        variable_gene_count=8,
        top_gene_count=3,
        rank_top_k=1,
    )


def test_t02a_freezes_complete_raw_and_mass_noise_floors(tmp_path: Path) -> None:
    pooled, counts = _fixture(tmp_path)
    output = qualify_raw_count_mass_noise(
        tmp_path / "T02A",
        pooled_bundle=pooled,
        count_store=counts,
        split_repeats=100,
        mass_bootstrap_repeats=100,
        variable_gene_count=8,
        top_gene_count=3,
        rank_top_k=1,
    )
    bundle = verify_raw_count_mass_noise(output, pooled_bundle=pooled, count_store=counts)
    assert isinstance(bundle, RawCountMassNoiseBundle)
    receipt = json.loads((output / "TEST_RECEIPT.json").read_text())
    assert receipt["status"] == "pass"
    assert receipt["raw_invariants_pass"]
    assert receipt["mass_invariants_pass"]
    raw = pd.read_parquet(output / "RAW_SPLIT_HALF_METRICS.parquet")
    mass = pd.read_parquet(output / "MASS_BOOTSTRAP_METRICS.parquet")
    assert len(raw) == 5 * 2 * 100
    assert len(mass) == 100
    assert (raw.left_cells - raw.right_cells).abs().max() <= 1
    assert (output / "FROZEN_THRESHOLDS.json").is_file()
    measures = pd.read_parquet(pooled / "finite-measures.parquet")
    pivot = measures.pivot(index="guide_id", columns="checkpoint", values="cell_count")
    source_probability = (pivot.P4 + 0.5) / (pivot.P4.sum() + 0.5 * len(pivot))
    terminal_probability = (pivot.P60 + 0.5) / (pivot.P60.sum() + 0.5 * len(pivot))
    expected_target_effect = np.log(terminal_probability.loc[["g1", "g2"]].sum()) - np.log(
        source_probability.loc[["g1", "g2"]].sum()
    )
    mean_guide_effect = np.mean(
        np.log(terminal_probability.loc[["g1", "g2"]])
        - np.log(source_probability.loc[["g1", "g2"]])
    )
    target_noise = pd.read_parquet(output / "MASS_TARGET_NOISE.parquet").set_index("target_id")
    assert target_noise.loc["T1", "observed_interval_log_mass"] == pytest.approx(
        expected_target_effect
    )
    assert expected_target_effect != pytest.approx(mean_guide_effect)


def test_t02a_is_byte_deterministic_in_one_environment(tmp_path: Path) -> None:
    pooled, counts = _fixture(tmp_path)
    first = qualify_raw_count_mass_noise(
        tmp_path / "first",
        pooled_bundle=pooled,
        count_store=counts,
        variable_gene_count=8,
        top_gene_count=3,
        rank_top_k=1,
    )
    second = qualify_raw_count_mass_noise(
        tmp_path / "second",
        pooled_bundle=pooled,
        count_store=counts,
        variable_gene_count=8,
        top_gene_count=3,
        rank_top_k=1,
    )
    assert (first / "raw-count-mass-noise.json").read_bytes() == (
        second / "raw-count-mass-noise.json"
    ).read_bytes()
    assert (first / "TEST_RECEIPT.json").read_bytes() == (
        second / "TEST_RECEIPT.json"
    ).read_bytes()


def test_t02a_rejects_underpowered_repeat_contract(tmp_path: Path) -> None:
    pooled, counts = _fixture(tmp_path)
    with pytest.raises(ContractError, match="at least 100"):
        qualify_raw_count_mass_noise(
            tmp_path / "too-few",
            pooled_bundle=pooled,
            count_store=counts,
            split_repeats=99,
        )


def test_t02a_full_verification_detects_payload_corruption(tmp_path: Path) -> None:
    pooled, counts = _fixture(tmp_path)
    output = qualify_raw_count_mass_noise(
        tmp_path / "T02A",
        pooled_bundle=pooled,
        count_store=counts,
        variable_gene_count=8,
        top_gene_count=3,
        rank_top_k=1,
    )
    threshold = output / "FROZEN_THRESHOLDS.json"
    threshold.write_text(threshold.read_text() + " ")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        verify_raw_count_mass_noise(output, pooled_bundle=pooled, count_store=counts)
