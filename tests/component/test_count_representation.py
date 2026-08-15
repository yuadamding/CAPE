from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from credo_count_sde_v4.contracts import ComponentTestReceipt, CountRepresentationBundle, FeatureKey
from credo_count_sde_v4.data import build_pooled_finite_measures
from credo_count_sde_v4.representation import (
    qualify_count_representation,
    verify_count_representation,
)
from credo_count_sde_v4.store import build_count_store


def _t01_fixture(root: Path, *, terminal_seed: int = 91) -> tuple[Path, Path, pd.DataFrame]:
    genes = 16
    features = tuple(
        FeatureKey(namespace="synthetic", namespace_version="1", feature_id=f"g{index}")
        for index in range(genes)
    )
    guides: list[tuple[str, str, bool, int]] = []
    for target in range(3):
        guides.extend(
            (f"t{target}-g{guide}", f"target-{target}", False, guide) for guide in range(3)
        )
    guides.extend((f"ctrl-{guide}", "control", True, guide) for guide in range(3))
    rows: list[np.ndarray] = []
    cells: list[dict[str, object]] = []
    row_id = 0
    source_generator = np.random.default_rng(73)
    terminal_generator = np.random.default_rng(terminal_seed)
    for guide, target, is_control, _ in guides:
        target_index = 3 if is_control else int(target.rsplit("-", maxsplit=1)[1])
        base = np.full(genes, 0.02, dtype=np.float64)
        base[4:] += 0.01
        if not is_control:
            base[target_index] += 0.42
            base[8 + target_index] += 0.16
        base /= base.sum()
        terminal = base.copy()
        terminal[12] += 0.015
        terminal /= terminal.sum()
        for checkpoint, generator, probabilities in (
            ("P4", source_generator, base),
            ("P60", terminal_generator, terminal),
        ):
            for cell in range(18):
                rows.append(generator.multinomial(180 + cell % 7, probabilities).astype(np.int32))
                cells.append(
                    {
                        "row_id": row_id,
                        "cell_id": f"cell-{row_id}",
                        "sample_id": f"source-{checkpoint}",
                        "guide_id": guide,
                        "checkpoint": checkpoint,
                    }
                )
                row_id += 1
    catalog = pd.DataFrame(
        [
            {"guide_id": guide, "target_id": target, "is_control": control}
            for guide, target, control, _ in guides
        ]
    )
    count_path = root / "counts.h5"
    build_count_store(
        count_path,
        sparse.csr_matrix(np.asarray(rows, dtype=np.int32)),
        row_ids=np.arange(len(rows), dtype=np.int64),
        features=features,
    )
    pooled_path = root / "T00"
    feature_hash = "a" * 64
    build_pooled_finite_measures(
        pooled_path,
        cells=pd.DataFrame(cells),
        guide_catalog=catalog,
        source_checkpoint="P4",
        terminal_checkpoint="P60",
        feature_order_hashes={"P4": feature_hash, "P60": feature_hash},
        minimum_source_cells=10,
    )
    folds = pd.DataFrame(
        [{"guide_id": guide, "outer_fold": f"fold{fold}"} for guide, _, _, fold in guides]
    )
    return pooled_path, count_path, folds


def _qualify(root: Path, *, terminal_seed: int = 91) -> Path:
    pooled, counts, folds = _t01_fixture(root, terminal_seed=terminal_seed)
    return qualify_count_representation(
        root / "T01",
        pooled_bundle=pooled,
        count_store=counts,
        outer_folds=folds,
        dimensions=(2, 3, 4),
        fit_max_rows=100,
        inner_validation_max_rows=30,
        support_max_rows=30,
        bootstrap_draws=200,
        null_repeats=2,
        required_nll_margin=1e-5,
        minimum_target_activity=0.0,
        maximum_split_half_ratio=2.0,
        minimum_terminal_support_ratio=0.1,
        seed=101,
    )


def test_t01_fits_only_source_and_publishes_complete_receipt(tmp_path: Path) -> None:
    output = _qualify(tmp_path)
    bundle = CountRepresentationBundle.model_validate_json(
        (output / "representation.json").read_text()
    )
    receipt = ComponentTestReceipt.model_validate_json((output / "TEST_RECEIPT.json").read_text())
    selected = json.loads((output / "SELECTED_MODEL.json").read_text())
    assert bundle.selection_uses_terminal_outcomes is False
    assert bundle.dynamics_gradients_enabled is False
    assert selected["terminal_used_for_selection"] is False
    assert set(bundle.selected_dimensions) == {"fold0", "fold1", "fold2"}
    assert all(value in {0, 2, 3, 4} for value in bundle.selected_dimensions.values())
    assert receipt.test_id == "T01_COUNT_NATIVE_REPRESENTATION"
    for required in (
        "TEST_CONTRACT.json",
        "INPUTS.sha256",
        "CONFIG.yaml",
        "NULL_CALIBRATION.json",
        "PER_GUIDE_METRICS.parquet",
        "PER_TARGET_METRICS.parquet",
        "BOOTSTRAP_RESULTS.json",
        "CHANNEL_ACTIVITY.json",
        "SELECTED_MODEL.json",
        "TEST_RECEIPT.json",
        "SHA256SUMS",
    ):
        assert (output / required).is_file()
    pooled = tmp_path / "T00"
    counts = tmp_path / "counts.h5"
    assert verify_count_representation(output, pooled_bundle=pooled, count_store=counts) == bundle


def test_t01_dimension_selection_is_invariant_to_terminal_counts(tmp_path: Path) -> None:
    first = _qualify(tmp_path / "a", terminal_seed=91)
    second = _qualify(tmp_path / "b", terminal_seed=9_191)
    first_selected = json.loads((first / "SELECTED_MODEL.json").read_text())
    second_selected = json.loads((second / "SELECTED_MODEL.json").read_text())
    assert first_selected["selected_dimensions"] == second_selected["selected_dimensions"]
    first_candidates = pd.read_parquet(first / "CANDIDATE_METRICS.parquet")
    second_candidates = pd.read_parquet(second / "CANDIDATE_METRICS.parquet")
    pd.testing.assert_frame_equal(first_candidates, second_candidates)
