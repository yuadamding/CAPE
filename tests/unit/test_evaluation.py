from __future__ import annotations

import numpy as np
from scipy import sparse

from credo_count_sde_v4.contracts import FeatureKey, SeriesRecord
from credo_count_sde_v4.evaluation.evaluator import (
    _independent_shrunk_target_prediction,
    _interaction_advancement_pass,
    _observed_gene_compositions,
    _target_balanced_bootstrap_differences,
)
from credo_count_sde_v4.store import CountStore, build_count_store


def test_target_bootstrap_preserves_equal_target_weighting() -> None:
    terminal = np.zeros((4, 1), dtype=np.float32)
    model = np.asarray([[1.0], [1.0], [1.0], [3.0]], dtype=np.float32)
    baseline = np.asarray([[2.0], [2.0], [2.0], [2.0]], dtype=np.float32)
    targets = np.asarray([0, 0, 0, 1], dtype=np.int64)
    values = _target_balanced_bootstrap_differences(
        model, baseline, terminal, targets, seed=7, draws=5_000
    )
    middle = np.sqrt((1.0 + 9.0) / 2.0) - 2.0
    expected = 0.25 * (-1.0) + 0.5 * middle + 0.25 * 1.0
    assert abs(float(values.mean()) - expected) < 0.04


def test_interaction_advancement_requires_interaction_family_and_effect_floor() -> None:
    common = {
        "interaction_bootstrap_upper": -0.02,
        "overall_bootstrap_upper": -0.03,
        "required_interaction_improvement": 0.01,
        "required_overall_improvement": 0.02,
        "interaction_displacement_rms": 0.2,
        "minimum_interaction_displacement_rms": 0.1,
    }
    assert _interaction_advancement_pass(
        selected_family="target_plus_source_target_interaction", **common
    )
    assert not _interaction_advancement_pass(
        selected_family="shrunk_sister_guide_target_terminal", **common
    )
    assert not _interaction_advancement_pass(
        selected_family="target_plus_source_target_interaction",
        **{**common, "interaction_displacement_rms": 0.05},
    )
    assert not _interaction_advancement_pass(
        selected_family="target_plus_source_target_interaction",
        **{**common, "overall_bootstrap_upper": -0.01},
    )


def test_shrunk_target_baseline_is_materialized_independently() -> None:
    train_terminal = np.asarray([[0.0], [0.0], [1.0], [1.2], [2.0], [2.2]], dtype=np.float32)
    train_target = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    train_control = np.asarray([True, True, False, False, False, False])
    prediction, alpha = _independent_shrunk_target_prediction(
        train_terminal=train_terminal,
        train_target=train_target,
        train_control=train_control,
        evaluation_target=np.asarray([0, 1, 2], dtype=np.int64),
        evaluation_control=np.asarray([True, False, False]),
        maximum_weight=1.0,
        scalar_ridge=1.0,
    )
    global_terminal = float(train_terminal.mean())
    assert 0.0 < alpha < 1.0
    assert prediction[0, 0] == global_terminal
    assert prediction[1, 0] != global_terminal
    assert prediction[2, 0] != global_terminal


def test_observed_gene_compositions_aggregate_terminal_cells(tmp_path) -> None:
    features = tuple(
        FeatureKey(namespace="test", namespace_version="v1", feature_id=str(index))
        for index in range(2)
    )
    path = tmp_path / "counts.h5"
    manifest = build_count_store(
        path,
        sparse.csr_matrix(np.asarray([[1, 0], [0, 3], [1, 1]], dtype=np.int32)),
        row_ids=np.asarray([10, 11, 12], dtype=np.int64),
        features=features,
    )
    records = (
        SeriesRecord(
            series_id="a",
            target_index=0,
            pool_index=0,
            is_control=True,
            source_rows=(10,),
            terminal_rows=(10, 11),
            source_count=1,
            terminal_count=2,
            duration=1.0,
        ),
        SeriesRecord(
            series_id="b",
            target_index=1,
            pool_index=0,
            is_control=False,
            source_rows=(12,),
            terminal_rows=(12,),
            source_count=1,
            terminal_count=1,
            duration=1.0,
        ),
    )
    result = _observed_gene_compositions(records, CountStore(path, manifest))
    np.testing.assert_allclose(result, [[0.25, 0.75], [0.5, 0.5]])
