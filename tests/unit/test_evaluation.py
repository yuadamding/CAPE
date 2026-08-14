from __future__ import annotations

import numpy as np
from scipy import sparse

from credo_count_sde_v4.contracts import FeatureKey, SeriesRecord
from credo_count_sde_v4.evaluation.evaluator import (
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
