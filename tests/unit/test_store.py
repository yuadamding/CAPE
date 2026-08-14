from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from credo_count_sde_v4.contracts import FeatureKey
from credo_count_sde_v4.store import CountStore, build_count_store


def _store(tmp_path: Path) -> CountStore:
    matrix = sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int32))
    features = tuple(
        FeatureKey(namespace="test", feature_id=str(index), namespace_version="v1")
        for index in range(3)
    )
    path = tmp_path / "counts.h5"
    manifest = build_count_store(path, matrix, row_ids=np.asarray([10, 20, 30]), features=features)
    return CountStore(path, manifest)


def test_rows_preserve_order_and_duplicates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = store.rows(np.asarray([30, 10, 30]))
    assert batch.row_ids.tolist() == [30, 10, 30]
    assert batch.matrix.toarray().tolist() == [[4, 0, 5], [1, 0, 2], [4, 0, 5]]
    assert batch.library_sizes.tolist() == [9, 3, 9]


def test_rows_contiguous_fast_path_preserves_values(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = store.rows(np.asarray([20, 30]))
    assert batch.row_ids.tolist() == [20, 30]
    assert batch.matrix.toarray().tolist() == [[0, 3, 0], [4, 0, 5]]


def test_rows_dense_interleaved_selection_preserves_duplicates_and_order(tmp_path: Path) -> None:
    matrix = sparse.csr_matrix(np.arange(10_000, dtype=np.int32).reshape(5_000, 2))
    features = tuple(
        FeatureKey(namespace="test", namespace_version="1", feature_id=f"g{i}") for i in range(2)
    )
    path = tmp_path / "counts.h5"
    manifest = build_count_store(
        path, matrix, row_ids=np.arange(10_000, 15_000, dtype=np.int64), features=features
    )
    store = CountStore(path, manifest)
    positions = np.arange(4_096, dtype=np.int64)[::-1]
    positions[0] = positions[1]
    requested = positions + 10_000
    batch = store.rows(requested)
    np.testing.assert_array_equal(batch.row_ids, requested)
    np.testing.assert_array_equal(batch.matrix.toarray(), matrix[positions].toarray())


def test_store_missing_rows_and_bounded_dense_conversion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.rows(np.asarray([99]))
    batch = store.rows(np.asarray([10, 20]))
    with pytest.raises(MemoryError):
        batch.to_dense(byte_limit=1)
    assert batch.to_dense(byte_limit=100).shape == (2, 3)


def test_store_detects_content_corruption(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store.path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(Exception, match="hash mismatch"):
        store.verify(full=True)


def test_iter_batches_is_cursor_deterministic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batches = list(store.iter_batches(np.asarray([30, 10, 20]), batch_size=2))
    assert [cursor for cursor, _ in batches] == [2, 3]
    assert batches[0][1].row_ids.tolist() == [30, 10]
    resumed = list(store.iter_batches(np.asarray([30, 10, 20]), batch_size=2, cursor=2))
    assert resumed[0][1].row_ids.tolist() == [20]


def test_builder_capacity_gate_fails_before_publication(tmp_path: Path) -> None:
    matrix = sparse.csr_matrix(np.asarray([[1]], dtype=np.int32))
    feature = (FeatureKey(namespace="test", feature_id="x", namespace_version="v1"),)
    destination = tmp_path / "too-large.h5"
    with pytest.raises(OSError, match="capacity gate"):
        build_count_store(
            destination,
            matrix,
            row_ids=np.asarray([1]),
            features=feature,
            projected_peak_bytes=10**30,
        )
    assert not destination.exists()
