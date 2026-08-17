from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.contracts import FeatureKey, ShardAppendContract
from credo_count_sde_v4.store import (
    BoundedCSRShardWriter,
    CountStore,
    ShardedCountStore,
    ShardedCountStoreBuilder,
    build_count_store,
)


def _append_plan(
    chunks: tuple[tuple[np.ndarray, sparse.csr_matrix, str], ...],
    *,
    build_plan_id: str = "plan-1",
    shard_index: int = 0,
    source_id: str = "D1_Rest",
) -> tuple[ShardAppendContract, ...]:
    cursor = 0
    cumulative = hashlib.sha256(b"").hexdigest()
    contracts: list[ShardAppendContract] = []
    for index, (row_ids, matrix, guide_run_id) in enumerate(chunks):
        values = np.asarray(row_ids, dtype="<i8")
        cumulative = hashlib.sha256(bytes.fromhex(cumulative) + values.tobytes()).hexdigest()
        payload = {
            "schema_version": 1,
            "chunk_contract_id": "pending",
            "build_plan_id": build_plan_id,
            "shard_index": shard_index,
            "chunk_index": index,
            "source_id": source_id,
            "source_cursor_start": cursor,
            "source_cursor_end": cursor + len(values),
            "row_ids_hash": hashlib.sha256(values.tobytes()).hexdigest(),
            "guide_run_id": guide_run_id,
            "chunk_rows": len(values),
            "chunk_nnz": matrix.nnz,
            "cumulative_row_identity_hash": cumulative,
        }
        payload["chunk_contract_id"] = contract_id(payload, id_field="chunk_contract_id")
        contracts.append(ShardAppendContract.model_validate(payload))
        cursor += len(values)
    return tuple(contracts)


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


def test_persistent_reader_reuses_handle_and_rejects_process_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    with store:
        assert store.persistent_open
        first = store.rows(np.asarray([30, 10]))
        second = store.rows(np.asarray([10, 30]))
        assert first.matrix.toarray().tolist() == [[4, 0, 5], [1, 0, 2]]
        assert second.matrix.toarray().tolist() == [[1, 0, 2], [4, 0, 5]]
        owner = store._owner_pid
        assert owner is not None
        monkeypatch.setattr("credo_count_sde_v4.store.csr.os.getpid", lambda: owner + 1)
        with pytest.raises(Exception, match="process boundary"):
            store.rows(np.asarray([10]))
        monkeypatch.setattr("credo_count_sde_v4.store.csr.os.getpid", lambda: owner)
    assert not store.persistent_open


def test_rows_contiguous_fast_path_preserves_values(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = store.rows(np.asarray([20, 30]))
    assert batch.row_ids.tolist() == [20, 30]
    assert batch.matrix.toarray().tolist() == [[0, 3, 0], [4, 0, 5]]


def test_rows_multiple_contiguous_runs_preserve_duplicates_and_order(tmp_path: Path) -> None:
    matrix = sparse.csr_matrix(np.arange(80, dtype=np.int32).reshape(40, 2))
    features = tuple(
        FeatureKey(namespace="test", namespace_version="1", feature_id=f"g{i}") for i in range(2)
    )
    path = tmp_path / "counts.h5"
    row_ids = np.arange(100, 140, dtype=np.int64)
    manifest = build_count_store(path, matrix, row_ids=row_ids, features=features)
    requested = np.asarray([131, 130, 106, 107, 108, 131], dtype=np.int64)
    batch = CountStore(path, manifest).rows(requested)
    np.testing.assert_array_equal(batch.row_ids, requested)
    np.testing.assert_array_equal(batch.matrix.toarray(), matrix[requested - 100].toarray())


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


def test_sharded_builder_resume_publish_and_exact_rows(tmp_path: Path) -> None:
    features = tuple(
        FeatureKey(namespace="test", namespace_version="1", feature_id=f"g{i}") for i in range(3)
    )
    staging = tmp_path / ".counts.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=4,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    builder.append(
        sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.int32)),
        row_ids=np.asarray([10, 20]),
    )
    resumed = ShardedCountStoreBuilder(staging)
    assert resumed.completed_shards == 1
    assert resumed.rows_written == 2
    resumed.append(
        sparse.csr_matrix(np.asarray([[4, 0, 5], [0, 6, 7]], dtype=np.int32)),
        row_ids=np.asarray([30, 40]),
    )
    destination = tmp_path / "counts.store"
    manifest = resumed.finalize(destination)
    assert manifest.rows == 4
    assert manifest.nnz == 7
    assert not staging.exists()
    assert not (destination / "BUILD_STATE.json").exists()
    store = ShardedCountStore(destination)
    assert store.verify(full=True) == manifest
    assert validate_contract(destination / "manifest.json")["contract_type"] == (
        "ShardedCountStoreManifest"
    )
    batch = store.rows(np.asarray([40, 10, 30, 40]))
    assert store.persistent_reader_count == 2
    assert batch.row_ids.tolist() == [40, 10, 30, 40]
    assert batch.matrix.toarray().tolist() == [
        [0, 6, 7],
        [1, 0, 2],
        [4, 0, 5],
        [0, 6, 7],
    ]
    store.close()
    assert store.persistent_reader_count == 0


def test_sharded_store_rejects_overlap_missing_rows_and_corruption(tmp_path: Path) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    staging = tmp_path / ".counts.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=2,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    builder.append(sparse.csr_matrix([[1]], dtype=np.int32), row_ids=np.asarray([10]))
    builder.append(sparse.csr_matrix([[2]], dtype=np.int32), row_ids=np.asarray([10]))
    with pytest.raises(Exception, match="overlap across shards"):
        builder.finalize(tmp_path / "invalid.store")

    staging = tmp_path / ".valid.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=2,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    builder.append(sparse.csr_matrix([[1]], dtype=np.int32), row_ids=np.asarray([30]))
    builder.append(sparse.csr_matrix([[3]], dtype=np.int32), row_ids=np.asarray([10]))
    destination = tmp_path / "counts.store"
    builder.finalize(destination)
    store = ShardedCountStore(destination)
    with pytest.raises(KeyError):
        store.rows(np.asarray([20]))
    with (destination / "shards/shard-000001.h5").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(Exception, match="hash mismatch"):
        store.verify(full=True)


def test_sharded_store_routes_canonical_ids_independent_of_physical_order(
    tmp_path: Path,
) -> None:
    features = tuple(
        FeatureKey(namespace="test", namespace_version="1", feature_id=f"g{i}") for i in range(2)
    )
    staging = tmp_path / ".counts.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=4,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    builder.append(
        sparse.csr_matrix([[3, 30], [1, 10]], dtype=np.int32),
        row_ids=np.asarray([300, 100]),
    )
    builder.append(
        sparse.csr_matrix([[4, 40], [2, 20]], dtype=np.int32),
        row_ids=np.asarray([400, 200]),
    )
    destination = tmp_path / "counts.store"
    builder.finalize(destination)
    store = ShardedCountStore(destination)
    batch = store.rows(np.asarray([200, 300, 100, 200]))
    assert batch.matrix.toarray().tolist() == [[2, 20], [3, 30], [1, 10], [2, 20]]
    assert (
        store.verify(full=True).row_ids_hash
        == hashlib.sha256(np.asarray([100, 200, 300, 400], dtype="<i8").tobytes()).hexdigest()
    )


def test_sharded_builder_capacity_gate_and_no_clobber(tmp_path: Path) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    with pytest.raises(OSError, match="capacity gate"):
        ShardedCountStoreBuilder.create(
            tmp_path / ".too-large",
            features=features,
            expected_rows=1,
            projected_final_bytes=10**30,
            projected_peak_bytes=10**30,
        )
    destination = tmp_path / "counts.store"
    destination.mkdir()
    staging = tmp_path / ".counts.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=1,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    builder.append(sparse.csr_matrix([[1]], dtype=np.int32), row_ids=np.asarray([1]))
    with pytest.raises(FileExistsError):
        builder.finalize(destination)


def test_bounded_shard_writer_reconciles_uncommitted_suffix_and_finalizes(
    tmp_path: Path,
) -> None:
    features = tuple(
        FeatureKey(namespace="test", namespace_version="1", feature_id=f"g{i}") for i in range(3)
    )
    destination = tmp_path / "shard.h5"
    checkpoint_path = tmp_path / "shard-build.json"
    first_ids = np.asarray([30, 10])
    first_matrix = sparse.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
    second_ids = np.asarray([20])
    second_matrix = sparse.csr_matrix([[0, 0, 4]], dtype=np.int32)
    append_plan = _append_plan(
        (
            (first_ids, first_matrix, "guide-run-a"),
            (second_ids, second_matrix, "guide-run-b"),
        )
    )
    writer = BoundedCSRShardWriter.create(
        destination,
        checkpoint_path,
        features=features,
        expected_rows=3,
        expected_nnz=4,
        append_plan=append_plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    checkpoint = writer.append(
        first_matrix,
        row_ids=first_ids,
        source_cursor=2,
    )
    assert checkpoint.active_shard_rows == 2
    assert checkpoint.active_shard_nonzeros == 3
    writer.close()

    with pytest.raises(Exception, match="build-plan identity changed"):
        BoundedCSRShardWriter(
            destination,
            checkpoint_path,
            features=features,
            expected_rows=3,
            expected_nnz=4,
            append_plan=append_plan,
            build_plan_id="different-plan",
            shard_index=0,
            source_id="D1_Rest",
        )

    # Simulate payload bytes flushed before the typed checkpoint advanced.
    with h5py.File(tmp_path / ".shard.h5.partial", "r+") as handle:
        handle["X/data"].resize((4,))
        handle["X/indices"].resize((4,))
        handle["X/indptr"].resize((4,))
        handle["row_ids"].resize((3,))
        handle["X/data"][3] = 99
        handle["X/indices"][3] = 2
        handle["X/indptr"][3] = 4
        handle["row_ids"][2] = 999
        handle.flush()

    resumed = BoundedCSRShardWriter(
        destination,
        checkpoint_path,
        features=features,
        expected_rows=3,
        expected_nnz=4,
        append_plan=append_plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    with pytest.raises(Exception, match="exact next P2 chunk"):
        resumed.append(
            second_matrix,
            row_ids=second_ids,
            source_cursor=2,
        )
    resumed.append(
        second_matrix,
        row_ids=second_ids,
        source_cursor=3,
    )
    manifest, final_checkpoint = resumed.finalize()
    assert manifest.rows == 3
    assert manifest.nnz == 4
    assert final_checkpoint.active_shard_index is None
    assert final_checkpoint.completed_shard_hashes == (manifest.content_sha256,)
    batch = CountStore(destination, manifest).rows(np.asarray([10, 20, 30]))
    assert batch.matrix.toarray().tolist() == [[0, 3, 0], [0, 0, 4], [1, 0, 2]]
    assert not (tmp_path / ".shard.h5.partial").exists()


def test_bounded_writer_rejects_wrong_planned_rows_before_payload_advance(
    tmp_path: Path,
) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    matrix = sparse.csr_matrix([[1], [2]], dtype=np.int32)
    planned_ids = np.asarray([10, 20])
    append_plan = _append_plan(((planned_ids, matrix, "guide-run-a"),))
    writer = BoundedCSRShardWriter.create(
        tmp_path / "shard.h5",
        tmp_path / "checkpoint.json",
        features=features,
        expected_rows=2,
        expected_nnz=2,
        append_plan=append_plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    with pytest.raises(Exception, match="exact next P2 chunk"):
        writer.append(matrix, row_ids=planned_ids[::-1], source_cursor=2)
    assert writer.checkpoint.row_count == 0
    assert writer.checkpoint.nonzero_count == 0
    writer.close()


@pytest.mark.parametrize(
    "boundary",
    (
        "after_verify_partial",
        "after_manifest",
        "after_finalizing_checkpoint",
        "after_rename",
        "after_directory_fsync",
        "after_finalized_checkpoint",
    ),
)
def test_bounded_writer_recovers_each_finalization_crash_boundary(
    tmp_path: Path, boundary: str
) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    ids = np.asarray([10, 20])
    matrix = sparse.csr_matrix([[1], [2]], dtype=np.int32)
    append_plan = _append_plan(((ids, matrix, "guide-run-a"),))

    def fail_at(observed: str) -> None:
        if observed == boundary:
            raise RuntimeError(boundary)

    destination = tmp_path / "shard.h5"
    checkpoint_path = tmp_path / "checkpoint.json"
    writer = BoundedCSRShardWriter.create(
        destination,
        checkpoint_path,
        features=features,
        expected_rows=2,
        expected_nnz=2,
        append_plan=append_plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
        fault_injector=fail_at,
    )
    writer.append(matrix, row_ids=ids, source_cursor=2)
    with pytest.raises(RuntimeError, match=boundary):
        writer.finalize()
    recovered = BoundedCSRShardWriter(
        destination,
        checkpoint_path,
        features=features,
        expected_rows=2,
        expected_nnz=2,
        append_plan=append_plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    manifest, checkpoint = recovered.finalize()
    assert checkpoint.phase == "FINALIZED"
    assert checkpoint.finalizing_manifest == manifest
    assert CountStore(destination, manifest).verify(full=True) == manifest


def test_sharded_reader_lru_bounds_handles_and_reports_index_memory(tmp_path: Path) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    staging = tmp_path / ".counts.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=3,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    for row_id in (10, 20, 30):
        builder.append(sparse.csr_matrix([[row_id]], dtype=np.int32), row_ids=np.asarray([row_id]))
    destination = tmp_path / "counts.store"
    builder.finalize(destination)
    store = ShardedCountStore(destination, max_open_shards=2)
    for row_id in (10, 20, 30):
        store.rows(np.asarray([row_id]))
    metrics = store.memory_metrics()
    assert metrics["open_shard_readers"] == 2
    assert metrics["maximum_open_shards"] == 2
    assert metrics["locator_cache_bytes"] == 3 * (8 + 4 + 8)
    assert metrics["shard_index_cache_bytes"] > 0
    assert metrics["hdf5_chunk_cache_bytes"] > 0
    store.close()


def test_sharded_reader_empty_cursor_context_and_constructor_guards(tmp_path: Path) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    staging = tmp_path / ".counts.build"
    builder = ShardedCountStoreBuilder.create(
        staging,
        features=features,
        expected_rows=2,
        projected_final_bytes=4_096,
        projected_peak_bytes=8_192,
    )
    builder.append(sparse.csr_matrix([[1]], dtype=np.int32), row_ids=np.asarray([10]))
    builder.append(sparse.csr_matrix([[2]], dtype=np.int32), row_ids=np.asarray([20]))
    destination = tmp_path / "counts.store"
    builder.finalize(destination)

    with pytest.raises(ValueError, match="max_open_shards"):
        ShardedCountStore(destination, max_open_shards=0)
    with pytest.raises(Exception, match="not a regular directory"):
        ShardedCountStore(tmp_path / "missing")
    marker = destination / "COMMITTED"
    original = marker.read_text()
    marker.write_text("wrong\n")
    with pytest.raises(Exception, match="commit marker mismatch"):
        ShardedCountStore(destination)
    marker.write_text(original)

    with ShardedCountStore(destination, max_open_shards=1) as store:
        assert store.locator_cache_bytes == 0
        empty = store.rows(np.asarray([], dtype=np.int64))
        assert empty.matrix.shape == (0, 1)
        list(store.iter_batches(np.asarray([10, 20]), batch_size=1))
        # Re-read the resident second shard to exercise the LRU hit path.
        assert store.rows(np.asarray([20])).matrix[0, 0] == 2
        for batch_size, cursor in ((0, 0), (1, -1)):
            with pytest.raises(ValueError, match="batch_size"):
                list(store.iter_batches(np.asarray([10]), batch_size=batch_size, cursor=cursor))


def test_bounded_writer_rejects_constructor_and_append_contract_errors(
    tmp_path: Path,
) -> None:
    features = (FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),)
    ids = np.asarray([10])
    matrix = sparse.csr_matrix([[1]], dtype=np.int32)
    plan = _append_plan(((ids, matrix, "guide-run-a"),))
    with pytest.raises(Exception, match="nonempty exact append plan"):
        BoundedCSRShardWriter(
            tmp_path / "missing.h5",
            tmp_path / "missing-checkpoint.json",
            features=features,
            expected_rows=1,
            expected_nnz=1,
            append_plan=(),
        )
    with pytest.raises(Exception, match="lacks its typed checkpoint"):
        BoundedCSRShardWriter(
            tmp_path / "missing.h5",
            tmp_path / "missing-checkpoint.json",
            features=features,
            expected_rows=1,
            expected_nnz=1,
            append_plan=plan,
        )

    destination = tmp_path / "shard.h5"
    checkpoint = tmp_path / "checkpoint.json"
    writer = BoundedCSRShardWriter.create(
        destination,
        checkpoint,
        features=features,
        expected_rows=1,
        expected_nnz=1,
        append_plan=plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    writer.close()
    for kwargs, message in (
        ({"shard_index": 1}, "shard index changed"),
        ({"source_id": "D2_Rest"}, "source identity changed"),
    ):
        with pytest.raises(Exception, match=message):
            BoundedCSRShardWriter(
                destination,
                checkpoint,
                features=features,
                expected_rows=1,
                expected_nnz=1,
                append_plan=plan,
                build_plan_id="plan-1",
                **kwargs,
            )

    dimensional = BoundedCSRShardWriter(
        destination,
        checkpoint,
        features=features,
        expected_rows=1,
        expected_nnz=1,
        append_plan=plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    with pytest.raises(Exception, match="dimensions disagree"):
        dimensional.append(
            sparse.csr_matrix([[1, 2]], dtype=np.int32),
            row_ids=ids,
            source_cursor=1,
        )
    dimensional.close()

    def reject(
        test_matrix: sparse.spmatrix,
        test_ids: np.ndarray,
        expected_rows: int,
        expected_nnz: int,
        message: str,
        suffix: str,
    ) -> None:
        test_plan = _append_plan(((test_ids, sparse.csr_matrix(test_matrix), "guide"),))
        bounded = BoundedCSRShardWriter.create(
            tmp_path / f"{suffix}.h5",
            tmp_path / f"{suffix}.json",
            features=features,
            expected_rows=expected_rows,
            expected_nnz=expected_nnz,
            append_plan=test_plan,
            build_plan_id="plan-1",
            shard_index=0,
            source_id="D1_Rest",
        )
        with pytest.raises(Exception, match=message):
            bounded.append(test_matrix, row_ids=test_ids, source_cursor=len(test_ids))
        bounded.close()

    reject(
        sparse.csr_matrix([[1], [2]], dtype=np.int32),
        np.asarray([10, 10]),
        2,
        2,
        "duplicated row IDs",
        "duplicate",
    )
    reject(sparse.csr_matrix([[1.5]], dtype=np.float32), ids, 1, 1, "nonnegative integer", "float")
    reject(sparse.csr_matrix([[-1]], dtype=np.int32), ids, 1, 1, "nonnegative integer", "negative")
    reject(
        sparse.csr_matrix([[np.iinfo(np.int32).max + 1]], dtype=np.int64),
        ids,
        1,
        1,
        "exceeds int32",
        "overflow",
    )

    wide_features = (
        FeatureKey(namespace="test", namespace_version="1", feature_id="g0"),
        FeatureKey(namespace="test", namespace_version="1", feature_id="g1"),
    )
    oversized_plan = _append_plan(((ids, sparse.csr_matrix([[1, 0]], dtype=np.int32), "guide"),))
    oversized = BoundedCSRShardWriter.create(
        tmp_path / "oversized.h5",
        tmp_path / "oversized.json",
        features=wide_features,
        expected_rows=1,
        expected_nnz=1,
        append_plan=oversized_plan,
        build_plan_id="plan-1",
        shard_index=0,
        source_id="D1_Rest",
    )
    with pytest.raises(Exception, match="frozen row/nonzero budget"):
        oversized.append(
            sparse.csr_matrix([[1, 2]], dtype=np.int32),
            row_ids=ids,
            source_cursor=1,
        )
    oversized.close()
