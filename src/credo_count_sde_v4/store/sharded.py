"""Restartable construction and exact reads for row-partitioned CSR stores."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from ..canonical import canonical_json_bytes, contract_id, sha256_bytes, sha256_file
from ..contracts import (
    CountStoreManifest,
    CountStoreShard,
    FeatureKey,
    ShardBuildCheckpoint,
    ShardedCountStoreManifest,
)
from ..errors import ContractError, IntegrityError
from .csr import CountStore, SparseCountBatch, _feature_hash, _row_hash, build_count_store


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _merkle_root(shards: tuple[CountStoreShard, ...], row_locator_sha256: str) -> str:
    leaves = [sha256_bytes(canonical_json_bytes(item.model_dump(mode="json"))) for item in shards]
    leaves.append(row_locator_sha256)
    if not leaves:
        raise ContractError("A sharded store needs at least one Merkle leaf.")
    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            sha256_bytes(bytes.fromhex(level[index]) + bytes.fromhex(level[index + 1]))
            for index in range(0, len(level), 2)
        ]
    return level[0]


def _count_manifest(shard: CountStoreShard, feature_index_hash: str) -> CountStoreManifest:
    return CountStoreManifest(
        store_id=shard.shard_id,
        rows=shard.rows,
        features=shard.features,
        nnz=shard.nnz,
        value_dtype="int32",
        row_ids_hash=shard.row_ids_hash,
        feature_index_hash=feature_index_hash,
        content_sha256=shard.content_sha256,
        relative_uri=Path(shard.relative_uri).name,
    )


def _active_offsets_hash(*, rows: int, nnz: int, chunks: int, last_offset: int) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"rows": rows, "nnz": nnz, "chunks": chunks, "last_offset": last_offset}
        )
    )


class BoundedCSRShardWriter:
    """Restartable, process-local writer for one bounded physical CSR shard.

    HDF5 payload bytes are flushed before the typed checkpoint advances. On
    resume, any uncheckpointed suffix is truncated to the last committed row
    and nonzero offsets. The final immutable file is never overwritten.
    """

    def __init__(
        self,
        destination: Path,
        checkpoint_path: Path,
        *,
        features: tuple[FeatureKey, ...],
        expected_rows: int,
        expected_nnz: int,
        build_plan_id: str | None = None,
        shard_index: int | None = None,
        source_id: str | None = None,
    ) -> None:
        self.destination = destination
        self.partial = destination.with_name(f".{destination.name}.partial")
        self.checkpoint_path = checkpoint_path
        self.features = features
        self.expected_rows = expected_rows
        self.expected_nnz = expected_nnz
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if not self.partial.is_file() or not checkpoint_path.is_file():
            raise IntegrityError("Bounded shard writer lacks its partial file or checkpoint.")
        self.checkpoint = ShardBuildCheckpoint.model_validate_json(checkpoint_path.read_text())
        if self.checkpoint.active_shard_index is None:
            raise IntegrityError("Bounded shard writer checkpoint is already finalized.")
        if build_plan_id is not None and self.checkpoint.build_plan_id != build_plan_id:
            raise IntegrityError("Bounded shard writer build-plan identity changed.")
        if shard_index is not None and self.checkpoint.active_shard_index != shard_index:
            raise IntegrityError("Bounded shard writer shard index changed.")
        if source_id is not None and self.checkpoint.source_id != source_id:
            raise IntegrityError("Bounded shard writer source identity changed.")
        if self.checkpoint.active_shard_rows > expected_rows:
            raise IntegrityError("Checkpoint rows exceed the frozen shard bound.")
        if self.checkpoint.active_shard_nonzeros > expected_nnz:
            raise IntegrityError("Checkpoint nonzeros exceed the frozen shard bound.")
        self._owner_pid = os.getpid()
        self._handle = h5py.File(self.partial, "r+")
        try:
            expected_shape = tuple(map(int, self._handle.attrs.get("expected_shape", ())))
            if expected_shape != (expected_rows, len(features)):
                raise IntegrityError("Partial shard expected shape differs from the frozen plan.")
            if int(self._handle.attrs.get("expected_nnz", -1)) != expected_nnz:
                raise IntegrityError("Partial shard nonzero budget differs from the frozen plan.")
            if self._handle.attrs.get("feature_index_hash") != _feature_hash(features):
                raise IntegrityError("Partial shard feature index differs from the frozen plan.")
            self._reconcile()
        except Exception:
            self._handle.close()
            raise

    @classmethod
    def create(
        cls,
        destination: Path,
        checkpoint_path: Path,
        *,
        features: tuple[FeatureKey, ...],
        expected_rows: int,
        expected_nnz: int,
        build_plan_id: str,
        shard_index: int,
        source_id: str,
    ) -> BoundedCSRShardWriter:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        partial = destination.with_name(f".{destination.name}.partial")
        if partial.exists() or checkpoint_path.exists():
            raise FileExistsError(partial if partial.exists() else checkpoint_path)
        if expected_rows <= 0 or expected_nnz < 0 or shard_index < 0:
            raise ValueError("Frozen shard rows/index must be positive and nonzeros nonnegative.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(partial, "x", libver="latest") as handle:
            group = handle.create_group("X")
            group.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.int32,
                compression="gzip",
                shuffle=True,
            )
            group.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.int32,
                compression="gzip",
            )
            group.create_dataset(
                "indptr",
                data=np.asarray([0], dtype=np.int64),
                maxshape=(None,),
                chunks=True,
                compression="gzip",
            )
            handle.create_dataset(
                "row_ids",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.int64,
                compression="gzip",
            )
            encoded = [
                json.dumps(feature.model_dump(mode="json"), sort_keys=True) for feature in features
            ]
            handle.create_dataset(
                "features", data=np.asarray(encoded, dtype=h5py.string_dtype("utf-8"))
            )
            handle.attrs["schema_id"] = "credo.count_store.partial"
            handle.attrs["schema_version"] = 1
            handle.attrs["feature_index_hash"] = _feature_hash(features)
            handle.attrs["expected_shape"] = (expected_rows, len(features))
            handle.attrs["expected_nnz"] = expected_nnz
            handle.flush()
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        checkpoint = ShardBuildCheckpoint(
            build_plan_id=build_plan_id,
            source_cursor=0,
            completed_shard_hashes=(),
            temporary_shard_ids=(partial.name,),
            row_count=0,
            nonzero_count=0,
            active_shard_index=shard_index,
            active_shard_rows=0,
            active_shard_nonzeros=0,
            committed_chunks=0,
            source_id=source_id,
            active_offsets_sha256=_active_offsets_hash(rows=0, nnz=0, chunks=0, last_offset=0),
        )
        _write_json_atomic(checkpoint_path, checkpoint.model_dump(mode="json"))
        return cls(
            destination,
            checkpoint_path,
            features=features,
            expected_rows=expected_rows,
            expected_nnz=expected_nnz,
            build_plan_id=build_plan_id,
            shard_index=shard_index,
            source_id=source_id,
        )

    def _assert_owner(self) -> None:
        if self._owner_pid != os.getpid():
            raise IntegrityError("A bounded shard writer cannot cross a process boundary.")

    def _reconcile(self) -> None:
        self._assert_owner()
        rows = self.checkpoint.active_shard_rows
        nnz = self.checkpoint.active_shard_nonzeros
        data = self._handle["X/data"]
        indices = self._handle["X/indices"]
        indptr = self._handle["X/indptr"]
        row_ids = self._handle["row_ids"]
        if len(data) < nnz or len(indices) < nnz or len(indptr) < rows + 1 or len(row_ids) < rows:
            raise IntegrityError("Partial shard is shorter than its committed checkpoint.")
        data.resize((nnz,))
        indices.resize((nnz,))
        indptr.resize((rows + 1,))
        row_ids.resize((rows,))
        if int(indptr[-1]) != nnz:
            raise IntegrityError("Partial shard offset disagrees with its committed checkpoint.")
        expected_hash = _active_offsets_hash(
            rows=rows,
            nnz=nnz,
            chunks=self.checkpoint.committed_chunks,
            last_offset=int(indptr[-1]),
        )
        if self.checkpoint.active_offsets_sha256 != expected_hash:
            raise IntegrityError("Partial shard offsets differ from the typed checkpoint.")
        self._handle.flush()

    def append(
        self, matrix: sparse.spmatrix, *, row_ids: np.ndarray, source_cursor: int
    ) -> ShardBuildCheckpoint:
        self._assert_owner()
        csr = sparse.csr_matrix(matrix)
        csr.sort_indices()
        ids = np.asarray(row_ids, dtype=np.int64)
        if csr.shape != (len(ids), len(self.features)) or not len(ids):
            raise ContractError("Bounded shard append dimensions disagree.")
        if len(np.unique(ids)) != len(ids):
            raise ContractError("Bounded shard append contains duplicated row IDs.")
        if not np.issubdtype(csr.dtype, np.integer) or np.any(csr.data < 0):
            raise ContractError("Bounded shard append requires nonnegative integer counts.")
        if csr.nnz and int(csr.data.max()) > np.iinfo(np.int32).max:
            raise ContractError("Bounded shard append exceeds int32 count storage.")
        rows_before = self.checkpoint.active_shard_rows
        nnz_before = self.checkpoint.active_shard_nonzeros
        rows_after = rows_before + len(ids)
        nnz_after = nnz_before + csr.nnz
        if rows_after > self.expected_rows or nnz_after > self.expected_nnz:
            raise ContractError("Bounded shard append exceeds its frozen row/nonzero budget.")
        if source_cursor <= self.checkpoint.source_cursor:
            raise ContractError("Source cursor must advance monotonically at every append.")
        data = self._handle["X/data"]
        indices = self._handle["X/indices"]
        indptr = self._handle["X/indptr"]
        stored_ids = self._handle["row_ids"]
        data.resize((nnz_after,))
        indices.resize((nnz_after,))
        indptr.resize((rows_after + 1,))
        stored_ids.resize((rows_after,))
        data[nnz_before:nnz_after] = csr.data.astype(np.int32, copy=False)
        indices[nnz_before:nnz_after] = csr.indices.astype(np.int32, copy=False)
        indptr[rows_before + 1 : rows_after + 1] = csr.indptr[1:].astype(np.int64) + nnz_before
        stored_ids[rows_before:rows_after] = ids
        self._handle.flush()
        with self.partial.open("rb") as handle:
            os.fsync(handle.fileno())
        checkpoint = self.checkpoint.model_copy(
            update={
                "source_cursor": source_cursor,
                "row_count": rows_after,
                "nonzero_count": nnz_after,
                "active_shard_rows": rows_after,
                "active_shard_nonzeros": nnz_after,
                "committed_chunks": self.checkpoint.committed_chunks + 1,
                "active_offsets_sha256": _active_offsets_hash(
                    rows=rows_after,
                    nnz=nnz_after,
                    chunks=self.checkpoint.committed_chunks + 1,
                    last_offset=nnz_after,
                ),
            }
        )
        _write_json_atomic(self.checkpoint_path, checkpoint.model_dump(mode="json"))
        self.checkpoint = checkpoint
        return checkpoint

    def close(self) -> None:
        self._assert_owner()
        if self._handle:
            self._handle.close()

    def finalize(self) -> tuple[CountStoreManifest, ShardBuildCheckpoint]:
        self._assert_owner()
        if self.checkpoint.active_shard_rows != self.expected_rows:
            raise ContractError("Bounded shard row count does not match its frozen plan.")
        if self.checkpoint.active_shard_nonzeros != self.expected_nnz:
            raise ContractError("Bounded shard nonzero count does not match its frozen plan.")
        ids = np.asarray(self._handle["row_ids"][:], dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        if len(sorted_ids) != len(np.unique(sorted_ids)):
            raise IntegrityError("Bounded shard row IDs overlap across committed chunks.")
        self._handle.create_dataset("row_ids_sorted", data=sorted_ids, compression="gzip")
        self._handle.create_dataset(
            "row_positions_sorted", data=order.astype(np.int64), compression="gzip"
        )
        self._handle.attrs["schema_id"] = "credo.count_store"
        self._handle.attrs["shape"] = (self.expected_rows, len(self.features))
        self._handle.flush()
        self._handle.close()
        with self.partial.open("rb") as handle:
            os.fsync(handle.fileno())
        content = sha256_file(self.partial)
        payload = {
            "schema_version": 1,
            "store_id": "pending",
            "backend": "csr_hdf5",
            "rows": self.expected_rows,
            "features": len(self.features),
            "nnz": self.expected_nnz,
            "value_dtype": "int32",
            "row_ids_hash": _row_hash(ids),
            "feature_index_hash": _feature_hash(self.features),
            "content_sha256": content,
            "relative_uri": self.destination.name,
        }
        payload["store_id"] = contract_id(payload, id_field="store_id")
        manifest = CountStoreManifest.model_validate(payload)
        CountStore(self.partial, manifest).verify(full=True)
        os.replace(self.partial, self.destination)
        descriptor = os.open(self.destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        checkpoint = self.checkpoint.model_copy(
            update={
                "completed_shard_hashes": (content,),
                "temporary_shard_ids": (),
                "active_shard_index": None,
                "active_shard_rows": 0,
                "active_shard_nonzeros": 0,
                "active_offsets_sha256": None,
            }
        )
        _write_json_atomic(self.checkpoint_path, checkpoint.model_dump(mode="json"))
        self.checkpoint = checkpoint
        return manifest, checkpoint

    def __enter__(self) -> BoundedCSRShardWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle:
            self.close()


class ShardedCountStoreBuilder:
    """Append bounded CSR shards and atomically publish one immutable directory.

    The staging directory is deliberately persistent and may be reopened after
    interruption. Publication requires the destination to share its filesystem
    so the final directory rename is atomic.
    """

    _STATE = "BUILD_STATE.json"
    _FEATURES = "FEATURES.json"

    def __init__(self, staging: Path) -> None:
        self.staging = staging
        state_path = staging / self._STATE
        features_path = staging / self._FEATURES
        if staging.is_symlink() or not state_path.is_file() or not features_path.is_file():
            raise IntegrityError("Incomplete or unsafe sharded-store staging directory.")
        self._state = json.loads(state_path.read_text())
        self.features = tuple(
            FeatureKey.model_validate(item) for item in json.loads(features_path.read_text())
        )
        if _feature_hash(self.features) != self._state["feature_index_hash"]:
            raise IntegrityError("Staged feature index hash mismatch.")

    @classmethod
    def create(
        cls,
        staging: Path,
        *,
        features: tuple[FeatureKey, ...],
        expected_rows: int,
        projected_final_bytes: int,
        projected_peak_bytes: int,
    ) -> ShardedCountStoreBuilder:
        if staging.exists():
            raise FileExistsError(staging)
        if expected_rows <= 0 or projected_final_bytes <= 0 or projected_peak_bytes <= 0:
            raise ValueError("Expected rows and projected byte counts must be positive.")
        feature_keys = {
            (item.namespace, item.namespace_version, item.feature_id) for item in features
        }
        if len(feature_keys) != len(features):
            raise ContractError("Composite feature keys must be unique.")
        capacity = shutil.disk_usage(staging.parent)
        if capacity.free < int(1.20 * projected_peak_bytes) or (
            capacity.free - projected_final_bytes < int(0.20 * capacity.total)
        ):
            raise OSError("Sharded count-store capacity gate failed.")
        staging.mkdir(parents=False)
        (staging / "shards").mkdir()
        feature_payload = [item.model_dump(mode="json") for item in features]
        _write_json_atomic(staging / cls._FEATURES, feature_payload)
        payload = {
            "schema_version": 1,
            "expected_rows": expected_rows,
            "projected_final_bytes": projected_final_bytes,
            "projected_peak_bytes": projected_peak_bytes,
            "feature_index_hash": _feature_hash(features),
            "rows": 0,
            "nnz": 0,
            "shards": [],
        }
        _write_json_atomic(staging / cls._STATE, payload)
        return cls(staging)

    @property
    def completed_shards(self) -> int:
        return len(self._state["shards"])

    @property
    def rows_written(self) -> int:
        return int(self._state["rows"])

    def append(self, matrix: sparse.spmatrix, *, row_ids: np.ndarray) -> CountStoreShard:
        csr = sparse.csr_matrix(matrix)
        ids = np.asarray(row_ids, dtype=np.int64)
        if csr.shape != (len(ids), len(self.features)) or not len(ids):
            raise ContractError("Shard matrix, row IDs, and feature dimensions disagree.")
        if len(np.unique(ids)) != len(ids):
            raise ContractError("Shard row IDs must be unique.")
        index = self.completed_shards
        relative_uri = f"shards/shard-{index:06d}.h5"
        path = self.staging / relative_uri
        manifest = build_count_store(
            path,
            csr,
            row_ids=ids,
            features=self.features,
            projected_peak_bytes=max(int(self._state["projected_peak_bytes"]), 4096),
        )
        shard = CountStoreShard(
            shard_id=manifest.store_id,
            relative_uri=relative_uri,
            rows=manifest.rows,
            features=manifest.features,
            nnz=manifest.nnz,
            row_id_min=int(ids.min()),
            row_id_max=int(ids.max()),
            row_ids_hash=manifest.row_ids_hash,
            content_sha256=manifest.content_sha256,
        )
        state = dict(self._state)
        state["rows"] = int(state["rows"]) + shard.rows
        state["nnz"] = int(state["nnz"]) + shard.nnz
        state["shards"] = [*state["shards"], shard.model_dump(mode="json")]
        _write_json_atomic(self.staging / self._STATE, state)
        self._state = state
        return shard

    def finalize(self, destination: Path) -> ShardedCountStoreManifest:
        if destination.exists():
            raise FileExistsError(destination)
        if destination.parent.resolve() != self.staging.parent.resolve():
            raise ContractError("Atomic publication requires staging beside the destination.")
        if int(self._state["rows"]) != int(self._state["expected_rows"]):
            raise ContractError("Staged row count does not match the frozen expected count.")
        shards = tuple(CountStoreShard.model_validate(item) for item in self._state["shards"])
        row_id_parts: list[np.ndarray] = []
        shard_index_parts: list[np.ndarray] = []
        row_position_parts: list[np.ndarray] = []
        for shard_index, shard in enumerate(shards):
            path = self.staging / shard.relative_uri
            CountStore(path, _count_manifest(shard, self._state["feature_index_hash"])).verify(
                full=True
            )
            with h5py.File(path, "r") as handle:
                ids = np.asarray(handle["row_ids"][:], dtype=np.int64)
            row_id_parts.append(ids)
            shard_index_parts.append(np.full(len(ids), shard_index, dtype=np.int32))
            row_position_parts.append(np.arange(len(ids), dtype=np.int64))
        all_row_ids = np.concatenate(row_id_parts)
        all_shard_indices = np.concatenate(shard_index_parts)
        all_row_positions = np.concatenate(row_position_parts)
        order = np.argsort(all_row_ids, kind="stable")
        sorted_ids = all_row_ids[order].astype("<i8", copy=False)
        if len(sorted_ids) != len(np.unique(sorted_ids)):
            raise IntegrityError("Canonical row IDs overlap across shards.")
        locator_path = self.staging / "row-locator.h5"
        with h5py.File(locator_path, "x", libver="latest") as handle:
            handle.create_dataset("row_ids_sorted", data=sorted_ids, compression="gzip")
            handle.create_dataset(
                "shard_indices_sorted",
                data=all_shard_indices[order].astype(np.int32, copy=False),
                compression="gzip",
            )
            handle.create_dataset(
                "row_positions_sorted",
                data=all_row_positions[order].astype(np.int64, copy=False),
                compression="gzip",
            )
            handle.attrs["schema_id"] = "credo.sharded_count_store.row_locator"
            handle.attrs["schema_version"] = 1
            handle.flush()
        with locator_path.open("rb") as handle:
            os.fsync(handle.fileno())
        locator_sha256 = sha256_file(locator_path)
        payload = {
            "schema_version": 1,
            "store_id": "pending",
            "backend": "csr_hdf5_sharded",
            "rows": int(self._state["rows"]),
            "features": len(self.features),
            "nnz": int(self._state["nnz"]),
            "value_dtype": "int32",
            "row_ids_hash": hashlib.sha256(sorted_ids.tobytes(order="C")).hexdigest(),
            "feature_index_hash": self._state["feature_index_hash"],
            "row_locator_sha256": locator_sha256,
            "row_locator_relative_uri": "row-locator.h5",
            "merkle_root": _merkle_root(shards, locator_sha256),
            "shards": [item.model_dump(mode="json") for item in shards],
        }
        payload["store_id"] = contract_id(payload, id_field="store_id")
        manifest = ShardedCountStoreManifest.model_validate(payload)
        _write_json_atomic(self.staging / "manifest.json", manifest.model_dump(mode="json"))
        (self.staging / "COMMITTED").write_text(manifest.store_id + "\n")
        with (self.staging / "COMMITTED").open("rb") as handle:
            os.fsync(handle.fileno())
        ShardedCountStore(self.staging).verify(full=True)
        (self.staging / self._STATE).unlink()
        (self.staging / self._FEATURES).unlink()
        staging_descriptor = os.open(self.staging, os.O_RDONLY)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        os.replace(self.staging, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.staging = destination
        return manifest


class ShardedCountStore:
    """Read-only facade over an immutable row-ordered CSR shard directory."""

    def __init__(self, path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise IntegrityError(f"Sharded count store is not a regular directory: {path}.")
        if not (path / "COMMITTED").is_file():
            raise IntegrityError("Sharded count store lacks its COMMITTED marker.")
        self.path = path
        self.manifest = ShardedCountStoreManifest.model_validate_json(
            (path / "manifest.json").read_text()
        )
        if (path / "COMMITTED").read_text().strip() != self.manifest.store_id:
            raise IntegrityError("Sharded count-store commit marker mismatch.")
        self._locator_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._shard_readers: dict[int, CountStore] = {}

    @property
    def persistent_reader_count(self) -> int:
        """Number of process-local shard handles opened by prior reads."""

        return len(self._shard_readers)

    def _reader(self, shard_index: int) -> CountStore:
        reader = self._shard_readers.get(shard_index)
        if reader is None:
            shard = self.manifest.shards[shard_index]
            reader = CountStore(
                self.path / shard.relative_uri,
                _count_manifest(shard, self.manifest.feature_index_hash),
            ).open()
            self._shard_readers[shard_index] = reader
        return reader

    def close(self) -> None:
        """Close all process-local HDF5 handles retained by this facade."""

        for reader in self._shard_readers.values():
            reader.close()
        self._shard_readers.clear()

    def __enter__(self) -> ShardedCountStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _locator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._locator_cache is None:
            with h5py.File(self.path / self.manifest.row_locator_relative_uri, "r") as handle:
                self._locator_cache = (
                    np.asarray(handle["row_ids_sorted"][:], dtype=np.int64),
                    np.asarray(handle["shard_indices_sorted"][:], dtype=np.int32),
                    np.asarray(handle["row_positions_sorted"][:], dtype=np.int64),
                )
        return self._locator_cache

    def verify(self, *, full: bool = True) -> ShardedCountStoreManifest:
        observed_rows = 0
        observed_nnz = 0
        per_shard_ids: list[np.ndarray] = []
        for shard in self.manifest.shards:
            path = self.path / shard.relative_uri
            store = CountStore(path, _count_manifest(shard, self.manifest.feature_index_hash))
            store.verify(full=full)
            ids = store.row_ids().astype("<i8", copy=False)
            if int(ids.min()) != shard.row_id_min or int(ids.max()) != shard.row_id_max:
                raise IntegrityError("Shard row bounds disagree with its manifest.")
            per_shard_ids.append(ids)
            observed_rows += shard.rows
            observed_nnz += shard.nnz
        if observed_rows != self.manifest.rows or observed_nnz != self.manifest.nnz:
            raise IntegrityError("Sharded count-store totals are inconsistent.")
        locator_path = self.path / self.manifest.row_locator_relative_uri
        if full and sha256_file(locator_path) != self.manifest.row_locator_sha256:
            raise IntegrityError("Row-locator content hash mismatch.")
        sorted_ids, shard_indices, row_positions = self._locator()
        if (
            len(sorted_ids) != self.manifest.rows
            or len(shard_indices) != self.manifest.rows
            or len(row_positions) != self.manifest.rows
            or np.any(np.diff(sorted_ids) <= 0)
        ):
            raise IntegrityError("Row locator is incomplete, duplicated, or unordered.")
        if hashlib.sha256(sorted_ids.astype("<i8", copy=False).tobytes()).hexdigest() != (
            self.manifest.row_ids_hash
        ):
            raise IntegrityError("Global row identity hash mismatch.")
        if np.any(shard_indices < 0) or np.any(shard_indices >= len(self.manifest.shards)):
            raise IntegrityError("Row locator references an unknown shard.")
        locator_order = np.lexsort((row_positions, shard_indices))
        located_ids = sorted_ids[locator_order]
        located_shards = shard_indices[locator_order]
        located_positions = row_positions[locator_order]
        cursor = 0
        for shard_index, source_ids in enumerate(per_shard_ids):
            count = len(source_ids)
            section = slice(cursor, cursor + count)
            if (
                np.any(located_shards[section] != shard_index)
                or not np.array_equal(located_positions[section], np.arange(count))
                or not np.array_equal(located_ids[section], source_ids)
            ):
                raise IntegrityError("Row locator disagrees with physical shard rows.")
            cursor += count
        if (
            _merkle_root(self.manifest.shards, self.manifest.row_locator_sha256)
            != self.manifest.merkle_root
        ):
            raise IntegrityError("Shard Merkle root mismatch.")
        return self.manifest

    def rows(self, row_ids: np.ndarray) -> SparseCountBatch:
        requested = np.asarray(row_ids, dtype=np.int64)
        if not len(requested):
            return SparseCountBatch(
                matrix=sparse.csr_matrix((0, self.manifest.features), dtype=np.int32),
                row_ids=requested.copy(),
                feature_index_hash=self.manifest.feature_index_hash,
            )
        sorted_ids, locator_shards, _ = self._locator()
        found = np.searchsorted(sorted_ids, requested)
        valid = found < len(sorted_ids)
        candidates = np.where(valid)[0]
        valid[candidates] = sorted_ids[found[candidates]] == requested[candidates]
        if not np.all(valid):
            missing = requested[~valid].astype(int).tolist()
            raise KeyError(f"Unknown sharded count-store row IDs: {missing[:10]}.")
        shard_indices = locator_shards[found]
        blocks: list[sparse.csr_matrix] = []
        positions: list[np.ndarray] = []
        for shard_index in np.unique(shard_indices):
            selected_positions = np.where(shard_indices == shard_index)[0]
            store = self._reader(int(shard_index))
            blocks.append(store.rows(requested[selected_positions]).matrix)
            positions.append(selected_positions)
        stacked = sparse.vstack(blocks, format="csr")
        concatenated = np.concatenate(positions)
        reordered = stacked[np.argsort(concatenated, kind="stable")].tocsr()
        return SparseCountBatch(
            matrix=reordered,
            row_ids=requested.copy(),
            feature_index_hash=self.manifest.feature_index_hash,
        )

    def iter_batches(
        self, ordered_row_ids: np.ndarray, *, batch_size: int, cursor: int = 0
    ) -> Iterator[tuple[int, SparseCountBatch]]:
        if batch_size <= 0 or cursor < 0:
            raise ValueError("batch_size must be positive and cursor nonnegative.")
        ids = np.asarray(ordered_row_ids, dtype=np.int64)
        while cursor < len(ids):
            end = min(cursor + batch_size, len(ids))
            yield end, self.rows(ids[cursor:end])
            cursor = end
