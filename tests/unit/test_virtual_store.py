from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter
from scipy import sparse

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.canonical import canonical_json_bytes, contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    G00SourcePlaneV2Amendment,
    ProtectedSourceAccessSemantics,
    SourceHashBinding,
    SourceNumericIntegrity,
    SourcePlaneDerivationReceipt,
    SourcePlaneDerivationRecord,
    VirtualCanonicalCountStoreManifestV1,
    VirtualCanonicalCountStoreManifestV2,
    VirtualCountSourceV1,
    VirtualCountSourceV2,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import VirtualCanonicalCountStore
from credo_count_sde_v4.store.virtual import (
    _csr_rows,
    _int64_hash,
    _source_row_pairs_hash,
)


def _write_source(path: Path, matrix: np.ndarray) -> None:
    csr = sparse.csr_matrix(matrix, dtype=np.int32)
    with h5py.File(path, "x") as handle:
        group = handle.create_group("X")
        group.create_dataset("data", data=csr.data)
        group.create_dataset("indices", data=csr.indices.astype(np.int32))
        group.create_dataset("indptr", data=csr.indptr.astype(np.int64))


def _artifact(path: Path, *, root: Path, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.artifact",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=str(path.relative_to(root)),
    )


def _virtual_store(tmp_path: Path) -> VirtualCanonicalCountStore:
    source_root = tmp_path / "sources"
    source_root.mkdir(parents=True)
    first = source_root / "s0.h5ad"
    second = source_root / "s1.h5ad"
    _write_source(first, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    # Physical feature order is [g2, g0, g1].
    _write_source(second, np.asarray([[30, 10, 20], [60, 40, 50]], dtype=np.int32))
    store_root = tmp_path / "virtual"
    store_root.mkdir()
    locator = store_root / "row-locator.h5"
    with h5py.File(locator, "x") as handle:
        handle.create_dataset("row_ids_sorted", data=np.asarray([10, 20, 30, 40], dtype=np.int64))
        handle.create_dataset(
            "source_indices_sorted", data=np.asarray([0, 0, 1, 1], dtype=np.int16)
        )
        handle.create_dataset("source_rows_sorted", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
        handle.create_dataset("guide_codes_sorted", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
        handle.create_dataset("target_codes_sorted", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
        handle.create_dataset(
            "guide_ids", data=np.asarray(["guide-a", "guide-b"], dtype=h5py.string_dtype())
        )
        handle.create_dataset(
            "target_ids", data=np.asarray(["target-a", "target-b"], dtype=h5py.string_dtype())
        )
    permutations = store_root / "feature-permutations.npz"
    identity = np.asarray([0, 1, 2], dtype=np.int32)
    rotated = np.asarray([1, 2, 0], dtype=np.int32)
    np.savez(permutations, source_000000=identity, source_000001=rotated)
    sources = (
        VirtualCountSourceV1(
            source_id="D1_Rest",
            donor_id="D1",
            checkpoint="Rest",
            physical_time_hours=0.0,
            relative_uri="s0.h5ad",
            source_file_sha256=sha256_file(first),
            rows=2,
            features=3,
            nnz=6,
            eligible_rows=2,
            eligible_nnz=6,
            source_feature_order_hash=hashlib.sha256(b"g0,g1,g2").hexdigest(),
            canonical_permutation_hash=hashlib.sha256(identity.astype("<i4").tobytes()).hexdigest(),
        ),
        VirtualCountSourceV1(
            source_id="D2_Stim8hr",
            donor_id="D2",
            checkpoint="Stim8hr",
            physical_time_hours=8.0,
            relative_uri="s1.h5ad",
            source_file_sha256=sha256_file(second),
            rows=2,
            features=3,
            nnz=6,
            eligible_rows=2,
            eligible_nnz=6,
            source_feature_order_hash=hashlib.sha256(b"g2,g0,g1").hexdigest(),
            canonical_permutation_hash=hashlib.sha256(rotated.astype("<i4").tobytes()).hexdigest(),
        ),
    )
    payload = {
        "schema_version": 1,
        "virtual_store_id": "pending",
        "source_authority_id": "authority-1",
        "canonical_feature_index_hash": hashlib.sha256(b"canonical").hexdigest(),
        "guide_catalog_hash": hashlib.sha256(b'["guide-a","guide-b"]').hexdigest(),
        "target_catalog_hash": hashlib.sha256(b'["target-a","target-b"]').hexdigest(),
        "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
        "eligible_rows": 4,
        "features": 3,
        "row_locator": _artifact(
            locator, root=store_root, media_type="application/x-hdf5"
        ).model_dump(mode="json"),
        "feature_permutations": _artifact(
            permutations, root=store_root, media_type="application/x-npz"
        ).model_dump(mode="json"),
        "sources": [source.model_dump(mode="json") for source in sources],
    }
    normalized_fields = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in VirtualCanonicalCountStoreManifestV1.model_fields.items()
        if name in payload
    }
    normalized = VirtualCanonicalCountStoreManifestV1.model_construct(
        **normalized_fields
    ).model_dump(mode="json")
    payload["virtual_store_id"] = contract_id(normalized, id_field="virtual_store_id")
    manifest = VirtualCanonicalCountStoreManifestV1.model_validate(payload)
    (store_root / "manifest.json").write_text(manifest.model_dump_json() + "\n")
    return VirtualCanonicalCountStore(store_root, source_root=source_root)


def _refresh_manifest_artifact(
    store: VirtualCanonicalCountStore, field: str, path: Path
) -> VirtualCanonicalCountStore:
    payload = store.manifest.model_dump(mode="json")
    payload[field]["sha256"] = sha256_file(path)
    payload[field]["size_bytes"] = path.stat().st_size
    payload["virtual_store_id"] = "pending"
    model = type(store.manifest)
    normalized_fields = {
        name: TypeAdapter(model_field.annotation).validate_python(payload[name])
        for name, model_field in model.model_fields.items()
        if name in payload
    }
    normalized = model.model_construct(**normalized_fields).model_dump(mode="json")
    payload["virtual_store_id"] = contract_id(normalized, id_field="virtual_store_id")
    manifest = model.model_validate(payload)
    (store.path / "manifest.json").write_text(manifest.model_dump_json() + "\n")
    return VirtualCanonicalCountStore(store.path, source_root=store.source_root)


def _rewrite_v2_manifest(
    store: VirtualCanonicalCountStore, payload: dict[str, object]
) -> VirtualCanonicalCountStore:
    payload["virtual_store_id"] = "pending"
    normalized = VirtualCanonicalCountStoreManifestV2.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in VirtualCanonicalCountStoreManifestV2.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload["virtual_store_id"] = contract_id(normalized, id_field="virtual_store_id")
    manifest = VirtualCanonicalCountStoreManifestV2.model_validate(payload)
    (store.path / "manifest.json").write_text(manifest.model_dump_json() + "\n")
    return VirtualCanonicalCountStore(store.path, source_root=store.source_root)


def _rewire_amendment(
    store: VirtualCanonicalCountStore,
    payload: dict[str, object],
    *,
    amendment_field: str,
    manifest_field: str,
) -> None:
    amendment_path = store.path / store.manifest.source_plane_amendment.relative_uri
    amendment_payload = G00SourcePlaneV2Amendment.model_validate_json(
        amendment_path.read_text()
    ).model_dump(mode="json")
    amendment_payload[amendment_field] = payload[manifest_field]
    amendment_payload["amendment_id"] = "pending"
    normalized = G00SourcePlaneV2Amendment.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(amendment_payload[name])
            for name, field in G00SourcePlaneV2Amendment.model_fields.items()
            if name in amendment_payload
        }
    ).model_dump(mode="json")
    amendment_payload["amendment_id"] = contract_id(normalized, id_field="amendment_id")
    amendment = G00SourcePlaneV2Amendment.model_validate(amendment_payload)
    amendment_path.write_text(amendment.model_dump_json() + "\n")
    payload["source_plane_amendment_id"] = amendment.amendment_id
    payload["source_plane_amendment"] = _artifact(
        amendment_path, root=store.path, media_type="application/json"
    ).model_dump(mode="json")


def _upgrade_virtual_store_to_v2(store: VirtualCanonicalCountStore) -> VirtualCanonicalCountStore:
    crosswalk = store.path / "GUIDE_TARGET_CROSSWALK.parquet"
    pd.DataFrame(
        {
            "guide_id": ["guide-a", "guide-b"],
            "target_id": ["target-a", "target-b"],
            "is_control": [False, False],
            "raw_guide_group": ["targeting single sgRNA", "targeting single sgRNA"],
            "eligible_cell_count": [2, 2],
            "observed_source_count": [2, 2],
        }
    ).to_parquet(crosswalk, index=False)
    numeric = store.path / "SOURCE_NUMERIC_AUDIT.parquet"
    integrity = SourceNumericIntegrity(
        storage_value_dtype="int32",
        indices_dtype="int32",
        indptr_dtype="int64",
        maximum_observed_count=60,
    )
    pd.DataFrame(
        [
            {"source_id": source.source_id, **integrity.model_dump(mode="python")}
            for source in store.manifest.sources
        ]
    ).to_parquet(numeric, index=False)
    sources = tuple(
        VirtualCountSourceV2(**source.model_dump(mode="python"), numeric_integrity=integrity)
        for source in store.manifest.sources
    )
    row_ids, source_indices, source_rows = store._locator()
    records = tuple(
        SourcePlaneDerivationRecord(
            source_id=source.source_id,
            selected_row_count=source.eligible_rows,
            selected_nnz=source.eligible_nnz,
            selected_row_ids_hash=_int64_hash(row_ids[source_indices == index]),
            source_row_pairs_hash=_source_row_pairs_hash(
                index, source_rows[source_indices == index]
            ),
            scanner_implementation_sha256="f" * 64,
            source_file_sha256=source.source_file_sha256,
        )
        for index, source in enumerate(sources)
    )
    derivation_payload = {
        "schema_version": 1,
        "receipt_id": "pending",
        "records": [record.model_dump(mode="json") for record in records],
    }
    derivation_normalized = SourcePlaneDerivationReceipt.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(derivation_payload[name])
            for name, field in SourcePlaneDerivationReceipt.model_fields.items()
            if name in derivation_payload
        }
    ).model_dump(mode="json")
    derivation_payload["receipt_id"] = contract_id(derivation_normalized, id_field="receipt_id")
    derivation_receipt = SourcePlaneDerivationReceipt.model_validate(derivation_payload)
    derivation = store.path / "SOURCE_PLANE_DERIVATION_RECEIPT.json"
    derivation.write_text(derivation_receipt.model_dump_json() + "\n")
    crosswalk_ref = _artifact(
        crosswalk, root=store.path, media_type="application/vnd.apache.parquet"
    )
    numeric_ref = _artifact(numeric, root=store.path, media_type="application/vnd.apache.parquet")
    derivation_ref = _artifact(derivation, root=store.path, media_type="application/json")
    amendment_payload = {
        "schema_version": 1,
        "amendment_id": "pending",
        "parent_g00a_v1_authority_id": store.manifest.source_authority_id,
        "parent_g00a_v1": ArtifactRef(
            schema_id="test.g00a-v1",
            schema_version=1,
            sha256="a" * 64,
            size_bytes=1,
            media_type="application/json",
            relative_uri="parent-g00a-v1.json",
        ).model_dump(mode="json"),
        "parent_g00b_v1_virtual_store_id": store.manifest.virtual_store_id,
        "parent_g00b_v1_manifest": ArtifactRef(
            schema_id="test.g00b-v1",
            schema_version=1,
            sha256="b" * 64,
            size_bytes=1,
            media_type="application/json",
            relative_uri="parent-g00b-v1.json",
        ).model_dump(mode="json"),
        "immutable_source_hashes": [
            SourceHashBinding(
                source_id=(source.source_id if index < len(sources) else f"extra-{index}"),
                source_file_sha256=(
                    source.source_file_sha256 if index < len(sources) else f"{index:x}" * 64
                ),
            ).model_dump(mode="json")
            for index, source in (
                (index, sources[index] if index < len(sources) else sources[0])
                for index in range(12)
            )
        ],
        "v2_guide_target_crosswalk": crosswalk_ref.model_dump(mode="json"),
        "v2_numerical_audit": numeric_ref.model_dump(mode="json"),
        "v2_source_derivation_receipt": derivation_ref.model_dump(mode="json"),
        "v2_row_locator": store.manifest.row_locator.model_dump(mode="json"),
        "builder_implementation_sha256": "c" * 64,
        "environment_hash": "d" * 64,
    }
    amendment_normalized = G00SourcePlaneV2Amendment.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(amendment_payload[name])
            for name, field in G00SourcePlaneV2Amendment.model_fields.items()
            if name in amendment_payload
        }
    ).model_dump(mode="json")
    amendment_payload["amendment_id"] = contract_id(amendment_normalized, id_field="amendment_id")
    amendment_model = G00SourcePlaneV2Amendment.model_validate(amendment_payload)
    amendment = store.path / "G00_SOURCE_PLANE_V2_AMENDMENT.json"
    amendment.write_text(amendment_model.model_dump_json() + "\n")
    payload = {
        "schema_version": 2,
        "virtual_store_id": "pending",
        "source_authority_id": "authority-v2",
        "source_plane_amendment_id": amendment_model.amendment_id,
        "source_plane_amendment": _artifact(
            amendment, root=store.path, media_type="application/json"
        ).model_dump(mode="json"),
        "canonical_feature_index_hash": store.manifest.canonical_feature_index_hash,
        "guide_catalog_hash": store.manifest.guide_catalog_hash,
        "target_catalog_hash": store.manifest.target_catalog_hash,
        "guide_target_crosswalk_hash": sha256_file(crosswalk),
        "guide_target_crosswalk": crosswalk_ref.model_dump(mode="json"),
        "source_numeric_audit": numeric_ref.model_dump(mode="json"),
        "source_derivation_receipt": derivation_ref.model_dump(mode="json"),
        "guide_count": 2,
        "target_control_count": 2,
        "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
        "eligible_row_ids_hash": _int64_hash(row_ids),
        "eligible_rows": 4,
        "eligible_nnz": 12,
        "features": 3,
        "row_locator": store.manifest.row_locator.model_dump(mode="json"),
        "feature_permutations": store.manifest.feature_permutations.model_dump(mode="json"),
        "sources": [source.model_dump(mode="json") for source in sources],
        "access_semantics": ProtectedSourceAccessSemantics().model_dump(mode="json"),
    }
    normalized_fields = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in VirtualCanonicalCountStoreManifestV2.model_fields.items()
        if name in payload
    }
    normalized = VirtualCanonicalCountStoreManifestV2.model_construct(
        **normalized_fields
    ).model_dump(mode="json")
    payload["virtual_store_id"] = contract_id(normalized, id_field="virtual_store_id")
    manifest = VirtualCanonicalCountStoreManifestV2.model_validate(payload)
    (store.path / "manifest.json").write_text(manifest.model_dump_json() + "\n")
    return VirtualCanonicalCountStore(store.path, source_root=store.source_root)


def test_virtual_canonical_store_preserves_rows_duplicates_and_feature_order(
    tmp_path: Path,
) -> None:
    store = _virtual_store(tmp_path)
    assert store.verify(full=True) == store.manifest
    batch = store.rows(np.asarray([40, 10, 30, 40], dtype=np.int64))
    assert batch.row_ids.tolist() == [40, 10, 30, 40]
    assert batch.matrix.toarray().tolist() == [
        [40, 50, 60],
        [1, 2, 3],
        [10, 20, 30],
        [40, 50, 60],
    ]
    assert store.locator_cache_bytes == 4 * (8 + 2 + 8)
    assert validate_contract(store.path / "manifest.json")["contract_type"] == (
        "VirtualCanonicalCountStoreManifest"
    )


def test_virtual_canonical_store_rejects_missing_and_corrupt_rows(tmp_path: Path) -> None:
    store = _virtual_store(tmp_path)
    with pytest.raises(KeyError, match="Unknown virtual row IDs"):
        store.rows(np.asarray([99]))
    permutation_path = store.path / store.manifest.feature_permutations.relative_uri
    with permutation_path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(Exception, match="metadata artifact hash"):
        store.verify(full=False)


def test_virtual_canonical_store_full_verification_rejects_changed_source(
    tmp_path: Path,
) -> None:
    store = _virtual_store(tmp_path)
    source = store.source_root / store.manifest.sources[0].relative_uri
    with source.open("ab") as handle:
        handle.write(b"changed-after-authority-freeze")
    with pytest.raises(Exception, match="Virtual source hash mismatch"):
        store.verify(full=True)


def test_virtual_store_empty_rows_cursor_batches_and_locator_cache(tmp_path: Path) -> None:
    store = _virtual_store(tmp_path)
    assert store.locator_cache_bytes == 0
    empty = store.rows(np.asarray([], dtype=np.int64))
    assert empty.matrix.shape == (0, 3)
    batches = list(store.iter_batches(np.asarray([40, 10, 20]), batch_size=2, cursor=1))
    assert [cursor for cursor, _ in batches] == [3]
    assert batches[0][1].row_ids.tolist() == [10, 20]
    for batch_size, cursor in ((0, 0), (1, -1)):
        with pytest.raises(ValueError, match="batch_size"):
            list(store.iter_batches(np.asarray([10]), batch_size=batch_size, cursor=cursor))


@pytest.mark.parametrize("missing", ("guide_ids", "target_codes_sorted"))
def test_virtual_store_rejects_incomplete_locator_catalog(tmp_path: Path, missing: str) -> None:
    store = _virtual_store(tmp_path)
    locator = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator, "r+") as handle:
        del handle[missing]
    store = _refresh_manifest_artifact(store, "row_locator", locator)
    with pytest.raises(Exception, match="lacks guide/target"):
        store.verify(full=False)


def test_virtual_store_rejects_bad_locator_codes_and_catalog_hash(tmp_path: Path) -> None:
    store = _virtual_store(tmp_path)
    locator = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator, "r+") as handle:
        handle["guide_codes_sorted"][0] = 99
    invalid_codes = _refresh_manifest_artifact(store, "row_locator", locator)
    with pytest.raises(Exception, match="codes are invalid"):
        invalid_codes.verify(full=False)

    store = _virtual_store(tmp_path / "catalog")
    locator = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator, "r+") as handle:
        handle["guide_ids"][0] = "changed-guide"
    changed_catalog = _refresh_manifest_artifact(store, "row_locator", locator)
    with pytest.raises(Exception, match="catalog hash mismatch"):
        changed_catalog.verify(full=False)


def test_virtual_store_rejects_invalid_locator_permutation_and_source(tmp_path: Path) -> None:
    store = _virtual_store(tmp_path)
    locator = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator, "r+") as handle:
        handle["row_ids_sorted"][1] = 10
    duplicated = _refresh_manifest_artifact(store, "row_locator", locator)
    with pytest.raises(Exception, match="row locator is invalid"):
        duplicated.verify(full=False)

    store = _virtual_store(tmp_path / "permutation")
    permutations = store.path / store.manifest.feature_permutations.relative_uri
    np.savez(
        permutations,
        source_000000=np.asarray([0, 0, 2], dtype=np.int32),
        source_000001=np.asarray([1, 2, 0], dtype=np.int32),
    )
    invalid_permutation = _refresh_manifest_artifact(store, "feature_permutations", permutations)
    with pytest.raises(Exception, match="Invalid canonical permutation"):
        invalid_permutation.verify(full=False)

    store = _virtual_store(tmp_path / "missing")
    source = store.source_root / store.manifest.sources[0].relative_uri
    source.rename(source.with_suffix(".moved"))
    with pytest.raises(Exception, match="Virtual source is missing"):
        store.verify(full=False)


def test_virtual_source_row_reader_rejects_out_of_bounds(tmp_path: Path) -> None:
    store = _virtual_store(tmp_path)
    source = store.manifest.sources[0]
    with pytest.raises(KeyError, match="outside .* bounds"):
        _csr_rows(store.source_root / source.relative_uri, source, np.asarray([source.rows]))


def test_dev30_virtual_store_verifies_crosswalk_and_numeric_authority(tmp_path: Path) -> None:
    store = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path))
    assert store.verify(full=True) == store.manifest
    assert validate_contract(store.path / "manifest.json")["schema_version"] == 2


def test_dev31_virtual_store_rejects_malformed_or_misidentified_amendment(
    tmp_path: Path,
) -> None:
    malformed = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path / "malformed"))
    amendment_path = malformed.path / malformed.manifest.source_plane_amendment.relative_uri
    amendment_path.write_text("{}\n")
    payload = malformed.manifest.model_dump(mode="json")
    payload["source_plane_amendment"] = _artifact(
        amendment_path, root=malformed.path, media_type="application/json"
    ).model_dump(mode="json")
    malformed = _rewrite_v2_manifest(malformed, payload)
    with pytest.raises(Exception, match="Field required"):
        malformed.verify(full=False)

    misidentified = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path / "misidentified"))
    amendment_path = misidentified.path / misidentified.manifest.source_plane_amendment.relative_uri
    amendment = G00SourcePlaneV2Amendment.model_validate_json(amendment_path.read_text())
    amendment_path.write_text(
        amendment.model_copy(update={"amendment_id": "wrong-amendment"}).model_dump_json() + "\n"
    )
    payload = misidentified.manifest.model_dump(mode="json")
    payload["source_plane_amendment"] = _artifact(
        amendment_path, root=misidentified.path, media_type="application/json"
    ).model_dump(mode="json")
    misidentified = _rewrite_v2_manifest(misidentified, payload)
    with pytest.raises(Exception, match="amendment_id mismatch"):
        misidentified.verify(full=False)


def test_dev30_virtual_store_rejects_crosswalk_row_assignment_drift(tmp_path: Path) -> None:
    store = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path))
    crosswalk = store.path / store.manifest.guide_target_crosswalk.relative_uri
    frame = pd.read_parquet(crosswalk)
    frame["target_id"] = frame["target_id"].iloc[::-1].to_numpy()
    frame.to_parquet(crosswalk, index=False)
    payload = store.manifest.model_dump(mode="json")
    payload["guide_target_crosswalk"]["sha256"] = sha256_file(crosswalk)
    payload["guide_target_crosswalk"]["size_bytes"] = crosswalk.stat().st_size
    payload["guide_target_crosswalk_hash"] = sha256_file(crosswalk)
    _rewire_amendment(
        store,
        payload,
        amendment_field="v2_guide_target_crosswalk",
        manifest_field="guide_target_crosswalk",
    )
    changed = _rewrite_v2_manifest(store, payload)
    with pytest.raises(Exception, match="guide-target assignments"):
        changed.verify(full=False)


def test_dev30_virtual_store_recomputes_row_hash_counts_and_source_row_uniqueness(
    tmp_path: Path,
) -> None:
    store = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path))
    payload = store.manifest.model_dump(mode="json")
    payload["eligible_row_ids_hash"] = "0" * 64
    changed = _rewrite_v2_manifest(store, payload)
    with pytest.raises(IntegrityError, match="eligible-row hash"):
        changed.verify(full=False)

    store = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path / "counts"))
    crosswalk = store.path / store.manifest.guide_target_crosswalk.relative_uri
    frame = pd.read_parquet(crosswalk)
    frame.loc[0, "eligible_cell_count"] = 3
    frame.to_parquet(crosswalk, index=False)
    payload = store.manifest.model_dump(mode="json")
    payload["guide_target_crosswalk"]["sha256"] = sha256_file(crosswalk)
    payload["guide_target_crosswalk"]["size_bytes"] = crosswalk.stat().st_size
    payload["guide_target_crosswalk_hash"] = sha256_file(crosswalk)
    _rewire_amendment(
        store,
        payload,
        amendment_field="v2_guide_target_crosswalk",
        manifest_field="guide_target_crosswalk",
    )
    changed = _rewrite_v2_manifest(store, payload)
    with pytest.raises(IntegrityError, match="support counts"):
        changed.verify(full=False)

    store = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path / "duplicate"))
    locator = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator, "r+") as handle:
        handle["source_rows_sorted"][-1] = 0
    changed = _refresh_manifest_artifact(store, "row_locator", locator)
    with pytest.raises(IntegrityError, match="rows are duplicated"):
        changed.verify(full=False)


def test_dev30_virtual_store_rejects_unused_target_catalog_category(tmp_path: Path) -> None:
    store = _upgrade_virtual_store_to_v2(_virtual_store(tmp_path))
    locator = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator, "r+") as handle:
        del handle["target_ids"]
        handle.create_dataset(
            "target_ids",
            data=np.asarray(["target-a", "target-b", "target-unused"], dtype=h5py.string_dtype()),
        )
    payload = store.manifest.model_dump(mode="json")
    payload["target_catalog_hash"] = hashlib.sha256(
        canonical_json_bytes(["target-a", "target-b", "target-unused"])
    ).hexdigest()
    payload["row_locator"]["sha256"] = sha256_file(locator)
    payload["row_locator"]["size_bytes"] = locator.stat().st_size
    _rewire_amendment(
        store,
        payload,
        amendment_field="v2_row_locator",
        manifest_field="row_locator",
    )
    changed = _rewrite_v2_manifest(store, payload)
    with pytest.raises(IntegrityError, match="target catalogs differ"):
        changed.verify(full=False)
