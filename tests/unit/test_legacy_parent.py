from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from credo_count_sde_v4.canonical import canonical_json_bytes, contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    G00SourceAuthorityV1,
    LegacyChecksumAttestedBoundary,
    LegacyParentAttestationReceiptV1,
    LegacyParentAttestationV1,
    LegacyPublicationSemantics,
    VirtualCanonicalCountStoreManifestV1,
    VirtualCountSourceV1,
)
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.persistence.artifacts import artifact_ref
from credo_count_sde_v4.store import (
    build_legacy_parent_attestation,
    resolve_parent_publication_boundary,
    verify_legacy_parent_attestation,
)
from credo_count_sde_v4.store import legacy_parent as legacy_module


def _identified(model: type[Any], payload: dict[str, Any], id_field: str) -> Any:
    payload[id_field] = "pending"
    normalized = model.model_construct(**payload).model_dump(mode="json", warnings=False)
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _hash_strings(values: list[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(values)).hexdigest()


def _ref(path: Path, root: Path, schema_id: str, media_type: str) -> ArtifactRef:
    return artifact_ref(root, path, schema_id=schema_id, media_type=media_type)


@pytest.fixture
def tiny_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        legacy_module,
        "_EXPECTED_RECEIPT_TOTALS",
        {
            "source_count": 12,
            "eligible_rows": 12,
            "eligible_nnz": 12,
            "features": 3,
            "guides": 12,
            "targets_including_control": 12,
        },
    )


def _make_parent(tmp_path: Path, name: str = "G00_SOURCE_PLANE_DEV29") -> Path:
    workspace = tmp_path / "workspace"
    parent = workspace / name
    store = parent / "G00B_VIRTUAL_CANONICAL_STORE"
    store.mkdir(parents=True)
    source_root = parent.parents[1]
    source_dir = source_root / "data"
    source_dir.mkdir()
    guide_ids = [f"guide-{index:02d}" for index in range(12)]
    target_ids = [f"target-{index:02d}" for index in range(12)]
    row_ids = np.asarray([(index << 32) for index in range(12)], dtype=np.int64)
    with h5py.File(store / "row-locator.h5", "w") as handle:
        handle.create_dataset("row_ids_sorted", data=row_ids)
        handle.create_dataset("source_indices_sorted", data=np.arange(12, dtype=np.int16))
        handle.create_dataset("source_rows_sorted", data=np.zeros(12, dtype=np.int32))
        handle.create_dataset("guide_codes_sorted", data=np.arange(12, dtype=np.int32))
        handle.create_dataset("target_codes_sorted", data=np.arange(12, dtype=np.int32))
        handle.create_dataset(
            "guide_ids", data=np.asarray(guide_ids, dtype=h5py.string_dtype("utf-8"))
        )
        handle.create_dataset(
            "target_ids", data=np.asarray(target_ids, dtype=h5py.string_dtype("utf-8"))
        )
    np.savez(
        store / "feature-permutations.npz",
        **{f"source_{index:06d}": np.arange(3, dtype=np.int32) for index in range(12)},
    )
    permutation_hash = hashlib.sha256(np.arange(3, dtype="<i4").tobytes()).hexdigest()
    sources: list[VirtualCountSourceV1] = []
    for index in range(12):
        source_path = source_dir / f"source-{index:02d}.h5ad"
        source_path.write_bytes(f"source-{index}".encode())
        sources.append(
            VirtualCountSourceV1(
                source_id=f"source-{index:02d}",
                donor_id=f"D{index // 3 + 1}",
                checkpoint=("Rest", "Stim8hr", "Stim48hr")[index % 3],
                physical_time_hours=(0.0, 8.0, 48.0)[index % 3],
                relative_uri=f"data/source-{index:02d}.h5ad",
                source_file_sha256=sha256_file(source_path),
                rows=1,
                features=3,
                nnz=1,
                eligible_rows=1,
                eligible_nnz=1,
                source_feature_order_hash=str(index).zfill(64),
                canonical_permutation_hash=permutation_hash,
            )
        )
    authority = _identified(
        G00SourceAuthorityV1,
        {
            "schema_version": 1,
            "sources": [source.model_dump(mode="json") for source in sources],
            "canonical_feature_index_hash": "1" * 64,
            "guide_catalog_hash": _hash_strings(guide_ids),
            "target_catalog_hash": _hash_strings(target_ids),
            "eligibility_rule": ("guide_group == targeting single sgRNA AND low_quality == false"),
            "eligible_row_ids_hash": hashlib.sha256(
                np.asarray(row_ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "eligible_rows": 12,
            "eligible_nnz": 12,
        },
        "authority_id",
    )
    (parent / "G00A_SOURCE_AUTHORITY.json").write_text(authority.model_dump_json() + "\n")
    manifest = _identified(
        VirtualCanonicalCountStoreManifestV1,
        {
            "schema_version": 1,
            "source_authority_id": authority.authority_id,
            "canonical_feature_index_hash": authority.canonical_feature_index_hash,
            "guide_catalog_hash": authority.guide_catalog_hash,
            "target_catalog_hash": authority.target_catalog_hash,
            "eligibility_rule": authority.eligibility_rule,
            "eligible_rows": 12,
            "features": 3,
            "row_locator": _ref(
                store / "row-locator.h5",
                store,
                "credo.virtual_row_locator",
                "application/x-hdf5",
            ).model_dump(mode="json"),
            "feature_permutations": _ref(
                store / "feature-permutations.npz",
                store,
                "credo.feature_permutations",
                "application/x-npz",
            ).model_dump(mode="json"),
            "sources": [source.model_dump(mode="json") for source in sources],
        },
        "virtual_store_id",
    )
    (store / "manifest.json").write_text(manifest.model_dump_json() + "\n")
    receipt = {
        "authority_id": authority.authority_id,
        "virtual_store_id": manifest.virtual_store_id,
        "source_count": 12,
        "eligible_rows": 12,
        "eligible_nnz": 12,
        "features": 3,
        "guides": 12,
        "targets_including_control": 12,
        "model_fitting": False,
        "scientific_evidence": False,
        "raw_counts_materialized": False,
    }
    (parent / "G00_SOURCE_PLANE_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )
    names = (
        "G00A_SOURCE_AUTHORITY.json",
        "G00B_VIRTUAL_CANONICAL_STORE/feature-permutations.npz",
        "G00B_VIRTUAL_CANONICAL_STORE/manifest.json",
        "G00B_VIRTUAL_CANONICAL_STORE/row-locator.h5",
        "G00_SOURCE_PLANE_RECEIPT.json",
    )
    (parent / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(parent / name)}  {name}\n" for name in names)
    )
    return parent


def _build(parent: Path, name: str = "G00_SOURCE_PLANE_DEV29_LEGACY_ATTESTATION") -> Path:
    return build_legacy_parent_attestation(
        parent,
        parent.parent / name,
        git_commit="9" * 40,
        distribution_sha256="a" * 64,
        environment_sha256="b" * 64,
    )


def _legacy_boundary(parent: Path, wrapper: Path) -> LegacyChecksumAttestedBoundary:
    attestation = LegacyParentAttestationV1.model_validate_json(
        (wrapper / "LEGACY_PARENT_ATTESTATION.json").read_text()
    )
    receipt = LegacyParentAttestationReceiptV1.model_validate_json(
        (wrapper / "TEST_RECEIPT.json").read_text()
    )
    return LegacyChecksumAttestedBoundary(
        parent_logical_name=parent.name,
        g00a_v1_authority_id=attestation.parent.g00a_v1_authority_id,
        g00b_v1_virtual_store_id=attestation.parent.g00b_v1_virtual_store_id,
        original_sha256sums=_ref(
            parent / "SHA256SUMS", parent, "credo.legacy_sha256sums", "text/plain"
        ),
        attestation_id=attestation.attestation_id,
        attestation=_ref(
            wrapper / "LEGACY_PARENT_ATTESTATION.json",
            wrapper,
            "credo.legacy_parent_attestation",
            "application/json",
        ),
        attestation_receipt_id=receipt.receipt_id,
        attestation_receipt=_ref(
            wrapper / "TEST_RECEIPT.json",
            wrapper,
            "credo.legacy_parent_attestation_receipt",
            "application/json",
        ),
    )


def test_build_verify_and_resolve_legacy_wrapper(tmp_path: Path, tiny_totals: None) -> None:
    parent = _make_parent(tmp_path)
    before = {path.relative_to(parent): path.stat().st_mtime_ns for path in parent.rglob("*")}
    wrapper = _build(parent)
    attestation, receipt = verify_legacy_parent_attestation(wrapper, parent_root=parent)
    boundary = _legacy_boundary(parent, wrapper)
    resolution = resolve_parent_publication_boundary(
        boundary, parent_root=parent, wrapper_root=wrapper
    )
    after = {path.relative_to(parent): path.stat().st_mtime_ns for path in parent.rglob("*")}
    assert before == after
    assert receipt.status == "pass"
    assert resolution.status == "pass"
    assert resolution.boundary.boundary_kind == "legacy_checksum_attested_v1"
    assert resolution.historical_semantics == LegacyPublicationSemantics()
    assert attestation.legacy_publication_semantics.atomic_publication_proven is False
    assert (wrapper / "artifacts.json").is_file()
    assert (wrapper / "COMMITTED").read_bytes() == b"CREDO-V4-COMMITTED\n"


@pytest.mark.parametrize("case", ["altered", "missing", "duplicate", "traversal", "extra"])
def test_wrapper_rejects_manifest_and_coverage_attacks(
    tmp_path: Path, tiny_totals: None, case: str
) -> None:
    parent = _make_parent(tmp_path)
    manifest = parent / "SHA256SUMS"
    if case == "altered":
        (parent / "G00_SOURCE_PLANE_RECEIPT.json").write_text("altered\n")
    elif case == "missing":
        (parent / "G00_SOURCE_PLANE_RECEIPT.json").unlink()
    elif case == "duplicate":
        first = manifest.read_text().splitlines()[0]
        manifest.write_text(manifest.read_text() + first + "\n")
    elif case == "traversal":
        manifest.write_text(manifest.read_text() + f"{'0' * 64}  ../escape\n")
    else:
        (parent / "UNLISTED.json").write_text("{}\n")
    with pytest.raises(IntegrityError):
        _build(parent)


def test_wrapper_rejects_symlink_and_parent_nesting(tmp_path: Path, tiny_totals: None) -> None:
    parent = _make_parent(tmp_path)
    os.symlink(parent / "G00A_SOURCE_AUTHORITY.json", parent / "UNLISTED_LINK")
    with pytest.raises(IntegrityError, match="symlink"):
        _build(parent)
    (parent / "UNLISTED_LINK").unlink()
    with pytest.raises(ContractError, match="sibling"):
        build_legacy_parent_attestation(
            parent,
            parent / "WRAPPER",
            git_commit="9" * 40,
            distribution_sha256="a" * 64,
            environment_sha256="b" * 64,
        )


def test_legacy_semantics_cannot_claim_original_atomicity() -> None:
    with pytest.raises(ValidationError):
        LegacyPublicationSemantics(atomic_publication_proven=True)  # type: ignore[arg-type]


def test_resolution_rejects_wrong_ids_failed_or_unrelated_wrapper(
    tmp_path: Path, tiny_totals: None
) -> None:
    parent = _make_parent(tmp_path, "PARENT_A")
    wrapper = _build(parent, "WRAPPER_A")
    boundary = _legacy_boundary(parent, wrapper)
    with pytest.raises(IntegrityError, match="semantic identities"):
        resolve_parent_publication_boundary(
            boundary.model_copy(update={"g00a_v1_authority_id": "wrong"}),
            parent_root=parent,
            wrapper_root=wrapper,
        )
    with pytest.raises(IntegrityError, match="unrelated or failed"):
        resolve_parent_publication_boundary(
            boundary.model_copy(
                update={
                    "attestation_receipt": boundary.attestation_receipt.model_copy(
                        update={"sha256": "0" * 64}
                    )
                }
            ),
            parent_root=parent,
            wrapper_root=wrapper,
        )
    other_parent = _make_parent(tmp_path / "other", "PARENT_B")
    other_wrapper = _build(other_parent, "WRAPPER_B")
    with pytest.raises(IntegrityError):
        resolve_parent_publication_boundary(
            boundary, parent_root=parent, wrapper_root=other_wrapper
        )


def test_native_parent_cannot_be_smuggled_through_legacy_mode(
    tmp_path: Path, tiny_totals: None
) -> None:
    parent = _make_parent(tmp_path)
    wrapper = _build(parent)
    boundary = _legacy_boundary(parent, wrapper)
    with pytest.raises(IntegrityError, match="wrapper"):
        resolve_parent_publication_boundary(boundary, parent_root=parent)
    with pytest.raises(IntegrityError, match="distinct exact sibling"):
        resolve_parent_publication_boundary(boundary, parent_root=parent, wrapper_root=parent)
