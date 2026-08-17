"""Non-retroactive admission of one historical checksum-only parent.

The wrapper proves current byte integrity and accepted parent semantics.  It
deliberately cannot prove that the historical directory was published
atomically or manifest-last, and it never opens the raw source matrices named
by the virtual store.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import pandas as pd

from ..canonical import atomic_json, contract_id, sha256_file
from ..contracts import (
    ArtifactRef,
    B0ParentResolutionReceiptV1,
    G00SourceAuthorityV1,
    LegacyAttestationArtifacts,
    LegacyAttestationBuilder,
    LegacyChecksumManifestBinding,
    LegacyChecksumVerificationSummary,
    LegacyExecutionBoundary,
    LegacyParentAttestationReceiptV1,
    LegacyParentAttestationTestContractV1,
    LegacyParentAttestationV1,
    LegacyParentDescriptor,
    LegacyPublicationSemantics,
    NativeManifestLastBoundary,
    ParentPublicationBoundary,
    StrictModel,
    VirtualCanonicalCountStoreManifestV1,
)
from ..errors import ContractError, IntegrityError
from ..persistence.artifacts import (
    artifact_ref,
    publish_directory,
    verify_directory,
)
from .qualification import validate_g00_source_plane
from .virtual import VirtualCanonicalCountStore

_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")
_REQUIRED_PARENT_FILES = frozenset(
    {
        "G00A_SOURCE_AUTHORITY.json",
        "G00B_VIRTUAL_CANONICAL_STORE/feature-permutations.npz",
        "G00B_VIRTUAL_CANONICAL_STORE/manifest.json",
        "G00B_VIRTUAL_CANONICAL_STORE/row-locator.h5",
        "G00_SOURCE_PLANE_RECEIPT.json",
    }
)
_WRAPPER_PAYLOADS = (
    "LEGACY_DIRECTORY_INVENTORY.parquet",
    "LEGACY_SHA256SUMS_VERIFICATION.json",
    "PARENT_LINK.json",
    "TEST_CONTRACT.json",
    "LEGACY_PARENT_ATTESTATION.json",
    "TEST_RECEIPT.json",
)
_EXPECTED_RECEIPT_TOTALS: dict[str, int] = {
    "source_count": 12,
    "eligible_rows": 21_996_842,
    "eligible_nnz": 90_997_745_441,
    "features": 18_130,
    "guides": 25_956,
    "targets_including_control": 12_732,
}
ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized = model.model_construct(**payload).model_dump(mode="json", warnings=False)
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _assert_regular_root(root: Path, *, label: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError(f"{label} is not one regular directory: {root}.")
    return root.resolve()


def _parse_legacy_manifest(parent_root: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    manifest_path = parent_root / "SHA256SUMS"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise IntegrityError("Legacy parent lacks one regular SHA256SUMS file.")
    entries: dict[str, str] = {}
    verification_rows: list[dict[str, Any]] = []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise IntegrityError("Legacy SHA256SUMS is empty.")
    for line_number, line in enumerate(lines, start=1):
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise IntegrityError(f"Invalid legacy SHA256SUMS syntax at line {line_number}.")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or "\\" in relative
            or pure.as_posix() != relative
        ):
            raise IntegrityError(f"Unsafe or noncanonical legacy path: {relative!r}.")
        if relative == "SHA256SUMS":
            raise IntegrityError("Legacy SHA256SUMS cannot contain a self-hash entry.")
        if relative in entries:
            raise IntegrityError(f"Duplicate legacy SHA256SUMS path: {relative}.")
        entries[relative] = digest
    for relative, expected in entries.items():
        path = parent_root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink():
            raise IntegrityError(f"Legacy manifest entry is a symlink: {relative}.")
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise IntegrityError(f"Legacy manifest entry is missing: {relative}.") from exc
        if not stat.S_ISREG(mode):
            raise IntegrityError(f"Legacy manifest entry is not a regular file: {relative}.")
        observed = sha256_file(path)
        if observed != expected:
            raise IntegrityError(f"Legacy checksum mismatch: {relative}.")
        verification_rows.append(
            {
                "relative_path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": True,
                "size_bytes": path.stat().st_size,
            }
        )
    return entries, verification_rows


def _inventory_parent(
    parent_root: Path, manifest_entries: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regular_paths: set[str] = set()
    for path in sorted(parent_root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(parent_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise IntegrityError(f"Legacy parent contains a symlink: {relative}.")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise IntegrityError(f"Legacy parent contains a special file: {relative}.")
        regular_paths.add(relative)
        rows.append(
            {
                "relative_path": relative,
                "file_type": "regular",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "listed_in_legacy_manifest": relative in manifest_entries,
                "consumed_by_b0": relative in _REQUIRED_PARENT_FILES,
            }
        )
    expected_regular = set(manifest_entries) | {"SHA256SUMS"}
    if regular_paths != expected_regular:
        missing = sorted(expected_regular - regular_paths)
        uncovered = sorted(regular_paths - expected_regular)
        raise IntegrityError(
            f"Legacy parent regular-file coverage differs; missing={missing}, "
            f"uncovered={uncovered}."
        )
    if not _REQUIRED_PARENT_FILES <= set(manifest_entries):
        missing = sorted(_REQUIRED_PARENT_FILES - set(manifest_entries))
        raise IntegrityError(f"B0-consumed parent files are not checksum-covered: {missing}.")
    return rows


def _parent_metadata_snapshot(parent_root: Path) -> tuple[tuple[str, int, int, int], ...]:
    rows: list[tuple[str, int, int, int]] = []
    for path in sorted(parent_root.rglob("*"), key=lambda value: value.as_posix()):
        stat_result = path.lstat()
        rows.append(
            (
                path.relative_to(parent_root).as_posix(),
                stat_result.st_mode,
                stat_result.st_size,
                stat_result.st_mtime_ns,
            )
        )
    return tuple(rows)


def _verify_parent_semantics(
    parent_root: Path,
) -> tuple[G00SourceAuthorityV1, VirtualCanonicalCountStoreManifestV1, dict[str, Any]]:
    authority = G00SourceAuthorityV1.model_validate_json(
        (parent_root / "G00A_SOURCE_AUTHORITY.json").read_text()
    )
    manifest = VirtualCanonicalCountStoreManifestV1.model_validate_json(
        (parent_root / "G00B_VIRTUAL_CANONICAL_STORE/manifest.json").read_text()
    )
    validate_g00_source_plane(authority, manifest)
    if len(authority.sources) != _EXPECTED_RECEIPT_TOTALS["source_count"]:
        raise IntegrityError("The accepted GSE314342 parent must contain exactly 12 sources.")
    store = VirtualCanonicalCountStore(
        parent_root / "G00B_VIRTUAL_CANONICAL_STORE",
        source_root=parent_root.parents[1],
    )
    store.verify(full=False)
    receipt = json.loads((parent_root / "G00_SOURCE_PLANE_RECEIPT.json").read_text())
    expected = {
        "authority_id": authority.authority_id,
        "virtual_store_id": manifest.virtual_store_id,
        **_EXPECTED_RECEIPT_TOTALS,
        "model_fitting": False,
        "scientific_evidence": False,
        "raw_counts_materialized": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise IntegrityError("Legacy Dev29 receipt differs from its accepted semantic totals.")
    return authority, manifest, receipt


def _artifact(path: Path, root: Path, schema_id: str, media_type: str) -> ArtifactRef:
    return artifact_ref(root, path, schema_id=schema_id, media_type=media_type)


def _write_wrapper_sha256sums(root: Path) -> None:
    lines = [f"{sha256_file(root / name)}  {name}\n" for name in _WRAPPER_PAYLOADS]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def build_legacy_parent_attestation(
    parent_root: Path,
    destination: Path,
    *,
    git_commit: str,
    distribution_sha256: str,
    environment_sha256: str,
) -> Path:
    """Publish one wrapper without writing the parent or opening raw matrices."""

    parent = _assert_regular_root(parent_root, label="Legacy parent")
    if destination.resolve().is_relative_to(parent):
        raise ContractError("Legacy wrapper must be a sibling, never inside its parent.")
    if destination.parent.resolve() != parent.parent:
        raise ContractError("Legacy wrapper must share the exact parent directory.")
    before = _parent_metadata_snapshot(parent)

    def writer(root: Path) -> None:
        entries, verification_rows = _parse_legacy_manifest(parent)
        inventory_rows = _inventory_parent(parent, entries)
        authority, manifest, _ = _verify_parent_semantics(parent)
        pd.DataFrame(inventory_rows).to_parquet(
            root / "LEGACY_DIRECTORY_INVENTORY.parquet", index=False
        )
        summary = LegacyChecksumVerificationSummary(
            listed_files=len(entries),
            matched_files=len(entries),
        )
        verification_payload = {
            "schema_version": 1,
            "parent_logical_name": parent.name,
            "sha256sums_sha256": sha256_file(parent / "SHA256SUMS"),
            "entries": verification_rows,
            "summary": summary.model_dump(mode="json"),
        }
        atomic_json(root / "LEGACY_SHA256SUMS_VERIFICATION.json", verification_payload)
        atomic_json(
            root / "PARENT_LINK.json",
            {
                "schema_version": 1,
                "evidence_role": "nonretroactive_legacy_checksum_parent_attestation",
                "parent_location_rule": "exact_sibling_directory",
                "wrapper_directory_name": destination.name,
                "parent_directory_name": parent.name,
                "parent_sha256sums_sha256": sha256_file(parent / "SHA256SUMS"),
                "g00a_v1_authority_id": authority.authority_id,
                "g00b_v1_virtual_store_id": manifest.virtual_store_id,
            },
        )
        contract = _identified(
            LegacyParentAttestationTestContractV1,
            {"schema_version": 1, "parent_logical_name": parent.name},
            "test_contract_id",
        )
        atomic_json(root / "TEST_CONTRACT.json", contract.model_dump(mode="json"))
        implementation_sha256 = sha256_file(Path(__file__))
        attestation = _identified(
            LegacyParentAttestationV1,
            {
                    "schema_id": "credo.legacy_parent_attestation",
                    "schema_version": 1,
                    "evidence_role": (
                        "nonretroactive_legacy_checksum_parent_attestation"
                    ),
                    "parent": LegacyParentDescriptor(
                        logical_name=parent.name,
                        g00a_v1_authority_id=authority.authority_id,
                        g00b_v1_virtual_store_id=manifest.virtual_store_id,
                        sha256sums=LegacyChecksumManifestBinding(
                            sha256=sha256_file(parent / "SHA256SUMS"),
                            size_bytes=(parent / "SHA256SUMS").stat().st_size,
                        ),
                    ).model_dump(mode="json"),
                    "legacy_publication_semantics": (
                        LegacyPublicationSemantics().model_dump(mode="json")
                    ),
                    "verification": summary.model_dump(mode="json"),
                    "artifacts": LegacyAttestationArtifacts(
                        directory_inventory=_artifact(
                            root / "LEGACY_DIRECTORY_INVENTORY.parquet",
                            root,
                            "credo.legacy_directory_inventory",
                            "application/x-parquet",
                        ),
                        checksum_verification=_artifact(
                            root / "LEGACY_SHA256SUMS_VERIFICATION.json",
                            root,
                            "credo.legacy_sha256sums_verification",
                            "application/json",
                        ),
                        parent_link=_artifact(
                            root / "PARENT_LINK.json",
                            root,
                            "credo.legacy_parent_link",
                            "application/json",
                        ),
                    ).model_dump(mode="json"),
                    "execution_boundary": LegacyExecutionBoundary().model_dump(mode="json"),
                    "builder": LegacyAttestationBuilder(
                        git_commit=git_commit,
                        distribution_sha256=distribution_sha256,
                        implementation_sha256=implementation_sha256,
                        environment_sha256=environment_sha256,
                    ).model_dump(mode="json"),
            },
            "attestation_id",
        )
        attestation_path = root / "LEGACY_PARENT_ATTESTATION.json"
        atomic_json(attestation_path, attestation.model_dump(mode="json"))
        attestation_ref = _artifact(
            attestation_path, root, "credo.legacy_parent_attestation", "application/json"
        )
        receipt = _identified(
            LegacyParentAttestationReceiptV1,
            {
                    "schema_version": 1,
                    "test_contract_id": contract.test_contract_id,
                    "attestation_id": attestation.attestation_id,
                    "attestation": attestation_ref.model_dump(mode="json"),
                    "legacy_sha256sums_verified": True,
                    "all_authoritative_files_covered": True,
                    "parent_g00a_v1_verified": True,
                    "parent_g00b_v1_verified": True,
                    "parent_g00a_g00b_relationship_verified": True,
                    "original_atomicity_not_claimed": True,
                    "status": "pass",
            },
            "receipt_id",
        )
        atomic_json(root / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        _write_wrapper_sha256sums(root)

    published = publish_directory(destination, writer)
    if _parent_metadata_snapshot(parent) != before:
        raise IntegrityError("Legacy parent metadata changed while publishing its wrapper.")
    verify_legacy_parent_attestation(published, parent_root=parent)
    return published


def _verify_wrapper_sha256sums(wrapper_root: Path) -> None:
    entries, _ = _parse_legacy_manifest(wrapper_root)
    if set(entries) != set(_WRAPPER_PAYLOADS):
        raise IntegrityError("Wrapper SHA256SUMS does not cover its exact payload set.")


def _matches_ref(path: Path, reference: ArtifactRef) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == reference.size_bytes
        and sha256_file(path) == reference.sha256
    )


def verify_legacy_parent_attestation(
    wrapper_root: Path, *, parent_root: Path
) -> tuple[LegacyParentAttestationV1, LegacyParentAttestationReceiptV1]:
    """Independently verify a committed wrapper and its unchanged sibling."""

    wrapper = _assert_regular_root(wrapper_root, label="Legacy wrapper")
    parent = _assert_regular_root(parent_root, label="Legacy parent")
    if wrapper.parent != parent.parent or wrapper == parent or wrapper.is_relative_to(parent):
        raise IntegrityError("Legacy wrapper must be a distinct exact sibling of its parent.")
    verify_directory(wrapper)
    _verify_wrapper_sha256sums(wrapper)
    attestation = LegacyParentAttestationV1.model_validate_json(
        (wrapper / "LEGACY_PARENT_ATTESTATION.json").read_text()
    )
    receipt = LegacyParentAttestationReceiptV1.model_validate_json(
        (wrapper / "TEST_RECEIPT.json").read_text()
    )
    contract = LegacyParentAttestationTestContractV1.model_validate_json(
        (wrapper / "TEST_CONTRACT.json").read_text()
    )
    link = json.loads((wrapper / "PARENT_LINK.json").read_text())
    if (
        link.get("parent_location_rule") != "exact_sibling_directory"
        or link.get("parent_directory_name") != parent.name
        or link.get("wrapper_directory_name") != wrapper.name
    ):
        raise IntegrityError("Legacy parent link resolves a different sibling relationship.")
    entries, verification_rows = _parse_legacy_manifest(parent)
    inventory_rows = _inventory_parent(parent, entries)
    authority, manifest, _ = _verify_parent_semantics(parent)
    if (
        attestation.parent.logical_name != parent.name
        or attestation.parent.g00a_v1_authority_id != authority.authority_id
        or attestation.parent.g00b_v1_virtual_store_id != manifest.virtual_store_id
        or attestation.parent.sha256sums.sha256 != sha256_file(parent / "SHA256SUMS")
        or attestation.parent.sha256sums.size_bytes != (parent / "SHA256SUMS").stat().st_size
        or link.get("parent_sha256sums_sha256") != attestation.parent.sha256sums.sha256
        or link.get("g00a_v1_authority_id") != authority.authority_id
        or link.get("g00b_v1_virtual_store_id") != manifest.virtual_store_id
    ):
        raise IntegrityError("Legacy attestation binds different parent bytes or identities.")
    for reference in (
        attestation.artifacts.directory_inventory,
        attestation.artifacts.checksum_verification,
        attestation.artifacts.parent_link,
    ):
        if not _matches_ref(wrapper / reference.relative_uri, reference):
            raise IntegrityError("Legacy attestation evidence ArtifactRef differs.")
    observed_inventory = pd.read_parquet(wrapper / "LEGACY_DIRECTORY_INVENTORY.parquet")
    expected_inventory = pd.DataFrame(inventory_rows)
    columns_differ = tuple(observed_inventory.columns) != tuple(expected_inventory.columns)
    values_differ = not observed_inventory.equals(expected_inventory)
    if columns_differ or values_differ:
        raise IntegrityError("Legacy directory inventory differs from its current parent.")
    verification = json.loads((wrapper / "LEGACY_SHA256SUMS_VERIFICATION.json").read_text())
    if verification.get("entries") != verification_rows or verification.get("summary") != (
        attestation.verification.model_dump(mode="json")
    ):
        raise IntegrityError("Legacy checksum verification evidence differs.")
    attestation_ref = _artifact(
        wrapper / "LEGACY_PARENT_ATTESTATION.json",
        wrapper,
        "credo.legacy_parent_attestation",
        "application/json",
    )
    if (
        receipt.status != "pass"
        or receipt.attestation_id != attestation.attestation_id
        or receipt.attestation != attestation_ref
        or receipt.test_contract_id != contract.test_contract_id
    ):
        raise IntegrityError("Legacy wrapper receipt binds different or failed evidence.")
    return attestation, receipt


def resolve_parent_publication_boundary(
    boundary: ParentPublicationBoundary,
    *,
    parent_root: Path,
    wrapper_root: Path | None = None,
) -> B0ParentResolutionReceiptV1:
    """Resolve a native or explicitly attested legacy parent without fallback."""

    parent = _assert_regular_root(parent_root, label="B0 parent")
    authority, manifest, _ = _verify_parent_semantics(parent)
    if (
        boundary.parent_logical_name != parent.name
        or boundary.g00a_v1_authority_id != authority.authority_id
        or boundary.g00b_v1_virtual_store_id != manifest.virtual_store_id
    ):
        raise IntegrityError("B0 parent boundary binds different semantic identities.")
    historical: LegacyPublicationSemantics | None = None
    if isinstance(boundary, NativeManifestLastBoundary):
        if wrapper_root is not None:
            raise IntegrityError("Native parent resolution cannot consume a legacy wrapper.")
        verify_directory(parent)
        expected_paths = {
            "artifacts.json": boundary.artifacts_manifest,
            "COMMITTED": boundary.committed_marker,
            "SHA256SUMS": boundary.sha256sums,
        }
        if any(not _matches_ref(parent / name, ref) for name, ref in expected_paths.items()):
            raise IntegrityError("Native parent publication ArtifactRef differs.")
    else:
        if wrapper_root is None:
            raise IntegrityError("Legacy parent resolution requires its exact wrapper.")
        attestation, receipt = verify_legacy_parent_attestation(
            wrapper_root, parent_root=parent
        )
        if (
            receipt.status != "pass"
            or boundary.attestation_id != attestation.attestation_id
            or boundary.attestation_receipt_id != receipt.receipt_id
            or not _matches_ref(
                Path(wrapper_root) / "LEGACY_PARENT_ATTESTATION.json", boundary.attestation
            )
            or not _matches_ref(
                Path(wrapper_root) / "TEST_RECEIPT.json", boundary.attestation_receipt
            )
            or not _matches_ref(parent / "SHA256SUMS", boundary.original_sha256sums)
        ):
            raise IntegrityError("Legacy B0 boundary binds unrelated or failed wrapper evidence.")
        historical = attestation.legacy_publication_semantics
    payload = {
        "schema_version": 1,
        "boundary": boundary.model_dump(mode="json"),
        "parent_checksum_boundary": True,
        "all_consumed_artifacts_covered": True,
        "g00a_v1_valid": True,
        "g00b_v1_valid": True,
        "g00a_g00b_relationship_valid": True,
        "historical_semantics": (
            historical.model_dump(mode="json") if historical is not None else None
        ),
        "status": "pass",
    }
    return _identified(B0ParentResolutionReceiptV1, payload, "receipt_id")


__all__ = [
    "build_legacy_parent_attestation",
    "resolve_parent_publication_boundary",
    "verify_legacy_parent_attestation",
]
