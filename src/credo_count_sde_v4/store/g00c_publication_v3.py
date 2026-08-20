"""Manifest-last, status-dependent publication seal for Dev36."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..canonical import atomic_json, sha256_file
from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV2,
    G00CExecutionBundleV4,
    G00CPublicationFreezeV1,
    G00CPublicationManifestV3,
)
from ..errors import IntegrityError
from .g00c_v3 import _path

CONTROL_NAMES = {"artifacts.json", "SHA256SUMS", "COMMITTED", "PUBLICATION_EVENT.json"}


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV2, role: str) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev36 authority has no unique {role} implementation.")
    return matches[0]


def expected_publication_payload_v3(
    freeze: G00CPublicationFreezeV1,
    terminal_status: str,
) -> set[str]:
    """Return exact non-control payload names for a terminal status."""

    expected = set(freeze.always_required_artifacts)
    if terminal_status == "pass":
        expected.update(freeze.pass_only_required_artifacts)
    elif terminal_status == "extension_required":
        expected.update(freeze.extension_required_artifacts)
    elif terminal_status == "fail_no_saturation":
        expected.update(freeze.fail_no_saturation_required_artifacts)
    elif terminal_status == "failed_integrity":
        expected.add("FAILURE_RECEIPT.json")
    else:
        raise IntegrityError("Dev36 publication has an unknown terminal status.")
    return expected - CONTROL_NAMES


def _parse_sha256sums(path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in observed:
            raise IntegrityError("Dev36 SHA256SUMS is malformed or contains duplicates.")
        observed[name] = digest
    return observed


def verify_g00c_publication_v3(
    publication_root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV2,
    freeze: G00CPublicationFreezeV1,
    bundle: G00CExecutionBundleV4,
    manifest: G00CPublicationManifestV3,
) -> set[str]:
    """Require expected = manifest = SHA inventory and validate every byte."""

    if (
        manifest.execution_authority_id != authority.authority_id
        or manifest.selection_freeze_id != bundle.selection_freeze_id
        or manifest.feature_selection_result_id != bundle.feature_selection_result_id
        or manifest.sample_size_selection_result_id != bundle.sample_size_selection_result_id
        or manifest.terminal_status != bundle.terminal_status
        or manifest.publisher_implementation_sha256
        != _implementation_hash(authority, "publication_verifier")
    ):
        raise IntegrityError("Dev36 publication manifest is cross-wired.")
    expected = expected_publication_payload_v3(freeze, bundle.terminal_status)
    entries = {entry.name: entry for entry in manifest.artifacts}
    sums = _parse_sha256sums(_path(publication_root, manifest.sha256sums))
    if set(entries) != expected or set(sums) != expected:
        raise IntegrityError("Dev36 expected, manifest, and SHA inventories differ.")
    for name in sorted(expected):
        path = publication_root / name
        entry = entries[name]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.size_bytes
            or sha256_file(path) != entry.sha256
            or sums[name] != entry.sha256
        ):
            raise IntegrityError(f"Dev36 publication payload failed verification: {name}.")
    committed = _path(publication_root, manifest.committed)
    if committed.read_text() != f"{bundle.terminal_status}\n":
        raise IntegrityError("Dev36 COMMITTED marker differs from terminal status.")
    event = json.loads(_path(publication_root, manifest.publication_event_receipt).read_text())
    if event != {
        "destination_preexisted": False,
        "directory_fsync_completed": True,
        "manifest_written_last": True,
        "no_clobber": True,
        "publisher_implementation_sha256": manifest.publisher_implementation_sha256,
        "status": "pass",
    }:
        raise IntegrityError("Dev36 publication event does not prove the frozen commit protocol.")
    allowed = expected | CONTROL_NAMES
    actual = {
        path.relative_to(publication_root).as_posix()
        for path in publication_root.rglob("*")
        if path.is_file()
    }
    if actual != allowed:
        raise IntegrityError("Dev36 publication directory contains missing or extra files.")
    return expected


def publish_g00c_v3(
    destination: Path,
    *,
    terminal_status: str,
    publisher_implementation_sha256: str,
    writer: Callable[[Path], G00CPublicationManifestV3],
) -> None:
    """Publish through a fresh sibling directory with control artifacts written last."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        manifest = writer(temporary)
        if manifest.terminal_status != terminal_status:
            raise IntegrityError("Dev36 publisher received a cross-wired manifest.")
        atomic_json(
            temporary / "PUBLICATION_EVENT.json",
            {
                "destination_preexisted": False,
                "directory_fsync_completed": True,
                "manifest_written_last": True,
                "no_clobber": True,
                "publisher_implementation_sha256": publisher_implementation_sha256,
                "status": "pass",
            },
        )
        (temporary / "COMMITTED").write_text(f"{terminal_status}\n")
        atomic_json(temporary / "artifacts.json", manifest.model_dump(mode="json"))
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
