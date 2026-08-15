"""Read-only frozen-CREDO compatibility preflight."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from ...canonical import sha256_file
from ...errors import IntegrityError

FROZEN_COMMIT = "6f4f57c114606476cb6e93c6d52dc34628173a4d"
FROZEN_SOURCE_SHA256 = "2020b0f2549bb98b2b0fbb9384783d39be3ce11aeef9ae2bfecf232ae07fc684"
FROZEN_ARCHIVE_ROOT = "credo-6f4f57c"


def _workspace_root() -> Path:
    override = os.environ.get("CREDO_V4_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[5]


def _metadata_head(checkout: Path) -> str:
    git_dir = checkout / ".git"
    head_path = git_dir / "HEAD"
    if not git_dir.is_dir() or not head_path.is_file():
        raise IntegrityError("Frozen CREDO checkout lacks Git metadata.")
    head = head_path.read_text().strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, name = line.split(" ", 1)
            if name == reference:
                return commit
    raise IntegrityError(f"Frozen CREDO reference is unresolved: {reference}.")


def _verify_checkout_against_archive(checkout: Path, archive: Path) -> None:
    expected: dict[str, bytes] = {}
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not member_path.parts
                or member_path.parts[0] != FROZEN_ARCHIVE_ROOT
                or not (member.isfile() or member.isdir())
            ):
                raise IntegrityError(f"Unsafe frozen CREDO archive member: {member.name}.")
            if member.isfile():
                relative = PurePosixPath(*member_path.parts[1:]).as_posix()
                stream = handle.extractfile(member)
                if stream is None or relative in expected:
                    raise IntegrityError("Frozen CREDO archive inventory is ambiguous.")
                expected[relative] = stream.read()
    observed: dict[str, Path] = {}
    for checkout_path in checkout.rglob("*"):
        relative_path = checkout_path.relative_to(checkout)
        if relative_path.parts and relative_path.parts[0] == ".git":
            continue
        if checkout_path.is_symlink() or (
            not checkout_path.is_file() and not checkout_path.is_dir()
        ):
            raise IntegrityError(f"Unsafe frozen CREDO checkout entry: {relative_path}.")
        if checkout_path.is_file():
            observed[relative_path.as_posix()] = checkout_path
    missing = set(expected) - set(observed)
    extras = set(observed) - set(expected)
    allowed_ignored = {
        name
        for name in extras
        if name.startswith(".pytest_cache/")
        or ("__pycache__" in PurePosixPath(name).parts and name.endswith(".pyc"))
    }
    if missing or extras != allowed_ignored:
        raise IntegrityError("Frozen CREDO checkout inventory differs from its archive.")
    mismatched = [
        name for name, payload in expected.items() if observed[name].read_bytes() != payload
    ]
    if mismatched:
        raise IntegrityError(f"Frozen CREDO checkout bytes differ: {mismatched[0]}.")


def verify_frozen_credo(workspace_root: Path | None = None) -> dict[str, str]:
    root = (workspace_root or _workspace_root()).resolve()
    checkout = root / "CREDO"
    archive_candidates = (
        root / "vendor" / "credo-6f4f57c.tar",
        root / "credo-count-sde-v4" / "vendor" / "credo-6f4f57c.tar",
        root / "credo_biology_validation" / "vendor" / "credo-6f4f57c.tar",
    )
    archive = next((path for path in archive_candidates if path.is_file()), None)
    if not checkout.is_dir() or archive is None:
        raise IntegrityError("Frozen CREDO checkout or compatibility artifact is unavailable.")
    digest = sha256_file(archive)
    if digest != FROZEN_SOURCE_SHA256:
        raise IntegrityError(f"Frozen CREDO artifact hash mismatch: {digest}.")
    if shutil.which("git"):
        try:
            head = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise IntegrityError("Frozen CREDO Git verification failed.") from error
        method = "git"
    else:
        head = _metadata_head(checkout)
        _verify_checkout_against_archive(checkout, archive)
        status = ""
        method = "archive_byte_comparison"
    if head != FROZEN_COMMIT:
        raise IntegrityError(f"Frozen CREDO commit mismatch: {head}.")
    if status:
        raise IntegrityError("Frozen CREDO checkout is not clean.")
    return {
        "commit": head,
        "artifact_sha256": digest,
        "checkout": str(checkout),
        "verification_method": method,
    }
