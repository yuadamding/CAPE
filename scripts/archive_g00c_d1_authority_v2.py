"""Create and independently re-audit one immutable Dev36 D1 authority archive."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import tarfile
import tempfile
from pathlib import Path

from credo_count_sde_v4.canonical import atomic_json, sha256_bytes, sha256_file
from credo_count_sde_v4.contracts import G00CD1ExecutionAuthorityFreezeV2
from credo_count_sde_v4.store.g00c_v3 import verify_g00c_d1_freeze_v1


def _manifest(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in observed:
            raise RuntimeError("The Dev36 authority SHA256SUMS is malformed.")
        observed[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(observed) != actual:
        raise RuntimeError("The Dev36 authority manifest does not cover the exact file set.")
    for name, digest in observed.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"The Dev36 authority file failed verification: {name}.")
    return observed


def _write_archive(root: Path, output: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(root).as_posix()
                    data = path.read_bytes()
                    info = tarfile.TarInfo(relative)
                    info.size = len(data)
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())


def _audit_archive(archive_path: Path, expected: dict[str, str]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="credo-dev36-archive-audit-") as temporary_name:
        temporary = Path(temporary_name)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if (
                names != sorted(names)
                or len(names) != len(set(names))
                or any(
                    not member.isfile() or member.issym() or member.islnk() for member in members
                )
                or set(names) != set(expected) | {"SHA256SUMS"}
            ):
                raise RuntimeError("The Dev36 authority archive inventory is unsafe or incomplete.")
            for member in members:
                payload = archive.extractfile(member)
                if payload is None:
                    raise RuntimeError("The Dev36 archive contains an unreadable member.")
                data = payload.read()
                destination = temporary / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                if member.name != "SHA256SUMS" and sha256_bytes(data) != expected[member.name]:
                    raise RuntimeError(
                        "A Dev36 archive member differs from the authority manifest."
                    )
        extracted = _manifest(temporary)
        if extracted != expected:
            raise RuntimeError("The extracted Dev36 archive manifest differs.")
        authority = G00CD1ExecutionAuthorityFreezeV2.model_validate_json(
            (temporary / "G00C_D1_EXECUTION_AUTHORITY.json").read_text()
        )
        verify_g00c_d1_freeze_v1(temporary, authority)
        return {
            "archive_sha256": sha256_file(archive_path),
            "authority_id": authority.authority_id,
            "authority_manifest_sha256": sha256_file(temporary / "SHA256SUMS"),
            "biological_claims": False,
            "expression_values_accessed": False,
            "member_count": len(expected) + 1,
            "schema_version": 1,
            "status": "pass_independent_archive_audit",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.authority_root.resolve()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise RuntimeError("Refusing to replace a Dev36 authority archive or checksum.")
    expected = _manifest(root)
    authority = G00CD1ExecutionAuthorityFreezeV2.model_validate_json(
        (root / "G00C_D1_EXECUTION_AUTHORITY.json").read_text()
    )
    verify_g00c_d1_freeze_v1(root, authority)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(root, output)
    audit = _audit_archive(output, expected)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{audit['archive_sha256']}  {output.name}\n")
    atomic_json(output.with_suffix(output.suffix + ".audit.json"), audit)


if __name__ == "__main__":
    main()
