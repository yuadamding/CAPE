"""Generate or verify REPOSITORY.sha256 from the committed Git index only."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from credo_count_sde_v4.canonical import sha256_file

MANIFEST_NAME = "REPOSITORY.sha256"


def tracked_paths(root: Path) -> tuple[Path, ...]:
    """Return regular tracked files, excluding the self-referential manifest."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    names = tuple(
        name.decode("utf-8")
        for name in completed.stdout.split(b"\0")
        if name and name.decode("utf-8") != MANIFEST_NAME
    )
    paths = tuple(root / name for name in sorted(names))
    missing = [path.relative_to(root).as_posix() for path in paths if not path.is_file()]
    symlinks = [path.relative_to(root).as_posix() for path in paths if path.is_symlink()]
    if missing or symlinks:
        raise RuntimeError(
            "Tracked manifest inputs must be regular files: "
            f"missing={missing}, symlinks={symlinks}."
        )
    return paths


def manifest_text(root: Path) -> str:
    return "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in tracked_paths(root)
    )


def write_manifest(root: Path) -> Path:
    destination = root / MANIFEST_NAME
    payload = manifest_text(root)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MANIFEST_NAME}.", suffix=".tmp", dir=root
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def check_manifest(root: Path) -> None:
    destination = root / MANIFEST_NAME
    observed = destination.read_text()
    expected = manifest_text(root)
    if observed != expected:
        raise SystemExit(
            "REPOSITORY.sha256 differs from the exact regular-file Git index; "
            "regenerate it after staging the intended tree."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.check:
        check_manifest(root)
    else:
        print(write_manifest(root))


if __name__ == "__main__":
    main()
