"""Audit built wheel namespace, license, and unsafe archive paths."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath


def main(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
    unsafe = [
        name
        for name in names
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    forbidden = [name for name in names if name == "credo" or name.startswith("credo/")]
    if unsafe or forbidden:
        raise SystemExit(f"Unsafe wheel paths={unsafe}; frozen namespace files={forbidden}")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        raise SystemExit("Wheel does not contain the AGPL license.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_package.py WHEEL")
    main(Path(sys.argv[1]))
