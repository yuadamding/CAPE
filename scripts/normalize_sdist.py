"""Normalize a setuptools sdist into deterministic tar/gzip bytes."""

from __future__ import annotations

import gzip
import io
import os
import sys
import tarfile
from pathlib import Path


def normalize(source: Path, destination: Path, *, epoch: int) -> None:
    with tarfile.open(source, "r:gz") as archive:
        members = sorted(archive.getmembers(), key=lambda item: item.name)
        payloads: list[tuple[tarfile.TarInfo, bytes | None]] = []
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Links are forbidden in source distributions: {member.name}")
            payload = archive.extractfile(member).read() if member.isfile() else None
            normalized = tarfile.TarInfo(member.name)
            normalized.type = member.type
            normalized.mode = member.mode
            normalized.size = len(payload) if payload is not None else 0
            normalized.mtime = epoch
            normalized.uid = normalized.gid = 0
            normalized.uname = normalized.gname = ""
            payloads.append((normalized, payload))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as output:
        for member, payload in payloads:
            output.addfile(member, io.BytesIO(payload) if payload is not None else None)
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            compressed.write(buffer.getvalue())


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: normalize_sdist.py INPUT OUTPUT")
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    normalize(Path(sys.argv[1]), Path(sys.argv[2]), epoch=epoch)


if __name__ == "__main__":
    main()
