"""Canonical JSON and content identities.

The scientific contract uses a deliberately small canonical JSON profile. It
rejects non-finite values and path escapes before hashing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ContractError


def _validate(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"Non-finite number at {path}.")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractError(f"Non-string mapping key at {path}.")
            _validate(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate(child, f"{path}[{index}]")
        return
    raise ContractError(f"Unsupported canonical value {type(value).__name__} at {path}.")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one validated object with stable UTF-8 JSON bytes."""

    _validate(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def contract_id(value: Any, *, id_field: str | None = None) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    if id_field and isinstance(value, dict):
        value = {key: child for key, child in value.items() if key != id_field}
    return sha256_bytes(canonical_json_bytes(value))


def validate_relative_uri(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ContractError(f"Artifact URI must be a safe POSIX relative path: {value!r}.")
    return value


def path_manifest(root: Path, *, ignore: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Return a byte-identity manifest without following symlinks."""

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in ignore or any(part in ignore for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ContractError(f"Symlink is forbidden in manifested directory: {relative}.")
        if path.is_file():
            rows.append(
                {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return rows


def directory_digest(root: Path, *, ignore: frozenset[str] = frozenset()) -> str:
    return sha256_bytes(canonical_json_bytes(path_manifest(root, ignore=ignore)))


def atomic_json(path: Path, value: Any) -> None:
    """Atomically publish one canonical JSON file on its target filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = canonical_json_bytes(value) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
