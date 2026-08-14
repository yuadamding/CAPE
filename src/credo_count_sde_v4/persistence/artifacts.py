"""Safe tensor files and manifest-last atomic directory publication."""

from __future__ import annotations

import json
import os
import shutil
import struct
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..canonical import canonical_json_bytes, path_manifest, sha256_file
from ..contracts import ArtifactRef
from ..errors import ContractError, IntegrityError

_TORCH_TO_CODE: dict[torch.dtype, str] = {
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_CODE_TO_NUMPY: dict[str, np.dtype[Any]] = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def save_tensor_file(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    """Write the documented safetensors wire format without pickle.

    This minimal codec supports the dtypes used by the v4 model and optimizer.
    It is intentionally deterministic: names are sorted and tensors are
    contiguous little-endian CPU values.
    """

    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.dtype not in _TORCH_TO_CODE:
            raise ContractError(f"Unsupported safe tensor dtype {tensor.dtype} for {name!r}.")
        array = tensor.numpy()
        dtype = _CODE_TO_NUMPY[_TORCH_TO_CODE[tensor.dtype]]
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        payload = array.tobytes(order="C")
        header[name] = {
            "dtype": _TORCH_TO_CODE[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    with path.open("xb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for payload in payloads:
            handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_tensor_file(path: Path, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load one safe tensor file after structural bounds validation."""

    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise IntegrityError(f"Truncated tensor header: {path}.")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > path.stat().st_size - 8:
            raise IntegrityError(f"Invalid tensor header size: {path}.")
        try:
            header = json.loads(handle.read(header_size))
        except Exception as exc:
            raise IntegrityError(f"Invalid tensor JSON header: {path}.") from exc
        data = handle.read()
    tensors: dict[str, torch.Tensor] = {}
    occupied: list[tuple[int, int]] = []
    for name, entry in header.items():
        if not isinstance(name, str) or entry.get("dtype") not in _CODE_TO_NUMPY:
            raise IntegrityError(f"Unsupported tensor entry {name!r} in {path}.")
        start, end = entry.get("data_offsets", [-1, -1])
        if not (0 <= start <= end <= len(data)):
            raise IntegrityError(f"Out-of-bounds tensor {name!r} in {path}.")
        occupied.append((start, end))
        dtype = _CODE_TO_NUMPY[entry["dtype"]]
        shape = tuple(int(value) for value in entry.get("shape", []))
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if end - start != expected:
            raise IntegrityError(f"Tensor byte count mismatch for {name!r} in {path}.")
        array = np.frombuffer(data[start:end], dtype=dtype).reshape(shape).copy()
        tensors[name] = torch.from_numpy(array).to(device)
    occupied.sort()
    if any(
        end > next_start for (_, end), (next_start, _) in zip(occupied, occupied[1:], strict=False)
    ):
        raise IntegrityError(f"Overlapping tensor ranges in {path}.")
    cursor = 0
    for start, end in occupied:
        if start != cursor:
            raise IntegrityError(f"Unreferenced tensor bytes in {path}.")
        cursor = end
    if cursor != len(data):
        raise IntegrityError(f"Trailing unreferenced tensor bytes in {path}.")
    return tensors


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"Symlink is forbidden in published bundle: {path}.")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def publish_directory(destination: Path, writer: Callable[[Path], None]) -> Path:
    """Publish an immutable directory with manifest-last, same-FS rename.

    A committed destination is never overwritten. A failure leaves only an
    ignored ``.tmp-*`` directory and the previous generation untouched.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        writer(temporary)
        manifest = path_manifest(temporary)
        artifacts_path = temporary / "artifacts.json"
        artifacts_path.write_bytes(
            canonical_json_bytes({"schema_version": 1, "files": manifest}) + b"\n"
        )
        _fsync_tree(temporary)
        verify_directory(temporary, require_committed=False)
        committed = temporary / "COMMITTED"
        committed.write_bytes(b"CREDO-V4-COMMITTED\n")
        with committed.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination
    except Exception:
        # Retain a nonempty temporary directory for receipt-gated diagnosis.
        if temporary.exists() and not any(temporary.iterdir()):
            temporary.rmdir()
        raise


def verify_directory(root: Path, *, require_committed: bool = True) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError(f"Bundle root is not a regular directory: {root}.")
    if require_committed and (root / "COMMITTED").read_bytes() != b"CREDO-V4-COMMITTED\n":
        raise IntegrityError(f"Bundle is not committed: {root}.")
    manifest_path = root / "artifacts.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise IntegrityError(f"Missing or invalid artifact manifest in {root}.") from exc
    expected = {row["path"]: row for row in manifest.get("files", [])}
    actual = {
        row["path"]: row
        for row in path_manifest(root, ignore=frozenset({"artifacts.json", "COMMITTED"}))
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise IntegrityError(f"Bundle file set mismatch; missing={missing}, extra={extra}.")
    for relative, row in expected.items():
        if (
            actual[relative]["size_bytes"] != row["size_bytes"]
            or actual[relative]["sha256"] != row["sha256"]
        ):
            raise IntegrityError(f"Artifact mismatch: {relative}.")
    return manifest


def artifact_ref(root: Path, path: Path, *, schema_id: str, media_type: str) -> ArtifactRef:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"Artifact is not a regular file: {path}.")
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.relative_to(root).as_posix(),
    )


def quarantine_orphan(path: Path, quarantine: Path) -> Path:
    """Move one exact uncommitted temporary directory to quarantine."""

    if not path.name.startswith(".") or ".tmp-" not in path.name or (path / "COMMITTED").exists():
        raise ContractError(f"Refusing to quarantine non-orphan path: {path}.")
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / path.name
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(path), str(target))
    return target
