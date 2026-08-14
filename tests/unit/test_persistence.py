from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.persistence import (
    artifact_ref,
    load_tensor_file,
    publish_directory,
    save_tensor_file,
    verify_directory,
)
from credo_count_sde_v4.persistence.artifacts import quarantine_orphan


def test_safe_tensor_roundtrip_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "state.safetensors"
    tensors = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.tensor([2])}
    save_tensor_file(path, tensors)
    loaded = load_tensor_file(path)
    assert set(loaded) == {"a", "b"}
    assert torch.equal(loaded["a"], tensors["a"])
    assert not path.read_bytes().startswith(b"PK")


def test_atomic_publish_and_extra_file_detection(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    publish_directory(destination, lambda root: (root / "value.txt").write_text("ok"))
    verify_directory(destination)
    (destination / "extra.txt").write_text("bad")
    with pytest.raises(IntegrityError, match="file set mismatch"):
        verify_directory(destination)


def test_writer_failure_never_publishes_destination(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"

    def fail(root: Path) -> None:
        (root / "partial").write_text("partial")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        publish_directory(destination, fail)
    assert not destination.exists()
    orphans = list(tmp_path.glob(".bundle.tmp-*"))
    assert orphans
    moved = quarantine_orphan(orphans[0], tmp_path / "quarantine")
    assert moved.exists() and not orphans[0].exists()


def test_tensor_loader_rejects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"short")
    with pytest.raises(IntegrityError, match="Truncated"):
        load_tensor_file(path)


def test_tensor_loader_rejects_unreferenced_trailing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "trailing.safetensors"
    save_tensor_file(path, {"value": torch.tensor([1.0])})
    with path.open("ab") as handle:
        handle.write(b"unreferenced")
    with pytest.raises(IntegrityError, match="Trailing unreferenced"):
        load_tensor_file(path)


def _raw_tensor(path: Path, header: object, data: bytes = b"") -> None:
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.mark.parametrize(
    ("header", "data", "message"),
    [
        ({"x": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 1]}}, b"x", "Unsupported"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}}, b"1234", "Out-of-bounds"),
        ({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"1234", "byte count"),
        (
            {
                "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
            },
            b"123",
            "Overlapping",
        ),
        ({"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}, b"12", "Unreferenced"),
    ],
)
def test_tensor_loader_rejects_malformed_ranges(
    tmp_path: Path, header: object, data: bytes, message: str
) -> None:
    path = tmp_path / f"bad-{message}.safetensors"
    _raw_tensor(path, header, data)
    with pytest.raises(IntegrityError, match=message):
        load_tensor_file(path)


def test_tensor_codec_rejects_bad_json_and_dtype(tmp_path: Path) -> None:
    invalid = tmp_path / "json.safetensors"
    invalid.write_bytes(struct.pack("<Q", 1) + b"{")
    with pytest.raises(IntegrityError, match="JSON"):
        load_tensor_file(invalid)
    with pytest.raises(Exception, match="Unsupported safe tensor dtype"):
        save_tensor_file(
            tmp_path / "complex.safetensors", {"x": torch.ones(1, dtype=torch.complex64)}
        )


def test_publish_rejects_existing_destination_and_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(destination, lambda root: None)

    linked = tmp_path / "linked"

    def writer(root: Path) -> None:
        target = root / "target"
        target.write_text("x")
        (root / "link").symlink_to(target.name)

    with pytest.raises(Exception, match="Symlink"):
        publish_directory(linked, writer)
    assert not linked.exists()


def test_verifier_and_artifact_reference_fail_closed(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.write_text("x")
    with pytest.raises(IntegrityError, match="regular directory"):
        verify_directory(plain)
    bundle = tmp_path / "bundle"
    publish_directory(bundle, lambda root: (root / "value").write_text("x"))
    reference = artifact_ref(tmp_path, bundle / "value", schema_id="test", media_type="text/plain")
    assert reference.relative_uri == "bundle/value"
    with pytest.raises(Exception, match="regular file"):
        artifact_ref(tmp_path, bundle, schema_id="test", media_type="text/plain")
    (bundle / "COMMITTED").write_text("wrong")
    with pytest.raises(IntegrityError, match="not committed"):
        verify_directory(bundle)


def test_quarantine_rejects_non_orphan_and_collision(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    with pytest.raises(Exception, match="Refusing"):
        quarantine_orphan(ordinary, tmp_path / "q")
    orphan = tmp_path / ".run.tmp-fixed"
    orphan.mkdir()
    target = tmp_path / "q" / orphan.name
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        quarantine_orphan(orphan, tmp_path / "q")
