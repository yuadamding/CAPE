from __future__ import annotations

import json
from pathlib import Path

import h5py

from credo_count_sde_v4.prepare import prepare_representation
from credo_count_sde_v4.synthetic import create_synthetic_project


def test_protected_endpoint_changes_do_not_change_representation_identity(tmp_path: Path) -> None:
    left = create_synthetic_project(tmp_path / "left", updates=2)
    right = create_synthetic_project(tmp_path / "right", updates=2)
    count_path = right.parent / "work" / "input" / "counts.h5"
    with h5py.File(count_path, "r+") as handle:
        indptr = handle["X/indptr"][:]
        terminal_start = int(indptr[96])
        data = handle["X/data"][:]
        data[terminal_start:] += 11
        handle["X/data"][:] = data
        handle.flush()
    left_prepared = prepare_representation(left)
    right_prepared = prepare_representation(right)
    left_manifest = json.loads((left_prepared / "prepared.json").read_text())
    right_manifest = json.loads((right_prepared / "prepared.json").read_text())
    assert left_manifest["prepared_id"] == right_manifest["prepared_id"]
    assert left_manifest["cache_generation_id"] != right_manifest["cache_generation_id"]
    assert (left_prepared / "encoder.safetensors").read_bytes() == (
        right_prepared / "encoder.safetensors"
    ).read_bytes()
