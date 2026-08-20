"""Emit the exact CPU environment authority used for development validation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import sysconfig
from pathlib import Path

import numpy as np
import torch

from credo_count_sde_v4.canonical import atomic_json, canonical_json_bytes, sha256_bytes

DIRECT_DEPENDENCIES = ("h5py", "numpy", "pandas", "pyarrow", "pydantic", "PyYAML", "scipy", "torch")


def _distribution_record(name: str) -> dict[str, object]:
    distribution = importlib.metadata.distribution(name)
    files = distribution.files or ()
    record = next((item for item in files if item.name == "RECORD"), None)
    metadata = next((item for item in files if item.name == "METADATA"), None)
    if record is None or metadata is None:
        raise RuntimeError(f"Installed distribution {name!r} has no RECORD/METADATA authority.")
    record_path = Path(distribution.locate_file(record))
    metadata_path = Path(distribution.locate_file(metadata))
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "record_sha256": sha256_bytes(record_path.read_bytes()),
        "metadata_sha256": sha256_bytes(metadata_path.read_bytes()),
        "installed_file_count": len(files),
    }


def _cpu_identity() -> dict[str, object]:
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in {"vendor_id", "model name", "cpu family", "model", "stepping", "microcode"}:
                fields.setdefault(key.replace(" ", "_"), value)
    fields["logical_cpu_count"] = str(os.cpu_count())
    return dict(sorted(fields.items()))


def tested_environment() -> dict[str, object]:
    numpy_config = getattr(np.__config__, "CONFIG", {})
    payload: dict[str, object] = {
        "schema_version": 1,
        "environment_kind": "exact_local_lock",
        "scope": "contract_and_cpu_validation_only",
        "expression_access": False,
        "gpu_execution": False,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "full_version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "compiler": platform.python_compiler(),
            "build": platform.python_build(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": _cpu_identity(),
        "numerical_runtime": {
            "numpy_build_dependencies": numpy_config.get("Build Dependencies", {}),
            "numpy_simd": numpy_config.get("SIMD Extensions", {}),
            "torch_parallel_info": torch.__config__.parallel_info(),
            "torch_build_config": torch.__config__.show(),
        },
        "distributions": [_distribution_record(name) for name in DIRECT_DEPENDENCIES],
    }
    digest = sha256_bytes(canonical_json_bytes(payload))
    return {**payload, "execution_environment_digest": f"sha256:{digest}"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "locks/tested-environment.v1.json"
    payload = tested_environment()
    atomic_json(destination, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
