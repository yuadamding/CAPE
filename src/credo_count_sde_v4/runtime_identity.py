"""Installed implementation and environment identities used by run contracts."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path

from .canonical import canonical_json_bytes, directory_digest, sha256_bytes, sha256_file


def implementation_tree_hash() -> str:
    return directory_digest(Path(__file__).resolve().parent, ignore=frozenset({"__pycache__"}))


def environment_identity() -> dict[str, object]:
    packages = {}
    names = (
        "h5py",
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "PyYAML",
        "scipy",
        "torch",
    )
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_identity: dict[str, object] = {}
    try:
        import torch

        torch_identity = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    except ImportError:
        torch_identity = {"version": None, "cuda_runtime": None, "cudnn": None}
    image_digest = os.environ.get("CREDO_V4_EXECUTION_IMAGE_DIGEST")
    if image_digest is not None and not image_digest.startswith("sha256:"):
        raise ValueError("CREDO_V4_EXECUTION_IMAGE_DIGEST must be a sha256: digest.")
    return {
        "schema_version": 2,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        # Node kernels are deliberately excluded: preparation and training may
        # run on different Kubernetes nodes under one immutable image.
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "execution_image_digest": image_digest,
        "packages": packages,
        "torch": torch_identity,
    }


def environment_lock_hash() -> str:
    return sha256_bytes(canonical_json_bytes(environment_identity()))


def recipe_distribution_hash() -> str:
    """Return the executing wheel hash when supplied by the launch receipt."""

    wheel_path = os.environ.get("CREDO_V4_WHEEL_PATH")
    if wheel_path is not None:
        path = Path(wheel_path).resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError("CREDO_V4_WHEEL_PATH must identify one regular wheel file.")
        return sha256_file(path)
    value = os.environ.get("CREDO_V4_WHEEL_SHA256")
    if value is None:
        return implementation_tree_hash()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("CREDO_V4_WHEEL_SHA256 must be a lowercase SHA-256 digest.")
    return value
