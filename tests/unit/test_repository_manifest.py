from __future__ import annotations

import importlib.util
from pathlib import Path


def _manifest_module():  # type: ignore[no-untyped-def]
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/generate_repository_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_repository_manifest", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_manifest_uses_only_regular_git_index_files() -> None:
    root = Path(__file__).resolve().parents[2]
    module = _manifest_module()
    paths = module.tracked_paths(root)
    relative = {path.relative_to(root).as_posix() for path in paths}
    assert ".coverage" not in relative
    assert "REPOSITORY.sha256" not in relative
    assert "src/credo_count_sde_v4/version.py" in relative
    assert all(path.is_file() and not path.is_symlink() for path in paths)
