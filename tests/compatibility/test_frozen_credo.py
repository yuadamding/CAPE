from __future__ import annotations

from pathlib import Path

import pytest

from credo_count_sde_v4.compat.credo3 import FROZEN_COMMIT, FROZEN_SOURCE_SHA256
from credo_count_sde_v4.compat.credo3 import verify as verify_module
from credo_count_sde_v4.compat.credo3.verify import verify_frozen_credo
from credo_count_sde_v4.recipe import recipe


def test_frozen_credo_preflight_when_workspace_is_available() -> None:
    root = Path(__file__).resolve().parents[3]
    if not (root / "CREDO").exists():
        pytest.skip("Sibling CREDO checkout is supplied by release CI.")
    receipt = verify_frozen_credo(root)
    assert receipt["commit"] == FROZEN_COMMIT
    assert receipt["artifact_sha256"] == FROZEN_SOURCE_SHA256


def test_development_descriptor_keeps_v4_loader_canonical() -> None:
    assert recipe.recipe_id == "credo.count_sde_v4"
    assert recipe.recipe_version == "4.0.dev34"
    assert recipe.loader == "credo-v4 open-run"
    assert recipe.discovery_only


def test_frozen_credo_preflight_without_git_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[3]
    if not (root / "CREDO").exists():
        pytest.skip("Sibling CREDO checkout is supplied by release CI.")
    monkeypatch.setattr(verify_module.shutil, "which", lambda _: None)
    receipt = verify_frozen_credo(root)
    assert receipt["commit"] == FROZEN_COMMIT
    assert receipt["verification_method"] == "archive_byte_comparison"
