from __future__ import annotations

from pathlib import Path

import pytest

from credo_count_sde_v4.contracts import RunIntent
from credo_count_sde_v4.synthetic import create_synthetic_project


@pytest.fixture
def synthetic_config(tmp_path: Path) -> Path:
    return create_synthetic_project(tmp_path / "project", intent=RunIntent.COUNT_CONTEXT, updates=8)
