from __future__ import annotations

import json
from pathlib import Path

from credo_count_sde_v4 import api
from credo_count_sde_v4.synthetic import create_synthetic_project


def test_fork_starts_new_attempt_with_parent_provenance(tmp_path: Path) -> None:
    parent_config = create_synthetic_project(tmp_path / "parent", updates=2)
    api.prepare(parent_config)
    api.compile_run(parent_config)
    api.train(parent_config, device="cpu")
    parent_latest = json.loads(
        (tmp_path / "parent/work/training/checkpoints/latest.json").read_text()
    )
    parent_generation = tmp_path / "parent/work/training" / parent_latest["relative_uri"]

    child_config = create_synthetic_project(tmp_path / "child", updates=2)
    api.prepare(child_config)
    api.compile_run(child_config)
    api.fork(child_config, from_checkpoint=parent_generation, device="cpu")
    first_generation = (
        tmp_path / "child/work/training/checkpoints/generation-000000001/checkpoint.json"
    )
    child_manifest = json.loads(first_generation.read_text())
    assert child_manifest["parent_checkpoint_id"] == parent_latest["checkpoint_id"]
