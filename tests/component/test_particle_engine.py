from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from credo_count_sde_v4.contracts import (
    ComponentTestContract,
    ComponentTestReceipt,
    ParticleEngineQualificationBundle,
    ParticleEngineTestReceipt,
)
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.model import DynamicPoolBank
from credo_count_sde_v4.numerics import (
    qualify_particle_engine,
    verify_particle_engine_qualification,
)


@pytest.fixture(scope="module")
def qualified_t04(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("t04") / "particle-engine"
    qualify_particle_engine(destination)
    return destination


def test_t04_fixed_truth_qualification_passes_every_channel_gate(
    qualified_t04: Path,
) -> None:
    bundle = verify_particle_engine_qualification(qualified_t04)
    receipt = ParticleEngineTestReceipt.model_validate_json(
        (qualified_t04 / "TEST_RECEIPT.json").read_text()
    )
    component = ComponentTestReceipt.model_validate_json(
        (qualified_t04 / "COMPONENT_RECEIPT.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json(
        (qualified_t04 / "TEST_CONTRACT.json").read_text()
    )
    assert isinstance(bundle, ParticleEngineQualificationBundle)
    assert bundle.particle_grid == (64, 256, 1024, 4096)
    assert bundle.step_grid == (8, 16, 32, 64)
    assert bundle.seed_count == 50
    assert bundle.environment_hash == receipt.environment_hash
    assert receipt.status == component.status == "pass"
    assert contract.test_id == "T04_PARTICLE_ENGINE"
    assert contract.drift == contract.diffusion == contract.reaction == "fixed"
    assert contract.ecology == contract.decoder == "off"
    assert receipt.deterministic_drift_max_abs_error <= 1e-12
    assert receipt.ou_largest_grid_variance_relative_error < 0.10
    assert receipt.reaction_max_relative_error < 1e-4
    assert receipt.ecology_absolute_weight_max_error <= 1e-12
    assert receipt.stabilized_absolute_log_mass_pass


def test_t04_ou_grid_and_negative_control_are_complete(qualified_t04: Path) -> None:
    ou = pd.read_parquet(qualified_t04 / "OU_GRID.parquet")
    assert len(ou) == 16
    assert set(ou.particles) == {64, 256, 1024, 4096}
    assert set(ou.steps) == {8, 16, 32, 64}
    assert set(ou.seed_count) == {50}
    ecology = json.loads((qualified_t04 / "ECOLOGY_RESULTS.json").read_text())
    assert ecology["normalized_context_negative_control_gap"] >= 1e-2
    assert ecology["stabilized_log_weight_pass"] is True
    lifecycle = json.loads((qualified_t04 / "LIFECYCLE_RESULTS.json").read_text())
    assert lifecycle["deterministic_cpu_replay_pass"] is True
    assert lifecycle["interrupted_resumed_parity_pass"] is True
    assert lifecycle["no_guide_switching_pass"] is True


def test_dynamic_pool_log_mass_is_stable_and_fails_closed() -> None:
    state = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    pools = torch.zeros(2, dtype=torch.int64)
    extreme = DynamicPoolBank.from_log_series(
        state,
        torch.tensor([1000.0, 998.0], dtype=torch.float64),
        pools,
        pool_count=1,
    )
    shifted = DynamicPoolBank.from_series(
        state,
        torch.exp(torch.tensor([0.0, -2.0], dtype=torch.float64)),
        pools,
        pool_count=1,
    )
    assert torch.allclose(extreme.state_mean, shifted.state_mean, atol=1e-12)
    assert torch.allclose(extreme.log_mass - 1000.0, shifted.log_mass, atol=1e-12)
    with pytest.raises(ContractError, match="zero total mass"):
        DynamicPoolBank.from_log_series(
            state,
            torch.full((2,), -torch.inf, dtype=torch.float64),
            pools,
            pool_count=1,
        )
    with pytest.raises(ContractError, match="finite and nonnegative"):
        DynamicPoolBank.from_series(
            state,
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            pools,
            pool_count=1,
        )


def test_t04_publication_is_no_clobber_and_tamper_evident(qualified_t04: Path) -> None:
    with pytest.raises(FileExistsError):
        qualify_particle_engine(qualified_t04)
    target = qualified_t04 / "REACTION_RESULTS.json"
    original = target.read_bytes()
    target.write_bytes(original + b"\n")
    try:
        with pytest.raises(IntegrityError, match="Artifact mismatch"):
            verify_particle_engine_qualification(qualified_t04)
    finally:
        target.write_bytes(original)
    verify_particle_engine_qualification(qualified_t04)
