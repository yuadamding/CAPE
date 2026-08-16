from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.claims import validate_g14_contracts
from credo_count_sde_v4.contracts import (
    ClaimRecord,
    ClaimRegistry,
    G14MultiplicityContract,
    G14MultiplicityFamily,
    G14RobustnessPlan,
    G14SealContract,
    RobustnessAxis,
    StrictModel,
)
from credo_count_sde_v4.errors import IntegrityError

OUTPUTS = (
    "ROBUSTNESS_MATRIX.parquet",
    "SIMULTANEOUS_INTERVALS.parquet",
    "MULTIPLICITY_DECISION.json",
    "CLAIM_LEDGER.parquet",
    "FINAL_EVIDENCE_GRAPH.json",
    "FINAL_CLAIM_SEAL_RECEIPT.json",
    "SHA256SUMS",
)
ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    payload[id_field] = contract_id(payload, id_field=id_field)
    return model.model_validate(payload)


def _contracts() -> tuple[
    ClaimRegistry,
    G14RobustnessPlan,
    G14MultiplicityContract,
    G14SealContract,
]:
    claim = ClaimRecord(
        claim_id="STATE_E8",
        component_id="G07",
        endpoint="heldout_donor_8h_state",
        candidate="qualified_state_model",
        comparator="strongest_training_only_baseline",
        direction="lower",
        primary_or_secondary="primary",
        claim_family="P",
        resampling_unit="target_within_donor",
        multiplicity_method="paired_max_statistic",
        eligible_wording="donor-inductive prediction in this four-donor cohort",
        forbidden_wording=("general donor-population prediction", "causal state effect"),
    )
    registry = _identified(
        ClaimRegistry,
        {
            "schema_version": 1,
            "frozen_before_first_relevant_outer_evaluation": True,
            "records": [claim.model_dump(mode="json")],
        },
        "registry_id",
    )
    robustness = _identified(
        G14RobustnessPlan,
        {
            "schema_version": 1,
            "axes": [
                RobustnessAxis(
                    axis_id="algorithmic_seed",
                    values=("base", "sensitivity"),
                ).model_dump(mode="json")
            ],
            "base_contract_hashes": ["a" * 64],
            "model_fitting": False,
            "model_selection": False,
            "threshold_adjustment": False,
            "target_discovery": False,
            "sensitivities_select_reported_model": False,
        },
        "plan_id",
    )
    multiplicity = _identified(
        G14MultiplicityContract,
        {
            "schema_version": 1,
            "claim_registry_id": registry.registry_id,
            "families": [
                G14MultiplicityFamily(
                    family_id="P",
                    claim_ids=("STATE_E8",),
                    method="paired_max_statistic",
                    conditional_on_observed_donors=True,
                ).model_dump(mode="json")
            ],
            "general_population_donor_claim_allowed": False,
        },
        "multiplicity_contract_id",
    )
    seal = _identified(
        G14SealContract,
        {
            "schema_version": 1,
            "claim_registry_id": registry.registry_id,
            "robustness_plan_id": robustness.plan_id,
            "multiplicity_contract_id": multiplicity.multiplicity_contract_id,
            "purpose": "robustness_multiplicity_and_claim_sealing",
            "model_fitting": False,
            "model_selection": False,
            "threshold_adjustment": False,
            "target_discovery": False,
            "required_outputs": OUTPUTS,
        },
        "g14_contract_id",
    )
    return registry, robustness, multiplicity, seal


def test_g14_contract_graph_is_complete_and_evidence_only() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    validate_g14_contracts(registry, robustness, multiplicity, seal)
    assert not robustness.model_fitting
    assert not robustness.model_selection
    assert not robustness.threshold_adjustment
    assert not robustness.target_discovery
    assert not seal.model_fitting
    assert set(seal.required_outputs) == set(OUTPUTS)


def test_g14_registry_rejects_unknown_parent_claim() -> None:
    claim = ClaimRecord(
        claim_id="MECHANISM",
        component_id="G09",
        parent_claim_ids=("MISSING",),
        endpoint="joint_model",
        candidate="joint",
        comparator="separate",
        direction="lower",
        primary_or_secondary="primary",
        claim_family="M",
        resampling_unit="target_within_donor",
        multiplicity_method="joint_max_statistic",
        eligible_wording="internal model mechanism",
        forbidden_wording=("biological causality",),
    )
    payload = {
        "schema_version": 1,
        "registry_id": "pending",
        "frozen_before_first_relevant_outer_evaluation": True,
        "records": [claim.model_dump(mode="json")],
    }
    payload["registry_id"] = contract_id(payload, id_field="registry_id")
    with pytest.raises(ValidationError, match="unknown parent"):
        ClaimRegistry.model_validate(payload)


def test_g14_cross_validation_rejects_uncovered_claim() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    altered_payload = multiplicity.model_dump(mode="json")
    altered_payload["families"][0]["claim_ids"] = ["UNKNOWN"]
    altered_payload["multiplicity_contract_id"] = "pending"
    altered_payload["multiplicity_contract_id"] = contract_id(
        altered_payload, id_field="multiplicity_contract_id"
    )
    altered = G14MultiplicityContract.model_validate(altered_payload)
    altered_seal_payload = seal.model_dump(mode="json")
    altered_seal_payload["multiplicity_contract_id"] = altered.multiplicity_contract_id
    altered_seal_payload["g14_contract_id"] = "pending"
    altered_seal_payload["g14_contract_id"] = contract_id(
        altered_seal_payload, id_field="g14_contract_id"
    )
    altered_seal = G14SealContract.model_validate(altered_seal_payload)
    with pytest.raises(IntegrityError, match="unknown claim"):
        validate_g14_contracts(registry, robustness, altered, altered_seal)


def test_g14_contracts_dispatch_through_public_validator(tmp_path: Path) -> None:
    registry, robustness, multiplicity, seal = _contracts()
    expected = (
        ("claim-registry.json", registry, "ClaimRegistry"),
        ("robustness.json", robustness, "G14RobustnessPlan"),
        ("multiplicity.json", multiplicity, "G14MultiplicityContract"),
        ("seal.json", seal, "G14SealContract"),
    )
    for name, contract, contract_type in expected:
        path = tmp_path / name
        path.write_text(contract.model_dump_json() + "\n")
        assert validate_contract(path)["contract_type"] == contract_type
