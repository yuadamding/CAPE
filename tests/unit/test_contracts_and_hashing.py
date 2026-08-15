from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from credo_count_sde_v4.canonical import (
    canonical_json_bytes,
    contract_id,
    path_manifest,
    validate_relative_uri,
)
from credo_count_sde_v4.contracts import (
    ComponentTestContract,
    CounterfactualBranch,
    CounterfactualDesign,
    DenominatorBlock,
    DenominatorContract,
    FeatureIndex,
    FeatureKey,
    InformationSet,
    ModelConfig,
    PhysicalGrid,
    PoolContract,
    PoolContributor,
    ResolvedRunCapabilities,
    RunIntent,
    SemanticStudySnapshot,
    SplitContract,
    TrainingConfig,
    checked_contract,
)
from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.runtime_identity import environment_identity


def test_canonical_json_is_order_stable_and_rejects_nonfinite() -> None:
    left = canonical_json_bytes({"b": 2, "a": [1, True]})
    right = canonical_json_bytes({"a": [1, True], "b": 2})
    assert left == right
    assert contract_id({"b": 2, "a": 1}) == contract_id({"a": 1, "b": 2})
    with pytest.raises(ContractError, match="Non-finite"):
        canonical_json_bytes({"bad": math.nan})
    with pytest.raises(ContractError, match="Non-string"):
        canonical_json_bytes({1: "bad"})
    with pytest.raises(ContractError, match="Unsupported"):
        canonical_json_bytes({"bad": object()})


def test_environment_identity_binds_image_without_node_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    monkeypatch.setenv("CREDO_V4_EXECUTION_IMAGE_DIGEST", digest)
    identity = environment_identity()
    assert identity["schema_version"] == 2
    assert identity["execution_image_digest"] == digest
    assert set(identity["platform"]) == {"system", "machine"}
    monkeypatch.setenv("CREDO_V4_EXECUTION_IMAGE_DIGEST", "mutable-tag")
    with pytest.raises(ValueError, match="sha256"):
        environment_identity()


@pytest.mark.parametrize("value", ["/absolute", "../escape", "a/../../b", r"a\b"])
def test_relative_uri_fails_closed(value: str) -> None:
    with pytest.raises(ContractError):
        validate_relative_uri(value)


def test_contracts_reject_unknown_fields_and_information_leakage() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(target_count=2, unknown=True)
    with pytest.raises(ValidationError, match="protected"):
        InformationSet(
            information_set_id="bad",
            fit_rows=(1, 2),
            protected_rows=(2, 3),
        )


def test_capabilities_are_derived_from_intent() -> None:
    state = ResolvedRunCapabilities.for_intent(RunIntent.COUNT_STATE)
    measure = ResolvedRunCapabilities.for_intent(RunIntent.COUNT_MEASURE)
    context = ResolvedRunCapabilities.for_intent(RunIntent.COUNT_CONTEXT)
    assert not state.predict_relative_mass
    assert measure.predict_relative_mass and not measure.dynamic_context_counterfactual
    assert context.dynamic_context_counterfactual


def test_contract_invariant_failures_are_explicit() -> None:
    with pytest.raises(ValidationError, match="pairwise disjoint"):
        SplitContract(
            split_id="bad",
            training_units=("same",),
            outer_evaluation_units=("same",),
            grouping_unit="unit",
        )
    feature = FeatureKey(namespace="n", namespace_version="v1", feature_id="x")
    with pytest.raises(ValidationError, match="unique"):
        FeatureIndex(features=(feature, feature), ordered_hash="0" * 64)
    with pytest.raises(ValidationError, match="nonempty"):
        DenominatorBlock(block_id="bad", category_ids=())
    with pytest.raises(ValidationError, match="remain"):
        DenominatorBlock(block_id="bad", category_ids=("a",), explicit_zero_categories=("b",))
    with pytest.raises(ValidationError, match="frozen"):
        DenominatorContract(denominator_id="bad", blocks=(), source_smoothing=1.0)
    with pytest.raises(ValidationError, match="physical pool"):
        PoolContract(
            pool_contract_id="bad",
            physical_pool_ids=("a", "b"),
            contributors=(PoolContributor(pool_id="a", series_id="x"),),
        )
    with pytest.raises(ValidationError, match="at least one series"):
        SemanticStudySnapshot(
            study_id="empty",
            series=(),
            observed_edges=(),
            feature_index_hash="0" * 64,
            row_universe_hash="1" * 64,
            exposure_registry={"records": []},
        )
    with pytest.raises(ValidationError, match="Shared diffusion"):
        ModelConfig(target_count=1, shared_diffusion=True)
    with pytest.raises(ValidationError, match="exceeds"):
        TrainingConfig(max_updates=2, selected_update=3)
    with pytest.raises(ValidationError, match="Physical grid"):
        PhysicalGrid(axis_unit="hour", duration=8, maximum_step=2, steps=3, step_size=8 / 3)
    branch = CounterfactualBranch(
        branch_id="same", effect_mode="factual", context_mode="source_fixed"
    )
    with pytest.raises(ValidationError, match="unique"):
        CounterfactualDesign(series_index=0, branches=(branch, branch))
    with pytest.raises(ContractError):
        checked_contract(ModelConfig, {"target_count": 0})


def test_path_manifest_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("x")
    (tmp_path / "link").symlink_to(target.name)
    with pytest.raises(ContractError, match="Symlink"):
        path_manifest(tmp_path)


def test_component_contract_enforces_channel_isolation_matrix() -> None:
    payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": "T06_DIFFUSION",
        "component": "diffusion",
        "primary_metric": "terminal_variance_error",
        "primary_baseline": "fixed_zero_diffusion",
        "required_margin": 0.0,
        "drift": "fixed",
        "diffusion": "trainable",
        "reaction": "off",
        "ecology": "off",
        "decoder": "off",
        "update_zero_selectable": True,
        "post_selection_refit_required": True,
    }
    payload["test_contract_id"] = contract_id(payload, id_field="test_contract_id")
    ComponentTestContract.model_validate(payload)
    payload["reaction"] = "trainable"
    payload["test_contract_id"] = contract_id(payload, id_field="test_contract_id")
    with pytest.raises(ValidationError, match="channel-isolation"):
        ComponentTestContract.model_validate(payload)
