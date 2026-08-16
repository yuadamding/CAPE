from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import credo_count_sde_v4.reaction.physical_pool_conditional as conditional
from credo_count_sde_v4.api import validate_contract
from credo_count_sde_v4.contracts import (
    PhysicalPoolConditionalReactionBundle,
    PhysicalPoolConditionalReactionReceipt,
)
from credo_count_sde_v4.errors import IntegrityError


def _mixed_catalog() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_index = {"__control__": 0, "T1": 1, "T2": 2, "T3": 3}
    design = {
        "__control__": ([0, 1, 2, 3], 0.0),
        "T1": ([0, 1, 2, 3], 0.7),
        "T2": ([0, 1, 2], -0.5),
        "T3": ([0, 2], 0.2),
    }
    guide = 0
    for target, (folds, effect) in design.items():
        for within_target, fold in enumerate(folds):
            rows.append(
                {
                    "guide_id": f"{target}:guide{within_target}",
                    "target_id": target,
                    "target_index": target_index[target],
                    "is_control": target == "__control__",
                    "source_count": 500 + 17 * guide,
                    "terminal_count": 0,
                    "held_out_fold": fold,
                    "role": (
                        "outer_evaluation"
                        if fold == 0
                        else "inner_validation"
                        if fold == 1
                        else "fit"
                    ),
                    "generating_effect": effect,
                }
            )
            guide += 1
    frame = pd.DataFrame(rows).sort_values("guide_id", kind="stable").reset_index(drop=True)
    logits = np.log(frame.source_count.to_numpy(dtype=np.float64) + 0.5)
    logits += frame.generating_effect.to_numpy(dtype=np.float64)
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    frame["terminal_count"] = np.random.default_rng(20260827).multinomial(100_000, probability)
    return frame.drop(columns="generating_effect")


def test_two_and_three_guide_targets_have_required_sister_support() -> None:
    audit, summary = conditional._sister_support(_mixed_catalog())
    assert summary == {
        "minimum_inner_fit_sisters": 1,
        "zero_inner_fit_sister_guides": 0,
        "minimum_outer_nonouter_sisters": 1,
        "zero_outer_nonouter_sister_guides": 0,
    }
    assert set(audit.sister_count) == {1, 2, 3}


def test_inner_validation_without_fit_sister_fails_closed() -> None:
    catalog = _mixed_catalog()
    unsupported = pd.DataFrame(
        [
            {
                "guide_id": "T4:guide0",
                "target_id": "T4",
                "target_index": 4,
                "is_control": False,
                "source_count": 500,
                "terminal_count": 500,
                "held_out_fold": 1,
                "role": "inner_validation",
            }
        ]
    )
    with pytest.raises(IntegrityError, match="at least one permitted sister"):
        conditional._sister_support(pd.concat([catalog, unsupported], ignore_index=True))


def test_outer_terminal_changes_cannot_change_selection_or_estimator_parity() -> None:
    original = _mixed_catalog()
    changed = original.copy()
    outer = changed.held_out_fold.eq(0)
    changed.loc[outer, "terminal_count"] = np.arange(1, int(outer.sum()) + 1) * 10_000
    selected_a, curve_a = conditional._fit_and_select(original)
    selected_b, curve_b = conditional._fit_and_select(changed)
    assert selected_a == selected_b
    pd.testing.assert_frame_equal(curve_a, curve_b, check_exact=True)
    direct_a = conditional._direct_reference(original, (1, 2, 3))
    direct_b = conditional._direct_reference(changed, (1, 2, 3))
    assert np.array_equal(direct_a, direct_b)


def test_scipy_and_production_conditional_estimators_agree() -> None:
    catalog = _mixed_catalog()
    reference = conditional._direct_reference(catalog, (1, 2, 3))
    production = conditional._fit_production(catalog, (1, 2, 3), 200)
    production_effect = production.target_fitness.detach().numpy()
    assert np.max(np.abs(production_effect - reference)) < 5e-6
    direct_probability = conditional._probabilities(catalog, reference)
    production_probability = conditional._production_probabilities(production, catalog)
    assert np.max(np.abs(production_probability - direct_probability)) < 1e-6
    assert production_probability.sum() == pytest.approx(1.0, abs=1e-15)


def test_parent_crosslink_mismatch_is_rejected(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in ("t02", "amend", "t07"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "artifacts.json").write_text("{}\n")
    (tmp_path / "t02" / "raw-count-mass-noise.json").write_text("{}\n")
    (tmp_path / "amend" / "raw-count-mass-noise-amendment.json").write_text("{}\n")
    (tmp_path / "amend" / "VERIFICATION_RECEIPT.json").write_text("{}\n")
    (tmp_path / "t07" / "reaction-recovery-amendment.json").write_text("{}\n")
    (tmp_path / "t07" / "TEST_RECEIPT.json").write_text("{}\n")
    pooled = SimpleNamespace(
        pooled_data_id="p",
        source_checkpoint="P4",
        terminal_checkpoint="P60",
    )
    raw_noise = SimpleNamespace(
        noise_id="noise",
        pooled_data_id="p",
        source_checkpoint="P4",
        terminal_checkpoint="P60",
    )
    noise = SimpleNamespace(
        amendment_id="amendment",
        parent_noise_id="noise",
        parent_bundle_sha256=conditional.sha256_file(
            tmp_path / "t02" / "raw-count-mass-noise.json"
        ),
        source_checkpoint="P4",
        terminal_checkpoint="P60",
    )
    noise_receipt = SimpleNamespace(
        status="pass",
        parent_bundle_verified=True,
        amendment_id="WRONG",
        parent_noise_id="noise",
    )
    receipt_ref = SimpleNamespace(
        relative_uri="TEST_RECEIPT.json",
        size_bytes=(tmp_path / "t07" / "TEST_RECEIPT.json").stat().st_size,
        sha256=conditional.sha256_file(tmp_path / "t07" / "TEST_RECEIPT.json"),
    )
    reaction = SimpleNamespace(
        method="t07s_interval_metric_amendment_v2",
        parent_qualification_id="qualification",
        test_receipt=receipt_ref,
    )
    reaction_receipt = SimpleNamespace(
        status="pass", optimizer_rerun=False, parent_qualification_id="qualification"
    )
    monkeypatch.setattr(conditional, "verify_pooled_finite_measures", lambda _: pooled)
    monkeypatch.setattr(conditional, "verify_directory", lambda _: None)
    monkeypatch.setattr(
        conditional.RawCountMassNoiseBundle,
        "model_validate_json",
        lambda _: raw_noise,
    )
    monkeypatch.setattr(
        conditional.RawCountMassNoiseAmendment,
        "model_validate_json",
        lambda _: noise,
    )
    monkeypatch.setattr(
        conditional.RawCountMassNoiseAmendmentReceipt,
        "model_validate_json",
        lambda _: noise_receipt,
    )
    monkeypatch.setattr(
        conditional.ReactionRecoveryMetricAmendment,
        "model_validate_json",
        lambda _: reaction,
    )
    monkeypatch.setattr(
        conditional.ReactionRecoveryTestReceiptV3,
        "model_validate_json",
        lambda _: reaction_receipt,
    )
    with pytest.raises(IntegrityError, match="not exactly cross-linked"):
        conditional._verify_parents(
            tmp_path / "unused", tmp_path / "t02", tmp_path / "amend", tmp_path / "t07"
        )


def _fake_parent_files(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    pooled = root / "T00"
    raw_noise = root / "T02A"
    amendment = root / "T02A-amendment"
    reaction = root / "T07S"
    for directory in (pooled, raw_noise, amendment, reaction):
        directory.mkdir()
        (directory / "artifacts.json").write_text("{}\n")
    (pooled / "pooled-data.json").write_text("{}\n")
    (raw_noise / "raw-count-mass-noise.json").write_text("{}\n")
    (amendment / "raw-count-mass-noise-amendment.json").write_text("{}\n")
    (reaction / "reaction-recovery-amendment.json").write_text("{}\n")
    folds = root / "folds.parquet"
    _mixed_catalog()[["guide_id", "held_out_fold"]].to_parquet(folds, index=False)
    return pooled, raw_noise, amendment, reaction, folds


def test_forensic_bundle_is_typed_recomputed_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _mixed_catalog()
    audit, support = conditional._sister_support(catalog)
    pooled_object = SimpleNamespace(
        pooled_data_id="p" * 64,
        source_checkpoint="P4",
        terminal_checkpoint="P60",
    )
    raw_noise_object = SimpleNamespace(noise_id="n" * 64)
    noise_object = SimpleNamespace(amendment_id="a" * 64)
    reaction_object = SimpleNamespace(
        amendment_id="r" * 64, method="t07s_interval_metric_amendment_v2"
    )
    monkeypatch.setattr(
        conditional,
        "_verify_parents",
        lambda *args: (
            pooled_object,
            raw_noise_object,
            noise_object,
            reaction_object,
        ),
    )
    monkeypatch.setattr(
        conditional,
        "_load_catalog",
        lambda *args: (pooled_object, catalog.copy(), audit.copy(), support.copy()),
    )
    pooled, raw_noise, amendment, reaction, folds = _fake_parent_files(tmp_path)
    output = conditional.qualify_physical_pool_conditional_reaction(
        tmp_path / "T07R-v2",
        pooled_bundle=pooled,
        t02a_bundle=raw_noise,
        t02a_amendment=amendment,
        t07s_amendment=reaction,
        fold_assignment=folds,
    )
    bundle = conditional.verify_physical_pool_conditional_reaction(
        output,
        pooled_bundle=pooled,
        t02a_bundle=raw_noise,
        t02a_amendment=amendment,
        t07s_amendment=reaction,
        fold_assignment=folds,
    )
    receipt = PhysicalPoolConditionalReactionReceipt.model_validate_json(
        (output / "TEST_RECEIPT.json").read_text()
    )
    assert isinstance(bundle, PhysicalPoolConditionalReactionBundle)
    assert receipt.estimator_parity_pass
    assert receipt.factorization_pass
    assert receipt.status == f"forensic_{receipt.predictive_decision}"
    assert (
        validate_contract(output / "physical-pool-conditional-reaction.json")["contract_type"]
        == "PhysicalPoolConditionalReactionBundle"
    )
    assert validate_contract(output / "TEST_RECEIPT.json")["contract_type"] == (
        "PhysicalPoolConditionalReactionReceipt"
    )
    metrics = output / "OUTER_GUIDE_METRICS.parquet"
    original = metrics.read_bytes()
    metrics.write_bytes(original + b"tamper")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        conditional.verify_physical_pool_conditional_reaction(
            output,
            pooled_bundle=pooled,
            t02a_bundle=raw_noise,
            t02a_amendment=amendment,
            t07s_amendment=reaction,
            fold_assignment=folds,
        )
    metrics.write_bytes(original)
