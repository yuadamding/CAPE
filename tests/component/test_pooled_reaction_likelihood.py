from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import credo_count_sde_v4.reaction.pooled_likelihood as pooled_likelihood
from credo_count_sde_v4.api import validate_contract
from credo_count_sde_v4.contracts import (
    PooledReactionLikelihoodBundle,
    PooledReactionLikelihoodReceipt,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.persistence import verify_directory


def _catalog() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    effects = {"__control__": 0.0, "T1": 0.7, "T2": -0.5}
    source = {"__control__": 500, "T1": 700, "T2": 600}
    guide_index = 0
    for fold in range(4):
        for target_index, target_id in enumerate(effects):
            is_control = target_id == "__control__"
            source_count = source[target_id] + 17 * fold
            rows.append(
                {
                    "guide_id": f"{target_id}:guide{fold}",
                    "target_id": target_id,
                    "target_index": target_index,
                    "is_control": is_control,
                    "source_count": source_count,
                    "terminal_count": 0,
                    "held_out_fold": fold,
                    "role": (
                        "outer_evaluation"
                        if fold == 0
                        else "inner_validation"
                        if fold == 1
                        else "fit"
                    ),
                    "guide_index": guide_index,
                }
            )
            guide_index += 1
    frame = pd.DataFrame(rows).sort_values("guide_id", kind="stable").reset_index(drop=True)
    raw = frame.target_id.map(effects).to_numpy(dtype=np.float64)
    probability = np.exp(np.log(frame.source_count.to_numpy() + 0.5) + raw)
    probability /= probability.sum()
    frame["terminal_count"] = np.random.default_rng(1107).multinomial(60_000, probability)
    return frame.drop(columns="guide_index")


def test_independent_reference_and_production_likelihood_recover_same_estimator() -> None:
    frame = _catalog().loc[lambda value: value.held_out_fold != 0].copy()
    reference = pooled_likelihood._direct_reference(frame)
    production = pooled_likelihood._fit_production(frame, 200)
    production_effect = production.target_fitness.detach().numpy()
    assert np.max(np.abs(production_effect - reference)) < 5e-6
    direct_probability = pooled_likelihood._direct_probabilities(frame, reference)
    production_probability = pooled_likelihood._production_probabilities(production, frame)
    assert np.max(np.abs(production_probability - direct_probability)) < 1e-6
    assert production_effect[0] == 0.0
    assert production_probability.sum() == pytest.approx(1.0, abs=1e-15)


def _parent_files(root: Path) -> tuple[Path, Path, Path, Path]:
    pooled = root / "T00"
    noise = root / "T02A"
    reaction = root / "T07S"
    for path in (pooled, noise, reaction):
        path.mkdir()
        (path / "artifacts.json").write_text("{}\n")
    (pooled / "pooled-data.json").write_text("{}\n")
    (noise / "raw-count-mass-noise-amendment.json").write_text("{}\n")
    (reaction / "reaction-recovery-amendment.json").write_text("{}\n")
    folds = root / "folds.parquet"
    _catalog()[["guide_id", "held_out_fold"]].to_parquet(folds, index=False)
    return pooled, noise, reaction, folds


def _patched_parents(monkeypatch: pytest.MonkeyPatch, catalog: pd.DataFrame) -> None:
    pooled = SimpleNamespace(pooled_data_id="p" * 64)
    noise = SimpleNamespace(amendment_id="n" * 64)
    reaction = SimpleNamespace(amendment_id="r" * 64)
    monkeypatch.setattr(
        pooled_likelihood,
        "_verify_parents",
        lambda *args: (pooled, noise, reaction),
    )
    monkeypatch.setattr(
        pooled_likelihood,
        "_load_catalog",
        lambda *args: (pooled, catalog.copy()),
    )


def test_t07r_bundle_is_typed_recomputed_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _catalog()
    _patched_parents(monkeypatch, catalog)
    pooled, noise, reaction, folds = _parent_files(tmp_path)
    output = tmp_path / "T07R"
    settings = {
        "pooled_bundle": pooled,
        "t02a_amendment": noise,
        "t07s_amendment": reaction,
        "fold_assignment": folds,
    }
    pooled_likelihood.qualify_pooled_reaction_likelihood(output, **settings)
    bundle = pooled_likelihood.verify_pooled_reaction_likelihood(output, **settings)
    receipt = PooledReactionLikelihoodReceipt.model_validate_json(
        (output / "TEST_RECEIPT.json").read_text()
    )
    assert isinstance(bundle, PooledReactionLikelihoodBundle)
    assert receipt.estimator_parity_pass
    assert receipt.selected_update in (0, 25, 50, 100, 200)
    assert receipt.paired_bootstrap_draws == 4000
    verify_directory(output)

    metrics = output / "OUTER_GUIDE_METRICS.parquet"
    original = metrics.read_bytes()
    metrics.write_bytes(original + b"tamper")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        pooled_likelihood.verify_pooled_reaction_likelihood(output, **settings)
    metrics.write_bytes(original)
    pooled_likelihood.verify_pooled_reaction_likelihood(output, **settings)


def test_t07r_contract_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _catalog()
    _patched_parents(monkeypatch, catalog)
    pooled, noise, reaction, folds = _parent_files(tmp_path)
    output = pooled_likelihood.qualify_pooled_reaction_likelihood(
        tmp_path / "T07R",
        pooled_bundle=pooled,
        t02a_amendment=noise,
        t07s_amendment=reaction,
        fold_assignment=folds,
    )
    bundle = json.loads((output / "pooled-reaction-likelihood.json").read_text())
    receipt = json.loads((output / "TEST_RECEIPT.json").read_text())
    assert bundle["method"] == "pooled_target_reaction_dm_likelihood_v1"
    assert "estimator_parity_pass" in receipt
    assert validate_contract(output / "pooled-reaction-likelihood.json")["contract_type"] == (
        "PooledReactionLikelihoodBundle"
    )
    assert validate_contract(output / "TEST_RECEIPT.json")["contract_type"] == (
        "PooledReactionLikelihoodReceipt"
    )
