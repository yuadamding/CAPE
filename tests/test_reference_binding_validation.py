from __future__ import annotations

import json
from dataclasses import replace

import pytest

import credo
from credo.contracts import SplitSpec
from credo.data import PerturbationReferenceBindingTable
from credo.registry import get_recipe


def _series_identities(study):
    series = study.series.to_pandas()
    perturbations = study.perturbations.to_pandas().set_index("perturbation_id")
    series["is_control"] = series["perturbation_id"].map(perturbations["is_control"])
    subjects = tuple(dict.fromkeys(series["subject_id"].astype(str)))
    control = str(
        series.loc[series["subject_id"].eq(subjects[0]) & series["is_control"], "series_id"].iloc[0]
    )
    held_control = str(
        series.loc[series["subject_id"].eq(subjects[-1]) & series["is_control"], "series_id"].iloc[
            0
        ]
    )
    held_intervention = str(
        series.loc[series["subject_id"].eq(subjects[-1]) & ~series["is_control"], "series_id"].iloc[
            0
        ]
    )
    return control, held_control, held_intervention


def _subject_checkpoint_binding(study) -> PerturbationReferenceBindingTable:
    references = study.reference_bindings.to_pandas()
    references["scope_kind"] = "subject"
    references["match_keys"] = json.dumps(
        ["subject_id", "checkpoint_id"],
        separators=(",", ":"),
    )
    return PerturbationReferenceBindingTable(references)


def _validate(study, tiny_config, selection=None):
    recipe = get_recipe(tiny_config.recipe)
    view = study.view(selection)
    return view.validate_for(
        recipe.requirements(tiny_config.recipe_config),
        SplitSpec(strategy="none"),
    )


def test_subject_checkpoint_reference_requires_a_matched_control(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        control, _, held_intervention = _series_identities(study)
        rebound = replace(
            study,
            reference_bindings=_subject_checkpoint_binding(study),
        )
        selection = credo.SelectionSpec(
            series_ids=(control, held_intervention),
            composition_policy="drop",
        )

        report = _validate(rebound, tiny_config, selection)

        issue = next(
            issue for issue in report.errors if issue.code == "recipe.reference_match_coverage"
        )
        assert "3 selected intervention observations" in issue.message
        assert "subject_id" in issue.message
        assert "checkpoint_id" in issue.message

        recipe = get_recipe(tiny_config.recipe)
        validation = tiny_config.recipe_config.validation.model_copy(
            update={"strategy": "train_self_eval", "values": (), "fraction": 0.0}
        )
        settings = tiny_config.recipe_config.model_copy(update={"validation": validation})
        view = rebound.view(selection)
        plan = recipe.plan_split(view, settings)
        with pytest.raises(ValueError, match="recipe.reference_match_coverage"):
            recipe.compile(view, plan, settings)
    finally:
        study.close()


def test_subject_checkpoint_reference_accepts_complete_matches(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        _, held_control, held_intervention = _series_identities(study)
        rebound = replace(
            study,
            reference_bindings=_subject_checkpoint_binding(study),
        )
        selection = credo.SelectionSpec(
            series_ids=(held_control, held_intervention),
            composition_policy="drop",
        )

        report = _validate(rebound, tiny_config, selection)

        assert not {
            "recipe.reference_match_coverage",
            "recipe.counterfactual_reference_effect",
        } & {issue.code for issue in report.errors}
    finally:
        study.close()


def test_compact_rejects_nonreference_counterfactual_effect(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        perturbations = study.perturbations.to_pandas().set_index("perturbation_id")
        effects = study.effect_bindings.to_pandas()
        nonreference_effect = str(
            effects.loc[
                ~effects["perturbation_id"].map(perturbations["is_control"]),
                "effect_id",
            ].iloc[0]
        )
        references = study.reference_bindings.to_pandas()
        references["counterfactual_effect_id"] = nonreference_effect
        rebound = replace(
            study,
            reference_bindings=PerturbationReferenceBindingTable(references),
        )

        report = _validate(rebound, tiny_config)

        assert any(
            issue.code == "recipe.counterfactual_reference_effect" for issue in report.errors
        )
        with pytest.raises(ValueError, match="counterfactual_reference_effect"):
            report.raise_for_errors()

        recipe = get_recipe(tiny_config.recipe)
        view = rebound.view()
        plan = recipe.plan_split(view, tiny_config.recipe_config)
        with pytest.raises(ValueError, match="recipe.counterfactual_reference_effect"):
            recipe.compile(view, plan, tiny_config.recipe_config)
    finally:
        study.close()
