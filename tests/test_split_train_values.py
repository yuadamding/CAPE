from __future__ import annotations

import pytest

import credo
from credo.contracts import SplitSpec
from credo.data.splits import validate_split_plan
from credo.registry import get_recipe


def _subject_ids(study: credo.PerturbSeqStudy) -> tuple[str, ...]:
    return tuple(dict.fromkeys(study.series.to_pandas()["subject_id"].astype(str)))


def test_explicit_train_values_reject_unknown_identity(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        subjects = _subject_ids(study)
        recipe = get_recipe(tiny_config.recipe)

        with pytest.raises(ValueError, match="Unknown training subject values"):
            recipe.plan_split(
                study.view(),
                tiny_config.recipe_config,
                SplitSpec(
                    strategy="subject",
                    train_values=("DOES_NOT_EXIST",),
                    validation_values=(subjects[-1],),
                ),
            )
    finally:
        study.close()


def test_explicit_train_values_exclude_undeclared_identities(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        series = study.series.to_pandas()
        effects = study.effect_bindings.to_pandas()[["perturbation_id", "effect_id"]]
        metadata = series.merge(
            effects,
            on="perturbation_id",
            how="left",
            validate="many_to_one",
        )
        candidates = next(
            rows["series_id"].astype(str).tolist()
            for _, rows in metadata.groupby("effect_id", observed=True)
            if len(rows) >= 2
        )
        declared_train = candidates[0]
        held_out = candidates[1]
        recipe = get_recipe(tiny_config.recipe)

        plan = recipe.plan_split(
            study.view(),
            tiny_config.recipe_config,
            SplitSpec(
                strategy="measure",
                train_values=(declared_train,),
                validation_values=(held_out,),
            ),
        )

        assert plan.train_series_ids == (declared_train,)
        assert plan.validation_series_ids == (held_out,)
        assert set(plan.train_series_ids) | set(plan.validation_series_ids) < set(
            study.series.series_ids
        )
        validate_split_plan(study.view(), plan)
    finally:
        study.close()


def test_explicit_train_values_match_resolved_training_partition(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        subjects = _subject_ids(study)
        held_out = subjects[-1]
        declared_train = subjects[:-1]
        recipe = get_recipe(tiny_config.recipe)

        plan = recipe.plan_split(
            study.view(),
            tiny_config.recipe_config,
            SplitSpec(
                strategy="subject",
                train_values=declared_train,
                validation_values=(held_out,),
            ),
        )

        metadata = study.series.to_pandas().set_index("series_id")
        actual_train = set(metadata.loc[list(plan.train_series_ids), "subject_id"].astype(str))
        actual_validation = set(
            metadata.loc[list(plan.validation_series_ids), "subject_id"].astype(str)
        )
        assert actual_train == set(declared_train)
        assert actual_validation == {held_out}
    finally:
        study.close()


def test_explicit_checkpoint_train_values_match_resolved_partition(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        downstream = study.design.ordered_checkpoint_ids[1:]
        assert len(downstream) >= 2
        held_out = downstream[-1]
        declared_train = downstream[:-1]
        recipe = get_recipe(tiny_config.recipe)

        plan = recipe.plan_split(
            study.view(),
            tiny_config.recipe_config,
            SplitSpec(
                strategy="checkpoint",
                train_values=declared_train,
                validation_values=(held_out,),
            ),
        )

        assert plan.train_checkpoint_ids == declared_train
        assert plan.validation_checkpoint_ids == (held_out,)
    finally:
        study.close()
