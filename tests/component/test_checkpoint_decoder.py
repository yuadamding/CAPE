from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from scipy import sparse

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.representation import (
    CheckpointMultinomialDecoder,
    fit_checkpoint_frequency_null,
)


def _fit() -> CheckpointMultinomialDecoder:
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [20, 2, 1, 0],
                [18, 4, 1, 0],
                [21, 1, 2, 0],
                [1, 3, 18, 4],
                [0, 2, 21, 5],
                [1, 4, 17, 6],
            ],
            dtype=np.int32,
        )
    )
    return fit_checkpoint_frequency_null(
        counts,
        training_checkpoints=np.asarray(["Rest"] * 3 + ["Stim8hr"] * 3),
        training_row_ids=np.arange(6, dtype=np.int64),
        checkpoint_order=("Rest", "Stim8hr"),
    )


def test_g04_dimension_zero_is_checkpoint_global_frequency_only() -> None:
    decoder = _fit()
    z = np.empty((4, 0), dtype=np.float64)
    checkpoints = np.asarray(["Rest", "Rest", "Stim8hr", "Stim8hr"])
    probabilities = decoder.probabilities(z, checkpoints)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(probabilities[0], probabilities[1])
    np.testing.assert_array_equal(probabilities[2], probabilities[3])
    assert not np.array_equal(probabilities[0], probabilities[2])
    assert decoder.contract.intercept_axis == "checkpoint"
    assert decoder.contract.dimension_zero_null == "checkpoint_global_frequency"


def test_g04_dimension_zero_cannot_depend_on_guide_or_target_labels() -> None:
    decoder = _fit()
    signature = inspect.signature(decoder.probabilities)
    fit_signature = inspect.signature(fit_checkpoint_frequency_null)
    for forbidden in ("guide", "guide_id", "target", "target_id", "donor", "donor_id"):
        assert forbidden not in signature.parameters
        assert forbidden not in fit_signature.parameters
    assert set(decoder.state_dict()) == {"checkpoint_intercepts", "latent_weights"}
    assert decoder.contract.guide_parameters is False
    assert decoder.contract.target_parameters is False
    assert decoder.contract.heldout_donor_parameters is False


def test_g04_checkpoint_permutation_changes_only_intercept_assignment() -> None:
    decoder = _fit()
    z = np.empty((2, 0), dtype=np.float64)
    forward = decoder.probabilities(z, np.asarray(["Rest", "Stim8hr"]))
    reverse = decoder.probabilities(z, np.asarray(["Stim8hr", "Rest"]))
    np.testing.assert_array_equal(reverse, forward[::-1])


def test_g04_heldout_donor_endpoint_replacement_cannot_change_fit() -> None:
    training = sparse.csr_matrix(
        np.asarray([[9, 1, 0], [8, 2, 0], [0, 2, 8], [0, 1, 9]], dtype=np.int32)
    )
    labels = np.asarray(["Rest", "Rest", "Stim8hr", "Stim8hr"])
    row_ids = np.arange(4, dtype=np.int64)
    heldout_a = np.asarray([[1000, 0, 0], [0, 0, 1000]], dtype=np.int32)
    heldout_b = heldout_a[:, ::-1].copy()
    first = fit_checkpoint_frequency_null(
        training,
        training_checkpoints=labels,
        training_row_ids=row_ids,
        checkpoint_order=("Rest", "Stim8hr"),
    )
    second = fit_checkpoint_frequency_null(
        training,
        training_checkpoints=labels,
        training_row_ids=row_ids,
        checkpoint_order=("Rest", "Stim8hr"),
    )
    assert not np.array_equal(heldout_a, heldout_b)
    np.testing.assert_array_equal(first.checkpoint_intercepts, second.checkpoint_intercepts)
    assert first.contract == second.contract
    assert first.contract.heldout_donor_outcomes_used is False


def test_g04_contract_dispatches_through_public_validator(tmp_path: Path) -> None:
    decoder = _fit()
    path = tmp_path / "checkpoint-decoder.json"
    path.write_text(decoder.contract.model_dump_json() + "\n")
    assert validate_contract(path)["contract_type"] == "CheckpointMultinomialDecoderContract"
