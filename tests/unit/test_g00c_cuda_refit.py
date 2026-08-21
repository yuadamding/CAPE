from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store.g00c_refit_cuda import (
    accumulate_checkpoint_counts_cpu_compact,
    compact_sampled_rows,
)


def test_compact_sampled_rows_is_stable_and_rejects_inconsistent_weights() -> None:
    compact = compact_sampled_rows(
        np.asarray([30, 10, 30, 20, 10, 30], dtype=np.int64),
        np.asarray([2, 6, 2, 4, 6, 2], dtype=np.float64),
    )
    assert compact.row_ids.tolist() == [10, 20, 30]
    assert compact.multiplicities.tolist() == [2, 1, 3]
    assert compact.inverse_probability_weights.tolist() == [6, 4, 2]

    with pytest.raises(IntegrityError, match="inconsistent hierarchical weights"):
        compact_sampled_rows(
            np.asarray([10, 10], dtype=np.int64),
            np.asarray([6, 7], dtype=np.float64),
        )
    with pytest.raises(IntegrityError, match="exact integral"):
        compact_sampled_rows(
            np.asarray([10], dtype=np.int64),
            np.asarray([1.5], dtype=np.float64),
        )


def test_compact_cpu_accumulation_is_integer_exact_and_restart_stable() -> None:
    compact = compact_sampled_rows(
        np.asarray([30, 10, 30, 20, 10, 30], dtype=np.int64),
        np.asarray([2, 6, 2, 4, 6, 2], dtype=np.float64),
    )
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [2, 0, 1, 3],
                [0, 4, 0, 1],
                [3, 1, 2, 0],
            ],
            dtype=np.int64,
        )
    )
    checkpoints = np.asarray([0, 1, 2], dtype=np.int64)
    first, first_hash = accumulate_checkpoint_counts_cpu_compact(
        matrix, checkpoints, compact, seed=1234, row_block_size=2
    )
    resumed, resumed_hash = accumulate_checkpoint_counts_cpu_compact(
        matrix, checkpoints, compact, seed=1234, row_block_size=2
    )
    assert np.array_equal(first, resumed)
    assert first_hash == resumed_hash
    assert first.tolist() == [[12, 0, 0, 24], [0, 4, 0, 4], [8, 4, 6, 0]]
    assert first_hash == "9617c1edb97bd1ea5ba1cab8f36779f6d99af46675792afc4c83b44ed6db0de9"


def test_compact_cpu_accumulation_rejects_noncounts() -> None:
    compact = compact_sampled_rows(
        np.asarray([10], dtype=np.int64), np.asarray([1], dtype=np.float64)
    )
    with pytest.raises(IntegrityError, match="count contract"):
        accumulate_checkpoint_counts_cpu_compact(
            sparse.csr_matrix(np.asarray([[1.5]], dtype=np.float64)),
            np.asarray([0]),
            compact,
            seed=1,
        )
