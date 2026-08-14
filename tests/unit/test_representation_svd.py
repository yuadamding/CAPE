from __future__ import annotations

import numpy as np
from scipy import sparse

from credo_count_sde_v4.prepare.pipeline import _randomized_right_singular_vectors


def test_randomized_sparse_basis_is_deterministic_and_orthonormal() -> None:
    generator = np.random.default_rng(7)
    matrix = sparse.random(
        300,
        120,
        density=0.08,
        format="csr",
        random_state=generator,
        data_rvs=lambda size: generator.random(size, dtype=np.float32),
        dtype=np.float32,
    )
    left = _randomized_right_singular_vectors(matrix, 12, seed=19)
    right = _randomized_right_singular_vectors(matrix, 12, seed=19)
    assert left.shape == (12, 120)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(left @ left.T, np.eye(12), atol=2e-5, rtol=2e-5)
