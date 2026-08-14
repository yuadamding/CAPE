from __future__ import annotations

import numpy as np

from credo_count_sde_v4.contracts import ArtifactRef, ContextAuditContract
from credo_count_sde_v4.evaluation import evaluate_context_audit


def _contract() -> ContextAuditContract:
    return ContextAuditContract(
        held_out_pool_information_set="0" * 64,
        observation_operator=ArtifactRef(
            schema_id="test.operator",
            schema_version=1,
            sha256="1" * 64,
            size_bytes=1,
            media_type="application/json",
            relative_uri="operator.json",
        ),
    )


def test_context_audit_passes_only_all_fixed_gates() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    receipt = evaluate_context_audit(
        _contract(),
        matrix,
        selected_rank=2,
        context_parameters=100,
        comparator_parameters=100,
        held_out_access_pass=True,
        observation_operator_pass=True,
    )
    assert receipt.status == "eligible"
    failed = evaluate_context_audit(
        _contract(),
        matrix[:, :1],
        selected_rank=1,
        context_parameters=100,
        comparator_parameters=98,
        held_out_access_pass=False,
        observation_operator_pass=True,
    )
    assert failed.status == "diagnostic_only"
