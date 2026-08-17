"""Cross-contract validation for the integrated G00D loader gate."""

from __future__ import annotations

from ..contracts import (
    IntegratedLoaderQualificationContract,
    IntegratedLoaderQualificationReceipt,
)
from ..errors import IntegrityError


def validate_integrated_loader_qualification(
    contract: IntegratedLoaderQualificationContract,
    receipt: IntegratedLoaderQualificationReceipt,
) -> None:
    """Require wait, utilization, parity, correctness, and bounded memory together."""

    if receipt.qualification_contract_id != contract.qualification_contract_id:
        raise IntegrityError("G00D receipt binds a different qualification contract.")
    metric_parity = (
        receipt.training_metric_max_absolute_error <= contract.training_metric_absolute_tolerance
        and receipt.training_metric_max_relative_error
        <= contract.training_metric_relative_tolerance
    )
    if receipt.training_metric_parity_pass != metric_parity:
        raise IntegrityError("G00D metric-parity flag differs from its numerical errors.")
    passed = (
        receipt.data_wait_fraction <= contract.maximum_data_wait_fraction
        and receipt.steady_state_gpu_utilization >= contract.minimum_steady_state_gpu_utilization
        and receipt.p95_batch_ready_seconds
        <= receipt.covered_compute_seconds * contract.prefetch_depth
        and receipt.peak_loader_rss_bytes <= contract.maximum_loader_rss_bytes
        and receipt.peak_open_shards <= contract.maximum_open_shards
        and receipt.row_count_order_errors == 0
        and not receipt.unbounded_memory_growth_detected
        and metric_parity
    )
    expected = "pass" if passed else "fail"
    if receipt.status != expected:
        raise IntegrityError(f"G00D receipt status must be {expected} for its observed metrics.")
