"""Cross-contract validation for the G00 source plane and loader gate."""

from __future__ import annotations

from ..contracts import (
    FoldNativeCompactViewContractV2,
    G00SourceAuthorityV1,
    G00SourceAuthorityV2,
    IntegratedLoaderQualificationContractV1,
    IntegratedLoaderQualificationContractV2,
    IntegratedLoaderQualificationReceiptV1,
    IntegratedLoaderQualificationReceiptV2,
    VirtualCanonicalCountStoreManifestV1,
    VirtualCanonicalCountStoreManifestV2,
)
from ..errors import IntegrityError


def validate_g00_source_plane(
    authority: G00SourceAuthorityV2 | G00SourceAuthorityV1,
    manifest: VirtualCanonicalCountStoreManifestV2 | VirtualCanonicalCountStoreManifestV1,
) -> None:
    """Require G00B to be an exact metadata projection of its G00A parent."""

    if isinstance(authority, G00SourceAuthorityV1):
        if not isinstance(manifest, VirtualCanonicalCountStoreManifestV1):
            raise IntegrityError("G00A and G00B must use the same schema version.")
        if manifest.source_authority_id != authority.authority_id:
            raise IntegrityError("G00B binds a different G00A source authority.")
        v1_fields = (
            "canonical_feature_index_hash",
            "guide_catalog_hash",
            "target_catalog_hash",
            "eligibility_rule",
            "eligible_rows",
            "sources",
        )
        for field in v1_fields:
            if getattr(manifest, field) != getattr(authority, field):
                raise IntegrityError(f"G00B {field} differs from its G00A authority.")
        return
    if not isinstance(manifest, VirtualCanonicalCountStoreManifestV2):
        raise IntegrityError("G00A and G00B must use the same schema version.")
    if manifest.source_authority_id != authority.authority_id:
        raise IntegrityError("G00B binds a different G00A source authority.")
    v2_fields = (
        "canonical_feature_index_hash",
        "guide_catalog_hash",
        "target_catalog_hash",
        "guide_target_crosswalk_hash",
        "guide_target_crosswalk",
        "source_numeric_audit",
        "guide_count",
        "target_control_count",
        "eligibility_rule",
        "eligibility_uses_heldout_stimulated_outcomes",
        "eligible_row_ids_hash",
        "eligible_rows",
        "eligible_nnz",
        "sources",
        "access_semantics",
    )
    for field in v2_fields:
        if getattr(manifest, field) != getattr(authority, field):
            raise IntegrityError(f"G00B {field} differs from its G00A authority.")


def validate_g00_fold_view_parent(
    fold: FoldNativeCompactViewContractV2,
    authority: G00SourceAuthorityV2,
    manifest: VirtualCanonicalCountStoreManifestV2,
) -> None:
    """Require a G00C contract to inherit the exact qualified G00A/G00B universe."""

    validate_g00_source_plane(authority, manifest)
    if fold.parent_source_authority_id != authority.authority_id:
        raise IntegrityError("G00C binds a different G00A authority.")
    if fold.parent_virtual_store_id != manifest.virtual_store_id:
        raise IntegrityError("G00C binds a different G00B virtual store.")
    if fold.parent_eligible_row_ids_hash != manifest.eligible_row_ids_hash:
        raise IntegrityError("G00C eligible row universe differs from G00B.")
    if fold.parent_guide_target_crosswalk_hash != manifest.guide_target_crosswalk_hash:
        raise IntegrityError("G00C guide-target crosswalk differs from G00B.")


def _validate_v1_loader_qualification(
    contract: IntegratedLoaderQualificationContractV1,
    receipt: IntegratedLoaderQualificationReceiptV1,
) -> None:
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


def validate_integrated_loader_qualification(
    contract: IntegratedLoaderQualificationContractV2 | IntegratedLoaderQualificationContractV1,
    receipt: IntegratedLoaderQualificationReceiptV2 | IntegratedLoaderQualificationReceiptV1,
) -> None:
    """Require environment, performance, parity, correctness, and bounded memory."""

    if isinstance(contract, IntegratedLoaderQualificationContractV1):
        if not isinstance(receipt, IntegratedLoaderQualificationReceiptV1):
            raise IntegrityError("G00D v1 contract requires a v1 receipt.")
        _validate_v1_loader_qualification(contract, receipt)
        return
    if not isinstance(receipt, IntegratedLoaderQualificationReceiptV2):
        raise IntegrityError("G00D v2 contract requires a v2 receipt.")
    if receipt.qualification_contract_id != contract.qualification_contract_id:
        raise IntegrityError("G00D receipt binds a different qualification contract.")
    exact_fields = (
        ("gpu_name", "expected_gpu_name"),
        ("cuda_version", "expected_cuda_version"),
        ("torch_version", "expected_torch_version"),
        ("container_digest", "expected_container_digest"),
        ("worker_count", "worker_count"),
        ("cpu_count", "cpu_count"),
        ("storage_authority_hash", "storage_authority_hash"),
        ("microbatch_cells", "microbatch_cells"),
        ("macrobatch_cells", "macrobatch_cells"),
        ("prefetch_depth", "prefetch_depth"),
        ("warmup_updates", "warmup_updates"),
        ("measured_updates", "measured_updates"),
        ("cache_policy", "cache_policy"),
        ("telemetry_interval_seconds", "telemetry_interval_seconds"),
    )
    for observed, expected in exact_fields:
        if getattr(receipt, observed) != getattr(contract, expected):
            raise IntegrityError(f"G00D receipt {observed} differs from its contract.")
    parity_fields = (
        receipt.row_ids_parity_pass,
        receipt.raw_counts_parity_pass,
        receipt.sample_weights_parity_pass,
        receipt.thinning_rng_parity_pass,
        receipt.loss_parity_pass,
        receipt.gradient_parity_pass,
        receipt.parameter_parity_pass,
        receipt.interrupted_resume_parity_pass,
    )
    numerical_parity = (
        receipt.maximum_parity_absolute_error <= contract.parity_absolute_tolerance
        and receipt.maximum_parity_relative_error <= contract.parity_relative_tolerance
    )
    excursion_limit = contract.maximum_loader_rss_bytes * contract.maximum_rss_excursion_fraction
    memory_growth = (
        receipt.peak_loader_rss_bytes <= contract.maximum_loader_rss_bytes
        and receipt.rss_slope_upper_ci_bytes_per_second
        <= contract.maximum_rss_slope_upper_bytes_per_second
        and receipt.maximum_rss_excursion_bytes <= excursion_limit
    )
    if receipt.memory_growth_pass != memory_growth:
        raise IntegrityError("G00D memory-growth flag differs from its numerical evidence.")
    passed = (
        receipt.gpu_name == contract.expected_gpu_name
        and receipt.data_wait_fraction <= contract.maximum_data_wait_fraction
        and receipt.steady_state_gpu_utilization >= contract.minimum_steady_state_gpu_utilization
        and receipt.p95_batch_ready_seconds <= contract.maximum_p95_batch_ready_seconds
        and receipt.peak_open_shards <= contract.maximum_open_shards
        and receipt.cold_start_measured
        and receipt.steady_state_measured
        and all(parity_fields)
        and numerical_parity
        and memory_growth
        and receipt.lru_bound_pass
        and receipt.loader_error_count == 0
        and receipt.cuda_error_count == 0
        and receipt.monitor_error_count == 0
    )
    expected_status = "pass" if passed else "fail"
    if receipt.status != expected_status:
        raise IntegrityError(
            f"G00D receipt status must be {expected_status} for its observed evidence."
        )
