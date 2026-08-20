"""Dev36 G00C execution seal with no callback or opaque evidence escape hatch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..contracts import (
    ArtifactRef,
    G00CD1ExecutionAuthorityFreezeV2,
    G00CDecisionReceiptV4,
    G00CExecutionBundleV4,
    G00CFeatureSelectionResultV3,
    G00CMaterializationReceiptV3,
    G00CPublicationManifestV3,
    G00CRefitReplayReceiptV3,
    G00CRefitSeedScheduleV1,
    G00CSamplerEvidenceV3,
    G00CSampleSizeSelectionResultV3,
    G00CSupportAuditReceiptV3,
)
from ..errors import IntegrityError
from .g00c_materialization_v3 import verify_g00c_materialization_v3
from .g00c_publication_v3 import verify_g00c_publication_v3
from .g00c_refit_replay_v3 import verify_g00c_refit_replay_v3
from .g00c_sampler_v3 import verify_g00c_sampler_v3, verify_g00c_support_v3
from .g00c_v3 import (
    _path,
    _read_model,
    _verify_feature_selection_v3,
    _verify_sample_size_selection_v3,
    verify_g00c_d1_freeze_v1,
)
from .virtual import VirtualCanonicalCountStore


@dataclass(frozen=True)
class VerifiedG00CExecutionV4:
    """Only values recomputed through the complete Dev36 execution seal."""

    terminal_status: str
    execution_bundle_id: str
    execution_authority_id: str
    selection_freeze_id: str
    feature_selection_result_id: str
    sample_size_selection_result_id: str
    sampler_evidence_id: str
    feature_refit_replay_receipt_id: str
    sample_refit_replay_receipt_id: str
    support_audit_receipt_ids: tuple[str, ...]
    materialization_receipt_id: str | None
    publication_manifest_id: str
    verified_artifact_sha256s: tuple[str, ...]


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV2, role: str) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev36 authority has no unique {role} implementation.")
    return matches[0]


def _artifact_refs(value: Any) -> list[ArtifactRef]:
    if isinstance(value, ArtifactRef):
        return [value]
    if isinstance(value, BaseModel):
        refs: list[ArtifactRef] = []
        for field in type(value).model_fields:
            refs.extend(_artifact_refs(getattr(value, field)))
        return refs
    if isinstance(value, (tuple, list)):
        refs = []
        for item in value:
            refs.extend(_artifact_refs(item))
        return refs
    if isinstance(value, dict):
        refs = []
        for item in value.values():
            refs.extend(_artifact_refs(item))
        return refs
    return []


def _derived_eligibility(
    tables: tuple[pd.DataFrame, ...], *, kind: str, candidates: tuple[int, ...]
) -> tuple[bool, ...]:
    audit = pd.concat(tables, ignore_index=True)
    required_dimensions = {
        "donor_checkpoint",
        "target",
        "guide",
        "control_vs_targeting",
        "sampler_stratum",
    }
    selected = audit.loc[audit["candidate_kind"].astype(str) == kind]
    if set(selected["candidate_value"].astype(int)) != set(candidates):
        raise IntegrityError("Dev36 derived support omits a frozen candidate.")
    answer: list[bool] = []
    for candidate in candidates:
        rows = selected.loc[selected["candidate_value"].astype(int) == candidate]
        if (
            set(rows["dimension"].astype(str)) != required_dimensions
            or rows.duplicated(["dimension", "stratum_id"]).any()
            or (rows["cells"].astype(np.int64) < 0).any()
            or not np.isfinite(rows["weighted_effective_sample_size"].astype(float)).all()
            or not np.isfinite(rows["maximum_to_median_weight_ratio"].astype(float)).all()
        ):
            raise IntegrityError("Dev36 derived support is incomplete or nonfinite.")
        answer.append(
            bool(
                rows["selection_eligible"].astype(bool).all()
                and not rows["zero_support"].astype(bool).any()
            )
        )
    return tuple(answer)


def _verify_refit_receipt_binding(
    receipt: G00CRefitReplayReceiptV3,
    *,
    result: G00CFeatureSelectionResultV3 | G00CSampleSizeSelectionResultV3,
    kind: str,
    selected: int,
    reference: int,
    modeled_features: int,
) -> None:
    if (
        receipt.candidate_kind != kind
        or receipt.refit_records != result.refit_records
        or receipt.selected_candidate != selected
        or receipt.reference_candidate != reference
        or receipt.modeled_feature_count != modeled_features
    ):
        raise IntegrityError("Dev36 refit replay binds another selection surface.")


def verify_g00c_execution_v4(
    root: Path,
    publication_root: Path,
    bundle: G00CExecutionBundleV4,
    *,
    source_store: VirtualCanonicalCountStore | None = None,
) -> VerifiedG00CExecutionV4:
    """Recompute every Dev36 gate from bytes and invoke the direct source verifier."""

    authority = _read_model(root, bundle.execution_authority, G00CD1ExecutionAuthorityFreezeV2)
    freeze = verify_g00c_d1_freeze_v1(root, authority)
    if (
        bundle.execution_authority_id != authority.authority_id
        or bundle.selection_freeze != authority.selection_freeze
        or bundle.selection_freeze_id != freeze.freeze_id
        or bundle.seed_schedule != authority.seed_schedule
        or bundle.seed_schedule_id != authority.seed_schedule_id
        or bundle.row_roles != authority.row_role_freeze
        or bundle.base_support_audit_contract != authority.base_support_audit_contract
    ):
        raise IntegrityError("Dev36 execution bundle is cross-wired to another authority.")
    schedule = _read_model(root, bundle.seed_schedule, G00CRefitSeedScheduleV1)
    if schedule.schedule_id != bundle.seed_schedule_id:
        raise IntegrityError("Dev36 execution uses another seed schedule.")
    feature = _read_model(root, bundle.feature_selection_result, G00CFeatureSelectionResultV3)
    sample = _read_model(root, bundle.sample_size_selection_result, G00CSampleSizeSelectionResultV3)
    if (
        feature.result_id != bundle.feature_selection_result_id
        or sample.result_id != bundle.sample_size_selection_result_id
    ):
        raise IntegrityError("Dev36 result identities differ from their artifacts.")
    sampler = _read_model(root, bundle.sampler_evidence, G00CSamplerEvidenceV3)
    sample_candidates = (
        freeze.base_cell_grid
        if sample.grid_stage == "base"
        else (*freeze.base_cell_grid, *freeze.extension_additional_cell_grid)
    )
    plan, hierarchy, order, trace = verify_g00c_sampler_v3(
        root,
        authority,
        sampler,
        expected_candidate_values={
            "feature_count": freeze.feature_ranking.candidate_feature_counts,
            "training_cells": sample_candidates,
        },
    )
    base_support = _read_model(root, bundle.base_support_audit_receipt, G00CSupportAuditReceiptV3)
    if (
        base_support.support_contract != bundle.base_support_audit_contract
        or base_support.sampler_evidence != bundle.sampler_evidence
    ):
        raise IntegrityError("Dev36 base support receipt is cross-wired.")
    support_tables = [
        verify_g00c_support_v3(
            root,
            authority,
            sampler,
            base_support,
            plan=plan,
            hierarchy=hierarchy,
            order=order,
            trace=trace,
        )
    ]
    support_receipts = [base_support]
    if sample.grid_stage == "extension":
        if (
            bundle.extension_support_audit_contract is None
            or bundle.extension_support_audit_receipt is None
        ):
            raise IntegrityError("Dev36 extension execution lacks derived support evidence.")
        extension_support = _read_model(
            root, bundle.extension_support_audit_receipt, G00CSupportAuditReceiptV3
        )
        if (
            extension_support.support_contract != bundle.extension_support_audit_contract
            or extension_support.sampler_evidence != bundle.sampler_evidence
        ):
            raise IntegrityError("Dev36 extension support receipt is cross-wired.")
        support_tables.append(
            verify_g00c_support_v3(
                root,
                authority,
                sampler,
                extension_support,
                plan=plan,
                hierarchy=hierarchy,
                order=order,
                trace=trace,
            )
        )
        support_receipts.append(extension_support)
    feature_support = _derived_eligibility(
        tuple(support_tables),
        kind="feature_count",
        candidates=freeze.feature_ranking.candidate_feature_counts,
    )
    sample_support = _derived_eligibility(
        tuple(support_tables), kind="training_cells", candidates=sample_candidates
    )
    selected_features, _ = _verify_feature_selection_v3(
        root,
        freeze=freeze,
        bundle=bundle,
        result=feature,
        schedule=schedule,
        support_eligible_override=feature_support,
    )
    selected_rows = _verify_sample_size_selection_v3(
        root,
        freeze=freeze,
        authority=authority,
        bundle=bundle,
        feature_result=feature,
        result=sample,
        schedule=schedule,
        support_eligible_override=sample_support,
    )
    feature_replay = _read_model(
        root, bundle.feature_refit_replay_receipt, G00CRefitReplayReceiptV3
    )
    sample_replay = _read_model(root, bundle.sample_refit_replay_receipt, G00CRefitReplayReceiptV3)
    _verify_refit_receipt_binding(
        feature_replay,
        result=feature,
        kind="feature_count",
        selected=feature.selected_feature_count,
        reference=4096,
        modeled_features=4096,
    )
    selected_sample = sample.selected_training_cells or sample_candidates[-1]
    _verify_refit_receipt_binding(
        sample_replay,
        result=sample,
        kind="training_cells",
        selected=selected_sample,
        reference=sample_candidates[-1],
        modeled_features=feature.selected_feature_count,
    )
    verify_g00c_refit_replay_v3(root, authority, feature_replay)
    verify_g00c_refit_replay_v3(root, authority, sample_replay)
    expected_status = "pass" if sample.selection_status == "selected" else sample.selection_status
    if bundle.terminal_status != "failed_integrity" and bundle.terminal_status != expected_status:
        raise IntegrityError("Dev36 terminal status differs from recomputed selection.")
    materialization: G00CMaterializationReceiptV3 | None = None
    if bundle.terminal_status == "pass":
        if source_store is None or selected_rows is None or bundle.materialization_receipt is None:
            raise IntegrityError("Dev36 pass requires the direct source-backed verifier.")
        materialization = _read_model(
            root, bundle.materialization_receipt, G00CMaterializationReceiptV3
        )
        verify_g00c_materialization_v3(
            root,
            source_store,
            authority,
            materialization,
            expected_selected_feature_ids=selected_features,
            expected_selected_rows=selected_rows,
        )
    if bundle.terminal_status == "failed_integrity":
        if bundle.failure_receipt is None:
            raise IntegrityError("Dev36 integrity failure lacks a failure receipt.")
        failure = json.loads(_path(root, bundle.failure_receipt).read_text())
        if failure != {
            "biological_claims": False,
            "execution_authority_id": authority.authority_id,
            "schema_version": 1,
            "selection_freeze_id": freeze.freeze_id,
            "status": "failed_integrity",
        }:
            raise IntegrityError("Dev36 integrity-failure receipt is not exact.")
    publication = _read_model(root, bundle.publication_manifest, G00CPublicationManifestV3)
    verify_g00c_publication_v3(publication_root, authority, freeze.publication, bundle, publication)
    root_models: tuple[BaseModel, ...] = (
        bundle,
        authority,
        freeze,
        schedule,
        feature,
        sample,
        sampler,
        *support_receipts,
        feature_replay,
        sample_replay,
        *((materialization,) if materialization is not None else ()),
    )
    verified: set[str] = {freeze.dev33_canary.authority_archive_sha256}
    for model in root_models:
        for artifact in _artifact_refs(model):
            _path(root, artifact)
            verified.add(artifact.sha256)
    verified.add(bundle.publication_manifest.sha256)
    verified.update(entry.sha256 for entry in publication.artifacts)
    return VerifiedG00CExecutionV4(
        terminal_status=bundle.terminal_status,
        execution_bundle_id=bundle.bundle_id,
        execution_authority_id=authority.authority_id,
        selection_freeze_id=freeze.freeze_id,
        feature_selection_result_id=feature.result_id,
        sample_size_selection_result_id=sample.result_id,
        sampler_evidence_id=sampler.evidence_id,
        feature_refit_replay_receipt_id=feature_replay.receipt_id,
        sample_refit_replay_receipt_id=sample_replay.receipt_id,
        support_audit_receipt_ids=tuple(receipt.receipt_id for receipt in support_receipts),
        materialization_receipt_id=(
            materialization.receipt_id if materialization is not None else None
        ),
        publication_manifest_id=publication.manifest_id,
        verified_artifact_sha256s=tuple(sorted(verified)),
    )


def build_g00c_decision_receipt_v4(
    verified: VerifiedG00CExecutionV4,
) -> G00CDecisionReceiptV4:
    """Construct the sole G00D parent receipt from verified values only."""

    payload: dict[str, Any] = {
        "receipt_id": "0" * 64,
        "execution_bundle_id": verified.execution_bundle_id,
        "execution_authority_id": verified.execution_authority_id,
        "selection_freeze_id": verified.selection_freeze_id,
        "feature_selection_result_id": verified.feature_selection_result_id,
        "sample_size_selection_result_id": verified.sample_size_selection_result_id,
        "sampler_evidence_id": verified.sampler_evidence_id,
        "feature_refit_replay_receipt_id": verified.feature_refit_replay_receipt_id,
        "sample_refit_replay_receipt_id": verified.sample_refit_replay_receipt_id,
        "support_audit_receipt_ids": verified.support_audit_receipt_ids,
        "materialization_receipt_id": verified.materialization_receipt_id,
        "publication_manifest_id": verified.publication_manifest_id,
        "verified_artifact_sha256s": verified.verified_artifact_sha256s,
        "terminal_status": verified.terminal_status,
        "may_parent_g00d": verified.terminal_status == "pass",
    }
    provisional = G00CDecisionReceiptV4.model_construct(**payload)
    payload["receipt_id"] = provisional.identity(id_field="receipt_id")
    return G00CDecisionReceiptV4.model_validate(payload)


def verify_g00c_decision_v4(
    root: Path,
    publication_root: Path,
    bundle: G00CExecutionBundleV4,
    receipt: G00CDecisionReceiptV4,
    *,
    source_store: VirtualCanonicalCountStore | None = None,
) -> None:
    """Reject a receipt unless the complete direct Dev36 seal reproduces it."""

    verified = verify_g00c_execution_v4(root, publication_root, bundle, source_store=source_store)
    authority = _read_model(root, bundle.execution_authority, G00CD1ExecutionAuthorityFreezeV2)
    if (
        _implementation_hash(authority, "decision_verifier")
        not in verified.verified_artifact_sha256s
    ):
        raise IntegrityError("Dev36 decision verifier bytes were not independently verified.")
    if receipt != build_g00c_decision_receipt_v4(verified):
        raise IntegrityError("Dev36 decision receipt differs from recomputed sealed evidence.")
