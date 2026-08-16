"""Strict v4 contract models.

All contracts reject unknown fields and resolve defaults before identity is
computed. Paths are portable relative URIs; exact bytes remain authoritative.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical import contract_id, validate_relative_uri
from ..errors import ContractError

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    def identity(self, id_field: str | None = None) -> str:
        return contract_id(self, id_field=id_field)


class RunIntent(StrEnum):
    COUNT_STATE = "count_state"
    COUNT_MEASURE = "count_measure"
    COUNT_CONTEXT = "count_context"


class VerifyLevel(StrEnum):
    MANIFEST = "manifest"
    CONTENT = "content"
    RELOAD = "reload"
    RESUME = "resume"
    FULL = "full"


class LifecycleState(StrEnum):
    RESOLVED = "resolved"
    PREPARED = "prepared"
    COMPILED = "compiled"
    TRAINED = "trained"
    FINALIZED = "finalized"
    EVALUATED = "evaluated"
    SEALED = "sealed"


class EvidenceRole(StrEnum):
    DEVELOPMENT = "development"
    SEALED = "sealed"
    EXTERNAL_CONFIRMATION = "external_confirmation"


class ArtifactRef(StrictModel):
    schema_id: str
    schema_version: int = Field(ge=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: str
    relative_uri: str

    @field_validator("relative_uri")
    @classmethod
    def safe_uri(cls, value: str) -> str:
        return validate_relative_uri(value)


class FeatureKey(StrictModel):
    namespace: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    namespace_version: str = Field(min_length=1)


class InformationSet(StrictModel):
    information_set_id: str
    fit_rows: tuple[int, ...]
    validation_rows: tuple[int, ...] = ()
    query_rows: tuple[int, ...] = ()
    protected_rows: tuple[int, ...] = ()

    @model_validator(mode="after")
    def disjoint(self) -> InformationSet:
        fit, validation, protected = map(
            set, (self.fit_rows, self.validation_rows, self.protected_rows)
        )
        if fit & protected or validation & protected:
            raise ValueError("Fit/validation rows overlap protected rows.")
        if fit & validation:
            raise ValueError("Fit and validation rows must be disjoint.")
        return self


class SplitContract(StrictModel):
    schema_version: int = 1
    split_id: str
    training_units: tuple[str, ...]
    inner_validation_units: tuple[str, ...] = ()
    outer_evaluation_units: tuple[str, ...]
    grouping_unit: str

    @model_validator(mode="after")
    def unit_disjointness(self) -> SplitContract:
        groups = [
            set(self.training_units),
            set(self.inner_validation_units),
            set(self.outer_evaluation_units),
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Split unit sets must be pairwise disjoint.")
        return self


class RowIndex(StrictModel):
    schema_version: int = 1
    row_ids_hash: Sha256
    row_count: int = Field(gt=0)
    dtype: Literal["int64"] = "int64"
    unique: Literal[True] = True


class FeatureIndex(StrictModel):
    schema_version: int = 1
    features: tuple[FeatureKey, ...]
    ordered_hash: Sha256

    @model_validator(mode="after")
    def unique_features(self) -> FeatureIndex:
        keys = [(item.namespace, item.namespace_version, item.feature_id) for item in self.features]
        if len(keys) != len(set(keys)):
            raise ValueError("Composite feature keys must be unique.")
        return self


class PooledFiniteMeasureBundle(StrictModel):
    """Immutable T00 pooled guide-by-checkpoint finite-measure data contract."""

    schema_version: int = 1
    pooled_data_id: str
    sample_id: Literal["pooled"] = "pooled"
    source_checkpoint: str = Field(min_length=1)
    terminal_checkpoint: str = Field(min_length=1)
    feature_order_hash: Sha256
    input_cell_universe_hash: Sha256
    retained_cell_universe_hash: Sha256
    excluded_cell_universe_hash: Sha256
    source_eligibility_min_cells: int = Field(ge=1)
    eligibility_uses_terminal_counts: Literal[False] = False
    mass_pseudocount: float = Field(default=0.5, gt=0)
    retained_cells: int = Field(ge=1)
    retained_guides: int = Field(ge=1)
    targeting_guides: int = Field(ge=1)
    control_guides: int = Field(ge=1)
    perturbation_targets: int = Field(ge=1)
    cells: ArtifactRef
    guide_catalog: ArtifactRef
    eligibility: ArtifactRef
    finite_measures: ArtifactRef
    per_guide_metrics: ArtifactRef
    per_target_metrics: ArtifactRef

    @model_validator(mode="after")
    def validate_identity(self) -> PooledFiniteMeasureBundle:
        expected = self.identity(id_field="pooled_data_id")
        if self.pooled_data_id != expected:
            raise ValueError(f"pooled_data_id mismatch: expected {expected}.")
        if self.source_checkpoint == self.terminal_checkpoint:
            raise ValueError("Source and terminal checkpoints must differ.")
        if self.targeting_guides + self.control_guides != self.retained_guides:
            raise ValueError("Targeting/control guide counts do not cover the retained catalog.")
        return self


class CountRepresentationBundle(StrictModel):
    """Immutable T01 count-native representation qualification bundle."""

    schema_version: int = 1
    representation_id: str
    pooled_data_id: str
    method: Literal["multinomial_hellinger_pca_v1", "multinomial_centered_hellinger_pca_v2"]
    feature_index_hash: Sha256
    count_store_sha256: Sha256
    dimensions: tuple[int, ...]
    selected_dimensions: dict[str, int]
    outer_folds: tuple[str, ...]
    selection_uses_terminal_outcomes: Literal[False] = False
    dynamics_gradients_enabled: Literal[False] = False
    fit_checkpoint: str
    protected_checkpoint: str
    fold_index: ArtifactRef
    encoder_state: ArtifactRef
    candidate_metrics: ArtifactRef
    per_guide_metrics: ArtifactRef
    per_target_metrics: ArtifactRef
    support_metrics: ArtifactRef
    null_calibration: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_representation(self) -> CountRepresentationBundle:
        expected = self.identity(id_field="representation_id")
        if self.representation_id != expected:
            raise ValueError(f"representation_id mismatch: expected {expected}.")
        if not self.dimensions or tuple(sorted(set(self.dimensions))) != self.dimensions:
            raise ValueError("Representation dimensions must be unique and increasing.")
        if any(value <= 0 for value in self.dimensions):
            raise ValueError("Representation dimensions must be positive.")
        if set(self.selected_dimensions) != set(self.outer_folds):
            raise ValueError("Every outer fold must have exactly one selected dimension.")
        if any(value not in {0, *self.dimensions} for value in self.selected_dimensions.values()):
            raise ValueError("A selected dimension is outside the frozen candidate set.")
        if self.fit_checkpoint == self.protected_checkpoint:
            raise ValueError("Fit and protected checkpoints must differ.")
        return self


class ParticleEngineQualificationBundle(StrictModel):
    """Immutable T04 numerical particle-engine qualification bundle."""

    schema_version: int = 1
    qualification_id: str
    test_contract_id: str
    method: Literal["streaming_euler_maruyama_v1"]
    environment_hash: Sha256
    particle_grid: tuple[int, ...]
    step_grid: tuple[int, ...]
    seed_count: int = Field(ge=1)
    deterministic_drift: ArtifactRef
    ou_grid: ArtifactRef
    reaction_mass: ArtifactRef
    ecology: ArtifactRef
    lifecycle: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_qualification(self) -> ParticleEngineQualificationBundle:
        expected = self.identity(id_field="qualification_id")
        if self.qualification_id != expected:
            raise ValueError(f"qualification_id mismatch: expected {expected}.")
        if tuple(sorted(set(self.particle_grid))) != self.particle_grid:
            raise ValueError("Particle qualification grid must be increasing and unique.")
        if tuple(sorted(set(self.step_grid))) != self.step_grid:
            raise ValueError("Step qualification grid must be increasing and unique.")
        if any(value <= 0 for value in (*self.particle_grid, *self.step_grid)):
            raise ValueError("Particle and step qualification grids must be positive.")
        return self


class ParticleEngineTestReceipt(StrictModel):
    """Complete T04 fixed-truth decision surface."""

    schema_version: int = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    deterministic_drift_max_abs_error: float = Field(ge=0)
    drift_refinement_pass: bool
    ou_mean_within_two_standard_errors: bool
    ou_largest_grid_variance_relative_error: float = Field(ge=0)
    ou_convergence_pass: bool
    reaction_max_relative_error: float = Field(ge=0)
    ecology_absolute_weight_max_error: float = Field(ge=0)
    normalized_context_negative_control_detected: bool
    stabilized_log_weight_pass: bool
    deterministic_replay_pass: bool
    interrupted_resume_pass: bool
    no_guide_switching_pass: bool
    normalized_particle_weights_pass: bool
    declared_mass_pass: bool
    capacity_probe_exclusion_pass: bool
    protected_metrics_pass: bool
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_t04_receipt(self) -> ParticleEngineTestReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        gates = (
            self.drift_refinement_pass,
            self.ou_mean_within_two_standard_errors,
            self.ou_convergence_pass,
            self.normalized_context_negative_control_detected,
            self.stabilized_log_weight_pass,
            self.deterministic_replay_pass,
            self.interrupted_resume_pass,
            self.no_guide_switching_pass,
            self.normalized_particle_weights_pass,
            self.declared_mass_pass,
            self.capacity_probe_exclusion_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T04 receipt must satisfy every fixed numerical gate.")
        return self

    @property
    def stabilized_absolute_log_mass_pass(self) -> bool:
        """Correct interpretation of the frozen legacy wire-field name."""

        return self.stabilized_log_weight_pass


class ReactionRecoveryQualificationBundle(StrictModel):
    """Immutable T07S learned constant-reaction qualification bundle."""

    schema_version: int = 1
    qualification_id: str
    test_contract_id: str
    method: Literal[
        "complete_denominator_dm_reaction_recovery_v1",
        "complete_denominator_dm_reaction_recovery_v2",
    ]
    environment_hash: Sha256
    null_calibration_repeats: int = Field(ge=59)
    null_audit_repeats: int = Field(ge=59)
    candidate_updates: tuple[int, ...]
    target_count: int = Field(ge=4)
    pool_count: int = Field(ge=2)
    null_refits: ArtifactRef
    null_model_effects: ArtifactRef
    recovery_curve: ArtifactRef
    recovery_series: ArtifactRef
    target_metrics: ArtifactRef
    bootstrap_target_draws: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_reaction_qualification(self) -> ReactionRecoveryQualificationBundle:
        expected = self.identity(id_field="qualification_id")
        if self.qualification_id != expected:
            raise ValueError(f"qualification_id mismatch: expected {expected}.")
        if not self.candidate_updates or self.candidate_updates[0] != 0:
            raise ValueError("Reaction recovery must retain update 0 as a candidate.")
        if tuple(sorted(set(self.candidate_updates))) != self.candidate_updates:
            raise ValueError("Reaction candidate updates must be increasing and unique.")
        return self


class ReactionRecoveryTestReceiptV1(StrictModel):
    """Legacy dev23 T07S receipt retained for immutable-parent verification."""

    schema_version: int = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    r0_calibration_repeats: int = Field(ge=59)
    r0_audit_repeats: int = Field(ge=59)
    r0_required_margin: float = Field(ge=0)
    r0_audit_false_promotions: int = Field(ge=0)
    r0_audit_false_promotion_upper_95: float = Field(ge=0, le=1)
    r0_false_selection_guard_pass: bool
    r1_selected_update: int = Field(ge=0)
    r1_post_selection_refit_pass: bool
    r1_point_delta: float
    r1_target_bootstrap_interval: tuple[float, float]
    r1_margin_pass: bool
    r1_reaction_rmse: float = Field(ge=0)
    r1_sign_accuracy: float = Field(ge=0, le=1)
    r1_channel_activity: float = Field(ge=0)
    r1_channel_activity_pass: bool
    weighted_gauge_max_abs_error: float = Field(ge=0)
    rollout_mass_max_relative_error: float = Field(ge=0)
    probability_normalization_max_abs_error: float = Field(ge=0)
    control_target_mask_max_abs_error: float = Field(ge=0)
    fixed_channel_max_abs_change: float = Field(ge=0)
    protected_metrics_pass: bool
    update_zero_selectable: Literal[True] = True
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_reaction_receipt(self) -> ReactionRecoveryTestReceiptV1:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.r1_target_bootstrap_interval
        if lower > upper:
            raise ValueError("Reaction target-bootstrap interval must be ordered.")
        gates = (
            self.r0_false_selection_guard_pass,
            self.r1_post_selection_refit_pass,
            self.r1_margin_pass,
            self.r1_channel_activity_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T07S receipt must satisfy every recovery gate.")
        return self


class ReactionRecoveryTestReceipt(StrictModel):
    """Duration-correct dev24 T07S R0-null and R1-nonzero decision surface."""

    schema_version: Literal[2] = 2
    receipt_id: str
    test_contract_id: str
    parent_qualification_id: str | None = None
    status: Literal["pass", "fail_retired"]
    r0_calibration_repeats: int = Field(ge=59)
    r0_audit_repeats: int = Field(ge=59)
    r0_required_margin: float = Field(ge=0)
    r0_calibration_nonzero_checkpoint_selections: int = Field(ge=0)
    r0_audit_nonzero_checkpoint_selections: int = Field(ge=0)
    r0_audit_nonzero_checkpoint_selection_rate: float = Field(ge=0, le=1)
    r0_audit_false_promotions: int = Field(ge=0)
    r0_audit_false_promotion_upper_95: float = Field(ge=0, le=1)
    r0_false_promotion_guard_pass: bool
    r1_metric_estimand: Literal["centered_interval_log_frequency_change"]
    r1_selected_update: int = Field(ge=0)
    r1_post_selection_refit_pass: bool
    r1_interval_effect_target_balanced_rmse: float = Field(ge=0)
    r1_zero_baseline_interval_effect_target_balanced_rmse: float = Field(ge=0)
    r1_point_delta: float
    r1_target_bootstrap_interval: tuple[float, float]
    r1_margin_pass: bool
    r1_reaction_rmse: float = Field(ge=0)
    r1_sign_accuracy: float = Field(ge=0, le=1)
    r1_channel_activity: float = Field(ge=0)
    r1_channel_activity_pass: bool
    weighted_gauge_max_abs_error: float = Field(ge=0)
    rollout_mass_max_relative_error: float = Field(ge=0)
    probability_normalization_max_abs_error: float = Field(ge=0)
    control_target_mask_max_abs_error: float = Field(ge=0)
    fixed_channel_max_abs_change: float = Field(ge=0)
    protected_metrics_pass: bool
    update_zero_selectable: Literal[True] = True
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_reaction_receipt(self) -> ReactionRecoveryTestReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.r1_target_bootstrap_interval
        if lower > upper:
            raise ValueError("Reaction target-bootstrap interval must be ordered.")
        if self.r0_audit_nonzero_checkpoint_selections > self.r0_audit_repeats:
            raise ValueError("Audit nonzero-checkpoint selections exceed the audit repeats.")
        expected_rate = self.r0_audit_nonzero_checkpoint_selections / self.r0_audit_repeats
        if abs(self.r0_audit_nonzero_checkpoint_selection_rate - expected_rate) > 1e-15:
            raise ValueError("Audit nonzero-checkpoint selection rate is inconsistent.")
        gates = (
            self.r0_false_promotion_guard_pass,
            self.r1_post_selection_refit_pass,
            self.r1_margin_pass,
            self.r1_channel_activity_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T07S receipt must satisfy every recovery gate.")
        return self


class ReactionRecoveryMetricAmendment(StrictModel):
    """Immutable duration-correct metric amendment over a dev23 T07S parent."""

    schema_version: Literal[1] = 1
    amendment_id: str
    method: Literal["t07s_interval_metric_amendment_v1"]
    parent_qualification_id: str
    parent_bundle_sha256: Sha256
    parent_artifacts_manifest_sha256: Sha256
    metric_estimand: Literal["centered_interval_log_frequency_change"]
    false_promotion_upper_limit: Literal[0.05]
    environment_hash: Sha256
    null_refits: ArtifactRef
    null_model_effects: ArtifactRef
    recovery_curve: ArtifactRef
    recovery_series: ArtifactRef
    target_metrics: ArtifactRef
    bootstrap_target_draws: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef
    component_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_reaction_metric_amendment(self) -> ReactionRecoveryMetricAmendment:
        expected = self.identity(id_field="amendment_id")
        if self.amendment_id != expected:
            raise ValueError(f"amendment_id mismatch: expected {expected}.")
        return self


class RawCountMassNoiseBundle(StrictModel):
    """Immutable T02A raw-count and relative-mass noise-floor bundle."""

    schema_version: int = 1
    noise_id: str
    test_contract_id: str
    pooled_data_id: str
    count_store_sha256: Sha256
    feature_index_hash: Sha256
    environment_hash: Sha256
    source_checkpoint: str = Field(min_length=1)
    terminal_checkpoint: str = Field(min_length=1)
    split_repeats: int = Field(ge=100)
    mass_bootstrap_repeats: int = Field(ge=100)
    split_seed_start: int = Field(ge=0)
    mass_seed_start: int = Field(ge=0)
    variable_gene_count: int = Field(ge=2)
    top_gene_count: int = Field(ge=1)
    rank_top_k: int = Field(ge=1)
    variable_genes: ArtifactRef
    raw_split_metrics: ArtifactRef
    raw_repeat_summary: ArtifactRef
    raw_target_summary: ArtifactRef
    mass_bootstrap: ArtifactRef
    mass_guide_noise: ArtifactRef
    mass_target_noise: ArtifactRef
    frozen_thresholds: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_noise_bundle(self) -> RawCountMassNoiseBundle:
        expected = self.identity(id_field="noise_id")
        if self.noise_id != expected:
            raise ValueError(f"noise_id mismatch: expected {expected}.")
        if self.source_checkpoint == self.terminal_checkpoint:
            raise ValueError("T02A source and terminal checkpoints must differ.")
        if self.top_gene_count > self.variable_gene_count:
            raise ValueError("Top-gene overlap cannot exceed its frozen gene universe.")
        return self


class RawCountMassNoiseReceipt(StrictModel):
    """Complete T02A calibration decision and frozen noise floors."""

    schema_version: int = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    retained_guides: int = Field(ge=1)
    perturbation_targets: int = Field(ge=1)
    split_repeats: int = Field(ge=100)
    mass_bootstrap_repeats: int = Field(ge=100)
    target_balanced_hellinger_q95: float = Field(ge=0)
    control_hellinger_q95: float = Field(ge=0)
    target_balanced_js_q95: float = Field(ge=0)
    target_balanced_deviance_q95: float = Field(ge=0)
    target_balanced_spearman_q05: float = Field(ge=-1, le=1)
    target_balanced_top_gene_overlap_q05: float = Field(ge=0, le=1)
    interval_log_mass_rmse_q95: float = Field(ge=0)
    expansion_sign_accuracy_q05: float = Field(ge=0, le=1)
    guide_rank_spearman_q05: float = Field(ge=-1, le=1)
    target_rank_spearman_q05: float = Field(ge=-1, le=1)
    top_k_overlap_q05: float = Field(ge=0, le=1)
    bottom_k_overlap_q05: float = Field(ge=0, le=1)
    raw_invariants_pass: bool
    mass_invariants_pass: bool
    protected_metrics_frozen: bool
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_noise_receipt(self) -> RawCountMassNoiseReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        if self.status == "pass" and not (
            self.raw_invariants_pass and self.mass_invariants_pass and self.protected_metrics_frozen
        ):
            raise ValueError("A passing T02A receipt must freeze both complete noise floors.")
        return self


class RawCountMassNoiseAmendment(StrictModel):
    """Derived T02A interpretation amendment bound to one immutable bundle."""

    schema_version: Literal[1] = 1
    amendment_id: str
    parent_noise_id: str
    parent_bundle_sha256: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256
    source_checkpoint: str = Field(min_length=1)
    terminal_checkpoint: str = Field(min_length=1)
    thresholds_by_checkpoint: ArtifactRef
    recomputed_thresholds: ArtifactRef
    target_rank_stability: ArtifactRef
    threshold_semantics: ArtifactRef
    implementation_identity: ArtifactRef
    environment_identity: ArtifactRef

    @model_validator(mode="after")
    def validate_amendment(self) -> RawCountMassNoiseAmendment:
        expected = self.identity(id_field="amendment_id")
        if self.amendment_id != expected:
            raise ValueError(f"amendment_id mismatch: expected {expected}.")
        if self.source_checkpoint == self.terminal_checkpoint:
            raise ValueError("T02A amendment checkpoints must differ.")
        return self


class RawCountMassNoiseAmendmentReceipt(StrictModel):
    """Fail-closed verification result for a derived T02A amendment."""

    schema_version: Literal[1] = 1
    receipt_id: str
    amendment_id: str
    parent_noise_id: str
    implementation_hash: Sha256
    environment_hash: Sha256
    status: Literal["pass", "fail_retired"]
    parent_bundle_verified: bool
    table_invariants_pass: bool
    thresholds_recomputed: bool
    checkpoint_thresholds_recomputed: bool
    misleading_labels_retired: bool

    @model_validator(mode="after")
    def validate_amendment_receipt(self) -> RawCountMassNoiseAmendmentReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        gates = (
            self.parent_bundle_verified,
            self.table_invariants_pass,
            self.thresholds_recomputed,
            self.checkpoint_thresholds_recomputed,
            self.misleading_labels_retired,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T02A amendment must satisfy every verification gate.")
        return self


class ComponentTestContract(StrictModel):
    """Frozen component-wise qualification contract."""

    schema_version: int = 1
    test_contract_id: str
    test_id: str = Field(pattern=r"^T\d{2}[A-Z]?_[A-Z0-9_]+$")
    component: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    primary_baseline: str = Field(min_length=1)
    required_margin: float = Field(ge=0)
    drift: Literal["off", "fixed", "trainable"] = "off"
    diffusion: Literal["off", "fixed", "trainable"] = "off"
    reaction: Literal["off", "fixed", "trainable"] = "off"
    ecology: Literal["off", "fixed", "trainable"] = "off"
    decoder: Literal["off", "fixed", "trainable"] = "off"
    update_zero_selectable: bool
    post_selection_refit_required: bool

    @model_validator(mode="after")
    def validate_contract_identity(self) -> ComponentTestContract:
        expected = self.identity(id_field="test_contract_id")
        if self.test_contract_id != expected:
            raise ValueError(f"test_contract_id mismatch: expected {expected}.")
        stage = self.test_id[:3]
        exact: dict[str, tuple[str, str, str, str, str]] = {
            "T00": ("off", "off", "off", "off", "off"),
            "T01": ("off", "off", "off", "off", "off"),
            "T02": ("off", "off", "off", "off", "off"),
            "T03": ("off", "off", "off", "off", "off"),
            "T04": ("fixed", "fixed", "fixed", "off", "off"),
            "T05": ("trainable", "fixed", "off", "off", "off"),
            "T06": ("fixed", "trainable", "off", "off", "off"),
            "T07": ("fixed", "fixed", "trainable", "off", "off"),
            "T08": ("trainable", "trainable", "trainable", "off", "off"),
            "T09": ("fixed", "fixed", "fixed", "trainable", "off"),
            "T11": ("fixed", "fixed", "fixed", "fixed", "off"),
            "T12": ("fixed", "fixed", "fixed", "fixed", "trainable"),
            "T13": ("fixed", "fixed", "fixed", "fixed", "fixed"),
        }
        observed = (self.drift, self.diffusion, self.reaction, self.ecology, self.decoder)
        if stage in exact and observed != exact[stage]:
            raise ValueError(f"{stage} channel-isolation matrix mismatch.")
        if stage == "T10" and (
            observed[:3] != ("fixed", "fixed", "fixed")
            or self.ecology not in {"off", "fixed"}
            or self.decoder != "off"
        ):
            raise ValueError("T10 channel-isolation matrix mismatch.")
        return self


class ComponentTestReceipt(StrictModel):
    """Fail-closed result for one independently qualified component."""

    schema_version: int = 1
    receipt_id: str
    test_id: str = Field(pattern=r"^T\d{2}[A-Z]?_[A-Z0-9_]+$")
    status: Literal["pass", "fail_retired", "not_run"]
    primary_metric: str
    primary_baseline: str
    point_delta: float
    bootstrap_interval: tuple[float, float]
    required_margin: float = Field(ge=0)
    channel_activity: float = Field(ge=0)
    protected_metrics_pass: bool
    selected_update: int = Field(ge=0)
    input_hashes: dict[str, Sha256]
    config_hash: Sha256
    implementation_hash: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> ComponentTestReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.bootstrap_interval
        if lower > upper:
            raise ValueError("Bootstrap interval must be ordered.")
        if self.status == "pass" and not self.protected_metrics_pass:
            raise ValueError("A passing component must preserve protected metrics.")
        return self


class ComponentTestReceiptV2(StrictModel):
    """Role-aware component receipt without overloaded comparison fields."""

    schema_version: Literal[2] = 2
    receipt_id: str
    test_id: str = Field(pattern=r"^T\d{2}[A-Z]?_[A-Z0-9_]+$")
    receipt_role: Literal["model_comparison", "calibration", "invariant"]
    status: Literal["pass", "fail_retired", "not_run"]
    primary_metric: str
    primary_baseline: str
    point_delta: float | None = None
    bootstrap_interval: tuple[float, float] | None = None
    required_margin: float | None = Field(default=None, ge=0)
    channel_activity: float | None = Field(default=None, ge=0)
    estimand: str | None = None
    quantile_probability: float | None = Field(default=None, ge=0, le=1)
    quantile_value: float | None = None
    repeat_count: int | None = Field(default=None, ge=1)
    sampling_method: str | None = None
    protected_metrics_pass: bool
    selected_update: int = Field(ge=0)
    input_hashes: dict[str, Sha256]
    config_hash: Sha256
    implementation_hash: Sha256

    @model_validator(mode="after")
    def validate_role_receipt(self) -> ComponentTestReceiptV2:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        numeric = (
            self.point_delta,
            *(self.bootstrap_interval or ()),
            self.required_margin,
            self.channel_activity,
            self.quantile_probability,
            self.quantile_value,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("Component receipt numerical fields must be finite.")
        comparison = (
            self.point_delta,
            self.bootstrap_interval,
            self.required_margin,
            self.channel_activity,
        )
        calibration = (
            self.estimand,
            self.quantile_probability,
            self.quantile_value,
            self.repeat_count,
            self.sampling_method,
        )
        if self.receipt_role == "model_comparison":
            if any(value is None for value in comparison) or any(
                value is not None for value in calibration
            ):
                raise ValueError("Model-comparison receipts require only comparison fields.")
            assert self.bootstrap_interval is not None
            if self.bootstrap_interval[0] > self.bootstrap_interval[1]:
                raise ValueError("Bootstrap interval must be ordered.")
        elif self.receipt_role == "calibration":
            if any(value is not None for value in comparison) or any(
                value is None for value in calibration
            ):
                raise ValueError("Calibration receipts require only calibration fields.")
        elif any(value is not None for value in (*comparison, *calibration)):
            raise ValueError("Invariant receipts do not carry comparison or calibration fields.")
        if self.status == "pass" and not self.protected_metrics_pass:
            raise ValueError("A passing component must preserve protected metrics.")
        return self


class InputViewArtifact(StrictModel):
    schema_version: int = 1
    view_id: str
    method: Literal["identity_library_normalized", "matched_control_offset_v1"]
    raw_parent_hash: Sha256
    fit_rows_hash: Sha256
    parameter_artifact: ArtifactRef | None
    zero_preserving: Literal[True] = True
    estimability_pass: bool
    inner_validation_pass: bool
    selected: bool


class EffectHierarchyRow(StrictModel):
    perturbation_id: str
    target_id: str
    target_index: int = Field(ge=0)
    is_control: bool = False
    source_efficacy: float | None = Field(default=None, gt=0)


class EffectHierarchyContract(StrictModel):
    schema_version: int = 1
    hierarchy_id: str
    rows: tuple[EffectHierarchyRow, ...]
    guide_specific_drift: Literal[False] = False
    guide_specific_diffusion: Literal[False] = False


class TopologySupport(StrictModel):
    series_id: str
    partition_id: str
    source_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)


class TransportTopologyContract(StrictModel):
    schema_version: int = 1
    topology_id: str
    partitions: tuple[str, ...]
    allowed_edges: tuple[tuple[str, str], ...]
    support: tuple[TopologySupport, ...] = ()
    immigration_enabled: bool = False

    @model_validator(mode="after")
    def supported_terminal_states(self) -> TransportTopologyContract:
        allowed = set(self.partitions)
        if any(
            source not in allowed or target not in allowed for source, target in self.allowed_edges
        ):
            raise ValueError("Topology edge references an undeclared partition.")
        if not self.immigration_enabled:
            invalid = [
                row for row in self.support if row.source_count == 0 and row.terminal_count > 0
            ]
            if invalid:
                raise ValueError(
                    "Positive terminal support from a zero-source partition requires an "
                    "explicit immigration mechanism."
                )
        return self


class DenominatorBlock(StrictModel):
    block_id: str
    category_ids: tuple[str, ...]
    explicit_zero_categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def complete_zeros(self) -> DenominatorBlock:
        if not self.category_ids or len(self.category_ids) != len(set(self.category_ids)):
            raise ValueError("Denominator categories must be nonempty and unique.")
        if not set(self.explicit_zero_categories) <= set(self.category_ids):
            raise ValueError("Explicit zero categories must remain in the denominator.")
        return self


class DenominatorContract(StrictModel):
    schema_version: int = 1
    denominator_id: str
    blocks: tuple[DenominatorBlock, ...]
    relative_within_group: Literal[True] = True
    complete_categories: Literal[True] = True
    source_smoothing: float = 0.5

    @field_validator("source_smoothing")
    @classmethod
    def smoothing_frozen(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("v4.0 source smoothing is frozen at 0.5.")
        return value


class PoolContributor(StrictModel):
    pool_id: str
    series_id: str
    contributor_role: Literal["modeled", "background"] = "modeled"


class PoolContract(StrictModel):
    schema_version: int = 1
    pool_contract_id: str
    physical_pool_ids: tuple[str, ...]
    contributors: tuple[PoolContributor, ...]
    complete_contributor_policy: Literal[True] = True

    @model_validator(mode="after")
    def contributors_cover_pools(self) -> PoolContract:
        declared = set(self.physical_pool_ids)
        observed = {item.pool_id for item in self.contributors}
        if observed != declared:
            raise ValueError("Every declared physical pool needs at least one contributor.")
        return self


class ExposureRecord(StrictModel):
    unit_type: str
    unit_id: str
    endpoint_seen: bool
    used_for_architecture: bool = False
    used_for_hyperparameters: bool = False
    used_for_thresholds: bool = False
    used_for_biological_story: bool = False
    first_exposure_date: str | None = None
    source_artifact_hash: Sha256 | None = None
    exposure_role: EvidenceRole = EvidenceRole.DEVELOPMENT


class ExposureRegistry(StrictModel):
    schema_version: int = 1
    records: tuple[ExposureRecord, ...]


class EligibilityManifest(StrictModel):
    schema_version: int = 1
    manifest_id: str
    abundance_eligible_units: tuple[str, ...]
    state_evaluable_units: tuple[str, ...]
    endpoint_existence_used_for_abundance: Literal[False] = False
    source_information_set_hash: Sha256


class PreregistrationRef(StrictModel):
    schema_version: int = 1
    preregistration_id: str
    artifact: ArtifactRef
    frozen_before_evaluation: bool


class SeriesRecord(StrictModel):
    series_id: str
    target_index: int = Field(ge=0)
    pool_index: int = Field(ge=0)
    is_control: bool = False
    source_rows: tuple[int, ...]
    terminal_rows: tuple[int, ...]
    source_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)
    duration: float = Field(gt=0)


class SemanticStudySnapshot(StrictModel):
    schema_version: int = 1
    study_id: str
    series: tuple[SeriesRecord, ...]
    observed_edges: tuple[tuple[str, str], ...]
    feature_index_hash: Sha256
    row_universe_hash: Sha256
    exposure_registry: ExposureRegistry

    @model_validator(mode="after")
    def validate_series(self) -> SemanticStudySnapshot:
        if not self.series:
            raise ValueError("Semantic snapshot needs at least one series.")
        ids = [series.series_id for series in self.series]
        if len(ids) != len(set(ids)):
            raise ValueError("Series IDs must be unique.")
        return self


class CountStoreManifest(StrictModel):
    schema_version: int = 1
    store_id: str
    backend: Literal["csr_hdf5"] = "csr_hdf5"
    rows: int = Field(gt=0)
    features: int = Field(gt=0)
    nnz: int = Field(ge=0)
    value_dtype: str
    row_ids_hash: Sha256
    feature_index_hash: Sha256
    content_sha256: Sha256
    relative_uri: str

    _safe_uri = field_validator("relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )


class PreparedRepresentation(StrictModel):
    schema_version: int = 1
    prepared_id: str
    cache_generation_id: str
    count_store: ArtifactRef
    information_set: ArtifactRef
    fit_selection: ArtifactRef
    input_view: ArtifactRef
    feature_index: ArtifactRef
    encoder_state: ArtifactRef
    decoder_state: ArtifactRef
    latent_cache: ArtifactRef
    fit_rows_hash: Sha256
    validation_rows_hash: Sha256
    state_dim: int = Field(gt=0)


class ResolvedRunCapabilities(StrictModel):
    predict_state: bool = True
    # The development SVD representation has no calibrated count decoder.
    # Recipes must opt in only after a reconstruction/likelihood gate passes.
    decode_gene_composition: bool = False
    predict_relative_mass: bool = False
    resume_training: bool = True
    target_reference_counterfactual: bool = True
    fixed_context_counterfactual: bool = False
    dynamic_context_counterfactual: bool = False
    stream_terminal_particles: bool = True

    @classmethod
    def for_intent(cls, intent: RunIntent) -> ResolvedRunCapabilities:
        measure = intent in {RunIntent.COUNT_MEASURE, RunIntent.COUNT_CONTEXT}
        context = intent is RunIntent.COUNT_CONTEXT
        return cls(
            predict_relative_mass=measure,
            fixed_context_counterfactual=context,
            dynamic_context_counterfactual=context,
        )


class ImplementationCapabilities(StrictModel):
    schema_version: int = 1
    count_state: Literal[True] = True
    count_measure: Literal[True] = True
    count_context: Literal[True] = True
    sparse_count_store: Literal[True] = True
    exact_complete_block_dm: Literal[True] = True
    typed_resume: Literal[True] = True
    canonical_loader: Literal["credo-v4 open-run"] = "credo-v4 open-run"


class ModelConfig(StrictModel):
    state_dim: int = Field(default=8, gt=0)
    target_count: int = Field(gt=0)
    pool_count: int = Field(default=1, gt=0)
    hidden_dim: int = Field(default=32, gt=0)
    gene_decoder_features: int = Field(default=0, ge=0)
    gene_decoder_hidden_dim: int = Field(default=0, ge=0)
    state_dependent_drift: bool = False
    freeze_closed_form_drift: bool = False
    terminal_anchor_drift: bool = False
    source_carryover_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    source_conditioned_anchor: bool = False
    source_anchor_residual_scale: float = Field(default=0.25, gt=0.0)
    source_target_interaction_rank: int = Field(default=0, ge=0, le=8)
    source_target_interaction_scale: float = Field(default=0.25, gt=0.0)
    source_target_whitening_ridge: float = Field(default=1e-3, gt=0.0)
    source_target_main_max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    trainable_terminal_anchor: bool = True
    trainable_target_anchor: bool = True
    target_anchor_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptive_target_anchor: bool = False
    adaptive_target_anchor_max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    adaptive_target_anchor_min_guide_gain: float = Field(default=0.0, ge=0.0)
    shared_diffusion: bool = False
    shared_diffusion_inner_validation_pass: bool = False
    centered_selection: bool = False
    selection_inner_validation_pass: bool = False
    source_efficacy_sensitivity: bool = False
    context_rank: int = Field(default=2, ge=1, le=4)

    @model_validator(mode="after")
    def ablation_gates(self) -> ModelConfig:
        if self.gene_decoder_hidden_dim and not self.gene_decoder_features:
            raise ValueError("A hidden gene decoder requires gene_decoder_features.")
        if self.terminal_anchor_drift and self.state_dependent_drift:
            raise ValueError("Terminal-anchor and neural state drift are mutually exclusive.")
        if not self.terminal_anchor_drift and (
            self.source_carryover_alpha != 1.0
            or self.source_conditioned_anchor
            or not self.trainable_terminal_anchor
            or not self.trainable_target_anchor
            or self.target_anchor_weight != 0.0
        ):
            raise ValueError("Anchor shrinkage parameters require terminal_anchor_drift.")
        if self.source_conditioned_anchor and self.source_carryover_alpha != 0.0:
            raise ValueError(
                "A source-conditioned terminal anchor requires zero recurrent carryover."
            )
        if self.source_target_interaction_rank and (
            not self.terminal_anchor_drift or self.source_carryover_alpha != 0.0
        ):
            raise ValueError(
                "A source-target interaction requires a zero-carryover terminal anchor."
            )
        if self.source_target_interaction_rank and self.source_conditioned_anchor:
            raise ValueError(
                "Source-only and source-target anchor residuals are mutually exclusive."
            )
        if self.adaptive_target_anchor and (
            not self.terminal_anchor_drift
            or self.source_carryover_alpha != 0.0
            or self.target_anchor_weight != 0.0
        ):
            raise ValueError(
                "Adaptive target anchors require a zero-carryover terminal anchor "
                "with no fixed target weight."
            )
        if self.freeze_closed_form_drift and not self.state_dependent_drift:
            raise ValueError("A frozen closed-form drift requires a state-dependent residual.")
        if self.shared_diffusion and not self.shared_diffusion_inner_validation_pass:
            raise ValueError("Shared diffusion requires a passing inner-validation ablation.")
        if self.centered_selection and not self.selection_inner_validation_pass:
            raise ValueError("State selection requires a passing inner-validation ablation.")
        return self


class TrainingConfig(StrictModel):
    max_updates: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=1e-2, gt=0)
    state_batch_size: int = Field(default=16, gt=0)
    checkpoint_every: int = Field(default=25, gt=0)
    seed: int = Field(default=0, ge=0)
    initialization_seed: int = Field(default=0, ge=0)
    state_split_seed: int = Field(default=0, ge=0)
    dtype: Literal["float32", "float64"] = "float32"
    deterministic: bool = True
    state_full_batch: bool = False
    optimizer_name: Literal["AdamW"] = "AdamW"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = Field(default=1e-8, gt=0.0)
    optimizer_weight_decay: float = Field(default=0.01, ge=0.0)
    pilot_device_type: Literal["cpu", "cuda"] | None = None
    selected_update: int | None = None
    analytic_fit: bool = False
    gene_decoder_batch_size: int = Field(default=0, ge=0)
    gene_decoder_loss_weight: float = Field(default=0.0, ge=0.0)
    gene_decoder_validation_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    gene_decoder_validation_max_rows: int = Field(default=8_192, gt=0)
    train_state_with_gene_decoder: bool = False
    state_validation_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    state_validation_max_per_target: int = Field(default=1, gt=0)
    support_weight_power: float = Field(default=0.0, ge=0.0, le=1.0)
    support_weight_cap: int = Field(default=1_000, gt=0)
    target_drift_penalty: float = Field(default=0.0, ge=0.0)
    source_drift_penalty: float = Field(default=0.0, ge=0.0)
    source_target_main_penalty: float = Field(default=0.0, ge=0.0)
    source_target_interaction_penalty: float = Field(default=0.0, ge=0.0)
    noninteraction_linear_ridge: float = Field(default=1.0, gt=0.0)
    state_checkpoint_updates: tuple[int, ...] = ()
    state_validation_target_minimum_improvement: float = Field(default=0.0, ge=0.0)
    state_validation_interaction_minimum_improvement: float = Field(default=0.0, ge=0.0)
    post_selection_state_refit: bool = False
    checkpoint_selection: Literal[
        "final",
        "minimum_gene_decoder_validation",
        "minimum_state_validation",
        "minimum_state_validation_null_guarded",
    ] = "final"

    @model_validator(mode="after")
    def selected_in_budget(self) -> TrainingConfig:
        beta1, beta2 = self.optimizer_betas
        if not (0.0 <= beta1 < beta2 < 1.0):
            raise ValueError("optimizer_betas must satisfy 0 <= beta1 < beta2 < 1.")
        if (self.gene_decoder_batch_size == 0) != (self.gene_decoder_loss_weight == 0.0):
            raise ValueError("Gene decoder batch size and loss weight must be enabled together.")
        if self.gene_decoder_validation_fraction and not self.gene_decoder_batch_size:
            raise ValueError("Gene decoder validation requires decoder training.")
        if self.train_state_with_gene_decoder and not self.gene_decoder_batch_size:
            raise ValueError("Joint state/decoder training requires an enabled gene decoder.")
        if self.checkpoint_selection == "minimum_gene_decoder_validation":
            if not self.gene_decoder_validation_fraction:
                raise ValueError("Validation checkpoint selection requires a validation split.")
            if self.selected_update is not None:
                raise ValueError(
                    "Validation checkpoint selection resolves selected_update after training."
                )
        if self.checkpoint_selection in {
            "minimum_state_validation",
            "minimum_state_validation_null_guarded",
        }:
            if not self.state_validation_fraction:
                raise ValueError("State validation selection requires a validation split.")
            if self.selected_update is not None:
                raise ValueError(
                    "State-validation checkpoint selection resolves selected_update after training."
                )
        if (
            self.checkpoint_selection == "minimum_state_validation_null_guarded"
            and not self.post_selection_state_refit
        ):
            raise ValueError("Null-guarded state selection requires post-selection refitting.")
        if tuple(sorted(set(self.state_checkpoint_updates))) != self.state_checkpoint_updates:
            raise ValueError("state_checkpoint_updates must be strictly increasing and unique.")
        if any(
            update <= 0 or update > self.max_updates for update in self.state_checkpoint_updates
        ):
            raise ValueError("state_checkpoint_updates must lie within 1..max_updates.")
        if self.analytic_fit and self.gene_decoder_batch_size:
            raise ValueError("Analytic fitting cannot train a gene decoder.")
        if self.analytic_fit and (self.max_updates != 1 or self.selected_update not in {None, 1}):
            raise ValueError("Analytic fitting uses exactly one selected checkpoint generation.")
        if self.selected_update is not None and self.selected_update > self.max_updates:
            raise ValueError("selected_update exceeds max_updates.")
        if (
            self.selected_update is not None
            and self.selected_update != self.max_updates
            and self.selected_update % self.checkpoint_every != 0
        ):
            raise ValueError("selected_update must be a persisted checkpoint generation.")
        return self


class EvaluationConfig(StrictModel):
    particles: int = Field(default=128, gt=0)
    steps: int = Field(default=8, gt=0)
    seed: int = Field(default=10_000, ge=0)
    output_bytes_limit: int = Field(default=1_000_000_000, gt=0)
    max_step_duration: float = Field(default=1.0, gt=0)


class PhysicalGrid(StrictModel):
    schema_version: int = 1
    axis_unit: str = Field(min_length=1)
    duration: float = Field(gt=0)
    maximum_step: float = Field(gt=0)
    steps: int = Field(gt=0)
    step_size: float = Field(gt=0)

    @model_validator(mode="after")
    def exact_grid(self) -> PhysicalGrid:
        expected_steps = math.ceil(self.duration / self.maximum_step)
        expected_size = self.duration / expected_steps
        if self.steps != expected_steps or not math.isclose(
            self.step_size, expected_size, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("Physical grid does not match ceil(duration / maximum_step).")
        return self

    @classmethod
    def resolve(cls, *, axis_unit: str, duration: float, maximum_step: float) -> PhysicalGrid:
        steps = math.ceil(duration / maximum_step)
        return cls(
            axis_unit=axis_unit,
            duration=duration,
            maximum_step=maximum_step,
            steps=steps,
            step_size=duration / steps,
        )


class ResolvedConfig(StrictModel):
    schema_version: int = 1
    workspace: str
    semantic_snapshot: str
    count_store: str
    information_set: str
    split_contract: str
    eligibility_manifest: str
    effect_hierarchy: str
    topology_contract: str
    denominator_contract: str | None = None
    pool_contract: str | None = None
    preregistration: str
    multiplicity_plan: str
    candidate_selection_plan: str
    state_selection_calibration: str | None = None
    pooled_estimand: Literal["pooled_known_target_heldout_guide"] | None = None
    outer_fold_id: str | None = None
    inner_split_id: str | None = None
    pooled_outer_fold_ids: tuple[str, ...] = ()
    pooled_inner_split_ids: tuple[str, ...] = ()
    pooled_optimization_seeds: tuple[int, ...] = ()
    baseline_registry: str
    intent: RunIntent
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    feature_index: str
    source_manifest: str
    feature_permutation: str
    input_view: Literal["identity_library_normalized", "matched_control_offset_v1"] = (
        "identity_library_normalized"
    )
    correction_metadata: str | None = None
    source_smoothing: float = 0.5
    representation_fit_max_rows: int = Field(default=250_000, gt=0)
    representation_encode_batch_size: int = Field(default=8_192, gt=0)
    representation_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def correction_dependency(self) -> ResolvedConfig:
        if self.source_smoothing != 0.5:
            raise ValueError("v4.0 source smoothing is frozen at 0.5.")
        if self.input_view == "matched_control_offset_v1" and not self.correction_metadata:
            raise ValueError("matched_control_offset_v1 requires correction_metadata.")
        if (
            self.intent in {RunIntent.COUNT_MEASURE, RunIntent.COUNT_CONTEXT}
            and not self.denominator_contract
        ):
            raise ValueError(f"{self.intent.value} requires a complete denominator contract.")
        if self.intent is RunIntent.COUNT_CONTEXT and not self.pool_contract:
            raise ValueError("count_context requires a physical pool contract.")
        if self.intent is RunIntent.COUNT_STATE and (
            self.denominator_contract is not None or self.pool_contract is not None
        ):
            raise ValueError("count_state forbids denominator and pool contracts.")
        if self.model.terminal_anchor_drift and self.intent is not RunIntent.COUNT_STATE:
            raise ValueError("Development terminal-anchor drift is count_state-only.")
        if self.model.source_target_interaction_rank:
            if self.intent is not RunIntent.COUNT_STATE:
                raise ValueError("Source-target state interactions are count_state-only.")
            if (
                self.model.trainable_terminal_anchor
                or self.model.trainable_target_anchor
                or self.model.target_anchor_weight != 0.0
                or self.model.adaptive_target_anchor
            ):
                raise ValueError("Source-target pilots require frozen null and target anchors.")
            if self.training.source_target_main_penalty <= 0.0:
                raise ValueError("Source-target pilots require positive target-main shrinkage.")
            if self.training.source_target_interaction_penalty <= 0.0:
                raise ValueError(
                    "Source-target interactions require positive interaction shrinkage."
                )
            if self.training.state_validation_target_minimum_improvement <= 0.0:
                raise ValueError("Source-target pilots require a positive target-only margin.")
            if self.training.state_validation_interaction_minimum_improvement <= 0.0:
                raise ValueError("Source-target pilots require a positive interaction margin.")
            if self.training.checkpoint_selection != "minimum_state_validation_null_guarded":
                raise ValueError("Source-target interactions require null-guarded selection.")
            if not self.training.state_checkpoint_updates:
                raise ValueError(
                    "Source-target pilots require an explicit early checkpoint schedule."
                )
            if self.training.state_checkpoint_updates[0] > 25:
                raise ValueError("Source-target pilot checkpointing must begin by update 25.")
            if self.training.max_updates > 500:
                raise ValueError("The first pooled source-target pilot is capped at 500 updates.")
            if self.state_selection_calibration is None:
                raise ValueError("Source-target pilots require a bound selection calibration.")
            if self.pooled_estimand != "pooled_known_target_heldout_guide":
                raise ValueError(
                    "Source-target pilots require the pooled known-target held-out-guide estimand."
                )
            if not self.outer_fold_id or not self.inner_split_id:
                raise ValueError(
                    "Source-target pilots require fixed outer-fold and inner-split IDs."
                )
            if (
                len(set(self.pooled_outer_fold_ids)) < 2
                or self.outer_fold_id not in self.pooled_outer_fold_ids
            ):
                raise ValueError("Pooled pilots require at least two frozen outer guide folds.")
            if (
                len(self.pooled_inner_split_ids) != len(self.pooled_outer_fold_ids)
                or self.inner_split_id not in self.pooled_inner_split_ids
            ):
                raise ValueError("Pooled pilots require one frozen inner split per outer fold.")
            if (
                len(set(self.pooled_optimization_seeds)) < 3
                or self.training.seed not in self.pooled_optimization_seeds
            ):
                raise ValueError("Pooled pilots require a frozen plan of at least three seeds.")
            if self.model.pool_count != 1:
                raise ValueError(
                    "Pooled source-target pilots require exactly one model-facing pool."
                )
            if not self.training.state_full_batch:
                raise ValueError(
                    "Pooled source-target pilots require full-batch state optimization."
                )
            if self.training.pilot_device_type is None:
                raise ValueError("Source-target pilots require a frozen device type.")
            forbidden_channels = {
                "shared_diffusion": self.model.shared_diffusion,
                "centered_selection": self.model.centered_selection,
                "state_dependent_drift": self.model.state_dependent_drift,
                "source_conditioned_anchor": self.model.source_conditioned_anchor,
                "analytic_fit": self.training.analytic_fit,
            }
            enabled = sorted(name for name, value in forbidden_channels.items() if value)
            if enabled:
                raise ValueError(
                    "Source-target pilots forbid uncalibrated channels: " + ", ".join(enabled)
                )
            if self.training.support_weight_power != 0.0:
                raise ValueError(
                    "Source-target pilots freeze support_weight_power=0 until weighted "
                    "calibration is implemented."
                )
            if any(
                (
                    self.model.gene_decoder_features,
                    self.model.gene_decoder_hidden_dim,
                    self.training.gene_decoder_batch_size,
                    self.training.gene_decoder_loss_weight,
                    self.training.gene_decoder_validation_fraction,
                )
            ):
                raise ValueError(
                    "Source-target state pilots must disable the gene decoder entirely."
                )
        return self


class CompiledRunContract(StrictModel):
    schema_version: int = 1
    compiled_run_id: str
    recipe_id: Literal["credo.count_sde_v4"] = "credo.count_sde_v4"
    recipe_version: Literal["4.0.dev24"] = "4.0.dev24"
    recipe_wheel_hash: Sha256
    frozen_credo_artifact_hash: Sha256
    environment_lock_hash: Sha256
    source_manifest_hash: Sha256
    count_store_merkle_root: Sha256
    row_universe_hash: Sha256
    feature_index_hash: Sha256
    feature_permutation_hash: Sha256
    exposure_registry_hash: Sha256
    split_manifest_hash: Sha256
    information_set_hash: Sha256
    eligibility_manifest_hash: Sha256
    target_hierarchy_hash: Sha256
    denominator_manifest_hash: Sha256 | None
    experimental_topology_hash: Sha256
    physical_pool_manifest_hash: Sha256 | None
    preregistration_hash: Sha256
    multiplicity_plan_hash: Sha256
    candidate_selection_plan_hash: Sha256
    state_selection_calibration_hash: Sha256 | None
    state_selection_calibration_stage: Literal["development", "locked_audit"] | None
    correction_contract_hash: Sha256
    representation_id: str
    latent_cache_index_hash: Sha256
    compiled_problem_hash: Sha256
    resolved_config_hash: Sha256
    mathematical_contract_hash: Sha256
    count_estimator_contract_hash: Sha256 | None
    loss_scale_hash: Sha256
    baseline_registry_hash: Sha256
    compute_budget_hash: Sha256
    output_quota_hash: Sha256
    implementation_tree_hash: Sha256
    capabilities: ResolvedRunCapabilities

    @model_validator(mode="after")
    def validate_compiled_id(self) -> CompiledRunContract:
        expected = self.identity(id_field="compiled_run_id")
        if self.compiled_run_id != expected:
            raise ValueError(f"compiled_run_id mismatch: expected {expected}.")
        return self


class CheckpointManifest(StrictModel):
    schema_version: int = 1
    checkpoint_id: str
    compiled_run_id: str
    generation: int = Field(ge=0)
    update: int = Field(ge=0)
    stage: str
    parent_checkpoint_id: str | None = None
    selection_source_checkpoint_id: str | None = None
    model: ArtifactRef
    optimizer: ArtifactRef
    optimizer_tree: ArtifactRef
    rng: ArtifactRef
    sampler: ArtifactRef
    training_state: ArtifactRef


class InferenceBundleManifest(StrictModel):
    schema_version: int = 1
    run_id: str
    compiled_run_id: str
    selected_checkpoint_id: str
    recipe_id: Literal["credo.count_sde_v4"] = "credo.count_sde_v4"
    recipe_version: Literal["4.0.dev24"] = "4.0.dev24"
    selected_family: Literal[
        "configured_checkpoint",
        "gene_decoder_selected",
        "state_validation_selected",
        "global_terminal_null",
        "shrunk_sister_guide_target_terminal",
        "selected_training_only_target_main",
        "target_plus_source_target_interaction",
    ]
    selection: ArtifactRef
    model: ArtifactRef
    run_contract: ArtifactRef
    evaluation_seed_plan: ArtifactRef
    output_schema: ArtifactRef
    capabilities: ResolvedRunCapabilities


class EvaluationBundleManifest(StrictModel):
    schema_version: int = 1
    evaluation_id: str
    run_id: str
    plan: ArtifactRef
    metrics: ArtifactRef
    predictions: ArtifactRef
    audit: ArtifactRef


class SealedRunManifest(StrictModel):
    schema_version: int = 1
    sealed_id: str
    inference: ArtifactRef
    evaluations: tuple[ArtifactRef, ...]
    claim_audit: ArtifactRef


class CounterfactualBranch(StrictModel):
    branch_id: str
    effect_mode: Literal["factual", "reference"]
    context_mode: Literal["source_fixed", "pool_dynamic", "reference_dynamic"]


class CounterfactualDesign(StrictModel):
    series_index: int = Field(ge=0)
    branches: tuple[CounterfactualBranch, ...]
    particles: int = Field(default=128, gt=0)
    steps: int = Field(default=8, gt=0)
    seed: int = Field(default=0, ge=0)

    @field_validator("branches")
    @classmethod
    def unique_branches(
        cls, value: tuple[CounterfactualBranch, ...]
    ) -> tuple[CounterfactualBranch, ...]:
        ids = [branch.branch_id for branch in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Counterfactual branch IDs must be unique.")
        return value


class PredictionQuery(StrictModel):
    schema_version: int = 1
    series_indices: tuple[int, ...]
    particles: int = Field(gt=0)
    physical_grid: PhysicalGrid
    seed: int = Field(ge=0)
    include_terminal_particles: bool = False
    output_bytes_limit: int = Field(gt=0)


class EvaluationPlan(StrictModel):
    schema_version: int = 1
    evaluation_plan_id: str
    run_id: str
    information_set_hash: Sha256
    seed_plan_hash: Sha256
    baseline_registry_hash: Sha256
    multiplicity_plan_hash: Sha256
    one_shot_outer_access: Literal[True] = True


class BaselineRegistry(StrictModel):
    schema_version: int = 1
    registry_id: str
    baselines: tuple[BaselineInformationSet, ...]


class MultiplicityPlan(StrictModel):
    schema_version: int = 1
    plan_id: str
    procedure: Literal["BH", "hierarchical_BH"] = "BH"
    q: float = Field(default=0.05, gt=0, lt=1)
    families: tuple[str, ...]
    reporting_universe: str


class CandidateSelectionPlan(StrictModel):
    schema_version: int = 1
    plan_id: str
    information_set_hash: Sha256
    score: str
    direction: Literal["higher", "lower"]
    tie_rule: str
    maximum_candidates: int = Field(gt=0)
    minimum_support: int = Field(ge=1)
    protected_endpoint_selection: Literal[False] = False


class StateSelectionCalibration(StrictModel):
    """Frozen pooled calibration for nested state-family selection."""

    schema_version: int = 1
    calibration_id: str
    method: Literal["pooled_nested_two_null"] = "pooled_nested_two_null"
    calibration_stage: Literal["development", "locked_audit"]
    development_calibration_sha256: Sha256 | None = None
    repeated_per_null: int = Field(ge=119)
    false_target_main_count: Literal[0] = 0
    false_interaction_count: Literal[0] = 0
    false_joint_interaction_count: Literal[0] = 0
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    confidence_method: Literal["clopper_pearson_one_sided"] = "clopper_pearson_one_sided"
    familywise_error_target: float = Field(default=0.05, gt=0.0, le=0.05)
    false_target_main_rate_upper_bound: float = Field(ge=0.0, le=0.05)
    false_interaction_rate_upper_bound: float = Field(ge=0.0, le=0.05)
    false_joint_interaction_rate_upper_bound: float = Field(ge=0.0, le=0.05)
    target_minimum_improvement: float = Field(gt=0.0)
    interaction_minimum_improvement: float = Field(gt=0.0)
    checkpoint_updates: tuple[int, ...]
    results_artifact: ArtifactRef
    calibration_protocol_hash: Sha256

    @model_validator(mode="after")
    def valid_schedule(self) -> StateSelectionCalibration:
        if not self.checkpoint_updates:
            raise ValueError("Calibration must bind a nonempty checkpoint schedule.")
        if tuple(sorted(set(self.checkpoint_updates))) != self.checkpoint_updates:
            raise ValueError("Calibration checkpoint updates must be increasing and unique.")
        if self.calibration_stage == "locked_audit" and self.repeated_per_null < 199:
            raise ValueError("Locked audit calibration requires at least 199 fits per null.")
        if (self.calibration_stage == "locked_audit") != (
            self.development_calibration_sha256 is not None
        ):
            raise ValueError("Only locked audits must bind the threshold-development calibration.")
        exact_upper = 1.0 - (1.0 - self.confidence_level) ** (1.0 / self.repeated_per_null)
        observed_bounds = (
            self.false_target_main_rate_upper_bound,
            self.false_interaction_rate_upper_bound,
            self.false_joint_interaction_rate_upper_bound,
        )
        if any(
            not math.isclose(value, exact_upper, rel_tol=0.0, abs_tol=1e-12)
            for value in observed_bounds
        ):
            raise ValueError(
                "A null-family upper bound is inconsistent with zero-failure one-sided "
                "Clopper-Pearson calibration."
            )
        return self


class StateCalibrationCheckpointScore(StrictModel):
    update: int = Field(ge=0)
    global_null_score: float = Field(ge=0.0)
    shrunk_target_only_score: float = Field(ge=0.0)
    empirical_bayes_target_score: float = Field(ge=0.0)
    target_terminal_score: float = Field(ge=0.0)
    target_delta_score: float = Field(ge=0.0)
    linear_source_plus_target_score: float = Field(ge=0.0)
    best_target_only_score: float = Field(ge=0.0)
    best_target_only_baseline: Literal[
        "shrunk_target_only", "empirical_bayes_target", "target_terminal"
    ]
    best_noninteraction_score: float = Field(ge=0.0)
    best_noninteraction_baseline: Literal[
        "shrunk_target_only",
        "empirical_bayes_target",
        "target_terminal",
        "target_delta",
        "linear_source_plus_target",
    ]
    interaction_score: float = Field(ge=0.0)

    @model_validator(mode="after")
    def exact_best_baselines(self) -> StateCalibrationCheckpointScore:
        target_scores = {
            "shrunk_target_only": self.shrunk_target_only_score,
            "empirical_bayes_target": self.empirical_bayes_target_score,
            "target_terminal": self.target_terminal_score,
        }
        all_scores = {
            **target_scores,
            "target_delta": self.target_delta_score,
            "linear_source_plus_target": self.linear_source_plus_target_score,
        }
        if not math.isclose(
            self.best_target_only_score,
            target_scores[self.best_target_only_baseline],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or self.best_target_only_score != min(target_scores.values()):
            raise ValueError("Best target-only calibration score is inconsistent.")
        if not math.isclose(
            self.best_noninteraction_score,
            all_scores[self.best_noninteraction_baseline],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or self.best_noninteraction_score != min(all_scores.values()):
            raise ValueError("Best noninteraction calibration score is inconsistent.")
        return self


class StateSelectionCalibrationRow(StrictModel):
    """One randomization fit conditional on a fixed pooled training split."""

    replicate_index: int = Field(ge=0)
    null_family: Literal["global_target_main", "conditional_interaction", "joint_nested"]
    outer_fold_id: str
    inner_split_id: str
    permutation_seed: int = Field(ge=0)
    optimizer_seed: int = Field(ge=0)
    initialization_seed: int = Field(ge=0)
    permutation_sha256: Sha256
    fit_series_sha256: Sha256
    validation_series_sha256: Sha256
    checkpoint_scores: tuple[StateCalibrationCheckpointScore, ...]
    selected_update: int = Field(ge=0)
    selected_family: Literal[
        "global_terminal_null",
        "shrunk_sister_guide_target_terminal",
        "selected_training_only_target_main",
        "target_plus_source_target_interaction",
    ]
    maximum_target_main_gain: float
    maximum_interaction_gain: float
    false_target_main_selected: bool
    false_interaction_selected: bool
    false_joint_interaction_selected: bool

    @model_validator(mode="after")
    def exact_false_selection_label(self) -> StateSelectionCalibrationRow:
        if len({self.permutation_seed, self.optimizer_seed, self.initialization_seed}) != 3:
            raise ValueError(
                "Calibration randomization, optimizer, and initialization seeds must differ."
            )
        m1_or_m2 = self.selected_family != "global_terminal_null"
        m2 = self.selected_family == "target_plus_source_target_interaction"
        expected_target = self.null_family == "global_target_main" and m1_or_m2
        expected_interaction = self.null_family == "conditional_interaction" and m2
        expected_joint = self.null_family == "joint_nested" and m2
        if (
            self.false_target_main_selected != expected_target
            or self.false_interaction_selected != expected_interaction
            or self.false_joint_interaction_selected != expected_joint
        ):
            raise ValueError("Calibration false-selection labels disagree with the null family.")
        updates = tuple(item.update for item in self.checkpoint_scores)
        if not updates or tuple(sorted(set(updates))) != updates:
            raise ValueError("Calibration checkpoint scores must be ordered and unique.")
        if self.selected_family == "target_plus_source_target_interaction":
            if self.selected_update == 0 or self.selected_update not in updates:
                raise ValueError("Selected interaction update is absent from checkpoint scores.")
        elif self.selected_update != 0:
            raise ValueError("A noninteraction calibration family must select update zero.")
        return self


class StateSelectionCalibrationResults(StrictModel):
    """Row-level evidence and exact execution surface for null calibration."""

    schema_version: int = 1
    calibration_id: str
    method: Literal["pooled_nested_two_null"] = "pooled_nested_two_null"
    calibration_stage: Literal["development", "locked_audit"]
    development_calibration_sha256: Sha256 | None = None
    repeated_per_null: int = Field(ge=119)
    pooled_estimand: Literal["pooled_known_target_heldout_guide"]
    outer_fold_id: str
    inner_split_id: str
    pooled_outer_fold_ids: tuple[str, ...]
    pooled_inner_split_ids: tuple[str, ...]
    pooled_optimization_seeds: tuple[int, ...]
    state_split_seed: int = Field(ge=0)
    implementation_tree_hash: Sha256
    calibration_code_hash: Sha256
    environment_lock_hash: Sha256
    optimizer_fingerprint: Sha256
    device_type: Literal["cpu", "cuda"]
    dtype: Literal["float32", "float64"]
    deterministic_algorithms: bool
    representation_id: str
    split_manifest_hash: Sha256
    compiled_problem_hash: Sha256
    interaction_rank: int = Field(ge=1, le=8)
    interaction_scale: float = Field(gt=0.0)
    learning_rate: float = Field(gt=0.0)
    state_batch_size: int = Field(gt=0)
    state_full_batch: Literal[True] = True
    noninteraction_linear_ridge: float = Field(gt=0.0)
    source_target_main_penalty: float = Field(gt=0.0)
    source_target_interaction_penalty: float = Field(gt=0.0)
    target_minimum_improvement: float = Field(gt=0.0)
    interaction_minimum_improvement: float = Field(gt=0.0)
    checkpoint_updates: tuple[int, ...]
    guide_per_target_distribution_hash: Sha256
    support_distribution_hash: Sha256
    rows: tuple[StateSelectionCalibrationRow, ...]

    @model_validator(mode="after")
    def complete_rows(self) -> StateSelectionCalibrationResults:
        if tuple(sorted(set(self.checkpoint_updates))) != self.checkpoint_updates:
            raise ValueError("Result checkpoint updates must be increasing and unique.")
        if self.calibration_stage == "locked_audit" and self.repeated_per_null < 199:
            raise ValueError("Locked audit results require at least 199 fits per null.")
        if (self.calibration_stage == "locked_audit") != (
            self.development_calibration_sha256 is not None
        ):
            raise ValueError("Only locked audit results bind a development calibration.")
        if (
            len(set(self.pooled_outer_fold_ids)) < 2
            or len(self.pooled_inner_split_ids) != len(self.pooled_outer_fold_ids)
            or self.outer_fold_id not in self.pooled_outer_fold_ids
            or self.inner_split_id not in self.pooled_inner_split_ids
            or len(set(self.pooled_optimization_seeds)) < 3
        ):
            raise ValueError("Calibration results have an incomplete pooled validation plan.")
        if len(self.rows) != 3 * self.repeated_per_null:
            raise ValueError("Every null family must contain repeated_per_null result rows.")
        if tuple(row.replicate_index for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("Calibration replicate indices must be contiguous and ordered.")
        if (
            len({row.outer_fold_id for row in self.rows}) != 1
            or len({row.inner_split_id for row in self.rows}) != 1
        ):
            raise ValueError("Calibration rows must use one fixed pooled split identity.")
        if any(
            row.outer_fold_id != self.outer_fold_id or row.inner_split_id != self.inner_split_id
            for row in self.rows
        ):
            raise ValueError("Calibration row split identity differs from the result contract.")
        if (
            len({row.fit_series_sha256 for row in self.rows}) != 1
            or len({row.validation_series_sha256 for row in self.rows}) != 1
        ):
            raise ValueError("Calibration rows must use one fixed fit/validation series split.")
        expected_updates = (0, *self.checkpoint_updates)
        if any(
            tuple(score.update for score in row.checkpoint_scores) != expected_updates
            for row in self.rows
        ):
            raise ValueError("Calibration row checkpoint grids differ from the result contract.")
        for family in ("global_target_main", "conditional_interaction", "joint_nested"):
            local = [row for row in self.rows if row.null_family == family]
            if len(local) != self.repeated_per_null:
                raise ValueError(f"Calibration null family {family} is incomplete.")
            if len({row.permutation_seed for row in local}) != len(local):
                raise ValueError(f"Calibration null family {family} has duplicate permutations.")
        return self


class SelectionManifest(StrictModel):
    """Immutable nested-family decision and post-selection refit binding."""

    schema_version: int = 1
    selection_id: str
    compiled_run_id: str
    policy: str
    metric: str
    score: float | None
    selected_update: int = Field(ge=0)
    selected_checkpoint_id: str
    selected_checkpoint_relative_uri: str
    candidate_updates: tuple[int, ...]
    selected_family: Literal[
        "configured_checkpoint",
        "gene_decoder_selected",
        "state_validation_selected",
        "global_terminal_null",
        "shrunk_sister_guide_target_terminal",
        "selected_training_only_target_main",
        "target_plus_source_target_interaction",
    ]
    inner_selected_update: int | None = Field(default=None, ge=0)
    inner_selected_checkpoint_id: str | None = None
    refit_checkpoint_id: str | None = None
    post_selection_refit: bool = False
    refit_series_hash: Sha256 | None = None
    global_null_score: float | None = None
    shrunk_target_only_score: float | None = None
    best_target_only_score: float | None = None
    best_target_only_baseline: (
        Literal["shrunk_target_only", "empirical_bayes_target", "target_terminal"] | None
    ) = None
    best_noninteraction_score: float | None = None
    best_noninteraction_baseline: (
        Literal[
            "shrunk_target_only",
            "empirical_bayes_target",
            "target_terminal",
            "target_delta",
            "linear_source_plus_target",
        ]
        | None
    ) = None
    interaction_score: float | None = None
    interaction_incremental_gain: float | None = None
    target_incremental_gain: float | None = None
    target_minimum_required_improvement: float | None = None
    interaction_minimum_required_improvement: float | None = None
    selection_calibration_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> SelectionManifest:
        expected = self.identity(id_field="selection_id")
        if self.selection_id != expected:
            raise ValueError(f"selection_id mismatch: expected {expected}.")
        if self.policy == "minimum_state_validation_null_guarded":
            required = (
                self.global_null_score,
                self.shrunk_target_only_score,
                self.best_target_only_score,
                self.best_target_only_baseline,
                self.best_noninteraction_score,
                self.best_noninteraction_baseline,
                self.interaction_score,
                self.interaction_incremental_gain,
                self.target_incremental_gain,
                self.target_minimum_required_improvement,
                self.interaction_minimum_required_improvement,
                self.selection_calibration_hash,
            )
            if any(value is None for value in required):
                raise ValueError("Null-guarded selections require complete nested-family evidence.")
        if self.post_selection_refit:
            if self.inner_selected_checkpoint_id is None or self.refit_checkpoint_id is None:
                raise ValueError("Refit selections must bind inner and refit checkpoint IDs.")
            if self.selected_checkpoint_id != self.refit_checkpoint_id:
                raise ValueError("Selected checkpoint must be the committed refit checkpoint.")
        return self


class BaselineInformationSet(StrictModel):
    schema_version: int = 1
    baseline_id: str
    code_hash: Sha256
    allowed_training_rows_hash: Sha256
    allowed_test_source_rows_hash: Sha256
    forbidden_endpoint_rows_hash: Sha256
    split_hash: Sha256
    representation_id: str
    denominator_hash: Sha256 | None = None
    aggregation_hash: Sha256
    seed_plan_hash: Sha256


class ContextAuditContract(StrictModel):
    schema_version: int = 1
    effective_rank_threshold: int = Field(default=2, ge=2)
    singular_ratio_threshold: float = Field(default=0.05, gt=0, lt=1)
    maximum_interaction_rank: int = Field(default=4, ge=1, le=4)
    maximum_condition_number: float = Field(default=50.0, gt=0)
    parameter_match_fraction: float = Field(default=0.01, ge=0, le=0.01)
    held_out_pool_information_set: Sha256
    observation_operator: ArtifactRef


class ContextAuditReceipt(StrictModel):
    schema_version: int = 1
    contract_hash: Sha256
    effective_rank: int = Field(ge=0)
    selected_rank: int = Field(ge=0)
    condition_number: float | None
    parameter_difference_fraction: float = Field(ge=0)
    held_out_access_pass: bool
    observation_operator_pass: bool
    status: Literal["eligible", "diagnostic_only"]


class ShardBuildCheckpoint(StrictModel):
    schema_version: int = 1
    build_plan_id: str
    source_cursor: int = Field(ge=0)
    completed_shard_hashes: tuple[Sha256, ...]
    temporary_shard_ids: tuple[str, ...] = ()
    row_count: int = Field(ge=0)
    nonzero_count: int = Field(ge=0)
    rng_state: ArtifactRef | None = None


class RepresentationCheckpoint(StrictModel):
    schema_version: int = 1
    representation_contract_id: str
    update: int = Field(ge=0)
    model: ArtifactRef
    optimizer: ArtifactRef
    rng: ArtifactRef
    cell_sampler_cursor: int = Field(ge=0)
    loss_scale_hash: Sha256
    selected_inner_state: ArtifactRef | None = None


class DynamicsCheckpoint(StrictModel):
    schema_version: int = 1
    compiled_run_id: str
    update: int = Field(ge=0)
    model: ArtifactRef
    optimizer: ArtifactRef
    rng: ArtifactRef
    series_sampler_cursor: int = Field(ge=0)
    exact_count_cursor: int = Field(ge=0)
    brownian_plan_hash: Sha256
    bank_state: ArtifactRef | None = None
    compute_used: float = Field(ge=0)
    output_bytes_used: int = Field(ge=0)


class EvaluationCheckpoint(StrictModel):
    schema_version: int = 1
    run_id: str
    evaluation_id: str
    completed_chunks: tuple[Sha256, ...]
    particle_plan_hash: Sha256
    noise_plan_hash: Sha256
    output_bytes_used: int = Field(ge=0)
    compute_used: float = Field(ge=0)


def checked_contract(model: type[StrictModel], payload: dict[str, Any]) -> StrictModel:
    try:
        return model.model_validate(payload)
    except Exception as exc:  # pydantic aggregates the useful field diagnostics
        raise ContractError(str(exc)) from exc
