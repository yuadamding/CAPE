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
    dtype: Literal["float32", "float64"] = "float32"
    deterministic: bool = True
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
    checkpoint_selection: Literal[
        "final", "minimum_gene_decoder_validation", "minimum_state_validation"
    ] = "final"

    @model_validator(mode="after")
    def selected_in_budget(self) -> TrainingConfig:
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
        if self.checkpoint_selection == "minimum_state_validation":
            if not self.state_validation_fraction:
                raise ValueError("State validation selection requires a validation split.")
            if self.selected_update is not None:
                raise ValueError(
                    "State-validation checkpoint selection resolves selected_update after training."
                )
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
        return self


class CompiledRunContract(StrictModel):
    schema_version: int = 1
    compiled_run_id: str
    recipe_id: Literal["credo.count_sde_v4"] = "credo.count_sde_v4"
    recipe_version: Literal["4.0.dev14"] = "4.0.dev14"
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
    recipe_version: Literal["4.0.dev14"] = "4.0.dev14"
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
