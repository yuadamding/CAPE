"""Generate committed JSON Schemas from strict Pydantic contracts."""

from __future__ import annotations

import json
from pathlib import Path

from credo_count_sde_v4.contracts import (
    ArtifactRef,
    BaselineInformationSet,
    BaselineRegistry,
    CandidateSelectionPlan,
    CompiledRunContract,
    ComponentTestContract,
    ComponentTestReceipt,
    ComponentTestReceiptV2,
    ContextAuditContract,
    CountRepresentationBundle,
    CountStoreManifest,
    DenominatorContract,
    DynamicsCheckpoint,
    EffectHierarchyContract,
    EligibilityManifest,
    EvaluationBundleManifest,
    EvaluationCheckpoint,
    EvaluationPlan,
    ExposureRegistry,
    FeatureIndex,
    ImplementationCapabilities,
    InferenceBundleManifest,
    InputViewArtifact,
    MultiplicityPlan,
    ParticleEngineQualificationBundle,
    ParticleEngineTestReceipt,
    PhysicalGrid,
    PoolContract,
    PooledFiniteMeasureBundle,
    PredictionQuery,
    PreparedRepresentation,
    PreregistrationRef,
    RawCountMassNoiseAmendment,
    RawCountMassNoiseAmendmentReceipt,
    RawCountMassNoiseBundle,
    RawCountMassNoiseReceipt,
    RepresentationCheckpoint,
    RowIndex,
    SealedRunManifest,
    SelectionManifest,
    SemanticStudySnapshot,
    ShardBuildCheckpoint,
    SplitContract,
    StateSelectionCalibration,
    StateSelectionCalibrationResults,
    TransportTopologyContract,
)

MODELS = {
    "artifact-ref.v1.json": ArtifactRef,
    "semantic-snapshot.v1.json": SemanticStudySnapshot,
    "count-store.v1.json": CountStoreManifest,
    "count-representation-bundle.v1.json": CountRepresentationBundle,
    "row-index.v1.json": RowIndex,
    "feature-index.v1.json": FeatureIndex,
    "pooled-finite-measure-bundle.v1.json": PooledFiniteMeasureBundle,
    "component-test-contract.v1.json": ComponentTestContract,
    "component-test-receipt.v1.json": ComponentTestReceipt,
    "component-test-receipt.v2.json": ComponentTestReceiptV2,
    "particle-engine-qualification-bundle.v1.json": ParticleEngineQualificationBundle,
    "particle-engine-test-receipt.v1.json": ParticleEngineTestReceipt,
    "raw-count-mass-noise-bundle.v1.json": RawCountMassNoiseBundle,
    "raw-count-mass-noise-receipt.v1.json": RawCountMassNoiseReceipt,
    "raw-count-mass-noise-amendment.v1.json": RawCountMassNoiseAmendment,
    "raw-count-mass-noise-amendment-receipt.v1.json": RawCountMassNoiseAmendmentReceipt,
    "split-contract.v1.json": SplitContract,
    "input-view.v1.json": InputViewArtifact,
    "effect-hierarchy.v1.json": EffectHierarchyContract,
    "transport-topology.v1.json": TransportTopologyContract,
    "denominator-contract.v1.json": DenominatorContract,
    "pool-contract.v1.json": PoolContract,
    "physical-grid.v1.json": PhysicalGrid,
    "implementation-capabilities.v1.json": ImplementationCapabilities,
    "prepared-representation.v1.json": PreparedRepresentation,
    "compiled-run.v1.json": CompiledRunContract,
    "exposure-registry.v1.json": ExposureRegistry,
    "eligibility-manifest.v1.json": EligibilityManifest,
    "preregistration-ref.v1.json": PreregistrationRef,
    "multiplicity-plan.v1.json": MultiplicityPlan,
    "candidate-selection-plan.v1.json": CandidateSelectionPlan,
    "state-selection-calibration.v1.json": StateSelectionCalibration,
    "state-selection-calibration-results.v1.json": StateSelectionCalibrationResults,
    "selection-manifest.v1.json": SelectionManifest,
    "baseline-information-set.v1.json": BaselineInformationSet,
    "baseline-registry.v1.json": BaselineRegistry,
    "prediction-query.v1.json": PredictionQuery,
    "evaluation-plan.v1.json": EvaluationPlan,
    "context-audit.v1.json": ContextAuditContract,
    "shard-build-checkpoint.v1.json": ShardBuildCheckpoint,
    "representation-checkpoint.v1.json": RepresentationCheckpoint,
    "dynamics-checkpoint.v1.json": DynamicsCheckpoint,
    "evaluation-checkpoint.v1.json": EvaluationCheckpoint,
    "inference-bundle.v1.json": InferenceBundleManifest,
    "evaluation-bundle.v1.json": EvaluationBundleManifest,
    "sealed-run.v1.json": SealedRunManifest,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(exist_ok=True)
    for name, model in MODELS.items():
        payload = model.model_json_schema()
        payload["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        payload["$id"] = f"https://credo.local/schemas/{name}"
        (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
