"""Generate the non-claim development SBOM, validation receipt, and checksums."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch
from generate_repository_manifest import write_manifest

from credo_count_sde_v4.canonical import atomic_json, sha256_file
from credo_count_sde_v4.compat.credo3 import verify_frozen_credo
from credo_count_sde_v4.runtime_identity import (
    environment_lock_hash,
    implementation_tree_hash,
)
from credo_count_sde_v4.version import RECIPE_ID, RECIPE_VERSION, __version__

DIRECT_DEPENDENCIES = ("h5py", "numpy", "pandas", "pyarrow", "pydantic", "PyYAML", "scipy", "torch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-collected", type=int, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-skipped", type=int, required=True)
    parser.add_argument("--tests-failed", type=int, required=True)
    parser.add_argument("--coverage-percent", type=float, required=True)
    parser.add_argument("--distribution-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    components = [
        {
            "type": "library",
            "name": name,
            "version": importlib.metadata.version(name),
        }
        for name in DIRECT_DEPENDENCIES
    ]
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "credo-count-sde-v4",
                "version": __version__,
            }
        },
        "components": components,
    }
    (root / "sbom").mkdir(exist_ok=True)
    atomic_json(root / "sbom/development.cdx.json", sbom)
    distribution_dir = (args.distribution_dir or (root / "dist")).resolve()
    distributions = {
        path.name: sha256_file(path)
        for path in sorted(distribution_dir.glob("*"))
        if path.is_file()
    }
    blueprint = workspace / "docs/credo/count-sde-v4-repository-blueprint.md"
    review_hashes = json.loads((root / "provenance/reviews/review-hashes.v1.json").read_text())
    tested_environment_path = root / "locks/tested-environment.v1.json"
    tested_environment = json.loads(tested_environment_path.read_text())
    t07s_evidence = workspace / "credo_v4_t07s_reaction_recovery_20260815"
    t07s_report = workspace / "CREDO_V4_T07S_REACTION_RECOVERY_20260815_RUN_REPORT.md"
    t07s_metric_amendment = workspace / "credo_v4_t07s_reaction_metric_amendment_20260816_r2"
    t07s_metric_report = (
        workspace / "CREDO_V4_T07S_REACTION_METRIC_AMENDMENT_20260816_RUN_REPORT.md"
    )
    t07s_null_interval_amendment = workspace / "credo_v4_t07s_null_interval_amendment_20260816"
    t07s_null_interval_report = (
        workspace / "CREDO_V4_T07S_NULL_INTERVAL_AMENDMENT_20260816_RUN_REPORT.md"
    )
    t07r_evidence = workspace / "credo_v4_renz_t07r_a0_20260816"
    t07r_bundle = t07r_evidence / "T07R_A0_POOLED_LIKELIHOOD"
    t07r_v2_evidence = workspace / "credo_v4_renz_t07r_a0_v2_20260816"
    t07r_v2_bundle = t07r_v2_evidence / "T07R_A0_PHYSICAL_POOL_CONDITIONAL_DM_V2_R2"
    gse314342_g00 = workspace / "credo_v4_gse314342_g00_20260816"
    gse314342_g00_dev32_b0_provenance = root / "provenance/g00/dev32-b0-a2/PROVENANCE_INDEX.json"
    gse314342_g00_dev33_hardening = root / "docs/g00-dev33-g00c-hardening.md"
    gse314342_g00_dev33b_canary = root / "docs/g00-dev33b-extraction-canary.md"
    gse314342_g00_dev33b_provenance = root / "provenance/g00/dev33b-canary/PROVENANCE_INDEX.json"
    gse314342_g00_dev34a_claim_contract = root / "docs/g00-dev34a-claim-contract.md"
    gse314342_g00_source_plane = gse314342_g00 / "G00_SOURCE_PLANE_DEV29"
    receipt = {
        "schema_version": 1,
        "status": "engineering_only",
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "recipe_id": RECIPE_ID,
        "recipe_version": RECIPE_VERSION,
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "implementation_tree_hash": implementation_tree_hash(),
        "environment_lock_hash": environment_lock_hash(),
        "tested_environment_lock_sha256": sha256_file(tested_environment_path),
        "tested_execution_environment_digest": tested_environment["execution_environment_digest"],
        "frozen_credo": verify_frozen_credo(workspace),
        "blueprint_sha256": sha256_file(blueprint) if blueprint.is_file() else None,
        "amendment_sha256": review_hashes["amendment_sha256"],
        "component_qualification_decision_sha256": review_hashes[
            "component_qualification_decision_sha256"
        ],
        "component_qualification_assessment_sha256": review_hashes[
            "component_qualification_assessment_sha256"
        ],
        "component_status_assessment_sha256": review_hashes["component_status_assessment_sha256"],
        "t02a_interpretation_review_sha256": review_hashes["t02a_interpretation_review_sha256"],
        "t07s_review_sha256": review_hashes["t07s_review_sha256"],
        "t07s_qualification_sha256": (
            sha256_file(t07s_evidence / "reaction-recovery.json")
            if (t07s_evidence / "reaction-recovery.json").is_file()
            else None
        ),
        "t07s_test_receipt_sha256": (
            sha256_file(t07s_evidence / "TEST_RECEIPT.json")
            if (t07s_evidence / "TEST_RECEIPT.json").is_file()
            else None
        ),
        "t07s_run_report_sha256": sha256_file(t07s_report) if t07s_report.is_file() else None,
        "t07s_metric_review_sha256": review_hashes["t07s_metric_review_sha256"],
        "t07s_metric_amendment_sha256": (
            sha256_file(t07s_metric_amendment / "reaction-recovery-amendment.json")
            if (t07s_metric_amendment / "reaction-recovery-amendment.json").is_file()
            else None
        ),
        "t07s_metric_test_receipt_sha256": (
            sha256_file(t07s_metric_amendment / "TEST_RECEIPT.json")
            if (t07s_metric_amendment / "TEST_RECEIPT.json").is_file()
            else None
        ),
        "t07s_metric_run_report_sha256": (
            sha256_file(t07s_metric_report) if t07s_metric_report.is_file() else None
        ),
        "t07s_null_interval_review_sha256": review_hashes["t07s_null_interval_review_sha256"],
        "t07s_null_interval_amendment_sha256": (
            sha256_file(t07s_null_interval_amendment / "reaction-recovery-amendment.json")
            if (t07s_null_interval_amendment / "reaction-recovery-amendment.json").is_file()
            else None
        ),
        "t07s_null_interval_test_receipt_sha256": (
            sha256_file(t07s_null_interval_amendment / "TEST_RECEIPT.json")
            if (t07s_null_interval_amendment / "TEST_RECEIPT.json").is_file()
            else None
        ),
        "t07s_null_interval_run_report_sha256": (
            sha256_file(t07s_null_interval_report) if t07s_null_interval_report.is_file() else None
        ),
        "t07r_review_sha256": review_hashes["t07r_review_sha256"],
        "t07r_qualification_sha256": (
            sha256_file(t07r_bundle / "pooled-reaction-likelihood.json")
            if (t07r_bundle / "pooled-reaction-likelihood.json").is_file()
            else None
        ),
        "t07r_test_receipt_sha256": (
            sha256_file(t07r_bundle / "TEST_RECEIPT.json")
            if (t07r_bundle / "TEST_RECEIPT.json").is_file()
            else None
        ),
        "t07r_run_report_sha256": (
            sha256_file(t07r_evidence / "RUN_REPORT.md")
            if (t07r_evidence / "RUN_REPORT.md").is_file()
            else None
        ),
        "t07r_v2_review_sha256": review_hashes["t07r_v2_review_sha256"],
        "t07r_v2_qualification_sha256": (
            sha256_file(t07r_v2_bundle / "physical-pool-conditional-reaction.json")
            if (t07r_v2_bundle / "physical-pool-conditional-reaction.json").is_file()
            else None
        ),
        "t07r_v2_test_receipt_sha256": (
            sha256_file(t07r_v2_bundle / "TEST_RECEIPT.json")
            if (t07r_v2_bundle / "TEST_RECEIPT.json").is_file()
            else None
        ),
        "t07r_v2_run_report_sha256": (
            sha256_file(t07r_v2_evidence / "RUN_REPORT.md")
            if (t07r_v2_evidence / "RUN_REPORT.md").is_file()
            else None
        ),
        "gse314342_g00_artifacts_manifest_sha256": (
            sha256_file(gse314342_g00 / "ARTIFACTS.sha256")
            if (gse314342_g00 / "ARTIFACTS.sha256").is_file()
            else None
        ),
        "gse314342_g00_engineering_decision_sha256": (
            sha256_file(gse314342_g00 / "G00_ENGINEERING_DECISION.json")
            if (gse314342_g00 / "G00_ENGINEERING_DECISION.json").is_file()
            else None
        ),
        "gse314342_g00_execution_report_sha256": (
            sha256_file(gse314342_g00 / "G00_EXECUTION_REPORT.md")
            if (gse314342_g00 / "G00_EXECUTION_REPORT.md").is_file()
            else None
        ),
        "gse314342_g04_h100_evidence_archive_sha256": (
            sha256_file(gse314342_g00 / "FINAL_G04_EVIDENCE_A2.tar.gz")
            if (gse314342_g00 / "FINAL_G04_EVIDENCE_A2.tar.gz").is_file()
            else None
        ),
        "gse314342_g14_contract_manifest_sha256": (
            sha256_file(gse314342_g00 / "G14_FROZEN_CONTRACT/SHA256SUMS")
            if (gse314342_g00 / "G14_FROZEN_CONTRACT/SHA256SUMS").is_file()
            else None
        ),
        "gse314342_g00_dev29_review_sha256": review_hashes["gse314342_g00_dev29_review_sha256"],
        "gse314342_g00_dev30_review_sha256": review_hashes["gse314342_g00_dev30_review_sha256"],
        "gse314342_g00_dev30_r2_review_sha256": review_hashes[
            "gse314342_g00_dev30_r2_review_sha256"
        ],
        "gse314342_g00_dev31_review_sha256": review_hashes["gse314342_g00_dev31_review_sha256"],
        "gse314342_g00_dev31_b0_authorization_sha256": review_hashes[
            "gse314342_g00_dev31_b0_authorization_sha256"
        ],
        "gse314342_g00_dev32_legacy_review_sha256": review_hashes[
            "gse314342_g00_dev32_legacy_review_sha256"
        ],
        "gse314342_g00_dev32_b0_review_sha256": review_hashes[
            "gse314342_g00_dev32_b0_review_sha256"
        ],
        "gse314342_g00_dev32_b0_provenance_sha256": (
            sha256_file(gse314342_g00_dev32_b0_provenance)
            if gse314342_g00_dev32_b0_provenance.is_file()
            else None
        ),
        "gse314342_g00_dev33_hardening_sha256": (
            sha256_file(gse314342_g00_dev33_hardening)
            if gse314342_g00_dev33_hardening.is_file()
            else None
        ),
        "gse314342_g00_dev33b_canary_sha256": (
            sha256_file(gse314342_g00_dev33b_canary)
            if gse314342_g00_dev33b_canary.is_file()
            else None
        ),
        "gse314342_g00_dev33b_provenance_sha256": (
            sha256_file(gse314342_g00_dev33b_provenance)
            if gse314342_g00_dev33b_provenance.is_file()
            else None
        ),
        "gse314342_g00_dev33b_final_review_sha256": review_hashes[
            "gse314342_g00_dev33b_final_review_sha256"
        ],
        "gse314342_g00_dev34a_independent_review_sha256": review_hashes[
            "gse314342_g00_dev34a_independent_review_sha256"
        ],
        "gse314342_g00_dev34a_claim_contract_sha256": (
            sha256_file(gse314342_g00_dev34a_claim_contract)
            if gse314342_g00_dev34a_claim_contract.is_file()
            else None
        ),
        "gse314342_g00_dev35_independent_review_sha256": review_hashes[
            "gse314342_g00_dev35_independent_review_sha256"
        ],
        "gse314342_g00_dev36_independent_review_sha256": review_hashes[
            "gse314342_g00_dev36_independent_review_sha256"
        ],
        "gse314342_g00_dev29_builder_sha256": (
            sha256_file(gse314342_g00 / "freeze_g00_source_plane_dev29.py")
            if (gse314342_g00 / "freeze_g00_source_plane_dev29.py").is_file()
            else None
        ),
        "gse314342_g00_source_plane_manifest_sha256": (
            sha256_file(gse314342_g00_source_plane / "SHA256SUMS")
            if (gse314342_g00_source_plane / "SHA256SUMS").is_file()
            else None
        ),
        "gse314342_g00_source_authority_sha256": (
            sha256_file(gse314342_g00_source_plane / "G00A_SOURCE_AUTHORITY.json")
            if (gse314342_g00_source_plane / "G00A_SOURCE_AUTHORITY.json").is_file()
            else None
        ),
        "gse314342_g00_virtual_store_manifest_sha256": (
            sha256_file(gse314342_g00_source_plane / "G00B_VIRTUAL_CANONICAL_STORE/manifest.json")
            if (gse314342_g00_source_plane / "G00B_VIRTUAL_CANONICAL_STORE/manifest.json").is_file()
            else None
        ),
        "tests_collected": args.tests_collected,
        "tests_passed": args.tests_passed,
        "tests_skipped": args.tests_skipped,
        "tests_failed": args.tests_failed,
        "coverage_percent": args.coverage_percent,
        "distributions": distributions,
        "stable_discovery_registered": False,
        "biological_claims": False,
    }
    (root / "receipts").mkdir(exist_ok=True)
    atomic_json(root / "receipts/local-validation.json", receipt)
    write_manifest(root)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
