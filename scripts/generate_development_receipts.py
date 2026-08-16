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
    parser.add_argument("--tests-passed", type=int, required=True)
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
    amendment = Path(
        "/home/yding1995/.codex/attachments/52905259-b1cd-4e04-8fae-e2cb672514a8/pasted-text.txt"
    )
    component_decision = Path(
        "/home/yding1995/.codex/attachments/787edecc-16fc-44f4-95bb-0d5589155bbd/pasted-text.txt"
    )
    component_assessment = Path(
        "/home/yding1995/.codex/attachments/adcf3061-ef5f-44d5-abe7-51663447a868/pasted-text.txt"
    )
    component_status_assessment = Path(
        "/home/yding1995/.codex/attachments/419a4b6e-3a98-431d-8dce-6d90bac813f1/pasted-text.txt"
    )
    t02a_interpretation_review = Path(
        "/home/yding1995/.codex/attachments/801c74d4-66df-455e-8b14-0c6730dce794/pasted-text.txt"
    )
    t07s_review = Path(
        "/home/yding1995/.codex/attachments/8b562990-5737-48e1-8374-4dc608aa0541/pasted-text.txt"
    )
    t07s_evidence = workspace / "credo_v4_t07s_reaction_recovery_20260815"
    t07s_report = workspace / "CREDO_V4_T07S_REACTION_RECOVERY_20260815_RUN_REPORT.md"
    t07s_metric_review = Path(
        "/home/yding1995/.codex/attachments/75263066-934c-4192-96ad-aa9d6d980a75/pasted-text.txt"
    )
    t07s_metric_amendment = workspace / "credo_v4_t07s_reaction_metric_amendment_20260816_r2"
    t07s_metric_report = (
        workspace / "CREDO_V4_T07S_REACTION_METRIC_AMENDMENT_20260816_RUN_REPORT.md"
    )
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
        "frozen_credo": verify_frozen_credo(workspace),
        "blueprint_sha256": sha256_file(blueprint) if blueprint.is_file() else None,
        "amendment_sha256": sha256_file(amendment) if amendment.is_file() else None,
        "component_qualification_decision_sha256": (
            sha256_file(component_decision) if component_decision.is_file() else None
        ),
        "component_qualification_assessment_sha256": (
            sha256_file(component_assessment) if component_assessment.is_file() else None
        ),
        "component_status_assessment_sha256": (
            sha256_file(component_status_assessment)
            if component_status_assessment.is_file()
            else None
        ),
        "t02a_interpretation_review_sha256": (
            sha256_file(t02a_interpretation_review)
            if t02a_interpretation_review.is_file()
            else None
        ),
        "t07s_review_sha256": sha256_file(t07s_review) if t07s_review.is_file() else None,
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
        "t07s_metric_review_sha256": (
            sha256_file(t07s_metric_review) if t07s_metric_review.is_file() else None
        ),
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
        "tests_passed": args.tests_passed,
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
