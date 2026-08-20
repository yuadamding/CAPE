"""Publish the expression-free Dev37 D1 source-derived execution authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import freeze_g00c_d1_authority as dev35
import freeze_g00c_d1_authority_v2 as dev36
import numpy as np
import pandas as pd

from credo_count_sde_v4.canonical import atomic_json, canonical_json_bytes, sha256_file
from credo_count_sde_v4.contracts import (
    G00CD1ExecutionAuthorityFreezeV3,
    G00CHierarchyDerivationReceiptV4,
    G00CImplementationAuthorityV3,
    G00CImplementationBindingV3,
    G00CRefitSeedScheduleV1,
    G00CSamplerPlanEntryV4,
    G00CSamplerPlanV4,
    G00CSelectionFreezeContractV1,
    G00CSourceFileBindingV4,
    G00CSourcePlaneBindingV4,
    VirtualCanonicalCountStoreManifestV2,
)
from credo_count_sde_v4.store.g00c_source_v4 import derive_hierarchy_from_g00b_v4
from credo_count_sde_v4.store.g00c_v3 import verify_g00c_d1_freeze_v1
from credo_count_sde_v4.store.virtual import VirtualCanonicalCountStore


def _array_hash(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _copy_implementation(repository: Path, destination: Path, name: str, relative: str) -> Path:
    target = destination / name
    dev35._copy(repository / relative, target)
    return target


def _source_binding(
    temporary: Path,
    *,
    source_plane_root: Path,
    feature_index_path: Path,
) -> tuple[G00CSourcePlaneBindingV4, VirtualCanonicalCountStore]:
    store_root = source_plane_root / "G00B_VIRTUAL_CANONICAL_STORE_V2"
    manifest_path = store_root / "manifest.json"
    manifest = VirtualCanonicalCountStoreManifestV2.model_validate_json(manifest_path.read_text())
    accepted = temporary / "G00B_ACCEPTED_MANIFEST.json"
    feature_index = temporary / "FEATURE_INDEX.json"
    dev35._copy(manifest_path, accepted)
    dev35._copy(feature_index_path, feature_index)
    feature_payload = json.loads(feature_index.read_text())
    records = feature_payload["features"]
    if (
        feature_payload["feature_count"] != manifest.features
        or feature_payload["ordered_hash"] != manifest.canonical_feature_index_hash
        or hashlib.sha256(canonical_json_bytes(records)).hexdigest()
        != manifest.canonical_feature_index_hash
    ):
        raise RuntimeError("Dev37 feature-index metadata differs from accepted G00B.")
    puro = [
        index for index, record in enumerate(records) if record["feature_id"] == "CUSTOM001_PuroR"
    ]
    if len(puro) != 1:
        raise RuntimeError("Dev37 requires one exact CUSTOM001_PuroR feature.")
    payload = {
        "binding_id": "0" * 64,
        "accepted_g00b_parent": dev35._artifact(
            temporary,
            accepted,
            "credo.g00c.dev37.accepted-g00b-manifest",
            "application/json",
        ),
        "accepted_g00b_manifest_sha256": sha256_file(accepted),
        "virtual_store_id": manifest.virtual_store_id,
        "source_authority_id": manifest.source_authority_id,
        "virtual_store_relative_uri": store_root.name,
        "canonical_feature_index": dev35._artifact(
            temporary,
            feature_index,
            "credo.g00c.dev37.canonical-feature-index",
            "application/json",
        ),
        "canonical_feature_index_hash": manifest.canonical_feature_index_hash,
        "row_locator_sha256": manifest.row_locator.sha256,
        "feature_permutations_sha256": manifest.feature_permutations.sha256,
        "guide_target_crosswalk_sha256": manifest.guide_target_crosswalk.sha256,
        "source_files": tuple(
            G00CSourceFileBindingV4(
                source_id=source.source_id,
                checkpoint=source.checkpoint,
                source_file_sha256=source.source_file_sha256,
            )
            for source in manifest.sources
        ),
        "feature_count": manifest.features,
        "puro_r_canonical_index": puro[0],
    }
    binding = dev35._identified(G00CSourcePlaneBindingV4, payload, "binding_id")
    # Metadata-only hierarchy derivation opens neither rows() nor any source file.
    return binding, VirtualCanonicalCountStore(store_root, source_root=Path("/not-accessed"))


def _sampler_plan(
    temporary: Path,
    *,
    selection_freeze_id: str,
    schedule: G00CRefitSeedScheduleV1,
    macro_updates: int,
) -> G00CSamplerPlanV4:
    entries = tuple(
        G00CSamplerPlanEntryV4(
            candidate_kind=cast(Any, kind),
            candidate_value=value,
            refit_draw_id=draw,
            macro_updates=macro_updates,
            expected_trace_rows=macro_updates * 4096,
        )
        for kind, values in (
            ("feature_count", (256, 512, 1024, 2048, 4096)),
            ("training_cells", (50_000, 100_000, 250_000, 500_000, 1_000_000)),
        )
        for value in values
        for draw in range(59)
    )
    payload = {
        "plan_id": "0" * 64,
        "execution_authority_namespace": selection_freeze_id,
        "seed_schedule_id": schedule.schedule_id,
        "grid_stage": "base",
        "entries": entries,
        "resume_after_macro_update": macro_updates // 2,
        "expected_total_trace_rows": sum(item.expected_trace_rows for item in entries),
    }
    plan = dev35._identified(G00CSamplerPlanV4, payload, "plan_id")
    atomic_json(temporary / "G00C_BASE_SAMPLER_PLAN.json", plan.model_dump(mode="json"))
    return plan


def freeze(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"Refusing to replace existing Dev37 authority: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    bootstrap = output.parent / f".{output.name}.dev36-bootstrap"
    if bootstrap.exists():
        raise RuntimeError(f"Refusing stale bootstrap path: {bootstrap}.")
    bootstrap_args = argparse.Namespace(**vars(args))
    bootstrap_args.output = bootstrap
    try:
        dev36.freeze(bootstrap_args)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            excluded = {
                "G00C_D1_EXECUTION_AUTHORITY.json",
                "SOURCE_ACCESS_RECEIPT.json",
                "PROVENANCE_INDEX.json",
                "SHA256SUMS",
                "COMMITTED",
            }
            for source in bootstrap.rglob("*"):
                relative = source.relative_to(bootstrap)
                if source.is_dir():
                    (temporary / relative).mkdir(parents=True, exist_ok=True)
                elif relative.as_posix() not in excluded and not relative.as_posix().startswith(
                    "implementations/"
                ):
                    dev35._copy(source, temporary / relative)
            implementation_dir = temporary / "implementations"
            implementation_dir.mkdir(parents=True, exist_ok=True)
            # These semantic configuration bytes remain parents of the Dev34
            # selection freeze. Dev37 replaces executable verifier code, but
            # must preserve the exact non-code parents from the bootstrap
            # authority so the complete freeze can be independently reopened.
            for name in ("refit-model-config.json", "common-support-prior.json"):
                dev35._copy(bootstrap / "implementations" / name, implementation_dir / name)
            monitor_path = implementation_dir / "monitor.py"
            dev35._copy(args.monitor_implementation.resolve(), monitor_path)
            implementation_paths = {
                "feature_ranking": _copy_implementation(
                    repository,
                    implementation_dir,
                    "source-authority-v4.py",
                    "src/credo_count_sde_v4/store/g00c_source_v4.py",
                ),
                "source_authority": implementation_dir / "source-authority-v4.py",
                "hierarchy_derivation": implementation_dir / "source-authority-v4.py",
                "refit": _copy_implementation(
                    repository,
                    implementation_dir,
                    "selection.py",
                    "src/credo_count_sde_v4/store/g00c_selection.py",
                ),
                "sampler": _copy_implementation(
                    repository,
                    implementation_dir,
                    "sampler-v4.py",
                    "src/credo_count_sde_v4/store/g00c_sampler_v4.py",
                ),
                "support_auditor": implementation_dir / "sampler-v4.py",
                "monitor": monitor_path,
                "source_access_auditor": _copy_implementation(
                    repository,
                    implementation_dir,
                    "evidence-v4.py",
                    "src/credo_count_sde_v4/store/g00c_evidence_v4.py",
                ),
                "materialization_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "materialization-verifier-v4.py",
                    "src/credo_count_sde_v4/store/g00c_materialization_v4.py",
                ),
                "refit_replay_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "refit-replay-verifier-v4.py",
                    "src/credo_count_sde_v4/store/g00c_refit_replay_v4.py",
                ),
                "publication_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "publication-verifier-v4.py",
                    "src/credo_count_sde_v4/store/g00c_publication_v4.py",
                ),
                "restart_verifier": implementation_dir / "evidence-v4.py",
                "extension_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "extension-verifier-v4.py",
                    "src/credo_count_sde_v4/store/g00c_extension_v4.py",
                ),
                "execution_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "execution-verifier-v5.py",
                    "src/credo_count_sde_v4/store/g00c_v5.py",
                ),
                "decision_verifier": implementation_dir / "execution-verifier-v5.py",
            }
            selection = G00CSelectionFreezeContractV1.model_validate_json(
                (temporary / "G00C_D1_SELECTION_FREEZE.json").read_text()
            )
            schedule = G00CRefitSeedScheduleV1.model_validate_json(
                (temporary / "G00C_REFIT_SEED_SCHEDULE.json").read_text()
            )
            source_plane_root = args.source_root.resolve() / "01_SOURCE_PLANE_V2"
            binding, metadata_store = _source_binding(
                temporary,
                source_plane_root=source_plane_root,
                feature_index_path=args.feature_index.resolve(),
            )
            roles = pd.read_parquet(temporary / "D1_ROW_ROLES.parquet")
            training_rows = roles.loc[
                roles["role"].astype(str) == "training_fit", "row_id"
            ].to_numpy(dtype=np.int64)
            derived = derive_hierarchy_from_g00b_v4(metadata_store, training_rows)
            hierarchy_path = temporary / "D1_TRAINING_ROW_HIERARCHY.parquet"
            observed = pd.read_parquet(hierarchy_path)
            if not observed.equals(derived):
                raise RuntimeError("Dev37 independent hierarchy derivation differs from Dev36.")
            hierarchy_ref = dev35._artifact(
                temporary,
                hierarchy_path,
                "credo.g00c.d1-sampler-hierarchy",
                "application/vnd.apache.parquet",
            )
            hierarchy_receipt = dev35._identified(
                G00CHierarchyDerivationReceiptV4,
                {
                    "receipt_id": "0" * 64,
                    "source_binding_id": binding.binding_id,
                    "hierarchy": hierarchy_ref,
                    "row_count": len(derived),
                    "ordered_row_ids_sha256": _array_hash(derived["row_id"], "<i8"),
                    "source_index_sha256": _array_hash(derived["source_index"], "<i2"),
                    "target_code_sha256": _array_hash(derived["target_code"], "<i4"),
                    "guide_code_sha256": _array_hash(derived["guide_code"], "<i4"),
                    "is_control_sha256": _array_hash(derived["is_control"], "|b1"),
                },
                "receipt_id",
            )
            atomic_json(
                temporary / "G00C_HIERARCHY_DERIVATION_RECEIPT.json",
                hierarchy_receipt.model_dump(mode="json"),
            )
            plan = _sampler_plan(
                temporary,
                selection_freeze_id=selection.freeze_id,
                schedule=schedule,
                macro_updates=args.macro_updates,
            )
            environment = json.loads((temporary / "release/tested-environment.v1.json").read_text())
            role_annotation = G00CImplementationBindingV3.model_fields["role"].annotation
            expected_roles = tuple(role_annotation.__args__)
            implementations = tuple(
                G00CImplementationBindingV3(
                    role=cast(Any, role),
                    artifact=dev35._artifact(
                        temporary,
                        implementation_paths[role],
                        f"credo.g00c.dev37.implementation.{role}",
                        "text/x-python",
                    ),
                )
                for role in expected_roles
            )
            release_wheel = temporary / "release" / args.wheel.name
            release_sdist = temporary / "release" / args.normalized_sdist.name
            implementation = G00CImplementationAuthorityV3(
                dev37_code_commit=args.code_commit,
                wheel=dev35._artifact(
                    temporary,
                    release_wheel,
                    "credo.g00c.dev37-wheel",
                    "application/zip",
                ),
                normalized_sdist=dev35._artifact(
                    temporary,
                    release_sdist,
                    "credo.g00c.dev37-normalized-sdist",
                    "application/gzip",
                ),
                implementation_tree_sha256=args.implementation_tree_sha256,
                environment_lock=dev35._artifact(
                    temporary,
                    temporary / "release/tested-environment.v1.json",
                    "credo.g00c.tested-environment",
                    "application/json",
                ),
                environment_kind=environment["environment_kind"],
                execution_environment_digest=environment["execution_environment_digest"],
                implementations=implementations,
            )
            bootstrap_authority = json.loads(
                (bootstrap / "G00C_D1_EXECUTION_AUTHORITY.json").read_text()
            )
            authority_payload = {
                "authority_id": "0" * 64,
                "selection_freeze": dev35._artifact(
                    temporary,
                    temporary / "G00C_D1_SELECTION_FREEZE.json",
                    "credo.g00c.selection-freeze",
                    "application/json",
                ),
                "selection_freeze_id": selection.freeze_id,
                "row_role_freeze": dev35._artifact(
                    temporary,
                    temporary / "D1_ROW_ROLES.parquet",
                    "credo.g00c.d1-row-roles",
                    "application/vnd.apache.parquet",
                ),
                "row_roles": bootstrap_authority["row_roles"],
                "nested_training_row_order": dev35._artifact(
                    temporary,
                    temporary / "D1_NESTED_TRAINING_ROW_ORDER.parquet",
                    "credo.g00c.d1-training-order",
                    "application/vnd.apache.parquet",
                ),
                "nested_training_row_order_hash": bootstrap_authority[
                    "nested_training_row_order_hash"
                ],
                "feature_reference_rows": dev35._artifact(
                    temporary,
                    temporary / "D1_FEATURE_REFERENCE_ROWS.parquet",
                    "credo.g00c.d1-feature-reference",
                    "application/vnd.apache.parquet",
                ),
                "feature_reference_rows_hash": bootstrap_authority["feature_reference_rows_hash"],
                "sampler_row_hierarchy": hierarchy_ref,
                "sampler_hierarchy_rows": len(derived),
                "hierarchy_derivation_receipt": dev35._artifact(
                    temporary,
                    temporary / "G00C_HIERARCHY_DERIVATION_RECEIPT.json",
                    "credo.g00c.dev37.hierarchy-derivation-receipt",
                    "application/json",
                ),
                "source_plane": binding,
                "seed_schedule": dev35._artifact(
                    temporary,
                    temporary / "G00C_REFIT_SEED_SCHEDULE.json",
                    "credo.g00c.refit-seed-schedule",
                    "application/json",
                ),
                "seed_schedule_id": schedule.schedule_id,
                "base_sampler_plan": dev35._artifact(
                    temporary,
                    temporary / "G00C_BASE_SAMPLER_PLAN.json",
                    "credo.g00c.dev37.base-sampler-plan",
                    "application/json",
                ),
                "base_sampler_plan_id": plan.plan_id,
                "base_sampler_expected_trace_rows": plan.expected_total_trace_rows,
                "common_support_prior": bootstrap_authority["common_support_prior"],
                "base_support_audit_contract": dev35._artifact(
                    temporary,
                    temporary / "G00C_BASE_SUPPORT_AUDIT_CONTRACT.json",
                    "credo.g00c.support-audit-contract",
                    "application/json",
                ),
                "implementation": implementation,
                "fresh_attempt_id": args.attempt_id,
                "publication_root_uri": output.name,
            }
            authority = dev35._identified(
                G00CD1ExecutionAuthorityFreezeV3, authority_payload, "authority_id"
            )
            atomic_json(
                temporary / "G00C_D1_EXECUTION_AUTHORITY.json",
                authority.model_dump(mode="json"),
            )
            atomic_json(
                temporary / "SOURCE_ACCESS_RECEIPT.json",
                {
                    "biological_claims": False,
                    "datasets_read": [
                        "D1_ROW_ROLES",
                        "GUIDE_TARGET_CROSSWALK metadata",
                        "canonical FEATURE_INDEX metadata",
                        "guide_codes_sorted",
                        "guide_ids",
                        "row_ids_sorted",
                        "source_indices_sorted",
                        "target_codes_sorted",
                    ],
                    "expression_values_accessed": False,
                    "raw_source_files_opened": False,
                    "sampler_plan_entries": len(plan.entries),
                    "sampler_plan_expected_trace_rows": plan.expected_total_trace_rows,
                    "schema_version": 3,
                    "status": "metadata_only_preaccess_freeze",
                    "training_or_model_fitting_performed": False,
                },
            )
            atomic_json(
                temporary / "PROVENANCE_INDEX.json",
                {
                    "attempt_id": args.attempt_id,
                    "authority_id": authority.authority_id,
                    "biological_claims": False,
                    "code_commit": args.code_commit,
                    "expression_access": False,
                    "g00c_execution": "not_run",
                    "g00d": "blocked",
                    "hierarchy_derivation_receipt_id": hierarchy_receipt.receipt_id,
                    "package_version": "4.0.0.dev37",
                    "resource_ceiling_bytes": 64 * 1024**3,
                    "sampler_plan_id": plan.plan_id,
                    "schema_version": 3,
                    "seed_schedule_id": authority.seed_schedule_id,
                    "selection_freeze_id": authority.selection_freeze_id,
                    "source_binding_id": binding.binding_id,
                    "status": "finalized_preaccess_not_executed",
                },
            )
            verify_g00c_d1_freeze_v1(temporary, authority)
            (temporary / "COMMITTED").write_text("finalized_preaccess_not_executed\n")
            files = sorted(
                path
                for path in temporary.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            )
            (temporary / "SHA256SUMS").write_text(
                "".join(
                    f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
                    for path in files
                )
            )
            os.replace(temporary, output)
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if bootstrap.exists():
            shutil.rmtree(bootstrap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--feature-index", required=True, type=Path)
    parser.add_argument("--dev33-canary-root", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--normalized-sdist", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--monitor-implementation", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--implementation-tree-sha256", required=True)
    parser.add_argument("--macro-updates", type=int, default=8)
    parser.add_argument("--attempt-id", default="G00C_D1_AUTHORITY_DEV37_A3")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.macro_updates < 2 or args.macro_updates % 2:
        raise RuntimeError("Dev37 macro updates must be an even integer of at least two.")
    freeze(args)


if __name__ == "__main__":
    main()
