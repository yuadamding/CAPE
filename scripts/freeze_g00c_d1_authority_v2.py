"""Publish the metadata-only Dev36 D1 execution authority and sampler hierarchy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import freeze_g00c_d1_authority as dev35
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from credo_count_sde_v4.canonical import atomic_json, sha256_file
from credo_count_sde_v4.contracts import (
    G00CD1ExecutionAuthorityFreezeV2,
    G00CImplementationAuthorityV2,
    G00CImplementationBindingV2,
    G00CSelectionFreezeContractV1,
)
from credo_count_sde_v4.store.g00c_v3 import verify_g00c_d1_freeze_v1


def _write_hierarchy(authority_root: Path, locator_path: Path, crosswalk_path: Path) -> int:
    roles = pd.read_parquet(authority_root / "D1_ROW_ROLES.parquet")
    if tuple(roles.columns) != ("row_id", "role"):
        raise RuntimeError("The Dev36 hierarchy requires the exact D1 role schema.")
    training_ids = roles.loc[roles["role"].astype(str) == "training_fit", "row_id"].to_numpy(
        dtype=np.int64
    )
    with h5py.File(locator_path, "r") as handle:
        row_ids = np.asarray(handle["row_ids_sorted"][:], dtype=np.int64)
        positions = np.searchsorted(row_ids, training_ids)
        if np.any(positions >= len(row_ids)) or not np.array_equal(
            row_ids[positions], training_ids
        ):
            raise RuntimeError("Training-fit rows are absent from the finalized V2 locator.")
        source_indices = np.asarray(handle["source_indices_sorted"][positions], dtype=np.int16)
        guide_codes = np.asarray(handle["guide_codes_sorted"][positions], dtype=np.int32)
        target_codes = np.asarray(handle["target_codes_sorted"][positions], dtype=np.int32)
        guide_ids = handle["guide_ids"].asstr()[:]
    crosswalk = pd.read_parquet(crosswalk_path)
    if tuple(crosswalk.columns) != (
        "guide_id",
        "target_id",
        "is_control",
        "raw_guide_group",
        "eligible_cell_count",
        "observed_source_count",
    ):
        raise RuntimeError("The accepted guide/control crosswalk schema changed.")
    control_by_guide = crosswalk.set_index("guide_id")["is_control"].to_dict()
    if set(control_by_guide) != set(guide_ids.tolist()):
        raise RuntimeError("The crosswalk and locator guide catalogs differ.")
    control_codes = np.asarray(
        [bool(control_by_guide[str(guide_id)]) for guide_id in guide_ids], dtype=bool
    )
    table = pa.table(
        {
            "row_id": pa.array(training_ids),
            "source_index": pa.array(source_indices),
            "target_code": pa.array(target_codes),
            "guide_code": pa.array(guide_codes),
            "is_control": pa.array(control_codes[guide_codes]),
        }
    )
    destination = authority_root / "D1_TRAINING_ROW_HIERARCHY.parquet"
    pq.write_table(
        table,
        destination,
        compression="zstd",
        version="2.6",
        write_statistics=True,
        row_group_size=262_144,
    )
    return len(training_ids)


def _copy_implementation(root: Path, destination: Path, name: str, source: Path) -> Path:
    target = destination / name
    dev35._copy(source, target)
    return target


def freeze(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"Refusing to replace existing Dev36 authority: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    bootstrap = output.parent / f".{output.name}.dev35-bootstrap"
    if bootstrap.exists():
        raise RuntimeError(f"Refusing stale bootstrap path: {bootstrap}.")
    bootstrap_args = argparse.Namespace(**vars(args))
    bootstrap_args.output = bootstrap
    try:
        dev35.freeze(bootstrap_args)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            for source in bootstrap.rglob("*"):
                relative = source.relative_to(bootstrap)
                if source.is_dir():
                    (temporary / relative).mkdir(parents=True, exist_ok=True)
                elif relative.as_posix() not in {
                    "G00C_D1_EXECUTION_AUTHORITY.json",
                    "SOURCE_ACCESS_RECEIPT.json",
                    "PROVENANCE_INDEX.json",
                    "SHA256SUMS",
                    "COMMITTED",
                } and not relative.as_posix().startswith("implementations/"):
                    dev35._copy(source, temporary / relative)
            implementation_dir = temporary / "implementations"
            implementation_dir.mkdir(parents=True, exist_ok=True)
            # These bytes are also parents of the unchanged Dev34 selection freeze.
            for name in (
                "selection.py",
                "monitor.py",
                "refit-model-config.json",
                "common-support-prior.json",
            ):
                dev35._copy(bootstrap / "implementations" / name, implementation_dir / name)
            implementation_paths = {
                "feature_ranking": implementation_dir / "selection.py",
                "residual_frequency": implementation_dir / "selection.py",
                "refit": implementation_dir / "selection.py",
                "sampler": _copy_implementation(
                    repository,
                    implementation_dir,
                    "sampler-v3.py",
                    repository / "src/credo_count_sde_v4/store/g00c_sampler_v3.py",
                ),
                "support_auditor": implementation_dir / "sampler-v3.py",
                "monitor": implementation_dir / "monitor.py",
                "materialization_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "materialization-verifier-v3.py",
                    repository / "src/credo_count_sde_v4/store/g00c_materialization_v3.py",
                ),
                "refit_replay_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "refit-replay-verifier-v3.py",
                    repository / "src/credo_count_sde_v4/store/g00c_refit_replay_v3.py",
                ),
                "publication_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "publication-verifier-v3.py",
                    repository / "src/credo_count_sde_v4/store/g00c_publication_v3.py",
                ),
                "execution_verifier": _copy_implementation(
                    repository,
                    implementation_dir,
                    "execution-verifier-v4.py",
                    repository / "src/credo_count_sde_v4/store/g00c_v4.py",
                ),
                "decision_verifier": implementation_dir / "execution-verifier-v4.py",
            }
            source_root = args.source_root.resolve() / "01_SOURCE_PLANE_V2"
            store_root = source_root / "G00B_VIRTUAL_CANONICAL_STORE_V2"
            locator = store_root / "row-locator.h5"
            crosswalk = store_root / "GUIDE_TARGET_CROSSWALK.parquet"
            hierarchy_rows = _write_hierarchy(temporary, locator, crosswalk)
            selection = G00CSelectionFreezeContractV1.model_validate_json(
                (temporary / "G00C_D1_SELECTION_FREEZE.json").read_text()
            )
            environment = json.loads((temporary / "release/tested-environment.v1.json").read_text())
            release_wheel = temporary / "release" / args.wheel.name
            release_sdist = temporary / "release" / args.normalized_sdist.name
            implementations = tuple(
                G00CImplementationBindingV2(
                    role=cast(Any, role),
                    artifact=dev35._artifact(
                        temporary,
                        path,
                        f"credo.g00c.dev36.implementation.{role}",
                        "text/x-python",
                    ),
                )
                for role, path in implementation_paths.items()
            )
            implementation = G00CImplementationAuthorityV2(
                dev36_code_commit=args.code_commit,
                wheel=dev35._artifact(
                    temporary, release_wheel, "credo.g00c.dev36-wheel", "application/zip"
                ),
                normalized_sdist=dev35._artifact(
                    temporary,
                    release_sdist,
                    "credo.g00c.dev36-normalized-sdist",
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
                "sampler_row_hierarchy": dev35._artifact(
                    temporary,
                    temporary / "D1_TRAINING_ROW_HIERARCHY.parquet",
                    "credo.g00c.d1-sampler-hierarchy",
                    "application/vnd.apache.parquet",
                ),
                "sampler_hierarchy_rows": hierarchy_rows,
                "seed_schedule": dev35._artifact(
                    temporary,
                    temporary / "G00C_REFIT_SEED_SCHEDULE.json",
                    "credo.g00c.refit-seed-schedule",
                    "application/json",
                ),
                "seed_schedule_id": bootstrap_authority["seed_schedule_id"],
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
                G00CD1ExecutionAuthorityFreezeV2, authority_payload, "authority_id"
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
                        "guide_codes_sorted",
                        "guide_ids",
                        "row_ids_sorted",
                        "source_indices_sorted",
                        "target_codes_sorted",
                        "GUIDE_TARGET_CROSSWALK metadata",
                    ],
                    "expression_values_accessed": False,
                    "guide_target_crosswalk_sha256": sha256_file(crosswalk),
                    "raw_source_files_opened": False,
                    "row_locator_sha256": sha256_file(locator),
                    "schema_version": 2,
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
                    "package_version": "4.0.0.dev36",
                    "resource_ceiling_bytes": 64 * 1024**3,
                    "resource_ceiling_semantics": (
                        "claim_bearing_full_grid_process_tree_rss_ceiling"
                    ),
                    "schema_version": 2,
                    "seed_schedule_id": authority.seed_schedule_id,
                    "selection_freeze_id": authority.selection_freeze_id,
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
            parent_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if bootstrap.exists():
            shutil.rmtree(bootstrap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dev33-canary-root", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--normalized-sdist", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--monitor-implementation", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--implementation-tree-sha256", required=True)
    parser.add_argument("--attempt-id", default="G00C_D1_AUTHORITY_DEV36_A2")
    parser.add_argument("--output", required=True, type=Path)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
