"""Publish the concrete, expression-free Dev35 D1 G00C execution authority."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import TypeAdapter

from credo_count_sde_v4.canonical import (
    atomic_json,
    canonical_json_bytes,
    contract_id,
    sha256_bytes,
    sha256_file,
)
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    FoldRowRoleRecord,
    G00CCanaryPrerequisiteV1,
    G00CCommonSupportFeatureMetricV1,
    G00CCommonSupportPriorV2,
    G00CD1ExecutionAuthorityFreezeV1,
    G00CFeatureRankingFreezeV1,
    G00CImplementationAuthorityV1,
    G00CImplementationBindingV1,
    G00CMonitorFreezeV1,
    G00CParentBindingV1,
    G00CPublicationFreezeV1,
    G00CRefitFreezeV1,
    G00CSelectionFreezeContractV1,
    G00CSelectionMarginFreezeV1,
    G00CSerialSelectionFreezeV1,
    G00CSupportAuditContractV2,
    G00CSupportAuditFreezeV1,
    StrictModel,
)
from credo_count_sde_v4.store import derive_refit_seed_schedule

ModelT = TypeVar("ModelT", bound=StrictModel)
ROLE_NAMES = (
    "training_fit",
    "training_validation",
    "heldout_source_query",
    "protected_heldout_stimulated",
)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "0" * 64
    normalized = model.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in model.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _artifact(root: Path, path: Path, schema_id: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.relative_to(root).as_posix(),
    )


def _splitmix64(values: np.ndarray, namespace: int) -> np.ndarray:
    state = np.asarray(values, dtype=np.uint64) ^ np.uint64(namespace)
    state = state + np.uint64(0x9E3779B97F4A7C15)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return state ^ (state >> np.uint64(31))


def _row_set_hash(values: np.ndarray) -> str:
    ordered = np.sort(np.asarray(values, dtype="<i8"), kind="stable")
    return sha256_bytes(ordered.tobytes(order="C"))


def _ordered_row_hash(values: np.ndarray) -> str:
    return sha256_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Authority input must be one regular file: {source}.")
    shutil.copyfile(source, destination)


def _publication() -> G00CPublicationFreezeV1:
    return G00CPublicationFreezeV1(
        always_required_artifacts=(
            "G00C_REFIT_SEED_SCHEDULE.json",
            "ROW_HASHES.json",
            "FEATURE_SELECTION_CURVE.parquet",
            "CELL_SELECTION_CURVE.parquet",
            "REFIT_PROVENANCE.parquet",
            "REPLAY_AUDIT.json",
            "SUPPORT_AUDIT.parquet",
            "SAMPLER_RESTART.json",
            "DECISION_RECEIPT.json",
            "artifacts.json",
            "COMMITTED",
            "SHA256SUMS",
        ),
        pass_only_required_artifacts=(
            "SELECTED_FEATURES.parquet",
            "SELECTED_ROWS.parquet",
            "compact.h5",
            "puroR-sidecar.h5",
            "PHYSICAL_RUNS.parquet",
            "COMPACT_VERIFICATION.json",
            "WRITER_RESTART.json",
        ),
        extension_required_artifacts=("BASE_GRID_STOP_RECEIPT.json",),
        fail_no_saturation_required_artifacts=(
            "BASE_GRID_STOP_RECEIPT.json",
            "EXTENSION_RESULT.json",
        ),
    )


def _write_row_authority(
    temporary: Path,
    locator_path: Path,
    source_manifest: dict[str, Any],
) -> tuple[tuple[FoldRowRoleRecord, ...], str, str]:
    sources = source_manifest["sources"]
    heldout_source_indices = {
        index
        for index, source in enumerate(sources)
        if source["donor_id"] == "D1" and source["checkpoint"] == "Rest"
    }
    protected_indices = {
        index
        for index, source in enumerate(sources)
        if source["donor_id"] == "D1" and source["checkpoint"] != "Rest"
    }
    if heldout_source_indices != {0} or protected_indices != {1, 2}:
        raise RuntimeError("The exact D1 source/checkpoint index contract changed.")
    with h5py.File(locator_path, "r") as handle:
        if handle.attrs.get("phase") != "FINALIZED" or int(handle.attrs["schema_version"]) != 2:
            raise RuntimeError("The D1 authority requires the finalized V2 row locator.")
        row_ids = np.asarray(handle["row_ids_sorted"], dtype=np.int64)
        source_indices = np.asarray(handle["source_indices_sorted"], dtype=np.int16)
    if len(row_ids) != 21_996_842 or len(np.unique(row_ids)) != len(row_ids):
        raise RuntimeError("The D1 row locator does not expose the frozen eligible universe.")

    role_codes = np.zeros(len(row_ids), dtype=np.int8)
    role_codes[np.isin(source_indices, tuple(heldout_source_indices))] = 2
    role_codes[np.isin(source_indices, tuple(protected_indices))] = 3
    training_indices = np.flatnonzero(role_codes == 0)
    if len(training_indices) != 16_990_208:
        raise RuntimeError("The D2-D4 training universe changed before D1 freeze.")
    validation_priority = _splitmix64(row_ids[training_indices], 0xD135000000000001)
    validation_local = np.argpartition(validation_priority, 65_536 - 1)[:65_536]
    validation_indices = training_indices[validation_local]
    role_codes[validation_indices] = 1

    role_dictionary = pa.array(ROLE_NAMES, type=pa.string())
    role_array = pa.DictionaryArray.from_arrays(pa.array(role_codes), role_dictionary)
    role_table = pa.table({"row_id": pa.array(row_ids), "role": role_array})
    pq.write_table(
        role_table,
        temporary / "D1_ROW_ROLES.parquet",
        compression="zstd",
        version="2.6",
        write_statistics=True,
    )
    records = tuple(
        FoldRowRoleRecord(
            role=role,
            rows=int(np.count_nonzero(role_codes == code)),
            row_ids_hash=_row_set_hash(row_ids[role_codes == code]),
        )
        for code, role in enumerate(ROLE_NAMES)
    )
    if tuple(record.rows for record in records) != (
        16_924_672,
        65_536,
        1_752_037,
        3_254_597,
    ):
        raise RuntimeError("The concrete D1 row-role counts changed.")

    fit_rows = row_ids[role_codes == 0]
    order_priority = _splitmix64(fit_rows, 0xD135000000000002)
    selected_local = np.argpartition(order_priority, 2_000_000 - 1)[:2_000_000]
    selected_rows = fit_rows[selected_local]
    selected_priority = order_priority[selected_local]
    ordered_rows = selected_rows[np.lexsort((selected_rows, selected_priority))]
    order = pd.DataFrame({"rank": np.arange(1, 2_000_001, dtype=np.int64), "row_id": ordered_rows})
    order.to_parquet(temporary / "D1_NESTED_TRAINING_ROW_ORDER.parquet", index=False)
    order.iloc[:1_000_000].to_parquet(temporary / "D1_FEATURE_REFERENCE_ROWS.parquet", index=False)
    return records, _ordered_row_hash(ordered_rows), _ordered_row_hash(ordered_rows[:1_000_000])


def _parent_binding(
    temporary: Path,
    role: str,
    source: Path,
    identity_field: str,
) -> G00CParentBindingV1:
    destination = temporary / "parents" / f"{role}.json"
    _copy(source, destination)
    payload = json.loads(destination.read_text())
    return G00CParentBindingV1(
        role=role,
        identity_field=identity_field,
        identity_value=str(payload[identity_field]),
        artifact=_artifact(temporary, destination, f"credo.g00c.parent.{role}", "application/json"),
    )


def freeze(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing authority root: {output}.")
    if len(args.code_commit) != 40 or len(args.implementation_tree_sha256) != 64:
        raise RuntimeError("Dev35 release identities must be full lowercase digests.")
    parent_root = source_root / "01_SOURCE_PLANE_V2"
    store_root = parent_root / "G00B_VIRTUAL_CANONICAL_STORE_V2"
    locator_path = store_root / "row-locator.h5"
    source_manifest_path = store_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        parent_specs = (
            ("g00a_v2", parent_root / "G00A-v2.json", "authority_id"),
            ("g00b_v2", source_manifest_path, "virtual_store_id"),
            (
                "source_plane_amendment",
                store_root / "G00_SOURCE_PLANE_V2_AMENDMENT.json",
                "amendment_id",
            ),
            (
                "source_plane_amendment_receipt",
                parent_root / "G00_SOURCE_PLANE_V2_AMENDMENT_RECEIPT.json",
                "receipt_id",
            ),
            (
                "source_plane_decision_receipt",
                parent_root / "G00_SOURCE_PLANE_V2_DECISION_RECEIPT.json",
                "receipt_id",
            ),
            (
                "b0_a2_execution_amendment",
                parent_root / "B0_A2_EXECUTION_AMENDMENT.json",
                "amendment_id",
            ),
        )
        parents = tuple(
            _parent_binding(temporary, role, source, identity)
            for role, source, identity in parent_specs
        )
        row_roles, training_order_hash, reference_rows_hash = _write_row_authority(
            temporary, locator_path, source_manifest
        )

        release_dir = temporary / "release"
        for source, name in (
            (args.wheel.resolve(), args.wheel.name),
            (args.normalized_sdist.resolve(), args.normalized_sdist.name),
            (args.environment_lock.resolve(), "tested-environment.v1.json"),
        ):
            _copy(source, release_dir / name)
        implementation_dir = temporary / "implementations"
        implementation_sources = {
            "selection.py": root / "src/credo_count_sde_v4/store/g00c_selection.py",
            "verifier.py": root / "src/credo_count_sde_v4/store/g00c_v3.py",
            "authority-freezer.py": Path(__file__).resolve(),
            "monitor.py": args.monitor_implementation.resolve(),
        }
        for name, source in implementation_sources.items():
            _copy(source, implementation_dir / name)
        model_config = {
            "schema_version": 1,
            "model_family": "checkpoint_conditioned_multinomial_intercept_v1",
            "count_thinning_method": "paired_binomial_half_count_v1",
            "pseudocount_per_reference_feature": 0.5,
            "common_reference_feature_count": 4096,
            "optimizer": "closed_form_no_optimizer",
            "maximum_updates": 0,
        }
        atomic_json(implementation_dir / "refit-model-config.json", model_config)
        common_prior = G00CCommonSupportPriorV2()
        atomic_json(
            implementation_dir / "common-support-prior.json",
            common_prior.model_dump(mode="json"),
        )

        namespace_payload = {
            "attempt_id": args.attempt_id,
            "parents": [binding.model_dump(mode="json") for binding in parents],
            "row_roles": [record.model_dump(mode="json") for record in row_roles],
            "training_order_hash": training_order_hash,
            "reference_rows_hash": reference_rows_hash,
            "code_commit": args.code_commit,
            "wheel_sha256": sha256_file(args.wheel),
            "sdist_sha256": sha256_file(args.normalized_sdist),
            "implementation_tree_sha256": args.implementation_tree_sha256,
            "environment_lock_sha256": sha256_file(args.environment_lock),
        }
        namespace_id = sha256_bytes(canonical_json_bytes(namespace_payload))
        schedule = derive_refit_seed_schedule(namespace_id)
        atomic_json(
            temporary / "G00C_REFIT_SEED_SCHEDULE.json",
            schedule.model_dump(mode="json"),
        )

        support_payload = {
            "contract_id": "0" * 64,
            "stage": "base",
            "feature_candidate_counts": (256, 512, 1024, 2048, 4096),
            "cell_candidate_counts": (50_000, 100_000, 250_000, 500_000, 1_000_000),
        }
        support = _identified(G00CSupportAuditContractV2, support_payload, "contract_id")
        atomic_json(
            temporary / "G00C_BASE_SUPPORT_AUDIT_CONTRACT.json",
            support.model_dump(mode="json"),
        )
        schedule_ref = _artifact(
            temporary,
            temporary / "G00C_REFIT_SEED_SCHEDULE.json",
            "credo.g00c.refit-seed-schedule",
            "application/json",
        )
        row_role_ref = _artifact(
            temporary,
            temporary / "D1_ROW_ROLES.parquet",
            "credo.g00c.d1-row-roles",
            "application/vnd.apache.parquet",
        )
        common_prior_ref = _artifact(
            temporary,
            implementation_dir / "common-support-prior.json",
            "credo.g00c.common-support-prior",
            "application/json",
        )
        selection_impl_sha = sha256_file(implementation_dir / "selection.py")
        monitor_impl_sha = sha256_file(implementation_dir / "monitor.py")

        canary_root = args.dev33_canary_root.resolve()
        canary_contract = json.loads((canary_root / "contract.json").read_text())
        canary_verification = json.loads(
            (canary_root / "INDEPENDENT_CANARY_VERIFICATION.json").read_text()
        )
        canary_audit = json.loads((canary_root / "FINAL_CANARY_AUDIT.json").read_text())
        canary_archive = temporary / "parents/DEV33B_FINAL_CANARY_ARTIFACTS.sha256"
        _copy(canary_root / "FINAL_CANARY_ARTIFACTS.sha256", canary_archive)
        canary = G00CCanaryPrerequisiteV1(
            dev33_code_commit="d21c609771f3267a0fdd03d71ccf85463c6fcd36",
            dev33_wheel_sha256="d87b592f6675de0a615571b17e8aeaf165749595a1fe124df0f04e5f60902031",
            canary_contract_id=canary_contract["contract_id"],
            execution_record_commit="86d38a28c570cad9401693e4e4404906c15ddbb5",
            independent_verification_id=canary_verification["verification_id"],
            final_audit_id=canary_audit["audit_id"],
            authority_archive_uri=canary_archive.relative_to(temporary).as_posix(),
            authority_archive_sha256=sha256_file(canary_archive),
        )
        freeze_payload = {
            "freeze_id": "0" * 64,
            "parent_bindings": parents,
            "dev33_canary": canary,
            "row_roles": row_roles,
            "row_role_freeze": row_role_ref,
            "training_scale_row_order_hash": training_order_hash,
            "seed_schedule": schedule,
            "seed_schedule_artifact": schedule_ref,
            "feature_ranking": G00CFeatureRankingFreezeV1(
                tie_breaks=("total_umi_desc", "detection_count_desc", "feature_id_utf8_asc"),
                fit_reference_rows_hash=reference_rows_hash,
                validation_rows_hash=next(
                    record.row_ids_hash
                    for record in row_roles
                    if record.role == "training_validation"
                ),
            ),
            "common_support_metric": G00CCommonSupportFeatureMetricV1(
                residual_frequency_artifact=common_prior_ref,
                residual_frequency_fit_rows_hash=reference_rows_hash,
            ),
            "serial_selection": G00CSerialSelectionFreezeV1(
                frozen_nested_training_row_order_hash=training_order_hash,
                feature_selection_reference_rows_hash=reference_rows_hash,
            ),
            "selection_margins": G00CSelectionMarginFreezeV1(),
            "refits": G00CRefitFreezeV1(
                seed_schedule_id=schedule.schedule_id,
                seed_schedule_artifact=schedule_ref,
                model_config_hash=sha256_file(implementation_dir / "refit-model-config.json"),
                replay_implementation_sha256=selection_impl_sha,
            ),
            "support_audit": G00CSupportAuditFreezeV1(
                dimensions=(
                    "donor_checkpoint",
                    "target",
                    "guide",
                    "control_vs_targeting",
                    "sampler_stratum",
                )
            ),
            "monitor": G00CMonitorFreezeV1(
                implementation_sha256=monitor_impl_sha,
                polling_interval_milliseconds=50,
                maximum_unreadable_samples=10,
                maximum_unreadable_fraction=0.001,
                maximum_consecutive_unreadable_samples=3,
                maximum_temporal_gap_milliseconds=500,
                maximum_process_tree_rss_bytes=64 * 1024**3,
            ),
            "publication": _publication(),
        }
        selection_freeze = _identified(G00CSelectionFreezeContractV1, freeze_payload, "freeze_id")
        atomic_json(
            temporary / "G00C_D1_SELECTION_FREEZE.json",
            selection_freeze.model_dump(mode="json"),
        )

        binding_files = {
            "feature_ranking": implementation_dir / "selection.py",
            "residual_frequency": implementation_dir / "selection.py",
            "refit": implementation_dir / "selection.py",
            "sampler": implementation_dir / "authority-freezer.py",
            "monitor": implementation_dir / "monitor.py",
            "execution_verifier": implementation_dir / "verifier.py",
            "decision_verifier": implementation_dir / "verifier.py",
        }
        implementations = tuple(
            G00CImplementationBindingV1(
                role=role,
                artifact=_artifact(
                    temporary,
                    path,
                    f"credo.g00c.implementation.{role}",
                    "text/x-python",
                ),
            )
            for role, path in binding_files.items()
        )
        environment = json.loads((release_dir / "tested-environment.v1.json").read_text())
        implementation = G00CImplementationAuthorityV1(
            dev35_code_commit=args.code_commit,
            wheel=_artifact(
                temporary,
                release_dir / args.wheel.name,
                "credo.g00c.dev35-wheel",
                "application/zip",
            ),
            normalized_sdist=_artifact(
                temporary,
                release_dir / args.normalized_sdist.name,
                "credo.g00c.dev35-normalized-sdist",
                "application/gzip",
            ),
            implementation_tree_sha256=args.implementation_tree_sha256,
            environment_lock=_artifact(
                temporary,
                release_dir / "tested-environment.v1.json",
                "credo.g00c.tested-environment",
                "application/json",
            ),
            environment_kind=environment["environment_kind"],
            execution_environment_digest=environment["execution_environment_digest"],
            implementations=implementations,
        )
        authority_payload = {
            "authority_id": "0" * 64,
            "selection_freeze": _artifact(
                temporary,
                temporary / "G00C_D1_SELECTION_FREEZE.json",
                "credo.g00c.selection-freeze",
                "application/json",
            ),
            "selection_freeze_id": selection_freeze.freeze_id,
            "row_role_freeze": row_role_ref,
            "row_roles": row_roles,
            "nested_training_row_order": _artifact(
                temporary,
                temporary / "D1_NESTED_TRAINING_ROW_ORDER.parquet",
                "credo.g00c.d1-training-order",
                "application/vnd.apache.parquet",
            ),
            "nested_training_row_order_hash": training_order_hash,
            "feature_reference_rows": _artifact(
                temporary,
                temporary / "D1_FEATURE_REFERENCE_ROWS.parquet",
                "credo.g00c.d1-feature-reference",
                "application/vnd.apache.parquet",
            ),
            "feature_reference_rows_hash": reference_rows_hash,
            "seed_schedule": schedule_ref,
            "seed_schedule_id": schedule.schedule_id,
            "common_support_prior": common_prior,
            "base_support_audit_contract": _artifact(
                temporary,
                temporary / "G00C_BASE_SUPPORT_AUDIT_CONTRACT.json",
                "credo.g00c.support-audit-contract",
                "application/json",
            ),
            "implementation": implementation,
            "fresh_attempt_id": args.attempt_id,
            "publication_root_uri": output.name,
        }
        authority = _identified(G00CD1ExecutionAuthorityFreezeV1, authority_payload, "authority_id")
        atomic_json(
            temporary / "G00C_D1_EXECUTION_AUTHORITY.json",
            authority.model_dump(mode="json"),
        )
        atomic_json(
            temporary / "SOURCE_ACCESS_RECEIPT.json",
            {
                "schema_version": 1,
                "status": "metadata_only_preaccess_freeze",
                "row_locator_sha256": sha256_file(locator_path),
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "datasets_read": ["row_ids_sorted", "source_indices_sorted"],
                "expression_values_accessed": False,
                "raw_source_files_opened": False,
                "training_or_model_fitting_performed": False,
                "biological_claims": False,
            },
        )
        atomic_json(
            temporary / "PROVENANCE_INDEX.json",
            {
                "schema_version": 1,
                "status": "finalized_preaccess_not_executed",
                "authority_id": authority.authority_id,
                "selection_freeze_id": selection_freeze.freeze_id,
                "seed_schedule_id": schedule.schedule_id,
                "attempt_id": args.attempt_id,
                "package_version": "4.0.0.dev35",
                "code_commit": args.code_commit,
                "expression_access": False,
                "g00c_execution": "not_run",
                "g00d": "blocked",
                "biological_claims": False,
            },
        )
        files = sorted(
            path for path in temporary.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n" for path in files
            )
        )
        (temporary / "COMMITTED").write_text("finalized_preaccess_not_executed\n")
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


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
    parser.add_argument("--attempt-id", default="G00C_D1_AUTHORITY_DEV35_A1")
    parser.add_argument("--output", required=True, type=Path)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
