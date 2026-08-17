from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from credo_count_sde_v4.canonical import contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    G00DParityGateContract,
    G00DParityGateEvidence,
    IntegratedLoaderQualificationContractV2,
    IntegratedLoaderQualificationReceiptV2,
    StrictModel,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store import (
    validate_integrated_loader_qualification,
    verify_integrated_loader_qualification,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized = model.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in model.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _artifact(path: Path, root: Path) -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.g00d.artifact",
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=(
            "application/json" if path.suffix == ".json" else "application/vnd.apache.parquet"
        ),
        relative_uri=path.relative_to(root).as_posix(),
    )


def _write_evidence(
    root: Path,
) -> tuple[IntegratedLoaderQualificationContractV2, IntegratedLoaderQualificationReceiptV2]:
    gates = (
        ("row_ids", "exact_hash"),
        ("raw_counts", "exact_hash"),
        ("sample_weights", "exact_or_numerical"),
        ("thinning_rng", "exact_hash"),
        ("loss", "numerical_tolerance"),
        ("gradient", "numerical_tolerance"),
        ("parameter", "numerical_tolerance"),
        ("interrupted_resume", "exact_hash"),
    )
    contract = _identified(
        IntegratedLoaderQualificationContractV2,
        {
            "schema_version": 2,
            "fold_view_id": "fold-D4",
            "expected_gpu_name": "NVIDIA H100 80GB HBM3",
            "expected_cuda_version": "13.0",
            "expected_torch_version": "2.13.0+cu130",
            "expected_container_digest": "a" * 64,
            "worker_count": 8,
            "cpu_count": 32,
            "storage_authority_hash": "b" * 64,
            "prefetch_depth": 4,
            "warmup_updates": 50,
            "measured_updates": 500,
            "cache_policy": "bounded_lru_v1",
            "measurement_protocol_sha256": "c" * 64,
            "telemetry_interval_seconds": 0.1,
            "maximum_p95_batch_ready_seconds": 0.05,
            "maximum_loader_rss_bytes": 16_000_000_000,
            "maximum_process_loader_rss_bytes": 10_000_000_000,
            "maximum_aggregate_worker_rss_bytes": 16_000_000_000,
            "maximum_open_shards": 8,
            "maximum_open_file_handles": 128,
            "maximum_rss_slope_upper_bytes_per_second": 1_000.0,
            "maximum_rss_excursion_fraction": 0.10,
            "parity_gates": [
                G00DParityGateContract(gate=gate, comparison_mode=mode).model_dump(mode="json")
                for gate, mode in gates
            ],
        },
        "qualification_contract_id",
    )
    measurement_payload = {
        "measurement_protocol_sha256": contract.measurement_protocol_sha256,
        "gpu_name": contract.expected_gpu_name,
        "gpu_uuid": "GPU-0123",
        "gpu_count": 1,
        "cuda_version": contract.expected_cuda_version,
        "torch_version": contract.expected_torch_version,
        "container_digest": contract.expected_container_digest,
        "worker_count": contract.worker_count,
        "cpu_count": contract.cpu_count,
        "storage_authority_hash": contract.storage_authority_hash,
        "microbatch_cells": contract.microbatch_cells,
        "microbatches_per_update": contract.microbatches_per_update,
        "macrobatch_cells": contract.macrobatch_cells,
        "prefetch_depth": contract.prefetch_depth,
        "warmup_updates": contract.warmup_updates,
        "measured_updates": contract.measured_updates,
        "cold_start_measured": True,
        "steady_state_measured": True,
        "cache_policy": contract.cache_policy,
        "telemetry_interval_seconds": contract.telemetry_interval_seconds,
        "loader_error_count": 0,
        "cuda_error_count": 0,
        "monitor_error_count": 0,
    }
    measurement = root / "MEASUREMENT.json"
    measurement.write_text(json.dumps(measurement_payload, sort_keys=True) + "\n")
    telemetry = pd.DataFrame(
        {
            "update": np.arange(500, dtype=np.int64),
            "compute_seconds": np.full(500, 0.2),
            "data_wait_seconds": np.full(500, 0.01),
            "batch_ready_seconds": np.full(500, 0.04),
            "gpu_utilization": np.full(500, 0.9),
        }
    )
    telemetry_path = root / "TELEMETRY.parquet"
    telemetry.to_parquet(telemetry_path, index=False)
    parity_rows = tuple(
        G00DParityGateEvidence(
            gate=gate,
            comparison_mode=mode,
            reference_sha256="1" * 64,
            observed_sha256="1" * 64,
            maximum_absolute_error=0.0,
            maximum_relative_error=0.0,
        )
        for gate, mode in gates
    )
    parity_path = root / "PARITY.parquet"
    pd.DataFrame([row.model_dump(mode="json") for row in parity_rows]).to_parquet(
        parity_path, index=False
    )
    memory = pd.DataFrame(
        {
            "sample_time_seconds": np.arange(5, dtype=float),
            "loader_rss_bytes": np.full(5, 8_000_000_000, dtype=np.int64),
            "process_loader_rss_bytes": np.full(5, 7_000_000_000, dtype=np.int64),
            "aggregate_worker_rss_bytes": np.full(5, 8_000_000_000, dtype=np.int64),
            "open_shards": np.full(5, 4, dtype=np.int64),
            "open_file_handles": np.full(5, 64, dtype=np.int64),
        }
    )
    memory_path = root / "MEMORY.parquet"
    memory.to_parquet(memory_path, index=False)
    receipt = _identified(
        IntegratedLoaderQualificationReceiptV2,
        {
            "schema_version": 2,
            "qualification_contract_id": contract.qualification_contract_id,
            "gpu_name": contract.expected_gpu_name,
            "gpu_uuid": measurement_payload["gpu_uuid"],
            "gpu_count": 1,
            "cuda_version": contract.expected_cuda_version,
            "torch_version": contract.expected_torch_version,
            "container_digest": contract.expected_container_digest,
            "worker_count": contract.worker_count,
            "cpu_count": contract.cpu_count,
            "storage_authority_hash": contract.storage_authority_hash,
            "microbatch_cells": contract.microbatch_cells,
            "microbatches_per_update": contract.microbatches_per_update,
            "macrobatch_cells": contract.macrobatch_cells,
            "prefetch_depth": contract.prefetch_depth,
            "warmup_updates": contract.warmup_updates,
            "measured_updates": contract.measured_updates,
            "cold_start_measured": True,
            "steady_state_measured": True,
            "cache_policy": contract.cache_policy,
            "measurement_protocol_sha256": contract.measurement_protocol_sha256,
            "measurement_evidence": _artifact(measurement, root).model_dump(mode="json"),
            "telemetry_artifact": _artifact(telemetry_path, root).model_dump(mode="json"),
            "parity_artifact": _artifact(parity_path, root).model_dump(mode="json"),
            "memory_trace_artifact": _artifact(memory_path, root).model_dump(mode="json"),
            "telemetry_interval_seconds": contract.telemetry_interval_seconds,
            "median_compute_seconds": 0.2,
            "p95_compute_seconds": 0.2,
            "median_data_wait_seconds": 0.01,
            "p95_batch_ready_seconds": 0.04,
            "data_wait_fraction": 0.01 / 0.21,
            "steady_state_gpu_utilization": 0.9,
            "peak_loader_rss_bytes": 8_000_000_000,
            "peak_process_loader_rss_bytes": 7_000_000_000,
            "peak_aggregate_worker_rss_bytes": 8_000_000_000,
            "peak_open_shards": 4,
            "peak_open_file_handles": 64,
            "rss_slope_bytes_per_second": 0.0,
            "rss_slope_upper_ci_bytes_per_second": 0.0,
            "maximum_rss_excursion_bytes": 0,
            "parity_gates": [row.model_dump(mode="json") for row in parity_rows],
            "lru_bound_pass": True,
            "loader_error_count": 0,
            "cuda_error_count": 0,
            "monitor_error_count": 0,
            "maximum_parity_absolute_error": 0.0,
            "maximum_parity_relative_error": 0.0,
            "memory_growth_pass": True,
            "status": "pass",
        },
        "receipt_id",
    )
    return contract, receipt


def test_g00d_verifier_rederives_files_summaries_and_gate_policy(tmp_path: Path) -> None:
    contract, receipt = _write_evidence(tmp_path)
    verify_integrated_loader_qualification(tmp_path, contract, receipt)

    with pytest.raises(IntegrityError, match="telemetry summaries"):
        verify_integrated_loader_qualification(
            tmp_path,
            contract,
            receipt.model_copy(update={"median_compute_seconds": 0.19}),
        )

    exact_gate = receipt.parity_gates[0].model_copy(update={"observed_sha256": "2" * 64})
    with pytest.raises(IntegrityError, match="status must be fail"):
        validate_integrated_loader_qualification(
            contract,
            receipt.model_copy(update={"parity_gates": (exact_gate, *receipt.parity_gates[1:])}),
        )

    telemetry_path = tmp_path / receipt.telemetry_artifact.relative_uri
    telemetry_path.write_bytes(telemetry_path.read_bytes() + b"tamper")
    with pytest.raises(IntegrityError, match="artifact failed verification"):
        verify_integrated_loader_qualification(tmp_path, contract, receipt)
