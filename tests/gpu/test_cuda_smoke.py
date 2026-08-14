from __future__ import annotations

from pathlib import Path

import pytest
import torch

from credo_count_sde_v4 import api
from credo_count_sde_v4.synthetic import create_synthetic_project

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_reload_and_bf16_kernel(tmp_path: Path) -> None:
    torch.cuda.reset_peak_memory_stats()
    left = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite((left @ right).float()).all()
    config = create_synthetic_project(tmp_path / "cuda", updates=4)
    api.prepare(config)
    api.compile_run(config)
    api.train(config, device="cuda")
    api.finalize(config)
    run = api.open_run(tmp_path / "cuda/work/inference", device="cuda")
    mean, mass, _ = run.terminal(particles=32)
    assert mean.shape[0] == 6 and (mass > 0).all()
    assert torch.cuda.max_memory_allocated() < 4 * 1024**3
