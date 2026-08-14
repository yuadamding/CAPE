from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from credo_count_sde_v4.cli.app import main


def test_cli_full_surface(tmp_path: Path, capsys) -> None:
    project = tmp_path / "cli-project"
    assert main(["synthetic", "--output", str(project), "--updates", "4"]) == 0
    config = project / "config.yaml"
    assert main(["estimate", str(config)]) == 0
    assert main(["resolve", str(config)]) == 0
    assert main(["run-all", str(config), "--device", "cpu"]) == 0
    inference = project / "work" / "inference"
    assert main(["open-run", str(inference)]) == 0
    prediction = project / "prediction.npz"
    assert main(["predict", str(inference), "--output", str(prediction)]) == 0
    counterfactual = project / "counterfactual.json"
    assert (
        main(
            [
                "counterfactual",
                str(inference),
                "--output",
                str(counterfactual),
                "--series-index",
                "1",
            ]
        )
        == 0
    )
    assert main(["verify", str(project / "work" / "sealed"), "--level", "full"]) == 0
    assert (
        main(
            [
                "validate-contract",
                str(project / "work" / "compiled" / "contract.json"),
            ]
        )
        == 0
    )
    assert main(["audit"]) == 0
    assert prediction.exists()
    with np.load(prediction) as prediction_payload:
        assert "terminal_mean" in prediction_payload
        assert "gene_composition" not in prediction_payload
    payload = json.loads(counterfactual.read_text())
    assert len(payload["branches"]) == 4
    assert set(payload["contrasts"]) == {
        "delta_context_susceptibility",
        "delta_pool",
        "delta_reference",
    }
    assert "compatibility" in capsys.readouterr().out
