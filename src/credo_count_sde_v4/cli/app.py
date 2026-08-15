"""Argparse CLI mirroring the public lifecycle API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import api
from ..compat.credo3 import verify_frozen_credo
from ..contracts import CounterfactualBranch, CounterfactualDesign, RunIntent
from ..synthetic import create_synthetic_project
from ..version import RECIPE_DISPLAY, __version__


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _preflight() -> dict[str, str]:
    return verify_frozen_credo()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credo-v4", description=RECIPE_DISPLAY)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic", help="create a deterministic synthetic project")
    synthetic.add_argument("--output", type=_path, required=True)
    synthetic.add_argument(
        "--intent", choices=[intent.value for intent in RunIntent], default="count_context"
    )
    synthetic.add_argument("--updates", type=int, default=40)
    for name in (
        "prepare",
        "compile",
        "train",
        "resume",
        "finalize",
        "evaluate",
        "seal",
        "run-all",
    ):
        command = sub.add_parser(name)
        command.add_argument("config", type=_path)
        if name in {"train", "resume"}:
            command.add_argument("--device", default=None)
        elif name in {"evaluate", "run-all"}:
            command.add_argument("--device", default="cpu")
    fork = sub.add_parser("fork")
    fork.add_argument("config", type=_path)
    fork.add_argument("--from", dest="from_checkpoint", type=_path, required=True)
    fork.add_argument("--device", default=None)
    calibrate = sub.add_parser("calibrate-state")
    calibrate.add_argument("config", type=_path)
    calibrate.add_argument("--output", type=_path, required=True)
    calibrate.add_argument("--seed-start", type=int, default=100_000)
    calibrate.add_argument("--repeats", type=int, default=100)
    calibrate.add_argument("--device", default="cpu")
    open_parser = sub.add_parser("open-run")
    open_parser.add_argument("run", type=_path)
    open_parser.add_argument("--device", default="cpu")
    predict = sub.add_parser("predict")
    predict.add_argument("run", type=_path)
    predict.add_argument("--output", type=_path, required=True)
    predict.add_argument("--device", default="cpu")
    counterfactual = sub.add_parser("counterfactual")
    counterfactual.add_argument("run", type=_path)
    counterfactual.add_argument("--output", type=_path, required=True)
    counterfactual.add_argument("--series-index", type=int, default=0)
    counterfactual.add_argument("--device", default="cpu")
    verify = sub.add_parser("verify")
    verify.add_argument("path", type=_path)
    verify.add_argument(
        "--level", choices=["manifest", "content", "reload", "resume", "full"], default="content"
    )
    estimate = sub.add_parser("estimate")
    estimate.add_argument("config", type=_path)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("config", type=_path)
    validate = sub.add_parser("validate-contract")
    validate.add_argument("path", type=_path)
    sub.add_parser("audit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = _preflight()
    command = args.command
    result: object
    if command == "synthetic":
        result = create_synthetic_project(
            args.output, intent=RunIntent(args.intent), updates=args.updates
        )
    elif command == "prepare":
        result = api.prepare(args.config)
    elif command == "compile":
        result = api.compile_run(args.config)
    elif command == "train":
        result = api.train(args.config, device=args.device)
    elif command == "resume":
        result = api.resume(args.config, device=args.device)
    elif command == "fork":
        result = api.fork(args.config, from_checkpoint=args.from_checkpoint, device=args.device)
    elif command == "calibrate-state":
        results, calibration = api.calibrate_state_selection(
            args.config,
            args.output,
            seed_start=args.seed_start,
            repeats=args.repeats,
            device=args.device,
        )
        result = {"results": str(results), "calibration": str(calibration)}
    elif command == "finalize":
        result = api.finalize(args.config)
    elif command == "evaluate":
        result = api.evaluate(args.config, device=args.device)
    elif command == "seal":
        result = api.seal(args.config)
    elif command == "run-all":
        api.prepare(args.config)
        api.compile_run(args.config)
        api.train(args.config, device=args.device)
        api.finalize(args.config)
        api.evaluate(args.config, device=args.device)
        result = api.seal(args.config)
    elif command == "open-run":
        run = api.open_run(args.run, device=args.device)
        result = run.manifest.model_dump(mode="json")
    elif command == "predict":
        result = api.open_run(args.run, device=args.device).predict(args.output)
    elif command == "counterfactual":
        design = CounterfactualDesign(
            series_index=args.series_index,
            branches=(
                CounterfactualBranch(
                    branch_id="factual_pool", effect_mode="factual", context_mode="pool_dynamic"
                ),
                CounterfactualBranch(
                    branch_id="reference_pool", effect_mode="reference", context_mode="pool_dynamic"
                ),
                CounterfactualBranch(
                    branch_id="factual_reference",
                    effect_mode="factual",
                    context_mode="reference_dynamic",
                ),
                CounterfactualBranch(
                    branch_id="reference_reference",
                    effect_mode="reference",
                    context_mode="reference_dynamic",
                ),
            ),
            particles=64,
            steps=8,
            seed=20_001,
        )
        result = api.open_run(args.run, device=args.device).counterfactual(design, args.output)
    elif command == "verify":
        result = api.verify(args.path, level=args.level)
    elif command == "estimate":
        result = api.estimate(args.config)
    elif command == "resolve":
        result = api.resolve_config(args.config).model_dump(mode="json")
    elif command == "validate-contract":
        result = api.validate_contract(args.path)
    elif command == "audit":
        result = {"recipe": RECIPE_DISPLAY, "compatibility": receipt}
    else:  # pragma: no cover - argparse prevents it
        raise AssertionError(command)
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
