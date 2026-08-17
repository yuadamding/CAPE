"""Fail when committed schemas differ from strict contract models."""

from __future__ import annotations

import json
from pathlib import Path

from generate_schemas import LEGACY_SCHEMAS, MODELS


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    failures: list[str] = []
    for name, model in MODELS.items():
        payload = model.model_json_schema()
        payload["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        payload["$id"] = f"https://credo.local/schemas/{name}"
        expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path = root / name
        if not path.is_file() or path.read_text() != expected:
            failures.append(name)
    extras = sorted(
        path.name
        for path in root.glob("*.json")
        if path.name not in MODELS and path.name not in LEGACY_SCHEMAS
    )
    if failures or extras:
        raise SystemExit(f"Schema drift: changed={failures}, extra={extras}")


if __name__ == "__main__":
    main()
