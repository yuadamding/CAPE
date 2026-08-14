"""Append-only lifecycle ledger with strict transitions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from ..canonical import canonical_json_bytes
from ..contracts import LifecycleState
from ..errors import StateTransitionError

_ALLOWED: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.RESOLVED: frozenset({LifecycleState.PREPARED}),
    LifecycleState.PREPARED: frozenset({LifecycleState.COMPILED}),
    LifecycleState.COMPILED: frozenset({LifecycleState.TRAINED}),
    LifecycleState.TRAINED: frozenset({LifecycleState.FINALIZED}),
    LifecycleState.FINALIZED: frozenset({LifecycleState.EVALUATED}),
    LifecycleState.EVALUATED: frozenset({LifecycleState.SEALED}),
    LifecycleState.SEALED: frozenset(),
}


class LifecycleLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def events(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def state(self) -> LifecycleState:
        events = self.events()
        return LifecycleState(str(events[-1]["state"])) if events else LifecycleState.RESOLVED

    def transition(
        self, state: LifecycleState, *, artifact_id: str, details: dict[str, object] | None = None
    ) -> None:
        current = self.state()
        if state not in _ALLOWED[current]:
            raise StateTransitionError(
                f"Illegal lifecycle transition {current.value} -> {state.value}."
            )
        event = {
            "schema_version": 1,
            "sequence": len(self.events()),
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "previous_state": current.value,
            "state": state.value,
            "artifact_id": artifact_id,
            "details": details or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
