"""Structured trace of every agent step, one JSON object per line (logs/traces.jsonl)."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from energy_agent.sources import PROJECT_ROOT

TRACE_FILE = PROJECT_ROOT / "logs" / "traces.jsonl"


class Tracer:
    def __init__(
        self,
        path: Path | None = TRACE_FILE,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.path = path
        self.on_event = on_event
        self.events: list[dict] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, run_id: str, event_type: str, **fields) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": run_id,
            "event": event_type,
            **fields,
        }
        self.events.append(record)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.on_event:
            self.on_event(record)
