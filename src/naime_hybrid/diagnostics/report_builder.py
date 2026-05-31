from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .trace_context import TraceContext


def build_trace_summary(trace_context: TraceContext) -> dict[str, Any]:
    event_names = Counter(event.name for event in trace_context.events)
    event_kinds = Counter(event.kind for event in trace_context.events)
    return {
        "run_id": trace_context.run_id,
        "event_count": len(trace_context.events),
        "event_names": dict(event_names),
        "event_kinds": dict(event_kinds),
        "active_interventions": list(trace_context.active_interventions),
    }


def write_trace_artifacts(
    trace_context: TraceContext,
    *,
    output_dir: str | Path,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary = build_trace_summary(trace_context)
    if extra_summary:
        summary.update(extra_summary)
    manifest = {
        "run_id": trace_context.run_id,
        "event_count": len(trace_context.events),
        "artifacts": {
            "trace_events": "trace_events.jsonl",
            "summary": "summary.json",
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (path / "trace_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in trace_context.events:
            handle.write(
                json.dumps(
                    {
                        "name": event.name,
                        "kind": event.kind,
                        "stats": event.stats,
                        "tags": event.tags,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
