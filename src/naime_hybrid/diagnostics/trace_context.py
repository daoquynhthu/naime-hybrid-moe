from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .trace_config import TraceConfig


@dataclass(slots=True)
class TraceEvent:
    name: str
    kind: str
    stats: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceContext:
    config: TraceConfig = field(default_factory=TraceConfig)
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    sample_ids: list[str] = field(default_factory=list)
    segment_ids: list[str] = field(default_factory=list)
    active_interventions: list[str] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def emit(
        self,
        *,
        name: str,
        kind: str,
        stats: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        merged_tags = dict(self.config.tags)
        if tags:
            merged_tags.update(tags)
        self.events.append(
            TraceEvent(
                name=name,
                kind=kind,
                stats=stats or {},
                tags=merged_tags,
            )
        )
