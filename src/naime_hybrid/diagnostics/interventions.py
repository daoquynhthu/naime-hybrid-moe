from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class InterventionSpec:
    target: str
    operation: str
    value: Any = None


def describe_interventions(specs: list[InterventionSpec] | None) -> list[dict[str, Any]]:
    if not specs:
        return []
    return [{"target": spec.target, "operation": spec.operation, "value": spec.value} for spec in specs]
