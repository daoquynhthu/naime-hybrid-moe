from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TraceConfig:
    enabled: bool = False
    record_tensor_stats: bool = True
    record_metric_values: bool = True
    record_packet_fields: bool = True
    record_full_tensors: bool = False
    sample_limit: int | None = None
    batch_limit: int | None = None
    output_dir: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
