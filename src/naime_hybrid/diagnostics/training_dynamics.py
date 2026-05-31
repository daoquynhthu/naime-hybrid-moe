from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


CORE_KEYS = (
    "loss",
    "loss_lm",
    "loss_aux",
    "loss_total",
    "ppl_lm",
    "lr",
    "grad_norm",
    "bad_grad_window_count",
    "lr_safety_factor",
)

ROUTER_KEYS = (
    "alpha_downstream_mean",
    "router_entropy",
    "v5_router_world_ratio",
    "v5_router_world_cosine",
    "v5_router_world_gate",
    "v5_router_memory_ratio",
    "v5_router_effective_norm",
)

STATE_KEYS = (
    "v5_slot_delta",
    "v5_slot_update_gate",
    "v5_slot_write_entropy",
    "v5_slot_stability",
    "v6_reflection_norm",
    "v6_boundary_self",
    "v6_boundary_world",
    "v6_slot_cosine",
    "v7_thought_steps",
    "v7_latent_delta",
    "v7_latent_velocity",
    "v7_latent_acceleration",
    "v7_hidden_delta",
    "v7_hidden_write_ratio",
    "v7_world_delta",
    "v7_self_delta",
    "v7_controller_delta",
    "v7_world_write_gate",
    "v7_self_write_gate",
    "v7_controller_write_gate",
    "v7_dynamic_depth_mean",
    "v7_dynamic_halt_fraction",
    "v7_homeostatic_dhi",
    "v7_latent_rate_scale",
    "v7_world_rate_scale",
    "v7_self_rate_scale",
    "v7_ingress_compatibility",
    "v7_ingress_latent_gate",
    "v7_ingress_world_gate",
    "v7_ingress_self_gate",
    "v7_ingress_memory_gate",
    "v7_latent_tau",
    "v7_world_tau",
    "v7_self_tau",
    "v7_controller_tau",
)

PACKET_KEYS = (
    "diagnostics_event_count",
    "diagnostics_full_gain",
    "diagnostics_boundary_gain",
    "diagnostics_tail_gain",
    "diagnostics_output_dir",
)


def diagnostics_root(run_dir: Path, output_dir: str | None) -> Path:
    return Path(output_dir) if output_dir else run_dir / "training_diagnostics"


def _clean_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    try:
        item = value.item()
    except AttributeError:
        return str(value)
    return _clean_value(item)


def _select(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in keys:
        if key in payload:
            selected[key] = _clean_value(payload[key])
    return selected


def build_training_dynamics_event(
    *,
    step: int,
    phase: str,
    payload: dict[str, Any],
    tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "event": "training_dynamics",
        "phase": phase,
        "step": int(step),
        "core": _select(payload, CORE_KEYS),
        "router": _select(payload, ROUTER_KEYS),
        "state": _select(payload, STATE_KEYS),
        "packet": _select(payload, PACKET_KEYS),
    }
    if tags:
        event["tags"] = {str(key): _clean_value(value) for key, value in tags.items()}
    return event


def append_training_dynamics_event(root: Path, event: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dynamics_events.jsonl"
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".dynamics_events.", suffix=".tmp", dir=str(root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(line)
            tmp.write("\n")
        with path.open("a", encoding="utf-8") as target, open(tmp_name, "r", encoding="utf-8") as source:
            target.write(source.read())
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return path
