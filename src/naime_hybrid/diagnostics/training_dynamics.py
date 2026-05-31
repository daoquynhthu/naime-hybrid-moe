from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

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

GRADIENT_KEYS = (
    "grad_component_total_norm",
    "grad_component_max_abs",
    "grad_component_param_count",
    "loss_grad_component_norm",
    "loss_grad_component_cosine",
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
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(item) for item in value]
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


def _gradient_group(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered or "tok_embeddings" in lowered:
        return "embedding"
    if "lm_head" in lowered or "output" in lowered:
        return "lm_head"
    if "router" in lowered or "gate" in lowered:
        return "router_gate"
    if "expert" in lowered or ".moe" in lowered:
        return "moe_expert"
    if "attention" in lowered or ".attn" in lowered or "q_proj" in lowered or "k_proj" in lowered or "v_proj" in lowered:
        return "attention"
    if "world" in lowered:
        return "world_state"
    if "self_state" in lowered or "recursive_self" in lowered:
        return "self_state"
    if "typed_dynamics" in lowered or "latent" in lowered or "controller" in lowered:
        return "typed_dynamics"
    if "norm" in lowered:
        return "norm"
    return "other"


def collect_gradient_component_stats(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        group = _gradient_group(name)
        stats = groups.setdefault(group, {"sum_sq": 0.0, "max_abs": 0.0, "param_count": 0.0})
        detached = grad.detach().float()
        finite = torch.isfinite(detached)
        safe = detached.masked_fill(~finite, 0.0)
        stats["sum_sq"] += float(torch.sum(safe * safe).cpu().item())
        stats["max_abs"] = max(stats["max_abs"], float(safe.abs().max().cpu().item()) if safe.numel() else 0.0)
        stats["param_count"] += float(parameter.numel())
    return {
        group: {
            "total_norm": math.sqrt(max(values["sum_sq"], 0.0)),
            "max_abs": values["max_abs"],
            "param_count": values["param_count"],
        }
        for group, values in sorted(groups.items())
    }


def _collect_flat_group_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        group = _gradient_group(name)
        groups.setdefault(group, []).append(grad.detach().float().flatten().cpu())
    return {group: torch.cat(chunks) for group, chunks in groups.items() if chunks}


def flatten_gradient_component_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "grad_component_total_norm": {group: values["total_norm"] for group, values in stats.items()},
        "grad_component_max_abs": {group: values["max_abs"] for group, values in stats.items()},
        "grad_component_param_count": {group: values["param_count"] for group, values in stats.items()},
    }


def collect_loss_gradient_probe(
    model: torch.nn.Module,
    losses: dict[str, torch.Tensor],
) -> dict[str, dict[str, dict[str, float]]]:
    """Attribute selected loss components to broad parameter groups.

    This is intentionally diagnostics-only. It replays backward on retained
    graphs for scalar component losses and restores the original accumulated
    gradients before returning.
    """

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    original_grads = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
    baseline = _collect_flat_group_grads(model)
    norms: dict[str, dict[str, float]] = {}
    cosines: dict[str, dict[str, float]] = {}
    try:
        for loss_name, loss in losses.items():
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            current = _collect_flat_group_grads(model)
            norms[loss_name] = {}
            cosines[loss_name] = {}
            for group, vector in current.items():
                current_norm = float(vector.norm().item())
                norms[loss_name][group] = current_norm
                base = baseline.get(group)
                if base is not None and current_norm > 0.0:
                    base_norm = float(base.norm().item())
                    if base_norm > 0.0:
                        cosines[loss_name][group] = float(torch.dot(vector, base).item() / (current_norm * base_norm))
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, grad in zip(parameters, original_grads, strict=True):
            parameter.grad = grad
    return {
        "loss_grad_component_norm": norms,
        "loss_grad_component_cosine": cosines,
    }


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
        "gradients": _select(payload, GRADIENT_KEYS),
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
