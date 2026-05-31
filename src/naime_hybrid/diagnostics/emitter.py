from __future__ import annotations

from typing import Any

import torch

from naime_hybrid.models.state_packet import NAIMEStatePacket

from .trace_context import TraceContext


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_python_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_scalar(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize_tensor(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"present": False}
    detached = tensor.detach().float()
    finite = torch.isfinite(detached)
    finite_ratio = finite.float().mean().item() if detached.numel() else 1.0
    safe = detached.masked_fill(~finite, 0.0)
    stats: dict[str, Any] = {
        "present": True,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": safe.mean().item() if safe.numel() else 0.0,
        "abs_mean": safe.abs().mean().item() if safe.numel() else 0.0,
        "std": safe.std(unbiased=False).item() if safe.numel() else 0.0,
        "min": safe.min().item() if safe.numel() else 0.0,
        "max": safe.max().item() if safe.numel() else 0.0,
        "finite_ratio": finite_ratio,
    }
    if detached.ndim >= 1 and detached.size(-1) > 0:
        stats["norm_mean"] = detached.norm(dim=-1).mean().item()
    return stats


def summarize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    return {key: _to_python_scalar(value) for key, value in metrics.items()}


def summarize_packet(packet: NAIMEStatePacket | None) -> dict[str, Any]:
    if packet is None:
        return {"present": False}
    return {
        "present": True,
        "architecture_id": packet.architecture_id,
        "world_state": summarize_tensor(packet.world_state),
        "self_state": summarize_tensor(packet.self_state),
        "latent_field": summarize_tensor(packet.latent_field),
        "controller_state": summarize_tensor(packet.controller_state),
        "memory": summarize_tensor(packet.memory),
    }


def emit_trace_event(
    trace_context: TraceContext | None,
    *,
    name: str,
    kind: str,
    stats: dict[str, Any] | None = None,
    tensors: dict[str, torch.Tensor | None] | None = None,
    packet: NAIMEStatePacket | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    if trace_context is None or not trace_context.enabled:
        return
    payload = _to_python_scalar(stats or {})
    tag_payload = _to_python_scalar(tags or {})
    if tensors and trace_context.config.record_tensor_stats:
        payload["tensors"] = {key: summarize_tensor(value) for key, value in tensors.items()}
    if packet is not None and trace_context.config.record_packet_fields:
        payload["packet"] = summarize_packet(packet)
    trace_context.emit(name=name, kind=kind, stats=payload, tags=tag_payload)
