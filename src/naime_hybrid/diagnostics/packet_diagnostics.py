from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from naime_hybrid.models.state_packet import NAIMEStatePacket
from naime_hybrid.training.losses import (
    boundary_token_weights,
    masked_token_average,
    tail_token_weights,
    token_lm_loss,
)
from naime_hybrid.training.runtime import split_stateful_chunks

from .emitter import emit_trace_event, summarize_packet
from .report_builder import build_trace_summary, write_trace_artifacts
from .trace_context import TraceContext


PACKET_FIELDS = ("world_state", "self_state", "latent_field", "controller_state", "memory")


def _packet_like(packet: NAIMEStatePacket, **fields: torch.Tensor | None) -> NAIMEStatePacket:
    values = {name: getattr(packet, name) for name in PACKET_FIELDS}
    values.update(fields)
    return NAIMEStatePacket(
        world_state=values["world_state"],
        self_state=values["self_state"],
        latent_field=values["latent_field"],
        controller_state=values["controller_state"],
        memory=values["memory"],
        state_version=packet.state_version,
        protocol_version=packet.protocol_version,
        architecture_id=packet.architecture_id,
        causal_integrity_version=packet.causal_integrity_version,
        tokenizer_hash=packet.tokenizer_hash,
        created_step=packet.created_step,
        confidence=packet.confidence,
    )


def _field_erased_packet(packet: NAIMEStatePacket, field: str) -> NAIMEStatePacket:
    value = getattr(packet, field)
    if value is None:
        return packet
    return _packet_like(packet, **{field: torch.zeros_like(value)})


def _field_swapped_packet(packet: NAIMEStatePacket, field: str) -> NAIMEStatePacket:
    value = getattr(packet, field)
    if value is None or value.size(0) < 2:
        return packet
    return _packet_like(packet, **{field: value.roll(shifts=1, dims=0)})


def _loss_views(
    token_loss: torch.Tensor,
    labels: torch.Tensor,
    *,
    boundary_tokens: int,
) -> dict[str, float]:
    boundary = max(1, int(boundary_tokens))
    return {
        "full": float(masked_token_average(token_loss, labels).detach().cpu().item()),
        "boundary": float(
            masked_token_average(
                token_loss,
                labels,
                weights=boundary_token_weights(labels, boundary_tokens=boundary, decay=1.0),
            )
            .detach()
            .cpu()
            .item()
        ),
        "tail": float(
            masked_token_average(token_loss, labels, weights=tail_token_weights(labels, start=boundary))
            .detach()
            .cpu()
            .item()
        ),
    }


def _packet_intervention_report(
    model,
    *,
    packet: NAIMEStatePacket,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    baseline_token_loss: torch.Tensor,
    boundary_tokens: int,
    trace_context: TraceContext,
) -> dict[str, Any]:
    baseline_views = _loss_views(baseline_token_loss, labels, boundary_tokens=boundary_tokens)
    field_report: dict[str, Any] = {}
    for field in PACKET_FIELDS:
        value = getattr(packet, field)
        if value is None:
            field_report[field] = {"present": False}
            continue
        field_report[field] = {"present": True}
        for operation, mutated_packet in (
            ("erase", _field_erased_packet(packet, field)),
            ("swap", _field_swapped_packet(packet, field)),
        ):
            out = model(
                input_ids,
                attention_mask=attention_mask,
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
                past_state=mutated_packet,
                detach_past_state=False,
                trace_context=trace_context,
            )
            token_loss = token_lm_loss(out["logits"], labels)
            views = _loss_views(token_loss, labels, boundary_tokens=boundary_tokens)
            deltas = {name: views[name] - baseline_views[name] for name in baseline_views}
            field_report[field][operation] = {
                "loss": views,
                "delta_vs_full_packet": deltas,
            }
            emit_trace_event(
                trace_context,
                name=f"diagnostics.packet.{operation}.{field}",
                kind="intervention",
                stats={
                    "field": field,
                    "operation": operation,
                    "delta_full": deltas["full"],
                    "delta_boundary": deltas["boundary"],
                    "delta_tail": deltas["tail"],
                },
                packet=mutated_packet,
                tags={"phase": "packet_intervention", "field": field, "operation": operation},
            )
    return {
        "baseline_loss": baseline_views,
        "fields": field_report,
    }


def run_state_packet_diagnostics(
    model,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    infer_pad_mask: bool | None = None,
    chunk_len: int | None = None,
    boundary_tokens: int = 64,
    trace_context: TraceContext | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    trace_context = trace_context or TraceContext()
    chunks = split_stateful_chunks(
        input_ids,
        labels,
        attention_mask,
        chunk_len=chunk_len,
        target_chunks=2,
    )
    if len(chunks) < 2:
        raise ValueError("state packet diagnostics require at least two chunks")

    first = chunks[0]
    second = chunks[1]
    first_out = model(
        first["input_ids"],
        attention_mask=first.get("attention_mask"),
        infer_pad_mask=infer_pad_mask,
        return_aux=False,
        return_logits=False,
        return_state=True,
        trace_context=trace_context,
    )
    packet = first_out.get("state_packet")
    stateful_out = model(
        second["input_ids"],
        attention_mask=second.get("attention_mask"),
        infer_pad_mask=infer_pad_mask,
        return_aux=False,
        past_state=packet,
        detach_past_state=False,
        trace_context=trace_context,
    )
    fresh_out = model(
        second["input_ids"],
        attention_mask=second.get("attention_mask"),
        infer_pad_mask=infer_pad_mask,
        return_aux=False,
        trace_context=trace_context,
    )
    stateful_token_loss = token_lm_loss(stateful_out["logits"], second["labels"])
    fresh_token_loss = token_lm_loss(fresh_out["logits"], second["labels"])
    gain_curve = fresh_token_loss.detach() - stateful_token_loss.detach()
    boundary = int(max(boundary_tokens, 1))
    boundary_gain = masked_token_average(
        gain_curve,
        second["labels"],
        weights=boundary_token_weights(second["labels"], boundary_tokens=boundary, decay=1.0),
    )
    tail_gain = masked_token_average(gain_curve, second["labels"], weights=tail_token_weights(second["labels"], start=boundary))
    full_gain = masked_token_average(gain_curve, second["labels"])
    metrics = {
        "full_gain": float(full_gain.detach().cpu().item()),
        "boundary_gain": float(boundary_gain.detach().cpu().item()),
        "tail_gain": float(tail_gain.detach().cpu().item()),
        "token_gain_curve": gain_curve.detach().cpu().tolist(),
        "packet_summary": summarize_packet(packet),
    }
    if isinstance(packet, NAIMEStatePacket):
        metrics["packet_interventions"] = _packet_intervention_report(
            model,
            packet=packet,
            input_ids=second["input_ids"],
            labels=second["labels"],
            attention_mask=second.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            baseline_token_loss=stateful_token_loss,
            boundary_tokens=boundary,
            trace_context=trace_context,
        )
    emit_trace_event(
        trace_context,
        name="diagnostics.packet.compare",
        kind="report",
        stats={
            "full_gain": metrics["full_gain"],
            "boundary_gain": metrics["boundary_gain"],
            "tail_gain": metrics["tail_gain"],
        },
        packet=packet,
        tags={"phase": "packet_compare"},
    )
    summary = build_trace_summary(trace_context)
    if output_dir:
        output_path = Path(output_dir)
        write_trace_artifacts(trace_context, output_dir=output_path, extra_summary=metrics)
    return {
        "metrics": metrics,
        "summary": summary,
        "trace_context": trace_context,
    }
