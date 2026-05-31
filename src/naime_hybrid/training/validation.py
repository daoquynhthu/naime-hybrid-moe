import math

import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.models.state_packet import NAIMEStatePacket

from .losses import (
    IGNORE_INDEX,
    boundary_token_weights,
    collect_aux_losses,
    lm_loss,
    masked_token_average,
    tail_token_weights,
    token_lm_loss,
)
from .masks import prepare_attention_mask_for_device
from .runtime import split_stateful_chunks

BOUNDARY_WINDOWS = (16, 32, 64, 128)


def _supports_state_packet(model: torch.nn.Module) -> bool:
    native_model = getattr(model, "_orig_mod", model)
    return hasattr(native_model, "_initial_world_state")


def _native_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _packet_like(
    packet: NAIMEStatePacket,
    *,
    world_state: torch.Tensor | None,
    self_state: torch.Tensor | None,
    latent_field: torch.Tensor | None,
    memory: torch.Tensor | None,
    controller_state: torch.Tensor | None,
) -> NAIMEStatePacket:
    return NAIMEStatePacket(
        world_state=world_state,
        self_state=self_state,
        latent_field=latent_field,
        memory=memory,
        controller_state=controller_state,
        state_version=packet.state_version,
        protocol_version=packet.protocol_version,
        architecture_id=packet.architecture_id,
        causal_integrity_version=packet.causal_integrity_version,
        tokenizer_hash=packet.tokenizer_hash,
        created_step=packet.created_step,
        confidence=packet.confidence,
    )


def _roll_optional_batch(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.roll(shifts=1, dims=0) if value is not None else None


def _swap_packet_batch(packet: NAIMEStatePacket) -> NAIMEStatePacket:
    return _packet_like(
        packet,
        world_state=_roll_optional_batch(packet.world_state),
        self_state=_roll_optional_batch(packet.self_state),
        latent_field=_roll_optional_batch(packet.latent_field),
        memory=_roll_optional_batch(packet.memory),
        controller_state=_roll_optional_batch(packet.controller_state),
    )


def _stateful_chunk_pair(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    chunk_len: int | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
    chunks = split_stateful_chunks(
        input_ids,
        labels,
        attention_mask,
        chunk_len=chunk_len,
        target_chunks=2,
    )
    if len(chunks) < 2:
        return None
    return chunks[0], chunks[1]


def _prepare_stateful_probe_context(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    infer_pad_mask: bool | None,
    use_amp: bool,
    chunk_len: int | None,
    include_fresh: bool,
) -> dict[str, object] | None:
    if not _supports_state_packet(model) or input_ids.size(1) < 4:
        return None

    pair = _stateful_chunk_pair(input_ids, labels, attention_mask, chunk_len=chunk_len)
    if pair is None:
        return None
    first, second = pair
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        first_out = model(
            first["input_ids"],
            attention_mask=first.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            return_logits=False,
            return_state=True,
        )
        packet = first_out.get("state_packet")
        if packet is None:
            return None
        stateful_out = model(
            second["input_ids"],
            attention_mask=second.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=packet,
        )
        context: dict[str, object] = {
            "second": second,
            "packet": packet,
            "stateful_token_loss": token_lm_loss(stateful_out["logits"], second["labels"]),
        }
        if include_fresh:
            fresh_out = model(
                second["input_ids"],
                attention_mask=second.get("attention_mask"),
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
            )
            context["fresh_token_loss"] = token_lm_loss(fresh_out["logits"], second["labels"])
    return context


def _token_loss_views(
    token_loss: torch.Tensor,
    labels: torch.Tensor,
    *,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, torch.Tensor]:
    views = {"lm": masked_token_average(token_loss, labels)}
    for boundary in BOUNDARY_WINDOWS:
        views[f"boundary_{boundary}"] = masked_token_average(
            token_loss,
            labels,
            weights=boundary_token_weights(labels, boundary_tokens=boundary, decay=boundary_decay),
        )
    views["tail"] = masked_token_average(
        token_loss,
        labels,
        weights=tail_token_weights(labels, start=tail_start),
    )
    return views


def _pair_view_metrics(
    primary_token_loss: torch.Tensor,
    secondary_token_loss: torch.Tensor,
    labels: torch.Tensor,
    *,
    boundary_decay: float,
    tail_start: int,
    delta_name: str,
    primary_name: str,
    secondary_name: str,
) -> dict[str, float]:
    primary_views = _token_loss_views(
        primary_token_loss,
        labels,
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )
    secondary_views = _token_loss_views(
        secondary_token_loss,
        labels,
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )
    metrics: dict[str, float] = {}
    for name, primary_value in primary_views.items():
        secondary_value = secondary_views[name]
        metrics[f"{delta_name}_{name}"] = float((secondary_value - primary_value).detach().cpu())
        metrics[f"{primary_name}_{name}"] = float(primary_value.detach().cpu())
        metrics[f"{secondary_name}_{name}"] = float(secondary_value.detach().cpu())
    return metrics


def _estimate_state_carry_gain_from_context(
    context: dict[str, object],
    *,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    second = context.get("second")
    stateful_token_loss = context.get("stateful_token_loss")
    fresh_token_loss = context.get("fresh_token_loss")
    if not isinstance(second, dict) or not isinstance(stateful_token_loss, torch.Tensor) or not isinstance(
        fresh_token_loss, torch.Tensor
    ):
        return None
    return _pair_view_metrics(
        stateful_token_loss,
        fresh_token_loss,
        second["labels"],
        boundary_decay=boundary_decay,
        tail_start=tail_start,
        delta_name="gain",
        primary_name="stateful",
        secondary_name="fresh",
    )


def _estimate_v7_state_swap_penalty_from_context(
    model: torch.nn.Module,
    context: dict[str, object],
    *,
    infer_pad_mask: bool | None,
    use_amp: bool,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    second = context.get("second")
    packet = context.get("packet")
    correct_token_loss = context.get("stateful_token_loss")
    if not isinstance(second, dict) or not isinstance(packet, NAIMEStatePacket) or not isinstance(
        correct_token_loss, torch.Tensor
    ):
        return None
    if second["input_ids"].size(0) < 2:
        return None
    device_type = second["input_ids"].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        swapped_out = model(
            second["input_ids"],
            attention_mask=second.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_swap_packet_batch(packet),
        )
        swapped_token_loss = token_lm_loss(swapped_out["logits"], second["labels"])
    return _pair_view_metrics(
        correct_token_loss,
        swapped_token_loss,
        second["labels"],
        boundary_decay=boundary_decay,
        tail_start=tail_start,
        delta_name="delta",
        primary_name="correct",
        secondary_name="wrong",
    )


def _estimate_v7_state_erase_sensitivity_from_context(
    model: torch.nn.Module,
    context: dict[str, object],
    *,
    infer_pad_mask: bool | None,
    use_amp: bool,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    second = context.get("second")
    packet = context.get("packet")
    full_token_loss = context.get("stateful_token_loss")
    if not isinstance(second, dict) or not isinstance(packet, NAIMEStatePacket) or not isinstance(
        full_token_loss, torch.Tensor
    ):
        return None
    device_type = second["input_ids"].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        world_erased_out = model(
            second["input_ids"],
            attention_mask=second.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_packet_like(
                packet,
                world_state=None,
                self_state=packet.self_state,
                latent_field=packet.latent_field,
                memory=packet.memory,
                controller_state=packet.controller_state,
            ),
        )
        self_erased_out = model(
            second["input_ids"],
            attention_mask=second.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_packet_like(
                packet,
                world_state=packet.world_state,
                self_state=None,
                latent_field=packet.latent_field,
                memory=packet.memory,
                controller_state=packet.controller_state,
            ),
        )
        latent_erased_out = model(
            second["input_ids"],
            attention_mask=second.get("attention_mask"),
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_packet_like(
                packet,
                world_state=packet.world_state,
                self_state=packet.self_state,
                latent_field=None,
                memory=packet.memory,
                controller_state=packet.controller_state,
            ),
        )
        world_token_loss = token_lm_loss(world_erased_out["logits"], second["labels"])
        self_token_loss = token_lm_loss(self_erased_out["logits"], second["labels"])
        latent_token_loss = token_lm_loss(latent_erased_out["logits"], second["labels"])

    full_views = _token_loss_views(
        full_token_loss,
        second["labels"],
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )
    world_views = _token_loss_views(
        world_token_loss,
        second["labels"],
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )
    self_views = _token_loss_views(
        self_token_loss,
        second["labels"],
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )
    latent_views = _token_loss_views(
        latent_token_loss,
        second["labels"],
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )
    metrics: dict[str, float] = {}
    for name, full_value in full_views.items():
        metrics[f"world_delta_{name}"] = float((world_views[name] - full_value).detach().cpu())
        metrics[f"self_delta_{name}"] = float((self_views[name] - full_value).detach().cpu())
        metrics[f"latent_delta_{name}"] = float((latent_views[name] - full_value).detach().cpu())
        metrics[f"full_{name}"] = float(full_value.detach().cpu())
    return metrics


def _estimate_state_carry_gain(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
    chunk_len: int | None,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    context = _prepare_stateful_probe_context(
        model,
        input_ids,
        labels,
        attention_mask,
        infer_pad_mask=infer_pad_mask,
        use_amp=use_amp,
        chunk_len=chunk_len,
        include_fresh=True,
    )
    if context is None:
        return None
    return _estimate_state_carry_gain_from_context(
        context,
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )


def _estimate_latent_thought_gain(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
) -> tuple[float, float, float] | None:
    native_model = _native_model(model)
    self_state_slots = getattr(native_model, "self_state_slots", None)
    if self_state_slots is None or getattr(self_state_slots, "latent_thought_steps", 0) <= 0:
        return None
    if labels.ne(IGNORE_INDEX).sum().item() == 0:
        return None

    device_type = input_ids.device.type
    original_steps = self_state_slots.latent_thought_steps
    try:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            thought_out = model(
                input_ids,
                attention_mask=attention_mask,
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
            )
            thought_loss = lm_loss(thought_out["logits"], labels)
        self_state_slots.latent_thought_steps = 0
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            no_thought_out = model(
                input_ids,
                attention_mask=attention_mask,
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
            )
            no_thought_loss = lm_loss(no_thought_out["logits"], labels)
    finally:
        self_state_slots.latent_thought_steps = original_steps

    thought = float(thought_loss.detach().cpu())
    no_thought = float(no_thought_loss.detach().cpu())
    return no_thought - thought, thought, no_thought


def _estimate_v7_dynamics_gain(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
) -> tuple[float, float, float] | None:
    native_model = _native_model(model)
    if getattr(native_model, "typed_dynamics", None) is None:
        return None
    original_steps = int(getattr(native_model.config, "v7_dynamics_steps", 0))
    original_max_steps = int(getattr(native_model.config, "v7_max_dynamics_steps", 0))
    effective_steps = original_max_steps or original_steps
    if effective_steps <= 0 or labels.ne(IGNORE_INDEX).sum().item() == 0:
        return None

    device_type = input_ids.device.type
    try:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            dynamics_out = model(
                input_ids,
                attention_mask=attention_mask,
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
            )
            dynamics_loss = lm_loss(dynamics_out["logits"], labels)
        native_model.config.v7_dynamics_steps = 0
        native_model.config.v7_max_dynamics_steps = 0
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            disabled_out = model(
                input_ids,
                attention_mask=attention_mask,
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
            )
            disabled_loss = lm_loss(disabled_out["logits"], labels)
    finally:
        native_model.config.v7_dynamics_steps = original_steps
        native_model.config.v7_max_dynamics_steps = original_max_steps

    dynamics = float(dynamics_loss.detach().cpu())
    disabled = float(disabled_loss.detach().cpu())
    return disabled - dynamics, dynamics, disabled


def _estimate_v7_state_swap_penalty(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
    chunk_len: int | None,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    if input_ids.size(0) < 2:
        return None
    context = _prepare_stateful_probe_context(
        model,
        input_ids,
        labels,
        attention_mask,
        infer_pad_mask=infer_pad_mask,
        use_amp=use_amp,
        chunk_len=chunk_len,
        include_fresh=False,
    )
    if context is None:
        return None
    return _estimate_v7_state_swap_penalty_from_context(
        model,
        context,
        infer_pad_mask=infer_pad_mask,
        use_amp=use_amp,
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )


def _estimate_v7_state_erase_sensitivity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
    chunk_len: int | None,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    native_model = _native_model(model)
    if getattr(native_model, "typed_dynamics", None) is None:
        return None

    context = _prepare_stateful_probe_context(
        model,
        input_ids,
        labels,
        attention_mask,
        infer_pad_mask=infer_pad_mask,
        use_amp=use_amp,
        chunk_len=chunk_len,
        include_fresh=False,
    )
    if context is None:
        return None
    return _estimate_v7_state_erase_sensitivity_from_context(
        model,
        context,
        infer_pad_mask=infer_pad_mask,
        use_amp=use_amp,
        boundary_decay=boundary_decay,
        tail_start=tail_start,
    )


def _estimate_doc_continuity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
    *,
    chunk_len: int | None,
    target_chunks: int,
    boundary_decay: float,
    tail_start: int,
) -> dict[str, float] | None:
    if not _supports_state_packet(model) or target_chunks < 2:
        return None

    chunks = split_stateful_chunks(
        input_ids,
        labels,
        attention_mask,
        chunk_len=chunk_len,
        target_chunks=target_chunks,
    )
    if len(chunks) < 2:
        return None

    gains_full: list[float] = []
    gains_boundary_64: list[float] = []
    gains_tail: list[float] = []
    stateful_full_losses: list[float] = []
    fresh_full_losses: list[float] = []
    stateful_boundary_64_losses: list[float] = []
    fresh_boundary_64_losses: list[float] = []
    packet = None
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        for chunk in chunks:
            if packet is None:
                carry_out = model(
                    chunk["input_ids"],
                    attention_mask=chunk.get("attention_mask"),
                    infer_pad_mask=infer_pad_mask,
                    return_aux=False,
                    return_logits=False,
                    return_state=True,
                )
                packet = carry_out.get("state_packet")
                continue

            fresh_out = model(
                chunk["input_ids"],
                attention_mask=chunk.get("attention_mask"),
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
            )
            fresh_token_loss = token_lm_loss(fresh_out["logits"], chunk["labels"])
            stateful_out = model(
                chunk["input_ids"],
                attention_mask=chunk.get("attention_mask"),
                infer_pad_mask=infer_pad_mask,
                return_aux=False,
                return_state=True,
                past_state=packet,
            )
            stateful_token_loss = token_lm_loss(stateful_out["logits"], chunk["labels"])
            packet = stateful_out.get("state_packet")
            if packet is None:
                return None
            fresh_views = _token_loss_views(
                fresh_token_loss,
                chunk["labels"],
                boundary_decay=boundary_decay,
                tail_start=tail_start,
            )
            stateful_views = _token_loss_views(
                stateful_token_loss,
                chunk["labels"],
                boundary_decay=boundary_decay,
                tail_start=tail_start,
            )
            gains_full.append(float((fresh_views["lm"] - stateful_views["lm"]).detach().cpu()))
            gains_boundary_64.append(
                float((fresh_views["boundary_64"] - stateful_views["boundary_64"]).detach().cpu())
            )
            gains_tail.append(float((fresh_views["tail"] - stateful_views["tail"]).detach().cpu()))
            fresh_full_losses.append(float(fresh_views["lm"].detach().cpu()))
            stateful_full_losses.append(float(stateful_views["lm"].detach().cpu()))
            fresh_boundary_64_losses.append(float(fresh_views["boundary_64"].detach().cpu()))
            stateful_boundary_64_losses.append(float(stateful_views["boundary_64"].detach().cpu()))

    if not gains_full:
        return None
    slope_full = 0.0 if len(gains_full) < 2 else (gains_full[-1] - gains_full[0]) / float(len(gains_full) - 1)
    slope_boundary_64 = (
        0.0
        if len(gains_boundary_64) < 2
        else (gains_boundary_64[-1] - gains_boundary_64[0]) / float(len(gains_boundary_64) - 1)
    )
    slope_tail = 0.0 if len(gains_tail) < 2 else (gains_tail[-1] - gains_tail[0]) / float(len(gains_tail) - 1)
    return {
        "gain_mean": sum(gains_full) / len(gains_full),
        "gain_cumulative": sum(gains_full),
        "gain_slope": slope_full,
        "stateful_mean": sum(stateful_full_losses) / len(stateful_full_losses),
        "fresh_mean": sum(fresh_full_losses) / len(fresh_full_losses),
        "gain_boundary_64_mean": sum(gains_boundary_64) / len(gains_boundary_64),
        "gain_boundary_64_cumulative": sum(gains_boundary_64),
        "gain_boundary_64_slope": slope_boundary_64,
        "stateful_boundary_64_mean": sum(stateful_boundary_64_losses) / len(stateful_boundary_64_losses),
        "fresh_boundary_64_mean": sum(fresh_boundary_64_losses) / len(fresh_boundary_64_losses),
        "gain_tail_mean": sum(gains_tail) / len(gains_tail),
        "gain_tail_cumulative": sum(gains_tail),
        "gain_tail_slope": slope_tail,
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    model_config: NAIMEStateMoEConfig,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
    lambda_load: float = 0.0,
    lambda_sparse: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_semantic_pred: float = 0.0,
    lambda_state_pred: float = 0.0,
    lambda_slot_diversity: float = 0.0,
    lambda_slot_stability: float = 0.0,
    lambda_self_pred: float = 0.0,
    lambda_self_slot_diversity: float = 0.0,
    state_carry: bool = False,
    latent_thought_gain: bool = False,
    v7_dynamics_gain: bool = False,
    v7_state_swap: bool = False,
    v7_state_erase: bool = False,
    doc_continuity: bool = False,
    doc_continuity_docs: int = 32,
    doc_continuity_chunks: int = 4,
    stateful_chunk_len: int | None = None,
    stateful_boundary_tokens: int = 64,
    stateful_boundary_decay: float = 0.97,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        key: 0.0
        for key in [
            "lm",
            "alpha",
            "alpha_raw",
            "alpha_prob",
            "alpha_clean_prob",
            "alpha_capped",
            "alpha_downstream",
            "entropy",
            "prior_entropy",
            "kl",
            "load",
            "sparse",
            "semantic_pred",
            "fusion_mid",
            "fusion_global",
            "v4_layer_scale",
            "v4_state_norm",
            "v4_memory_norm",
            "v4_memory_gate",
            "v4_memory_attention_entropy",
            "v4_memory_read_strength",
            "v4_memory_novelty",
            "v4_state_gate",
            "v4_state_confidence",
            "v4_state_delta",
            "v4_state_agreement",
            "gate_mix_alpha_weight",
            "gate_mix_clean_weight",
            "gate_mix_state_weight",
            "v5_state_pred",
            "v5_slot_diversity",
            "v5_slot_stability",
            "v5_slot_update_gate",
            "v5_slot_write_max",
            "v5_slot_write_entropy",
            "v5_slot_write_min",
            "v5_slot_write_active",
            "v5_slot_confidence",
            "v5_slot_confidence_std",
            "v5_slot_delta",
            "v5_slot_cosine",
            "v5_slot_read_entropy",
            "v5_slot_read_max",
            "v5_router_semantic_norm",
            "v5_router_world_raw_norm",
            "v5_router_world_norm",
            "v5_router_world_ratio",
            "v5_router_world_cosine",
            "v5_router_world_gate",
            "v5_router_world_modulation",
            "v5_router_world_cap",
            "v5_router_memory_norm",
            "v5_router_memory_ratio",
            "v5_router_effective_norm",
            "v5_semantic_hidden_write_norm",
            "v5_semantic_hidden_write_scale",
            "v5_memory_hidden_write_norm",
            "v5_memory_hidden_write_scale",
            "v6_self_pred",
            "v6_slot_diversity",
            "v6_slot_cosine",
            "v6_slot_context_cosine",
            "v6_state_delta",
            "v6_state_norm",
            "v6_reflection_norm",
            "v6_world_explained_norm",
            "v6_hidden_residual_norm",
            "v6_world_residual_ratio",
            "v6_hidden_write_gate",
            "v6_hidden_write_norm",
            "v6_hidden_write_scale",
            "v6_boundary_entropy",
            "v6_boundary_self",
            "v6_boundary_world",
            "v6_boundary_other",
            "v6_boundary_unknown",
            "v6_latent_thought_delta",
            "v6_latent_thought_velocity",
            "v6_latent_thought_write_norm",
            "v6_latent_thought_steps",
            "v6_state_evolution_delta",
            "v6_state_evolution_world_delta",
            "v6_state_evolution_self_delta",
            "v6_state_evolution_memory_delta",
            "v6_state_evolution_steps",
            "v6_latent_field_token_delta_norm",
            "v6_latent_field_token_delta_ratio",
            "v6_latent_field_read_entropy",
            "v6_latent_field_read_max",
            "v6_latent_field_gate",
            "v7_thought_steps",
            "v7_latent_delta",
            "v7_latent_velocity",
            "v7_latent_acceleration",
            "v7_hidden_delta",
            "v7_latent_hidden_write_norm",
            "v7_hidden_write_ratio",
            "v7_hidden_write_gate",
            "v7_latent_read_entropy",
            "v7_latent_read_max",
            "v7_latent_state_norm",
            "v7_world_state_norm",
            "v7_self_state_norm",
            "v7_controller_state_norm",
            "v7_world_delta",
            "v7_self_delta",
            "v7_controller_delta",
            "v7_world_write_gate",
            "v7_self_write_gate",
            "v7_controller_write_gate",
            "v7_dynamic_depth_enabled",
            "v7_dynamic_depth_mean",
            "v7_dynamic_halt_fraction",
            "v7_dynamic_continue_score",
            "v7_dynamic_convergence_threshold",
            "v7_causal_segments",
            "v7_past_latent_adapt_steps",
            "v7_past_latent_read_suppressed",
            "v7_latent_timescale",
            "v7_world_timescale",
            "v7_self_timescale",
            "v7_controller_fixed",
            "v7_homeostatic_control_enabled",
            "v7_homeostatic_dhi",
            "v7_homeostatic_balance_pressure",
            "v7_homeostatic_accel_pressure",
            "v7_latent_rate_scale",
            "v7_world_rate_scale",
            "v7_self_rate_scale",
            "v7_hidden_read_rate_scale",
            "v7_state_compatibility_enabled",
            "v7_carry_compatibility",
            "v7_carry_latent_gate",
            "v7_carry_world_gate",
            "v7_carry_self_gate",
            "v7_carry_controller_gate",
            "v7_carry_memory_gate",
            "v7_carry_blend_delta",
            "v7_hyperspherical_state_enabled",
            "v7_causal_summary_enabled",
            "v7_causal_summary_decay",
            "v7_adaptive_tau_enabled",
            "v7_latent_tau",
            "v7_world_tau",
            "v7_self_tau",
            "v7_controller_tau",
            "v7_ingress_compatibility_enabled",
            "v7_ingress_compatibility",
            "v7_ingress_latent_gate",
            "v7_ingress_world_gate",
            "v7_ingress_self_gate",
            "v7_ingress_controller_gate",
            "v7_ingress_memory_gate",
            "v7_ingress_latent_blend_delta",
            "v7_ingress_world_blend_delta",
            "v7_ingress_self_blend_delta",
            "v7_ingress_controller_blend_delta",
            "v7_ingress_memory_blend_delta",
            "v7_effective_latent_write_scale",
            "v7_effective_world_write_scale",
            "v7_effective_self_write_scale",
            "v7_effective_controller_write_scale",
            "state_carry_gain_lm",
            "state_carry_stateful_lm",
            "state_carry_fresh_lm",
            "latent_thought_gain_lm",
            "latent_thought_lm",
            "latent_thought_disabled_lm",
            "v7_dynamics_gain_lm",
            "v7_dynamics_lm",
            "v7_dynamics_disabled_lm",
            "v7_state_swap_delta_lm",
            "v7_state_swap_correct_lm",
            "v7_state_swap_wrong_lm",
            "v7_world_erase_delta_lm",
            "v7_self_erase_delta_lm",
            "v7_latent_erase_delta_lm",
            "v7_state_erase_full_lm",
            "doc_carry_gain_mean",
            "doc_carry_gain_cumulative",
            "doc_carry_gain_slope",
            "doc_stateful_loss_mean",
            "doc_fresh_loss_mean",
        ]
    }
    for boundary in BOUNDARY_WINDOWS:
        totals[f"state_carry_gain_boundary_{boundary}"] = 0.0
    totals["state_carry_gain_tail"] = 0.0
    totals["v7_state_swap_delta_boundary_64"] = 0.0
    totals["v7_state_swap_delta_tail"] = 0.0
    totals["v7_world_erase_delta_boundary_64"] = 0.0
    totals["v7_self_erase_delta_boundary_64"] = 0.0
    totals["v7_latent_erase_delta_boundary_64"] = 0.0
    totals["doc_carry_gain_boundary_64_mean"] = 0.0
    totals["doc_carry_gain_boundary_64_cumulative"] = 0.0
    totals["doc_carry_gain_boundary_64_slope"] = 0.0
    totals["doc_stateful_boundary_64_loss_mean"] = 0.0
    totals["doc_fresh_boundary_64_loss_mean"] = 0.0
    totals["doc_carry_gain_tail_mean"] = 0.0
    totals["doc_carry_gain_tail_cumulative"] = 0.0
    totals["doc_carry_gain_tail_slope"] = 0.0
    batches = 0
    state_carry_batches = 0
    latent_thought_gain_batches = 0
    v7_dynamics_gain_batches = 0
    v7_state_swap_batches = 0
    v7_state_erase_batches = 0
    doc_continuity_batches = 0
    tokens = 0
    tail_start = max(1, int(stateful_boundary_tokens))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask, infer_pad_mask = prepare_attention_mask_for_device(
                batch.get("attention_mask"),
                device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids, attention_mask=attention_mask, infer_pad_mask=infer_pad_mask)
                loss = lm_loss(out["logits"], labels)
                aux = collect_aux_losses(
                    out.get("aux", []),
                    model_config.target_sparsity,
                    sparse_alpha=model_config.semantic_sparse_alpha,
                    alpha_cap=model_config.semantic_router_alpha_cap,
                )
            totals["lm"] += float(loss.detach().cpu())
            totals["alpha"] += float(aux["alpha_mean"].detach().cpu())
            totals["alpha_raw"] += float(aux["alpha_raw_mean"].detach().cpu())
            totals["alpha_prob"] += float(aux["alpha_prob_mean"].detach().cpu())
            totals["alpha_clean_prob"] += float(aux["alpha_clean_prob_mean"].detach().cpu())
            totals["alpha_capped"] += float(aux["alpha_capped_mean"].detach().cpu())
            totals["alpha_downstream"] += float(aux["alpha_downstream_mean"].detach().cpu())
            totals["entropy"] += float(aux["router_entropy"].detach().cpu())
            totals["prior_entropy"] += float(aux["semantic_prior_entropy"].detach().cpu())
            totals["kl"] += float(aux["kl"].detach().cpu())
            totals["load"] += float(aux["load"].detach().cpu())
            totals["sparse"] += float(aux["sparse"].detach().cpu())
            totals["semantic_pred"] += float(aux["semantic_pred"].detach().cpu())
            totals["fusion_mid"] += float(aux["fusion_mid_weight"].detach().cpu())
            totals["fusion_global"] += float(aux["fusion_global_weight"].detach().cpu())
            totals["v4_layer_scale"] += float(aux["v4_layer_scale"].detach().cpu())
            totals["v4_state_norm"] += float(aux["v4_state_norm"].detach().cpu())
            totals["v4_memory_norm"] += float(aux["v4_memory_norm"].detach().cpu())
            totals["v4_memory_gate"] += float(aux["v4_memory_gate"].detach().cpu())
            totals["v4_memory_attention_entropy"] += float(aux["v4_memory_attention_entropy"].detach().cpu())
            totals["v4_memory_read_strength"] += float(aux["v4_memory_read_strength"].detach().cpu())
            totals["v4_memory_novelty"] += float(aux["v4_memory_novelty"].detach().cpu())
            totals["v4_state_gate"] += float(aux["v4_state_gate"].detach().cpu())
            totals["v4_state_confidence"] += float(aux["v4_state_confidence"].detach().cpu())
            totals["v4_state_delta"] += float(aux["v4_state_delta"].detach().cpu())
            totals["v4_state_agreement"] += float(aux["v4_state_agreement"].detach().cpu())
            totals["gate_mix_alpha_weight"] += float(aux["gate_mix_alpha_weight"].detach().cpu())
            totals["gate_mix_clean_weight"] += float(aux["gate_mix_clean_weight"].detach().cpu())
            totals["gate_mix_state_weight"] += float(aux["gate_mix_state_weight"].detach().cpu())
            totals["v5_state_pred"] += float(aux["v5_state_pred"].detach().cpu())
            totals["v5_slot_diversity"] += float(aux["v5_slot_diversity"].detach().cpu())
            totals["v5_slot_stability"] += float(aux["v5_slot_stability"].detach().cpu())
            totals["v5_slot_update_gate"] += float(aux["v5_slot_update_gate"].detach().cpu())
            totals["v5_slot_write_max"] += float(aux["v5_slot_write_max"].detach().cpu())
            totals["v5_slot_write_entropy"] += float(aux["v5_slot_write_entropy"].detach().cpu())
            totals["v5_slot_write_min"] += float(aux["v5_slot_write_min"].detach().cpu())
            totals["v5_slot_write_active"] += float(aux["v5_slot_write_active"].detach().cpu())
            totals["v5_slot_confidence"] += float(aux["v5_slot_confidence"].detach().cpu())
            totals["v5_slot_confidence_std"] += float(aux["v5_slot_confidence_std"].detach().cpu())
            totals["v5_slot_delta"] += float(aux["v5_slot_delta"].detach().cpu())
            totals["v5_slot_cosine"] += float(aux["v5_slot_cosine"].detach().cpu())
            totals["v5_slot_read_entropy"] += float(aux["v5_slot_read_entropy"].detach().cpu())
            totals["v5_slot_read_max"] += float(aux["v5_slot_read_max"].detach().cpu())
            totals["v5_router_semantic_norm"] += float(aux["v5_router_semantic_norm"].detach().cpu())
            totals["v5_router_world_raw_norm"] += float(aux["v5_router_world_raw_norm"].detach().cpu())
            totals["v5_router_world_norm"] += float(aux["v5_router_world_norm"].detach().cpu())
            totals["v5_router_world_ratio"] += float(aux["v5_router_world_ratio"].detach().cpu())
            totals["v5_router_world_cosine"] += float(aux["v5_router_world_cosine"].detach().cpu())
            totals["v5_router_world_gate"] += float(aux["v5_router_world_gate"].detach().cpu())
            totals["v5_router_world_modulation"] += float(aux["v5_router_world_modulation"].detach().cpu())
            totals["v5_router_world_cap"] += float(aux["v5_router_world_cap"].detach().cpu())
            totals["v5_router_memory_norm"] += float(aux["v5_router_memory_norm"].detach().cpu())
            totals["v5_router_memory_ratio"] += float(aux["v5_router_memory_ratio"].detach().cpu())
            totals["v5_router_effective_norm"] += float(aux["v5_router_effective_norm"].detach().cpu())
            totals["v5_semantic_hidden_write_norm"] += float(aux["v5_semantic_hidden_write_norm"].detach().cpu())
            totals["v5_semantic_hidden_write_scale"] += float(aux["v5_semantic_hidden_write_scale"].detach().cpu())
            totals["v5_memory_hidden_write_norm"] += float(aux["v5_memory_hidden_write_norm"].detach().cpu())
            totals["v5_memory_hidden_write_scale"] += float(aux["v5_memory_hidden_write_scale"].detach().cpu())
            totals["v6_self_pred"] += float(aux["v6_self_pred"].detach().cpu())
            totals["v6_slot_diversity"] += float(aux["v6_slot_diversity"].detach().cpu())
            totals["v6_slot_cosine"] += float(aux["v6_slot_cosine"].detach().cpu())
            totals["v6_slot_context_cosine"] += float(aux["v6_slot_context_cosine"].detach().cpu())
            totals["v6_state_delta"] += float(aux["v6_state_delta"].detach().cpu())
            totals["v6_state_norm"] += float(aux["v6_state_norm"].detach().cpu())
            totals["v6_reflection_norm"] += float(aux["v6_reflection_norm"].detach().cpu())
            totals["v6_world_explained_norm"] += float(aux["v6_world_explained_norm"].detach().cpu())
            totals["v6_hidden_residual_norm"] += float(aux["v6_hidden_residual_norm"].detach().cpu())
            totals["v6_world_residual_ratio"] += float(aux["v6_world_residual_ratio"].detach().cpu())
            totals["v6_hidden_write_gate"] += float(aux["v6_hidden_write_gate"].detach().cpu())
            totals["v6_hidden_write_norm"] += float(aux["v6_hidden_write_norm"].detach().cpu())
            totals["v6_hidden_write_scale"] += float(aux["v6_hidden_write_scale"].detach().cpu())
            totals["v6_boundary_entropy"] += float(aux["v6_boundary_entropy"].detach().cpu())
            totals["v6_boundary_self"] += float(aux["v6_boundary_self"].detach().cpu())
            totals["v6_boundary_world"] += float(aux["v6_boundary_world"].detach().cpu())
            totals["v6_boundary_other"] += float(aux["v6_boundary_other"].detach().cpu())
            totals["v6_boundary_unknown"] += float(aux["v6_boundary_unknown"].detach().cpu())
            totals["v6_latent_thought_delta"] += float(aux["v6_latent_thought_delta"].detach().cpu())
            totals["v6_latent_thought_velocity"] += float(aux["v6_latent_thought_velocity"].detach().cpu())
            totals["v6_latent_thought_write_norm"] += float(aux["v6_latent_thought_write_norm"].detach().cpu())
            totals["v6_latent_thought_steps"] += float(aux["v6_latent_thought_steps"].detach().cpu())
            totals["v6_state_evolution_delta"] += float(aux["v6_state_evolution_delta"].detach().cpu())
            totals["v6_state_evolution_world_delta"] += float(aux["v6_state_evolution_world_delta"].detach().cpu())
            totals["v6_state_evolution_self_delta"] += float(aux["v6_state_evolution_self_delta"].detach().cpu())
            totals["v6_state_evolution_memory_delta"] += float(aux["v6_state_evolution_memory_delta"].detach().cpu())
            totals["v6_state_evolution_steps"] += float(aux["v6_state_evolution_steps"].detach().cpu())
            totals["v6_latent_field_token_delta_norm"] += float(
                aux["v6_latent_field_token_delta_norm"].detach().cpu()
            )
            totals["v6_latent_field_token_delta_ratio"] += float(
                aux["v6_latent_field_token_delta_ratio"].detach().cpu()
            )
            totals["v6_latent_field_read_entropy"] += float(aux["v6_latent_field_read_entropy"].detach().cpu())
            totals["v6_latent_field_read_max"] += float(aux["v6_latent_field_read_max"].detach().cpu())
            totals["v6_latent_field_gate"] += float(aux["v6_latent_field_gate"].detach().cpu())
            totals["v7_thought_steps"] += float(aux["v7_thought_steps"].detach().cpu())
            totals["v7_latent_delta"] += float(aux["v7_latent_delta"].detach().cpu())
            totals["v7_latent_velocity"] += float(aux["v7_latent_velocity"].detach().cpu())
            totals["v7_latent_acceleration"] += float(aux["v7_latent_acceleration"].detach().cpu())
            totals["v7_hidden_delta"] += float(aux["v7_hidden_delta"].detach().cpu())
            totals["v7_latent_hidden_write_norm"] += float(aux["v7_latent_hidden_write_norm"].detach().cpu())
            totals["v7_hidden_write_ratio"] += float(aux["v7_hidden_write_ratio"].detach().cpu())
            totals["v7_hidden_write_gate"] += float(aux["v7_hidden_write_gate"].detach().cpu())
            totals["v7_latent_read_entropy"] += float(aux["v7_latent_read_entropy"].detach().cpu())
            totals["v7_latent_read_max"] += float(aux["v7_latent_read_max"].detach().cpu())
            totals["v7_latent_state_norm"] += float(aux["v7_latent_state_norm"].detach().cpu())
            totals["v7_world_state_norm"] += float(aux["v7_world_state_norm"].detach().cpu())
            totals["v7_self_state_norm"] += float(aux["v7_self_state_norm"].detach().cpu())
            totals["v7_controller_state_norm"] += float(aux["v7_controller_state_norm"].detach().cpu())
            totals["v7_world_delta"] += float(aux["v7_world_delta"].detach().cpu())
            totals["v7_self_delta"] += float(aux["v7_self_delta"].detach().cpu())
            totals["v7_controller_delta"] += float(aux["v7_controller_delta"].detach().cpu())
            totals["v7_world_write_gate"] += float(aux["v7_world_write_gate"].detach().cpu())
            totals["v7_self_write_gate"] += float(aux["v7_self_write_gate"].detach().cpu())
            totals["v7_controller_write_gate"] += float(aux["v7_controller_write_gate"].detach().cpu())
            totals["v7_dynamic_depth_enabled"] += float(aux["v7_dynamic_depth_enabled"].detach().cpu())
            totals["v7_dynamic_depth_mean"] += float(aux["v7_dynamic_depth_mean"].detach().cpu())
            totals["v7_dynamic_halt_fraction"] += float(aux["v7_dynamic_halt_fraction"].detach().cpu())
            totals["v7_dynamic_continue_score"] += float(aux["v7_dynamic_continue_score"].detach().cpu())
            totals["v7_dynamic_convergence_threshold"] += float(
                aux["v7_dynamic_convergence_threshold"].detach().cpu()
            )
            totals["v7_causal_segments"] += float(aux["v7_causal_segments"].detach().cpu())
            totals["v7_past_latent_adapt_steps"] += float(aux["v7_past_latent_adapt_steps"].detach().cpu())
            totals["v7_past_latent_read_suppressed"] += float(
                aux["v7_past_latent_read_suppressed"].detach().cpu()
            )
            totals["v7_latent_timescale"] += float(aux["v7_latent_timescale"].detach().cpu())
            totals["v7_world_timescale"] += float(aux["v7_world_timescale"].detach().cpu())
            totals["v7_self_timescale"] += float(aux["v7_self_timescale"].detach().cpu())
            totals["v7_controller_fixed"] += float(aux["v7_controller_fixed"].detach().cpu())
            totals["v7_homeostatic_control_enabled"] += float(
                aux["v7_homeostatic_control_enabled"].detach().cpu()
            )
            totals["v7_homeostatic_dhi"] += float(aux["v7_homeostatic_dhi"].detach().cpu())
            totals["v7_homeostatic_balance_pressure"] += float(
                aux["v7_homeostatic_balance_pressure"].detach().cpu()
            )
            totals["v7_homeostatic_accel_pressure"] += float(
                aux["v7_homeostatic_accel_pressure"].detach().cpu()
            )
            totals["v7_latent_rate_scale"] += float(aux["v7_latent_rate_scale"].detach().cpu())
            totals["v7_world_rate_scale"] += float(aux["v7_world_rate_scale"].detach().cpu())
            totals["v7_self_rate_scale"] += float(aux["v7_self_rate_scale"].detach().cpu())
            totals["v7_hidden_read_rate_scale"] += float(aux["v7_hidden_read_rate_scale"].detach().cpu())
            totals["v7_state_compatibility_enabled"] += float(
                aux["v7_state_compatibility_enabled"].detach().cpu()
            )
            totals["v7_carry_compatibility"] += float(aux["v7_carry_compatibility"].detach().cpu())
            totals["v7_carry_latent_gate"] += float(aux["v7_carry_latent_gate"].detach().cpu())
            totals["v7_carry_world_gate"] += float(aux["v7_carry_world_gate"].detach().cpu())
            totals["v7_carry_self_gate"] += float(aux["v7_carry_self_gate"].detach().cpu())
            totals["v7_carry_controller_gate"] += float(aux["v7_carry_controller_gate"].detach().cpu())
            totals["v7_carry_memory_gate"] += float(aux["v7_carry_memory_gate"].detach().cpu())
            totals["v7_carry_blend_delta"] += float(aux["v7_carry_blend_delta"].detach().cpu())
            totals["v7_hyperspherical_state_enabled"] += float(
                aux["v7_hyperspherical_state_enabled"].detach().cpu()
            )
            totals["v7_causal_summary_enabled"] += float(aux["v7_causal_summary_enabled"].detach().cpu())
            totals["v7_causal_summary_decay"] += float(aux["v7_causal_summary_decay"].detach().cpu())
            totals["v7_adaptive_tau_enabled"] += float(aux["v7_adaptive_tau_enabled"].detach().cpu())
            totals["v7_latent_tau"] += float(aux["v7_latent_tau"].detach().cpu())
            totals["v7_world_tau"] += float(aux["v7_world_tau"].detach().cpu())
            totals["v7_self_tau"] += float(aux["v7_self_tau"].detach().cpu())
            totals["v7_controller_tau"] += float(aux["v7_controller_tau"].detach().cpu())
            totals["v7_ingress_compatibility_enabled"] += float(
                aux["v7_ingress_compatibility_enabled"].detach().cpu()
            )
            totals["v7_ingress_compatibility"] += float(aux["v7_ingress_compatibility"].detach().cpu())
            totals["v7_ingress_latent_gate"] += float(aux["v7_ingress_latent_gate"].detach().cpu())
            totals["v7_ingress_world_gate"] += float(aux["v7_ingress_world_gate"].detach().cpu())
            totals["v7_ingress_self_gate"] += float(aux["v7_ingress_self_gate"].detach().cpu())
            totals["v7_ingress_controller_gate"] += float(aux["v7_ingress_controller_gate"].detach().cpu())
            totals["v7_ingress_memory_gate"] += float(aux["v7_ingress_memory_gate"].detach().cpu())
            totals["v7_ingress_latent_blend_delta"] += float(
                aux["v7_ingress_latent_blend_delta"].detach().cpu()
            )
            totals["v7_ingress_world_blend_delta"] += float(aux["v7_ingress_world_blend_delta"].detach().cpu())
            totals["v7_ingress_self_blend_delta"] += float(aux["v7_ingress_self_blend_delta"].detach().cpu())
            totals["v7_ingress_controller_blend_delta"] += float(
                aux["v7_ingress_controller_blend_delta"].detach().cpu()
            )
            totals["v7_ingress_memory_blend_delta"] += float(aux["v7_ingress_memory_blend_delta"].detach().cpu())
            totals["v7_effective_latent_write_scale"] += float(
                aux["v7_effective_latent_write_scale"].detach().cpu()
            )
            totals["v7_effective_world_write_scale"] += float(
                aux["v7_effective_world_write_scale"].detach().cpu()
            )
            totals["v7_effective_self_write_scale"] += float(
                aux["v7_effective_self_write_scale"].detach().cpu()
            )
            totals["v7_effective_controller_write_scale"] += float(
                aux["v7_effective_controller_write_scale"].detach().cpu()
            )
            probe_context = None
            if state_carry or v7_state_swap or v7_state_erase:
                probe_context = _prepare_stateful_probe_context(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask=infer_pad_mask,
                    use_amp=use_amp,
                    chunk_len=stateful_chunk_len,
                    include_fresh=state_carry,
                )
            if state_carry:
                carry = _estimate_state_carry_gain_from_context(
                    probe_context or {},
                    boundary_decay=stateful_boundary_decay,
                    tail_start=tail_start,
                )
                if carry is not None:
                    totals["state_carry_gain_lm"] += carry["gain_lm"]
                    totals["state_carry_stateful_lm"] += carry["stateful_lm"]
                    totals["state_carry_fresh_lm"] += carry["fresh_lm"]
                    for boundary in BOUNDARY_WINDOWS:
                        totals[f"state_carry_gain_boundary_{boundary}"] += carry[f"gain_boundary_{boundary}"]
                    totals["state_carry_gain_tail"] += carry["gain_tail"]
                    state_carry_batches += 1
            if latent_thought_gain:
                thought = _estimate_latent_thought_gain(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                )
                if thought is not None:
                    gain, thought_loss, disabled_loss = thought
                    totals["latent_thought_gain_lm"] += gain
                    totals["latent_thought_lm"] += thought_loss
                    totals["latent_thought_disabled_lm"] += disabled_loss
                    latent_thought_gain_batches += 1
            if v7_dynamics_gain:
                dynamics = _estimate_v7_dynamics_gain(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                )
                if dynamics is not None:
                    gain, dynamics_loss, disabled_loss = dynamics
                    totals["v7_dynamics_gain_lm"] += gain
                    totals["v7_dynamics_lm"] += dynamics_loss
                    totals["v7_dynamics_disabled_lm"] += disabled_loss
                    v7_dynamics_gain_batches += 1
            if v7_state_swap:
                swap = _estimate_v7_state_swap_penalty(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                    stateful_chunk_len,
                    stateful_boundary_decay,
                    tail_start,
                ) if probe_context is None else _estimate_v7_state_swap_penalty_from_context(
                    model,
                    probe_context,
                    infer_pad_mask=infer_pad_mask,
                    use_amp=use_amp,
                    boundary_decay=stateful_boundary_decay,
                    tail_start=tail_start,
                )
                if swap is not None:
                    totals["v7_state_swap_delta_lm"] += swap["delta_lm"]
                    totals["v7_state_swap_correct_lm"] += swap["correct_lm"]
                    totals["v7_state_swap_wrong_lm"] += swap["wrong_lm"]
                    totals["v7_state_swap_delta_boundary_64"] += swap["delta_boundary_64"]
                    totals["v7_state_swap_delta_tail"] += swap["delta_tail"]
                    v7_state_swap_batches += 1
            if v7_state_erase:
                erase = _estimate_v7_state_erase_sensitivity(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                    stateful_chunk_len,
                    stateful_boundary_decay,
                    tail_start,
                ) if probe_context is None else _estimate_v7_state_erase_sensitivity_from_context(
                    model,
                    probe_context,
                    infer_pad_mask=infer_pad_mask,
                    use_amp=use_amp,
                    boundary_decay=stateful_boundary_decay,
                    tail_start=tail_start,
                )
                if erase is not None:
                    totals["v7_world_erase_delta_lm"] += erase["world_delta_lm"]
                    totals["v7_self_erase_delta_lm"] += erase["self_delta_lm"]
                    totals["v7_latent_erase_delta_lm"] += erase["latent_delta_lm"]
                    totals["v7_state_erase_full_lm"] += erase["full_lm"]
                    totals["v7_world_erase_delta_boundary_64"] += erase["world_delta_boundary_64"]
                    totals["v7_self_erase_delta_boundary_64"] += erase["self_delta_boundary_64"]
                    totals["v7_latent_erase_delta_boundary_64"] += erase["latent_delta_boundary_64"]
                    v7_state_erase_batches += 1
            if doc_continuity and doc_continuity_batches < max(1, doc_continuity_docs):
                continuity = _estimate_doc_continuity(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                    chunk_len=stateful_chunk_len,
                    target_chunks=doc_continuity_chunks,
                    boundary_decay=stateful_boundary_decay,
                    tail_start=tail_start,
                )
                if continuity is not None:
                    totals["doc_carry_gain_mean"] += continuity["gain_mean"]
                    totals["doc_carry_gain_cumulative"] += continuity["gain_cumulative"]
                    totals["doc_carry_gain_slope"] += continuity["gain_slope"]
                    totals["doc_stateful_loss_mean"] += continuity["stateful_mean"]
                    totals["doc_fresh_loss_mean"] += continuity["fresh_mean"]
                    totals["doc_carry_gain_boundary_64_mean"] += continuity["gain_boundary_64_mean"]
                    totals["doc_carry_gain_boundary_64_cumulative"] += continuity["gain_boundary_64_cumulative"]
                    totals["doc_carry_gain_boundary_64_slope"] += continuity["gain_boundary_64_slope"]
                    totals["doc_stateful_boundary_64_loss_mean"] += continuity["stateful_boundary_64_mean"]
                    totals["doc_fresh_boundary_64_loss_mean"] += continuity["fresh_boundary_64_mean"]
                    totals["doc_carry_gain_tail_mean"] += continuity["gain_tail_mean"]
                    totals["doc_carry_gain_tail_cumulative"] += continuity["gain_tail_cumulative"]
                    totals["doc_carry_gain_tail_slope"] += continuity["gain_tail_slope"]
                    doc_continuity_batches += 1
            tokens += int(labels.ne(IGNORE_INDEX).sum().item())
            batches += 1

    if was_training:
        model.train()
    if batches == 0:
        raise RuntimeError("evaluation loader produced no batches")

    val_loss = totals["lm"] / batches
    val_load = totals["load"] / batches
    val_sparse = totals["sparse"] / batches
    val_kl = totals["kl"] / batches
    val_semantic_pred = totals["semantic_pred"] / batches
    val_load_contrib = lambda_load * val_load
    val_sparse_contrib = lambda_sparse * val_sparse
    val_kl_contrib = lambda_kl * val_kl
    val_semantic_pred_contrib = lambda_semantic_pred * val_semantic_pred
    val_state_pred = totals["v5_state_pred"] / batches
    val_slot_diversity = totals["v5_slot_diversity"] / batches
    val_slot_stability = totals["v5_slot_stability"] / batches
    val_self_pred = totals["v6_self_pred"] / batches
    val_self_slot_diversity = totals["v6_slot_diversity"] / batches
    val_state_pred_contrib = lambda_state_pred * val_state_pred
    val_slot_diversity_contrib = lambda_slot_diversity * val_slot_diversity
    val_slot_stability_contrib = lambda_slot_stability * val_slot_stability
    val_self_pred_contrib = lambda_self_pred * val_self_pred
    val_self_slot_diversity_contrib = lambda_self_slot_diversity * val_self_slot_diversity
    val_total_loss = (
        val_loss
        + val_load_contrib
        + val_sparse_contrib
        + val_kl_contrib
        + val_semantic_pred_contrib
        + val_state_pred_contrib
        + val_slot_diversity_contrib
        + val_slot_stability_contrib
        + val_self_pred_contrib
        + val_self_slot_diversity_contrib
    )
    val_aux_loss = val_total_loss - val_loss
    metrics = {
        "val_total_loss": val_total_loss,
        "val_aux_loss": val_aux_loss,
        "val_lm_loss": val_loss,
        "val_ppl": math.exp(min(20.0, val_loss)),
        "val_alpha_mean": totals["alpha"] / batches,
        "val_alpha_raw_mean": totals["alpha_raw"] / batches,
        "val_alpha_prob_mean": totals["alpha_prob"] / batches,
        "val_alpha_clean_prob_mean": totals["alpha_clean_prob"] / batches,
        "val_alpha_capped_mean": totals["alpha_capped"] / batches,
        "val_alpha_downstream_mean": totals["alpha_downstream"] / batches,
        "val_router_entropy": totals["entropy"] / batches,
        "val_semantic_prior_entropy": totals["prior_entropy"] / batches,
        "val_kl": val_kl,
        "val_load": val_load,
        "val_sparse": val_sparse,
        "val_semantic_pred": val_semantic_pred,
        "val_v5_state_pred": val_state_pred,
        "val_v5_slot_diversity": val_slot_diversity,
        "val_v5_slot_stability": val_slot_stability,
        "val_v6_self_pred": val_self_pred,
        "val_v6_slot_diversity": val_self_slot_diversity,
        "val_load_contrib": val_load_contrib,
        "val_sparse_contrib": val_sparse_contrib,
        "val_kl_contrib": val_kl_contrib,
        "val_semantic_pred_contrib": val_semantic_pred_contrib,
        "val_v5_state_pred_contrib": val_state_pred_contrib,
        "val_v5_slot_diversity_contrib": val_slot_diversity_contrib,
        "val_v5_slot_stability_contrib": val_slot_stability_contrib,
        "val_v6_self_pred_contrib": val_self_pred_contrib,
        "val_v6_slot_diversity_contrib": val_self_slot_diversity_contrib,
        "val_fusion_mid_weight": totals["fusion_mid"] / batches,
        "val_fusion_global_weight": totals["fusion_global"] / batches,
        "val_v4_layer_scale": totals["v4_layer_scale"] / batches,
        "val_v4_state_norm": totals["v4_state_norm"] / batches,
        "val_v4_memory_norm": totals["v4_memory_norm"] / batches,
        "val_v4_memory_gate": totals["v4_memory_gate"] / batches,
        "val_v4_memory_attention_entropy": totals["v4_memory_attention_entropy"] / batches,
        "val_v4_memory_read_strength": totals["v4_memory_read_strength"] / batches,
        "val_v4_memory_novelty": totals["v4_memory_novelty"] / batches,
        "val_v4_state_gate": totals["v4_state_gate"] / batches,
        "val_v4_state_confidence": totals["v4_state_confidence"] / batches,
        "val_v4_state_delta": totals["v4_state_delta"] / batches,
        "val_v4_state_agreement": totals["v4_state_agreement"] / batches,
        "val_gate_mix_alpha_weight": totals["gate_mix_alpha_weight"] / batches,
        "val_gate_mix_clean_weight": totals["gate_mix_clean_weight"] / batches,
        "val_gate_mix_state_weight": totals["gate_mix_state_weight"] / batches,
        "val_v5_slot_update_gate": totals["v5_slot_update_gate"] / batches,
        "val_v5_slot_write_max": totals["v5_slot_write_max"] / batches,
        "val_v5_slot_write_entropy": totals["v5_slot_write_entropy"] / batches,
        "val_v5_slot_write_min": totals["v5_slot_write_min"] / batches,
        "val_v5_slot_write_active": totals["v5_slot_write_active"] / batches,
        "val_v5_slot_confidence": totals["v5_slot_confidence"] / batches,
        "val_v5_slot_confidence_std": totals["v5_slot_confidence_std"] / batches,
        "val_v5_slot_delta": totals["v5_slot_delta"] / batches,
        "val_v5_slot_cosine": totals["v5_slot_cosine"] / batches,
        "val_v5_slot_read_entropy": totals["v5_slot_read_entropy"] / batches,
        "val_v5_slot_read_max": totals["v5_slot_read_max"] / batches,
        "val_v5_router_semantic_norm": totals["v5_router_semantic_norm"] / batches,
        "val_v5_router_world_raw_norm": totals["v5_router_world_raw_norm"] / batches,
        "val_v5_router_world_norm": totals["v5_router_world_norm"] / batches,
        "val_v5_router_world_ratio": totals["v5_router_world_ratio"] / batches,
        "val_v5_router_world_cosine": totals["v5_router_world_cosine"] / batches,
        "val_v5_router_world_gate": totals["v5_router_world_gate"] / batches,
        "val_v5_router_world_modulation": totals["v5_router_world_modulation"] / batches,
        "val_v5_router_world_cap": totals["v5_router_world_cap"] / batches,
        "val_v5_router_memory_norm": totals["v5_router_memory_norm"] / batches,
        "val_v5_router_memory_ratio": totals["v5_router_memory_ratio"] / batches,
        "val_v5_router_effective_norm": totals["v5_router_effective_norm"] / batches,
        "val_v5_semantic_hidden_write_norm": totals["v5_semantic_hidden_write_norm"] / batches,
        "val_v5_semantic_hidden_write_scale": totals["v5_semantic_hidden_write_scale"] / batches,
        "val_v5_memory_hidden_write_norm": totals["v5_memory_hidden_write_norm"] / batches,
        "val_v5_memory_hidden_write_scale": totals["v5_memory_hidden_write_scale"] / batches,
        "val_v6_slot_cosine": totals["v6_slot_cosine"] / batches,
        "val_v6_slot_context_cosine": totals["v6_slot_context_cosine"] / batches,
        "val_v6_state_delta": totals["v6_state_delta"] / batches,
        "val_v6_state_norm": totals["v6_state_norm"] / batches,
        "val_v6_reflection_norm": totals["v6_reflection_norm"] / batches,
        "val_v6_world_explained_norm": totals["v6_world_explained_norm"] / batches,
        "val_v6_hidden_residual_norm": totals["v6_hidden_residual_norm"] / batches,
        "val_v6_world_residual_ratio": totals["v6_world_residual_ratio"] / batches,
        "val_v6_hidden_write_gate": totals["v6_hidden_write_gate"] / batches,
        "val_v6_hidden_write_norm": totals["v6_hidden_write_norm"] / batches,
        "val_v6_hidden_write_scale": totals["v6_hidden_write_scale"] / batches,
        "val_v6_boundary_entropy": totals["v6_boundary_entropy"] / batches,
        "val_v6_boundary_self": totals["v6_boundary_self"] / batches,
        "val_v6_boundary_world": totals["v6_boundary_world"] / batches,
        "val_v6_boundary_other": totals["v6_boundary_other"] / batches,
        "val_v6_boundary_unknown": totals["v6_boundary_unknown"] / batches,
        "val_v6_latent_thought_delta": totals["v6_latent_thought_delta"] / batches,
        "val_v6_latent_thought_velocity": totals["v6_latent_thought_velocity"] / batches,
        "val_v6_latent_thought_write_norm": totals["v6_latent_thought_write_norm"] / batches,
        "val_v6_latent_thought_steps": totals["v6_latent_thought_steps"] / batches,
        "val_v6_state_evolution_delta": totals["v6_state_evolution_delta"] / batches,
        "val_v6_state_evolution_world_delta": totals["v6_state_evolution_world_delta"] / batches,
        "val_v6_state_evolution_self_delta": totals["v6_state_evolution_self_delta"] / batches,
        "val_v6_state_evolution_memory_delta": totals["v6_state_evolution_memory_delta"] / batches,
        "val_v6_state_evolution_steps": totals["v6_state_evolution_steps"] / batches,
        "val_v6_latent_field_token_delta_norm": totals["v6_latent_field_token_delta_norm"] / batches,
        "val_v6_latent_field_token_delta_ratio": totals["v6_latent_field_token_delta_ratio"] / batches,
        "val_v6_latent_field_read_entropy": totals["v6_latent_field_read_entropy"] / batches,
        "val_v6_latent_field_read_max": totals["v6_latent_field_read_max"] / batches,
        "val_v6_latent_field_gate": totals["v6_latent_field_gate"] / batches,
        "val_v7_thought_steps": totals["v7_thought_steps"] / batches,
        "val_v7_latent_delta": totals["v7_latent_delta"] / batches,
        "val_v7_latent_velocity": totals["v7_latent_velocity"] / batches,
        "val_v7_latent_acceleration": totals["v7_latent_acceleration"] / batches,
        "val_v7_hidden_delta": totals["v7_hidden_delta"] / batches,
        "val_v7_latent_hidden_write_norm": totals["v7_latent_hidden_write_norm"] / batches,
        "val_v7_hidden_write_ratio": totals["v7_hidden_write_ratio"] / batches,
        "val_v7_hidden_write_gate": totals["v7_hidden_write_gate"] / batches,
        "val_v7_latent_read_entropy": totals["v7_latent_read_entropy"] / batches,
        "val_v7_latent_read_max": totals["v7_latent_read_max"] / batches,
        "val_v7_latent_state_norm": totals["v7_latent_state_norm"] / batches,
        "val_v7_world_state_norm": totals["v7_world_state_norm"] / batches,
        "val_v7_self_state_norm": totals["v7_self_state_norm"] / batches,
        "val_v7_controller_state_norm": totals["v7_controller_state_norm"] / batches,
        "val_v7_world_delta": totals["v7_world_delta"] / batches,
        "val_v7_self_delta": totals["v7_self_delta"] / batches,
        "val_v7_controller_delta": totals["v7_controller_delta"] / batches,
        "val_v7_world_write_gate": totals["v7_world_write_gate"] / batches,
        "val_v7_self_write_gate": totals["v7_self_write_gate"] / batches,
        "val_v7_controller_write_gate": totals["v7_controller_write_gate"] / batches,
        "val_v7_dynamic_depth_enabled": totals["v7_dynamic_depth_enabled"] / batches,
        "val_v7_dynamic_depth_mean": totals["v7_dynamic_depth_mean"] / batches,
        "val_v7_dynamic_halt_fraction": totals["v7_dynamic_halt_fraction"] / batches,
        "val_v7_dynamic_continue_score": totals["v7_dynamic_continue_score"] / batches,
        "val_v7_dynamic_convergence_threshold": totals["v7_dynamic_convergence_threshold"] / batches,
        "val_v7_causal_segments": totals["v7_causal_segments"] / batches,
        "val_v7_past_latent_adapt_steps": totals["v7_past_latent_adapt_steps"] / batches,
        "val_v7_past_latent_read_suppressed": totals["v7_past_latent_read_suppressed"] / batches,
        "val_v7_latent_timescale": totals["v7_latent_timescale"] / batches,
        "val_v7_world_timescale": totals["v7_world_timescale"] / batches,
        "val_v7_self_timescale": totals["v7_self_timescale"] / batches,
        "val_v7_controller_fixed": totals["v7_controller_fixed"] / batches,
        "val_v7_homeostatic_control_enabled": totals["v7_homeostatic_control_enabled"] / batches,
        "val_v7_homeostatic_dhi": totals["v7_homeostatic_dhi"] / batches,
        "val_v7_homeostatic_balance_pressure": totals["v7_homeostatic_balance_pressure"] / batches,
        "val_v7_homeostatic_accel_pressure": totals["v7_homeostatic_accel_pressure"] / batches,
        "val_v7_latent_rate_scale": totals["v7_latent_rate_scale"] / batches,
        "val_v7_world_rate_scale": totals["v7_world_rate_scale"] / batches,
        "val_v7_self_rate_scale": totals["v7_self_rate_scale"] / batches,
        "val_v7_hidden_read_rate_scale": totals["v7_hidden_read_rate_scale"] / batches,
        "val_v7_state_compatibility_enabled": totals["v7_state_compatibility_enabled"] / batches,
        "val_v7_carry_compatibility": totals["v7_carry_compatibility"] / batches,
        "val_v7_carry_latent_gate": totals["v7_carry_latent_gate"] / batches,
        "val_v7_carry_world_gate": totals["v7_carry_world_gate"] / batches,
        "val_v7_carry_self_gate": totals["v7_carry_self_gate"] / batches,
        "val_v7_carry_controller_gate": totals["v7_carry_controller_gate"] / batches,
        "val_v7_carry_memory_gate": totals["v7_carry_memory_gate"] / batches,
        "val_v7_carry_blend_delta": totals["v7_carry_blend_delta"] / batches,
        "val_v7_hyperspherical_state_enabled": totals["v7_hyperspherical_state_enabled"] / batches,
        "val_v7_causal_summary_enabled": totals["v7_causal_summary_enabled"] / batches,
        "val_v7_causal_summary_decay": totals["v7_causal_summary_decay"] / batches,
        "val_v7_adaptive_tau_enabled": totals["v7_adaptive_tau_enabled"] / batches,
        "val_v7_latent_tau": totals["v7_latent_tau"] / batches,
        "val_v7_world_tau": totals["v7_world_tau"] / batches,
        "val_v7_self_tau": totals["v7_self_tau"] / batches,
        "val_v7_controller_tau": totals["v7_controller_tau"] / batches,
        "val_v7_ingress_compatibility_enabled": totals["v7_ingress_compatibility_enabled"] / batches,
        "val_v7_ingress_compatibility": totals["v7_ingress_compatibility"] / batches,
        "val_v7_ingress_latent_gate": totals["v7_ingress_latent_gate"] / batches,
        "val_v7_ingress_world_gate": totals["v7_ingress_world_gate"] / batches,
        "val_v7_ingress_self_gate": totals["v7_ingress_self_gate"] / batches,
        "val_v7_ingress_controller_gate": totals["v7_ingress_controller_gate"] / batches,
        "val_v7_ingress_memory_gate": totals["v7_ingress_memory_gate"] / batches,
        "val_v7_ingress_latent_blend_delta": totals["v7_ingress_latent_blend_delta"] / batches,
        "val_v7_ingress_world_blend_delta": totals["v7_ingress_world_blend_delta"] / batches,
        "val_v7_ingress_self_blend_delta": totals["v7_ingress_self_blend_delta"] / batches,
        "val_v7_ingress_controller_blend_delta": totals["v7_ingress_controller_blend_delta"] / batches,
        "val_v7_ingress_memory_blend_delta": totals["v7_ingress_memory_blend_delta"] / batches,
        "val_v7_effective_latent_write_scale": totals["v7_effective_latent_write_scale"] / batches,
        "val_v7_effective_world_write_scale": totals["v7_effective_world_write_scale"] / batches,
        "val_v7_effective_self_write_scale": totals["v7_effective_self_write_scale"] / batches,
        "val_v7_effective_controller_write_scale": totals["v7_effective_controller_write_scale"] / batches,
        "val_state_carry_gain_lm": totals["state_carry_gain_lm"] / state_carry_batches
        if state_carry_batches
        else 0.0,
        "val_state_carry_stateful_lm": totals["state_carry_stateful_lm"] / state_carry_batches
        if state_carry_batches
        else 0.0,
        "val_state_carry_fresh_lm": totals["state_carry_fresh_lm"] / state_carry_batches
        if state_carry_batches
        else 0.0,
        "val_state_carry_batches": float(state_carry_batches),
        "val_doc_carry_gain_mean": totals["doc_carry_gain_mean"] / doc_continuity_batches if doc_continuity_batches else 0.0,
        "val_doc_carry_gain_cumulative": totals["doc_carry_gain_cumulative"] / doc_continuity_batches
        if doc_continuity_batches
        else 0.0,
        "val_doc_carry_gain_slope": totals["doc_carry_gain_slope"] / doc_continuity_batches if doc_continuity_batches else 0.0,
        "val_doc_stateful_loss_mean": totals["doc_stateful_loss_mean"] / doc_continuity_batches
        if doc_continuity_batches
        else 0.0,
        "val_doc_fresh_loss_mean": totals["doc_fresh_loss_mean"] / doc_continuity_batches
        if doc_continuity_batches
        else 0.0,
        "val_doc_continuity_batches": float(doc_continuity_batches),
        "val_latent_thought_gain_lm": totals["latent_thought_gain_lm"] / latent_thought_gain_batches
        if latent_thought_gain_batches
        else 0.0,
        "val_latent_thought_lm": totals["latent_thought_lm"] / latent_thought_gain_batches
        if latent_thought_gain_batches
        else 0.0,
        "val_latent_thought_disabled_lm": totals["latent_thought_disabled_lm"] / latent_thought_gain_batches
        if latent_thought_gain_batches
        else 0.0,
        "val_latent_thought_gain_batches": float(latent_thought_gain_batches),
        "val_v7_dynamics_gain_lm": totals["v7_dynamics_gain_lm"] / v7_dynamics_gain_batches
        if v7_dynamics_gain_batches
        else 0.0,
        "val_v7_dynamics_lm": totals["v7_dynamics_lm"] / v7_dynamics_gain_batches
        if v7_dynamics_gain_batches
        else 0.0,
        "val_v7_dynamics_disabled_lm": totals["v7_dynamics_disabled_lm"] / v7_dynamics_gain_batches
        if v7_dynamics_gain_batches
        else 0.0,
        "val_v7_dynamics_gain_batches": float(v7_dynamics_gain_batches),
        "val_v7_state_swap_delta_lm": totals["v7_state_swap_delta_lm"] / v7_state_swap_batches
        if v7_state_swap_batches
        else 0.0,
        "val_v7_state_swap_correct_lm": totals["v7_state_swap_correct_lm"] / v7_state_swap_batches
        if v7_state_swap_batches
        else 0.0,
        "val_v7_state_swap_wrong_lm": totals["v7_state_swap_wrong_lm"] / v7_state_swap_batches
        if v7_state_swap_batches
        else 0.0,
        "val_v7_state_swap_batches": float(v7_state_swap_batches),
        "val_v7_world_erase_delta_lm": totals["v7_world_erase_delta_lm"] / v7_state_erase_batches
        if v7_state_erase_batches
        else 0.0,
        "val_v7_self_erase_delta_lm": totals["v7_self_erase_delta_lm"] / v7_state_erase_batches
        if v7_state_erase_batches
        else 0.0,
        "val_v7_latent_erase_delta_lm": totals["v7_latent_erase_delta_lm"] / v7_state_erase_batches
        if v7_state_erase_batches
        else 0.0,
        "val_v7_state_erase_full_lm": totals["v7_state_erase_full_lm"] / v7_state_erase_batches
        if v7_state_erase_batches
        else 0.0,
        "val_v7_state_erase_batches": float(v7_state_erase_batches),
        "val_batches": float(batches),
        "val_tokens": float(tokens),
    }
    for boundary in BOUNDARY_WINDOWS:
        metrics[f"val_state_carry_gain_boundary_{boundary}"] = (
            totals[f"state_carry_gain_boundary_{boundary}"] / state_carry_batches if state_carry_batches else 0.0
        )
    metrics["val_state_carry_gain_tail"] = totals["state_carry_gain_tail"] / state_carry_batches if state_carry_batches else 0.0
    metrics["val_doc_carry_gain_boundary_64_mean"] = (
        totals["doc_carry_gain_boundary_64_mean"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_carry_gain_boundary_64_cumulative"] = (
        totals["doc_carry_gain_boundary_64_cumulative"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_carry_gain_boundary_64_slope"] = (
        totals["doc_carry_gain_boundary_64_slope"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_stateful_boundary_64_loss_mean"] = (
        totals["doc_stateful_boundary_64_loss_mean"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_fresh_boundary_64_loss_mean"] = (
        totals["doc_fresh_boundary_64_loss_mean"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_carry_gain_tail_mean"] = (
        totals["doc_carry_gain_tail_mean"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_carry_gain_tail_cumulative"] = (
        totals["doc_carry_gain_tail_cumulative"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_doc_carry_gain_tail_slope"] = (
        totals["doc_carry_gain_tail_slope"] / doc_continuity_batches if doc_continuity_batches else 0.0
    )
    metrics["val_v7_state_swap_delta_boundary_64"] = (
        totals["v7_state_swap_delta_boundary_64"] / v7_state_swap_batches if v7_state_swap_batches else 0.0
    )
    metrics["val_v7_state_swap_delta_tail"] = (
        totals["v7_state_swap_delta_tail"] / v7_state_swap_batches if v7_state_swap_batches else 0.0
    )
    metrics["val_v7_world_erase_delta_boundary_64"] = (
        totals["v7_world_erase_delta_boundary_64"] / v7_state_erase_batches if v7_state_erase_batches else 0.0
    )
    metrics["val_v7_self_erase_delta_boundary_64"] = (
        totals["v7_self_erase_delta_boundary_64"] / v7_state_erase_batches if v7_state_erase_batches else 0.0
    )
    metrics["val_v7_latent_erase_delta_boundary_64"] = (
        totals["v7_latent_erase_delta_boundary_64"] / v7_state_erase_batches if v7_state_erase_batches else 0.0
    )
    return metrics
