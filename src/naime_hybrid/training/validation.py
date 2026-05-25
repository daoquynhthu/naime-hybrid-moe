import math

import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.models.state_packet import NAIMEStatePacket

from .losses import IGNORE_INDEX, collect_aux_losses, lm_loss
from .masks import prepare_attention_mask_for_device


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
) -> NAIMEStatePacket:
    return NAIMEStatePacket(
        world_state=world_state,
        self_state=self_state,
        latent_field=latent_field,
        memory=memory,
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
    )


def _estimate_state_carry_gain(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
) -> tuple[float, float, float] | None:
    if not _supports_state_packet(model) or input_ids.size(1) < 4:
        return None

    split = input_ids.size(1) // 2
    first_ids = input_ids[:, :split]
    second_ids = input_ids[:, split:]
    second_labels = labels[:, split:]
    if second_labels.ne(IGNORE_INDEX).sum().item() == 0:
        return None

    first_mask = attention_mask[:, :split] if attention_mask is not None else None
    second_mask = attention_mask[:, split:] if attention_mask is not None else None
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        first_out = model(
            first_ids,
            attention_mask=first_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            return_logits=False,
            return_state=True,
        )
        packet = first_out.get("state_packet")
        if packet is None:
            return None
        stateful_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=packet,
        )
        fresh_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
        )
        stateful_loss = lm_loss(stateful_out["logits"], second_labels)
        fresh_loss = lm_loss(fresh_out["logits"], second_labels)
    stateful = float(stateful_loss.detach().cpu())
    fresh = float(fresh_loss.detach().cpu())
    return fresh - stateful, stateful, fresh


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
) -> tuple[float, float, float] | None:
    if not _supports_state_packet(model) or input_ids.size(0) < 2 or input_ids.size(1) < 4:
        return None

    split = input_ids.size(1) // 2
    first_ids = input_ids[:, :split]
    second_ids = input_ids[:, split:]
    second_labels = labels[:, split:]
    if second_labels.ne(IGNORE_INDEX).sum().item() == 0:
        return None

    first_mask = attention_mask[:, :split] if attention_mask is not None else None
    second_mask = attention_mask[:, split:] if attention_mask is not None else None
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        first_out = model(
            first_ids,
            attention_mask=first_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            return_logits=False,
            return_state=True,
        )
        packet = first_out.get("state_packet")
        if packet is None:
            return None
        correct_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=packet,
        )
        swapped_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_swap_packet_batch(packet),
        )
        correct_loss = lm_loss(correct_out["logits"], second_labels)
        swapped_loss = lm_loss(swapped_out["logits"], second_labels)
    correct = float(correct_loss.detach().cpu())
    swapped = float(swapped_loss.detach().cpu())
    return swapped - correct, correct, swapped


def _estimate_v7_state_erase_sensitivity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    infer_pad_mask: bool | None,
    use_amp: bool,
) -> tuple[float, float, float, float] | None:
    native_model = _native_model(model)
    if getattr(native_model, "typed_dynamics", None) is None or input_ids.size(1) < 4:
        return None

    split = input_ids.size(1) // 2
    first_ids = input_ids[:, :split]
    second_ids = input_ids[:, split:]
    second_labels = labels[:, split:]
    if second_labels.ne(IGNORE_INDEX).sum().item() == 0:
        return None

    first_mask = attention_mask[:, :split] if attention_mask is not None else None
    second_mask = attention_mask[:, split:] if attention_mask is not None else None
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        first_out = model(
            first_ids,
            attention_mask=first_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            return_logits=False,
            return_state=True,
        )
        packet = first_out.get("state_packet")
        if packet is None:
            return None
        full_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=packet,
        )
        world_erased_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_packet_like(
                packet,
                world_state=None,
                self_state=packet.self_state,
                latent_field=packet.latent_field,
                memory=packet.memory,
            ),
        )
        self_erased_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_packet_like(
                packet,
                world_state=packet.world_state,
                self_state=None,
                latent_field=packet.latent_field,
                memory=packet.memory,
            ),
        )
        latent_erased_out = model(
            second_ids,
            attention_mask=second_mask,
            infer_pad_mask=infer_pad_mask,
            return_aux=False,
            past_state=_packet_like(
                packet,
                world_state=packet.world_state,
                self_state=packet.self_state,
                latent_field=None,
                memory=packet.memory,
            ),
        )
        full_loss = lm_loss(full_out["logits"], second_labels)
        world_loss = lm_loss(world_erased_out["logits"], second_labels)
        self_loss = lm_loss(self_erased_out["logits"], second_labels)
        latent_loss = lm_loss(latent_erased_out["logits"], second_labels)
    full = float(full_loss.detach().cpu())
    return (
        float(world_loss.detach().cpu()) - full,
        float(self_loss.detach().cpu()) - full,
        float(latent_loss.detach().cpu()) - full,
        full,
    )


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
            "v7_world_delta",
            "v7_self_delta",
            "v7_world_write_gate",
            "v7_self_write_gate",
            "v7_dynamic_depth_enabled",
            "v7_dynamic_depth_mean",
            "v7_dynamic_halt_fraction",
            "v7_dynamic_continue_score",
            "v7_dynamic_convergence_threshold",
            "v7_past_latent_adapt_steps",
            "v7_past_latent_read_suppressed",
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
        ]
    }
    batches = 0
    state_carry_batches = 0
    latent_thought_gain_batches = 0
    v7_dynamics_gain_batches = 0
    v7_state_swap_batches = 0
    v7_state_erase_batches = 0
    tokens = 0
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
            totals["v7_world_delta"] += float(aux["v7_world_delta"].detach().cpu())
            totals["v7_self_delta"] += float(aux["v7_self_delta"].detach().cpu())
            totals["v7_world_write_gate"] += float(aux["v7_world_write_gate"].detach().cpu())
            totals["v7_self_write_gate"] += float(aux["v7_self_write_gate"].detach().cpu())
            totals["v7_dynamic_depth_enabled"] += float(aux["v7_dynamic_depth_enabled"].detach().cpu())
            totals["v7_dynamic_depth_mean"] += float(aux["v7_dynamic_depth_mean"].detach().cpu())
            totals["v7_dynamic_halt_fraction"] += float(aux["v7_dynamic_halt_fraction"].detach().cpu())
            totals["v7_dynamic_continue_score"] += float(aux["v7_dynamic_continue_score"].detach().cpu())
            totals["v7_dynamic_convergence_threshold"] += float(
                aux["v7_dynamic_convergence_threshold"].detach().cpu()
            )
            totals["v7_past_latent_adapt_steps"] += float(aux["v7_past_latent_adapt_steps"].detach().cpu())
            totals["v7_past_latent_read_suppressed"] += float(
                aux["v7_past_latent_read_suppressed"].detach().cpu()
            )
            if state_carry:
                carry = _estimate_state_carry_gain(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                )
                if carry is not None:
                    gain, stateful_loss, fresh_loss = carry
                    totals["state_carry_gain_lm"] += gain
                    totals["state_carry_stateful_lm"] += stateful_loss
                    totals["state_carry_fresh_lm"] += fresh_loss
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
                )
                if swap is not None:
                    delta, correct_loss, wrong_loss = swap
                    totals["v7_state_swap_delta_lm"] += delta
                    totals["v7_state_swap_correct_lm"] += correct_loss
                    totals["v7_state_swap_wrong_lm"] += wrong_loss
                    v7_state_swap_batches += 1
            if v7_state_erase:
                erase = _estimate_v7_state_erase_sensitivity(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    infer_pad_mask,
                    use_amp,
                )
                if erase is not None:
                    world_delta, self_delta, latent_delta, full_loss = erase
                    totals["v7_world_erase_delta_lm"] += world_delta
                    totals["v7_self_erase_delta_lm"] += self_delta
                    totals["v7_latent_erase_delta_lm"] += latent_delta
                    totals["v7_state_erase_full_lm"] += full_loss
                    v7_state_erase_batches += 1
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
    return {
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
        "val_v7_world_delta": totals["v7_world_delta"] / batches,
        "val_v7_self_delta": totals["v7_self_delta"] / batches,
        "val_v7_world_write_gate": totals["v7_world_write_gate"] / batches,
        "val_v7_self_write_gate": totals["v7_self_write_gate"] / batches,
        "val_v7_dynamic_depth_enabled": totals["v7_dynamic_depth_enabled"] / batches,
        "val_v7_dynamic_depth_mean": totals["v7_dynamic_depth_mean"] / batches,
        "val_v7_dynamic_halt_fraction": totals["v7_dynamic_halt_fraction"] / batches,
        "val_v7_dynamic_continue_score": totals["v7_dynamic_continue_score"] / batches,
        "val_v7_dynamic_convergence_threshold": totals["v7_dynamic_convergence_threshold"] / batches,
        "val_v7_past_latent_adapt_steps": totals["v7_past_latent_adapt_steps"] / batches,
        "val_v7_past_latent_read_suppressed": totals["v7_past_latent_read_suppressed"] / batches,
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
