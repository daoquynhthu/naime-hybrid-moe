import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=0,
    )


def _select_alpha(
    semantic: dict,
    sparse_alpha: str,
    alpha_cap: float,
) -> torch.Tensor:
    if sparse_alpha == "alpha":
        return semantic["alpha"].float()
    if sparse_alpha == "prob":
        return semantic["gate_prob"].float()
    if sparse_alpha == "clean_prob":
        return semantic["gate_clean_prob"].float()
    if sparse_alpha == "capped_alpha":
        alpha = semantic["alpha"].float()
        return alpha.clamp(max=alpha_cap) if alpha_cap > 0 else alpha
    if sparse_alpha == "downstream":
        if "token_alpha_downstream" in semantic:
            return semantic["token_alpha_downstream"].float()
        alpha = semantic["alpha"].float()
        return alpha.clamp(max=alpha_cap) if alpha_cap > 0 else alpha
    raise ValueError("semantic_sparse_alpha must be alpha, prob, clean_prob, capped_alpha, or downstream")


def collect_aux_losses(
    aux_by_layer: list[dict],
    target_sparsity: float,
    sparse_alpha: str = "alpha",
    alpha_cap: float = 0.0,
) -> dict[str, torch.Tensor]:
    device = None
    load_losses = []
    sparse_losses = []
    kl_losses = []
    semantic_pred_losses = []
    router_entropies = []
    semantic_prior_entropies = []
    fusion_mid_weights = []
    fusion_global_weights = []
    alpha_means = []
    alpha_raw_means = []
    alpha_prob_means = []
    alpha_clean_prob_means = []
    alpha_capped_means = []
    alpha_downstream_means = []
    v4_layer_scales = []
    v4_state_norms = []
    v4_memory_norms = []
    v4_memory_gates = []
    v4_memory_entropies = []
    v4_memory_read_strengths = []
    v4_memory_novelties = []
    v4_state_gates = []
    v4_state_confidences = []
    v4_state_deltas = []
    v4_state_agreements = []
    gate_mix_alpha_weights = []
    gate_mix_clean_weights = []
    gate_mix_state_weights = []
    v5_state_preds = []
    v5_slot_diversities = []
    v5_slot_stabilities = []
    v5_slot_update_gates = []
    v5_slot_write_maxes = []
    v5_slot_write_entropies = []
    v5_slot_write_mins = []
    v5_slot_write_actives = []
    v5_slot_confidences = []
    v5_slot_confidence_stds = []
    v5_slot_deltas = []
    v5_slot_cosines = []
    v5_slot_read_entropies = []
    v5_slot_read_maxes = []
    v6_self_preds = []
    v6_slot_diversities = []
    v6_slot_cosines = []
    v6_slot_context_cosines = []
    v6_state_deltas = []
    v6_state_norms = []
    v6_reflection_norms = []
    v6_boundary_entropies = []
    v6_boundary_selfs = []
    v6_boundary_worlds = []
    v6_boundary_others = []
    v6_boundary_unknowns = []
    dispatch_denses = []

    for layer_aux in aux_by_layer:
        if not layer_aux:
            continue
        moe = layer_aux.get("moe")
        if moe is not None:
            load_losses.append(moe["load_balance"])
            router_entropies.append(moe["router_entropy"])
            if "semantic_prior_entropy" in moe:
                semantic_prior_entropies.append(moe["semantic_prior_entropy"])
            if "dispatch_dense" in moe:
                dispatch_denses.append(moe["dispatch_dense"].float())
            device = moe["load_balance"].device

        semantic = layer_aux.get("semantic")
        if semantic is not None:
            alpha_raw = semantic["alpha"].float()
            alpha_prob = semantic["gate_prob"].float()
            alpha_clean_prob = semantic["gate_clean_prob"].float()
            alpha_capped = alpha_raw.clamp(max=alpha_cap) if alpha_cap > 0 else alpha_raw
            alpha_downstream = semantic.get("token_alpha_downstream")
            alpha = _select_alpha(semantic, sparse_alpha, alpha_cap)
            alpha_mean = alpha.mean()
            alpha_means.append(alpha_mean)
            alpha_raw_means.append(alpha_raw.mean())
            alpha_prob_means.append(alpha_prob.mean())
            alpha_clean_prob_means.append(alpha_clean_prob.mean())
            alpha_capped_means.append(alpha_capped.mean())
            if alpha_downstream is not None:
                alpha_downstream_means.append(alpha_downstream.float().mean())
            alpha_clamped = alpha_mean.clamp(1e-6, 1.0 - 1e-6)
            rho = torch.tensor(target_sparsity, device=alpha.device, dtype=alpha.dtype)
            sparse_losses.append(
                rho * torch.log(rho / alpha_clamped) + (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - alpha_clamped))
            )
            kl_losses.append(semantic["kl"].float().mean())
            if "semantic_pred_loss" in semantic:
                semantic_pred_losses.append(semantic["semantic_pred_loss"])
            if "fusion_weights" in semantic:
                fusion_weights = semantic["fusion_weights"].float()
                if fusion_weights.numel() > 0:
                    fusion_mid_weights.append(fusion_weights[..., 1].mean())
                    fusion_global_weights.append(fusion_weights[..., 2].mean())
            if "gate_mix_weights" in semantic:
                gate_mix_weights = semantic["gate_mix_weights"].float()
                if gate_mix_weights.numel() > 0:
                    gate_mix_alpha_weights.append(gate_mix_weights[..., 0].mean())
                    gate_mix_clean_weights.append(gate_mix_weights[..., 1].mean())
                    gate_mix_state_weights.append(gate_mix_weights[..., 2].mean())
            device = alpha.device

        v4 = layer_aux.get("v4")
        if v4 is not None:
            if "layer_scale" in v4:
                v4_layer_scales.append(v4["layer_scale"].float())
            if "state_norm" in v4:
                v4_state_norms.append(v4["state_norm"].float())
            if "memory_norm" in v4:
                v4_memory_norms.append(v4["memory_norm"].float())
            if "memory_gate" in v4:
                v4_memory_gates.append(v4["memory_gate"].float())
            if "memory_attention_entropy" in v4:
                v4_memory_entropies.append(v4["memory_attention_entropy"].float())
            if "memory_read_strength" in v4:
                v4_memory_read_strengths.append(v4["memory_read_strength"].float())
            if "memory_novelty" in v4:
                v4_memory_novelties.append(v4["memory_novelty"].float())
            if "state_gate" in v4:
                v4_state_gates.append(v4["state_gate"].float())
            if "state_confidence" in v4:
                v4_state_confidences.append(v4["state_confidence"].float())
            if "state_delta" in v4:
                v4_state_deltas.append(v4["state_delta"].float())
            if "state_agreement" in v4:
                v4_state_agreements.append(v4["state_agreement"].float())
            for value in v4.values():
                if torch.is_tensor(value):
                    device = value.device
                    break

        v5 = layer_aux.get("v5")
        if v5 is not None:
            if "state_pred" in v5:
                v5_state_preds.append(v5["state_pred"].float())
            if "slot_diversity" in v5:
                v5_slot_diversities.append(v5["slot_diversity"].float())
            if "slot_stability" in v5:
                v5_slot_stabilities.append(v5["slot_stability"].float())
            if "slot_update_gate" in v5:
                v5_slot_update_gates.append(v5["slot_update_gate"].float())
            if "slot_write_max" in v5:
                v5_slot_write_maxes.append(v5["slot_write_max"].float())
            if "slot_write_entropy" in v5:
                v5_slot_write_entropies.append(v5["slot_write_entropy"].float())
            if "slot_write_min" in v5:
                v5_slot_write_mins.append(v5["slot_write_min"].float())
            if "slot_write_active" in v5:
                v5_slot_write_actives.append(v5["slot_write_active"].float())
            if "slot_confidence" in v5:
                v5_slot_confidences.append(v5["slot_confidence"].float())
            if "slot_confidence_std" in v5:
                v5_slot_confidence_stds.append(v5["slot_confidence_std"].float())
            if "slot_delta" in v5:
                v5_slot_deltas.append(v5["slot_delta"].float())
            if "slot_cosine" in v5:
                v5_slot_cosines.append(v5["slot_cosine"].float())
            if "slot_read_entropy" in v5:
                v5_slot_read_entropies.append(v5["slot_read_entropy"].float())
            if "slot_read_max" in v5:
                v5_slot_read_maxes.append(v5["slot_read_max"].float())
            for value in v5.values():
                if torch.is_tensor(value):
                    device = value.device
                    break

        v6 = layer_aux.get("v6")
        if v6 is not None:
            if "self_pred" in v6:
                v6_self_preds.append(v6["self_pred"].float())
            if "slot_diversity" in v6:
                v6_slot_diversities.append(v6["slot_diversity"].float())
            if "slot_cosine" in v6:
                v6_slot_cosines.append(v6["slot_cosine"].float())
            if "slot_context_cosine" in v6:
                v6_slot_context_cosines.append(v6["slot_context_cosine"].float())
            if "state_delta" in v6:
                v6_state_deltas.append(v6["state_delta"].float())
            if "state_norm" in v6:
                v6_state_norms.append(v6["state_norm"].float())
            if "reflection_norm" in v6:
                v6_reflection_norms.append(v6["reflection_norm"].float())
            if "boundary_entropy" in v6:
                v6_boundary_entropies.append(v6["boundary_entropy"].float())
            if "boundary_self" in v6:
                v6_boundary_selfs.append(v6["boundary_self"].float())
            if "boundary_world" in v6:
                v6_boundary_worlds.append(v6["boundary_world"].float())
            if "boundary_other" in v6:
                v6_boundary_others.append(v6["boundary_other"].float())
            if "boundary_unknown" in v6:
                v6_boundary_unknowns.append(v6["boundary_unknown"].float())
            for value in v6.values():
                if torch.is_tensor(value):
                    device = value.device
                    break

    if device is None:
        device = torch.device("cpu")

    zero = torch.tensor(0.0, device=device)
    return {
        "load": torch.stack(load_losses).mean() if load_losses else zero,
        "sparse": torch.stack(sparse_losses).mean() if sparse_losses else zero,
        "kl": torch.stack(kl_losses).mean() if kl_losses else zero,
        "semantic_pred": torch.stack(semantic_pred_losses).mean() if semantic_pred_losses else zero,
        "router_entropy": torch.stack(router_entropies).mean() if router_entropies else zero,
        "semantic_prior_entropy": torch.stack(semantic_prior_entropies).mean() if semantic_prior_entropies else zero,
        "fusion_mid_weight": torch.stack(fusion_mid_weights).mean() if fusion_mid_weights else zero,
        "fusion_global_weight": torch.stack(fusion_global_weights).mean() if fusion_global_weights else zero,
        "alpha_mean": torch.stack(alpha_means).mean() if alpha_means else zero,
        "alpha_raw_mean": torch.stack(alpha_raw_means).mean() if alpha_raw_means else zero,
        "alpha_prob_mean": torch.stack(alpha_prob_means).mean() if alpha_prob_means else zero,
        "alpha_clean_prob_mean": torch.stack(alpha_clean_prob_means).mean() if alpha_clean_prob_means else zero,
        "alpha_capped_mean": torch.stack(alpha_capped_means).mean() if alpha_capped_means else zero,
        "alpha_downstream_mean": torch.stack(alpha_downstream_means).mean() if alpha_downstream_means else zero,
        "v4_layer_scale": torch.stack(v4_layer_scales).mean() if v4_layer_scales else zero,
        "v4_state_norm": torch.stack(v4_state_norms).mean() if v4_state_norms else zero,
        "v4_memory_norm": torch.stack(v4_memory_norms).mean() if v4_memory_norms else zero,
        "v4_memory_gate": torch.stack(v4_memory_gates).mean() if v4_memory_gates else zero,
        "v4_memory_attention_entropy": torch.stack(v4_memory_entropies).mean() if v4_memory_entropies else zero,
        "v4_memory_read_strength": torch.stack(v4_memory_read_strengths).mean() if v4_memory_read_strengths else zero,
        "v4_memory_novelty": torch.stack(v4_memory_novelties).mean() if v4_memory_novelties else zero,
        "v4_state_gate": torch.stack(v4_state_gates).mean() if v4_state_gates else zero,
        "v4_state_confidence": torch.stack(v4_state_confidences).mean() if v4_state_confidences else zero,
        "v4_state_delta": torch.stack(v4_state_deltas).mean() if v4_state_deltas else zero,
        "v4_state_agreement": torch.stack(v4_state_agreements).mean() if v4_state_agreements else zero,
        "gate_mix_alpha_weight": torch.stack(gate_mix_alpha_weights).mean() if gate_mix_alpha_weights else zero,
        "gate_mix_clean_weight": torch.stack(gate_mix_clean_weights).mean() if gate_mix_clean_weights else zero,
        "gate_mix_state_weight": torch.stack(gate_mix_state_weights).mean() if gate_mix_state_weights else zero,
        "v5_state_pred": torch.stack(v5_state_preds).mean() if v5_state_preds else zero,
        "v5_slot_diversity": torch.stack(v5_slot_diversities).mean() if v5_slot_diversities else zero,
        "v5_slot_stability": torch.stack(v5_slot_stabilities).mean() if v5_slot_stabilities else zero,
        "v5_slot_update_gate": torch.stack(v5_slot_update_gates).mean() if v5_slot_update_gates else zero,
        "v5_slot_write_max": torch.stack(v5_slot_write_maxes).mean() if v5_slot_write_maxes else zero,
        "v5_slot_write_entropy": torch.stack(v5_slot_write_entropies).mean() if v5_slot_write_entropies else zero,
        "v5_slot_write_min": torch.stack(v5_slot_write_mins).mean() if v5_slot_write_mins else zero,
        "v5_slot_write_active": torch.stack(v5_slot_write_actives).mean() if v5_slot_write_actives else zero,
        "v5_slot_confidence": torch.stack(v5_slot_confidences).mean() if v5_slot_confidences else zero,
        "v5_slot_confidence_std": torch.stack(v5_slot_confidence_stds).mean() if v5_slot_confidence_stds else zero,
        "v5_slot_delta": torch.stack(v5_slot_deltas).mean() if v5_slot_deltas else zero,
        "v5_slot_cosine": torch.stack(v5_slot_cosines).mean() if v5_slot_cosines else zero,
        "v5_slot_read_entropy": torch.stack(v5_slot_read_entropies).mean() if v5_slot_read_entropies else zero,
        "v5_slot_read_max": torch.stack(v5_slot_read_maxes).mean() if v5_slot_read_maxes else zero,
        "v6_self_pred": torch.stack(v6_self_preds).mean() if v6_self_preds else zero,
        "v6_slot_diversity": torch.stack(v6_slot_diversities).mean() if v6_slot_diversities else zero,
        "v6_slot_cosine": torch.stack(v6_slot_cosines).mean() if v6_slot_cosines else zero,
        "v6_slot_context_cosine": torch.stack(v6_slot_context_cosines).mean() if v6_slot_context_cosines else zero,
        "v6_state_delta": torch.stack(v6_state_deltas).mean() if v6_state_deltas else zero,
        "v6_state_norm": torch.stack(v6_state_norms).mean() if v6_state_norms else zero,
        "v6_reflection_norm": torch.stack(v6_reflection_norms).mean() if v6_reflection_norms else zero,
        "v6_boundary_entropy": torch.stack(v6_boundary_entropies).mean() if v6_boundary_entropies else zero,
        "v6_boundary_self": torch.stack(v6_boundary_selfs).mean() if v6_boundary_selfs else zero,
        "v6_boundary_world": torch.stack(v6_boundary_worlds).mean() if v6_boundary_worlds else zero,
        "v6_boundary_other": torch.stack(v6_boundary_others).mean() if v6_boundary_others else zero,
        "v6_boundary_unknown": torch.stack(v6_boundary_unknowns).mean() if v6_boundary_unknowns else zero,
        "dispatch_dense": torch.stack(dispatch_denses).mean() if dispatch_denses else zero,
    }
