import math

import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig

from .losses import collect_aux_losses, lm_loss


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
            "v6_self_pred",
            "v6_slot_diversity",
            "v6_slot_cosine",
            "v6_slot_context_cosine",
            "v6_state_delta",
            "v6_state_norm",
            "v6_reflection_norm",
            "v6_boundary_entropy",
            "v6_boundary_self",
            "v6_boundary_world",
            "v6_boundary_other",
            "v6_boundary_unknown",
        ]
    }
    batches = 0
    tokens = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids)
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
            totals["v6_self_pred"] += float(aux["v6_self_pred"].detach().cpu())
            totals["v6_slot_diversity"] += float(aux["v6_slot_diversity"].detach().cpu())
            totals["v6_slot_cosine"] += float(aux["v6_slot_cosine"].detach().cpu())
            totals["v6_slot_context_cosine"] += float(aux["v6_slot_context_cosine"].detach().cpu())
            totals["v6_state_delta"] += float(aux["v6_state_delta"].detach().cpu())
            totals["v6_state_norm"] += float(aux["v6_state_norm"].detach().cpu())
            totals["v6_reflection_norm"] += float(aux["v6_reflection_norm"].detach().cpu())
            totals["v6_boundary_entropy"] += float(aux["v6_boundary_entropy"].detach().cpu())
            totals["v6_boundary_self"] += float(aux["v6_boundary_self"].detach().cpu())
            totals["v6_boundary_world"] += float(aux["v6_boundary_world"].detach().cpu())
            totals["v6_boundary_other"] += float(aux["v6_boundary_other"].detach().cpu())
            totals["v6_boundary_unknown"] += float(aux["v6_boundary_unknown"].detach().cpu())
            tokens += int(input_ids.numel())
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
        "val_v6_slot_cosine": totals["v6_slot_cosine"] / batches,
        "val_v6_slot_context_cosine": totals["v6_slot_context_cosine"] / batches,
        "val_v6_state_delta": totals["v6_state_delta"] / batches,
        "val_v6_state_norm": totals["v6_state_norm"] / batches,
        "val_v6_reflection_norm": totals["v6_reflection_norm"] / batches,
        "val_v6_boundary_entropy": totals["v6_boundary_entropy"] / batches,
        "val_v6_boundary_self": totals["v6_boundary_self"] / batches,
        "val_v6_boundary_world": totals["v6_boundary_world"] / batches,
        "val_v6_boundary_other": totals["v6_boundary_other"] / batches,
        "val_v6_boundary_unknown": totals["v6_boundary_unknown"] / batches,
        "val_batches": float(batches),
        "val_tokens": float(tokens),
    }
