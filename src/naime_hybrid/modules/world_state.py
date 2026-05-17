import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class WorldStateSlots(nn.Module):
    """Structured latent state bank used by the V5 world-state architecture."""

    def __init__(
        self,
        d_model: int,
        slots: int,
        diversity_margin: float = 0.85,
        stability_threshold: float = 1e-3,
        write_top_k: int = 2,
        pred_detach_target: bool = True,
    ):
        super().__init__()
        if slots <= 0:
            raise ValueError("world state slots must be positive")
        if write_top_k <= 0:
            raise ValueError("world_state write_top_k must be positive")
        self.slots = slots
        self.diversity_margin = diversity_margin
        self.stability_threshold = stability_threshold
        self.write_top_k = min(write_top_k, slots)
        self.pred_detach_target = pred_detach_target
        self.initial = nn.Parameter(torch.zeros(slots, d_model))
        self.slot_identity = nn.Parameter(torch.zeros(slots, d_model))
        self.query_norm = RMSNorm(d_model)
        self.slot_norm = RMSNorm(d_model)
        self.summary_norm = RMSNorm(d_model)
        # Kept for checkpoint compatibility; write addressing now uses
        # summary-slot similarity instead of summary-only routing.
        self.slot_router = nn.Linear(d_model, slots)
        self.update = nn.Linear(d_model * 3, d_model)
        self.update_gate = nn.Linear(d_model * 3, d_model)
        self.confidence = nn.Linear(d_model, 1)
        self.transition = nn.Linear(d_model, d_model)
        nn.init.normal_(self.initial, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_identity, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def read(self, hidden_states: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_norm(hidden_states)
        key = self.slot_norm(slots)
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        weights = torch.softmax(scores.float(), dim=-1).type_as(hidden_states)
        context = torch.matmul(weights, slots)
        slot_quality = torch.sigmoid(self.confidence(key.float()))
        max_score_per_token = scores.detach().max(dim=-1, keepdim=True).values
        token_match = torch.sigmoid(max_score_per_token)
        weighted_slot_quality = torch.sum(weights.unsqueeze(-1) * slot_quality.unsqueeze(1), dim=2)
        confidence = (weighted_slot_quality * token_match).type_as(hidden_states)
        return context, weights, confidence

    def update_slots(
        self,
        slots: torch.Tensor,
        semantic_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summary = semantic_summary.detach() if self.pred_detach_target else semantic_summary
        identity = self.slot_identity.to(device=slots.device, dtype=slots.dtype).unsqueeze(0).expand_as(slots)
        normalized_slots = self.slot_norm(slots)
        normalized_summary = self.summary_norm(summary)

        # Predict before writing so state_pred trains the current state to
        # explain the new summary instead of rewarding immediate overwrite.
        write_scores = torch.matmul(
            normalized_summary.unsqueeze(1),
            normalized_slots.transpose(1, 2),
        ).squeeze(1) / (normalized_summary.size(-1) ** 0.5)
        if self.write_top_k < self.slots:
            topk_scores, topk_idx = torch.topk(write_scores.float(), k=self.write_top_k, dim=-1)
            sparse_scores = torch.full_like(write_scores.float(), torch.finfo(write_scores.float().dtype).min)
            sparse_scores.scatter_(1, topk_idx, topk_scores)
            slot_write_weights = torch.softmax(sparse_scores, dim=-1).type_as(slots)
        else:
            slot_write_weights = torch.softmax(write_scores.float(), dim=-1).type_as(slots)
        pred_context = torch.sum(normalized_slots * slot_write_weights.unsqueeze(-1), dim=1)
        pred_summary = self.transition(pred_context)
        state_pred_loss = F.smooth_l1_loss(pred_summary.float(), summary.float())

        summary_expanded = summary.unsqueeze(1).expand_as(slots)
        update_input = torch.cat([slots, summary_expanded, identity], dim=-1)
        candidate = torch.tanh(self.update(update_input))
        gate = torch.sigmoid(self.update_gate(update_input))
        slot_write = slot_write_weights.unsqueeze(-1)
        next_slots = slots + slot_write * gate * (candidate - slots)
        normalized_next = self.slot_norm(next_slots)
        confidence = torch.sigmoid(self.confidence(normalized_next.float())).type_as(slots)
        state_delta = (next_slots - slots).float().pow(2).mean(dim=-1).sqrt()
        slot_cosine = self._mean_pairwise_cosine(normalized_next)
        diversity_loss = self._diversity_loss(normalized_next)
        below_threshold = (diversity_loss.detach().float() < self.stability_threshold).float()
        slot_confidence_detached = confidence.detach()
        raw_stability = (slot_confidence_detached * (next_slots - slots).float().pow(2)).mean()
        stability_loss = below_threshold * raw_stability
        write_entropy = -(slot_write_weights.clamp_min(1e-6).float() * slot_write_weights.clamp_min(1e-6).float().log())
        write_entropy = write_entropy.sum(dim=-1).mean().type_as(slots)
        write_max = slot_write_weights.max(dim=-1).values.mean()
        active_mask = slot_write_weights > 0
        active_write_min = slot_write_weights.masked_fill(~active_mask, 1.0).min(dim=-1).values.mean()
        active_write_count = active_mask.sum(dim=-1).float().mean().type_as(slots)
        metrics = {
            "slot_update_gate": gate.mean(),
            "slot_write_max": write_max,
            "slot_write_min": active_write_min,
            "slot_write_active": active_write_count,
            "slot_write_entropy": write_entropy,
            "slot_confidence": confidence.mean(),
            "slot_confidence_std": confidence.float().std(unbiased=False),
            "slot_delta": state_delta.mean(),
            "slot_cosine": slot_cosine,
            "slot_diversity": diversity_loss,
            "slot_stability": stability_loss,
            "state_pred": state_pred_loss,
        }
        return normalized_next, metrics

    def _mean_pairwise_cosine(self, slots: torch.Tensor) -> torch.Tensor:
        if self.slots < 2:
            return torch.zeros((), device=slots.device, dtype=slots.dtype)
        normed = F.normalize(slots.float(), dim=-1)
        cosine = torch.matmul(normed, normed.transpose(1, 2))
        mask = ~torch.eye(self.slots, device=slots.device, dtype=torch.bool).unsqueeze(0)
        return cosine.masked_select(mask).mean().type_as(slots)

    def _diversity_loss(self, slots: torch.Tensor) -> torch.Tensor:
        if self.slots < 2:
            return torch.zeros((), device=slots.device, dtype=slots.dtype)
        normed = F.normalize(slots.float(), dim=-1)
        cosine = torch.matmul(normed, normed.transpose(1, 2))
        mask = ~torch.eye(self.slots, device=slots.device, dtype=torch.bool).unsqueeze(0)
        pairwise = cosine.masked_select(mask)
        return F.relu(pairwise - self.diversity_margin).mean().type_as(slots)
