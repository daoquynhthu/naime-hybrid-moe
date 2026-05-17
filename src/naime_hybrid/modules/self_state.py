import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class RecursiveSelfState(nn.Module):
    """Self-state slots with a lightweight recursive reflection update.

    The module is intentionally modest: it turns token/world summaries into a
    compact self-state, applies one or more reflection passes, and returns
    metrics that make the mechanism falsifiable during training.
    """

    def __init__(
        self,
        d_model: int,
        *,
        slots: int = 4,
        recursion_depth: int = 1,
        write_scale: float = 0.03,
        hidden_scale: float = 0.02,
        boundary_temperature: float = 1.0,
        diversity_margin: float = 0.85,
        identity_scale: float = 0.02,
        context_score_scale: float = 4.0,
        pred_detach_target: bool = True,
    ) -> None:
        super().__init__()
        if slots <= 0:
            raise ValueError("self-state slots must be positive")
        if recursion_depth <= 0:
            raise ValueError("self-state recursion depth must be positive")
        self.slots = slots
        self.recursion_depth = recursion_depth
        self.write_scale = write_scale
        self.hidden_scale = hidden_scale
        self.boundary_temperature = max(boundary_temperature, 1e-3)
        self.diversity_margin = diversity_margin
        self.identity_scale = identity_scale
        self.context_score_scale = context_score_scale
        self.pred_detach_target = pred_detach_target

        self.initial = nn.Parameter(torch.zeros(slots, d_model))
        self.slot_identity = nn.Parameter(torch.zeros(slots, d_model))
        self.hidden_norm = RMSNorm(d_model)
        self.world_norm = RMSNorm(d_model)
        self.self_norm = RMSNorm(d_model)
        self.boundary = nn.Linear(d_model, 4)
        self.reflect = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update = nn.Linear(d_model * 3, d_model)
        self.update_gate = nn.Linear(d_model * 3, d_model)
        self.transition = nn.Linear(d_model, d_model)
        self.hidden_modulation = nn.Linear(d_model, d_model, bias=False)
        self.reset_slot_parameters()

    def reset_slot_parameters(self) -> None:
        nn.init.normal_(self.slot_identity, mean=0.0, std=self.identity_scale)
        nn.init.normal_(self.initial, mean=0.0, std=self.identity_scale * 0.25)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _masked_mean(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype).unsqueeze(-1)
        if weights is not None:
            mask_f = mask_f * weights.to(dtype=values.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (values * mask_f).sum(dim=1) / denom

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden_states.size(0)
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
        if self_state is None:
            self_state = self.initial_state(batch_size, hidden_states.device, hidden_states.dtype)

        normed_hidden = self.hidden_norm(hidden_states)
        boundary_logits = self.boundary(normed_hidden.float()) / self.boundary_temperature
        boundary_probs = torch.softmax(boundary_logits, dim=-1).to(dtype=hidden_states.dtype)

        hidden_summary = self._masked_mean(normed_hidden, attention_mask)
        self_summary = self._masked_mean(normed_hidden, attention_mask, boundary_probs[..., 0:1])
        if world_state is None:
            world_summary = torch.zeros_like(hidden_summary)
        else:
            world_summary = self.world_norm(world_state).mean(dim=1)

        current = self_state
        reflection = torch.zeros_like(hidden_summary)
        recursion_delta = hidden_summary.new_tensor(0.0)
        identity = self.slot_identity.to(device=hidden_states.device, dtype=hidden_states.dtype)
        identity = identity.unsqueeze(0).expand_as(current)
        slot_queries = F.normalize((current + identity).float(), dim=-1).to(dtype=hidden_states.dtype)
        slot_scores = torch.einsum("bsd,btd->bst", slot_queries, normed_hidden) * self.context_score_scale
        slot_scores = slot_scores.masked_fill(~attention_mask.unsqueeze(1), torch.finfo(slot_scores.dtype).min)
        slot_weights = torch.softmax(slot_scores.float(), dim=-1).to(dtype=hidden_states.dtype)
        slot_context = torch.bmm(slot_weights, normed_hidden)

        for _ in range(self.recursion_depth):
            pooled_self = self.self_norm(current).mean(dim=1)
            reflection_input = torch.cat([hidden_summary, self_summary, world_summary, pooled_self], dim=-1)
            reflection = torch.tanh(self.reflect(reflection_input))
            reflection_slots = reflection.unsqueeze(1).expand_as(current)
            update_input = torch.cat([current, reflection_slots + slot_context, identity], dim=-1)
            candidate = torch.tanh(self.update(update_input))
            gate = torch.sigmoid(self.update_gate(update_input)) * self.write_scale
            next_state = current + gate * (candidate - current)
            recursion_delta = recursion_delta + (next_state - current).float().pow(2).mean()
            current = next_state

        hidden_states = hidden_states + self.hidden_modulation(reflection).unsqueeze(1) * self.hidden_scale

        previous_summary = self.self_norm(self_state).mean(dim=1)
        pred_summary = self.transition(previous_summary)
        pred_target = self_summary.detach() if self.pred_detach_target else self_summary
        self_pred_loss = F.smooth_l1_loss(pred_summary.float(), pred_target.float())

        normalized = F.normalize(current.float(), dim=-1)
        cosine = torch.bmm(normalized, normalized.transpose(1, 2))
        context_normalized = F.normalize(slot_context.float(), dim=-1)
        context_cosine = torch.bmm(context_normalized, context_normalized.transpose(1, 2))
        off_diag = ~torch.eye(self.slots, dtype=torch.bool, device=cosine.device).unsqueeze(0)
        off_diag_cosine = cosine.masked_select(off_diag)
        off_diag_context_cosine = context_cosine.masked_select(off_diag)
        if off_diag_cosine.numel() == 0:
            slot_diversity = cosine.new_tensor(0.0)
            slot_cosine = cosine.new_tensor(0.0)
            slot_context_cosine = context_cosine.new_tensor(0.0)
        else:
            slot_diversity = F.relu(off_diag_cosine - self.diversity_margin).mean()
            slot_cosine = off_diag_cosine.mean()
            slot_context_cosine = off_diag_context_cosine.mean()

        probs = boundary_probs.float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        mask_f = attention_mask.float()
        boundary_entropy = (entropy * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        boundary_means = self._masked_mean(boundary_probs, attention_mask)

        metrics = {
            "self_pred": self_pred_loss,
            "slot_diversity": slot_diversity,
            "slot_cosine": slot_cosine,
            "slot_context_cosine": slot_context_cosine,
            "state_delta": recursion_delta / self.recursion_depth,
            "state_norm": current.float().norm(dim=-1).mean(),
            "reflection_norm": reflection.float().norm(dim=-1).mean(),
            "boundary_entropy": boundary_entropy,
            "boundary_self": boundary_means[..., 0].mean(),
            "boundary_world": boundary_means[..., 1].mean(),
            "boundary_other": boundary_means[..., 2].mean(),
            "boundary_unknown": boundary_means[..., 3].mean(),
        }
        return hidden_states, current, metrics
