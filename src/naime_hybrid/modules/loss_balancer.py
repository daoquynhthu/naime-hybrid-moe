import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class LossBalancer(nn.Module):
    """Confidence-gated auxiliary loss weight balancer.

    Reads the structured world-state slots to produce scalar weights for
    each auxiliary loss term.  Internally computes per-slot confidence
    from the slot representations; low confidence → low auxiliary-loss
    pressure so the model focuses on language modelling.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int,
        n_losses: int = 4,
        w_default: tuple[float, ...] = (0.01, 0.003, 0.02, 0.01),
        w_max: tuple[float, ...] = (0.03, 0.005, 0.04, 0.02),
    ):
        super().__init__()
        if len(w_default) != n_losses or len(w_max) != n_losses:
            raise ValueError("w_default and w_max must have length n_losses")

        self.n_losses = n_losses
        self.slot_norm = RMSNorm(d_model)
        self.confidence_proj = nn.Linear(d_model, 1)
        self.slot_pool = nn.Linear(d_model, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_losses),
        )
        self.register_buffer("w_default_log", torch.tensor(w_default, dtype=torch.float32).log())
        self.register_buffer("w_max", torch.tensor(w_max, dtype=torch.float32))

    def forward(self, world_state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, slots, d = world_state.shape

        normed = self.slot_norm(world_state)
        slot_conf = torch.sigmoid(self.confidence_proj(normed.float())).type_as(world_state)
        conf = slot_conf.mean(dim=1)

        slot_scores = self.slot_pool(normed).squeeze(-1)
        slot_weights = torch.softmax(slot_scores.float(), dim=-1).type_as(world_state)
        pooled = (normed * slot_weights.unsqueeze(-1)).sum(dim=1)

        features = torch.cat([pooled, conf], dim=-1)
        log_w = self.proj(features).float() + self.w_default_log
        w_raw = F.softplus(log_w).type_as(world_state)

        conf_gate = conf.mean(dim=-1, keepdim=True).type_as(world_state)
        w = w_raw * conf_gate
        w = torch.min(w, self.w_max)

        aux = {
            "balancer_conf": conf.mean(),
            "balancer_w_raw": w_raw,
            "balancer_w": w,
        }
        return w, aux
