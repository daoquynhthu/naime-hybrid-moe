import math

import torch
from torch import nn


class GumbelBlockGate(nn.Module):
    """Straight-through Gumbel-Sigmoid block gate."""

    def __init__(self, d_model: int, target_sparsity: float):
        super().__init__()
        if not 0.0 < target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in (0, 1)")

        self.target_sparsity = target_sparsity
        self.proj = nn.Linear(d_model, 1)
        init_bias = math.log(target_sparsity / (1.0 - target_sparsity))
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
        eval_mode: str = "prob",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.proj(x).squeeze(-1)
        clean_prob = torch.sigmoid(logits.float()).type_as(logits)

        if self.training:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform.float()))
            prob = torch.sigmoid((logits.float() + gumbel) / tau)
            if hard:
                hard_gate = (prob > 0.5).to(prob.dtype)
                alpha = (hard_gate - prob).detach() + prob
            else:
                alpha = prob
            return alpha.type_as(logits), logits, prob.type_as(logits), clean_prob

        if eval_mode == "hard":
            alpha = (logits > 0).to(logits.dtype)
        elif eval_mode == "prob":
            alpha = clean_prob
        else:
            raise ValueError(f"unknown gate eval mode: {eval_mode}")
        return alpha, logits, clean_prob, clean_prob
