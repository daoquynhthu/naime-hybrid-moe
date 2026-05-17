import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class SemanticGateMixer(nn.Module):
    """Blend noisy, clean, and state-derived gates into one downstream alpha."""

    def __init__(
        self,
        d_model: int,
        temperature: float = 1.0,
        min_weight: float = 0.0,
        max_clean_weight: float = 0.0,
        max_state_weight: float = 0.35,
    ):
        super().__init__()
        if min_weight < 0.0 or min_weight >= 1.0 / 3.0:
            raise ValueError("semantic gate mixer min_weight must be in [0, 1/3)")
        if max_clean_weight < 0.0 or max_clean_weight > 1.0:
            raise ValueError("semantic gate mixer max_clean_weight must be in [0, 1]")
        if max_state_weight < 0.0 or max_state_weight > 1.0:
            raise ValueError("semantic gate mixer max_state_weight must be in [0, 1]")
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model + 3, 3)
        self.temperature = max(1e-3, temperature)
        self.min_weight = min_weight
        self.max_clean_weight = max_clean_weight
        self.max_state_weight = max_state_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        alpha: torch.Tensor,
        clean_prob: torch.Tensor,
        state_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [
                self.norm(hidden_states),
                alpha,
                clean_prob,
                state_confidence.expand_as(alpha),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.proj(features).float() / self.temperature, dim=-1)
        if self.min_weight > 0.0:
            weights = weights * (1.0 - 3.0 * self.min_weight) + self.min_weight
        if self.max_clean_weight > 0.0:
            clean = weights[..., 1:2].clamp_max(self.max_clean_weight)
            non_clean = torch.cat([weights[..., 0:1], weights[..., 2:3]], dim=-1)
            non_clean_sum = non_clean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            non_clean = non_clean / non_clean_sum * (1.0 - clean)
            weights = torch.cat([non_clean[..., 0:1], clean, non_clean[..., 1:2]], dim=-1)
        if self.max_state_weight > 0.0:
            state = weights[..., 2:3].clamp_max(self.max_state_weight)
            non_state = torch.cat([weights[..., 0:1], weights[..., 1:2]], dim=-1)
            non_state_sum = non_state.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            non_state = non_state / non_state_sum * (1.0 - state)
            weights = torch.cat([non_state[..., 0:1], non_state[..., 1:2], state], dim=-1)
        weights = weights.type_as(hidden_states)
        mixed = weights[..., 0:1] * alpha + weights[..., 1:2] * clean_prob + weights[..., 2:3] * state_confidence
        return mixed.clamp(0.0, 1.0), weights


class SemanticMemory(nn.Module):
    """Small per-forward working memory carried across V4 layers."""

    def __init__(self, d_model: int, slots: int):
        super().__init__()
        self.slots = slots
        self.initial = nn.Parameter(torch.zeros(slots, d_model))
        self.query_norm = RMSNorm(d_model)
        self.mem_norm = RMSNorm(d_model)
        self.update = nn.Linear(d_model * 2, d_model)
        self.update_gate = nn.Linear(d_model * 2, d_model)

    def initial_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def read(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_norm(hidden_states)
        key = self.mem_norm(memory)
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        weights = torch.softmax(scores.float(), dim=-1).type_as(hidden_states)
        context = torch.matmul(weights, memory)
        return context, weights

    def write(
        self,
        memory: torch.Tensor,
        semantic_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory_before = memory
        summary = semantic_summary.unsqueeze(1).expand_as(memory)
        update_input = torch.cat([memory, summary], dim=-1)
        candidate = torch.tanh(self.update(update_input))
        gate = torch.sigmoid(self.update_gate(update_input))
        next_memory = memory + gate * (candidate - memory)
        novelty = F.cosine_similarity(
            F.normalize(semantic_summary, dim=-1).unsqueeze(1),
            F.normalize(memory_before, dim=-1),
            dim=-1,
        )
        novelty = 1.0 - novelty.max(dim=-1).values
        return next_memory, gate, novelty


class CrossLayerSemanticState(nn.Module):
    """Gated recurrent semantic state shared by V4 semantic blocks."""

    def __init__(self, d_model: int, confidence_mode: str = "learned", confidence_temperature: float = 2.0):
        super().__init__()
        if confidence_mode not in {"learned", "calibrated", "hybrid"}:
            raise ValueError("confidence_mode must be learned, calibrated, or hybrid")
        self.initial = nn.Parameter(torch.zeros(d_model))
        self.norm = RMSNorm(d_model)
        self.update = nn.Linear(d_model * 2, d_model)
        self.update_gate = nn.Linear(d_model * 2, d_model)
        self.confidence = nn.Linear(d_model, 1)
        self.confidence_mode = confidence_mode
        self.confidence_temperature = confidence_temperature

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        state: torch.Tensor,
        semantic_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        update_input = torch.cat([state, semantic_summary], dim=-1)
        candidate = torch.tanh(self.update(update_input))
        gate = torch.sigmoid(self.update_gate(update_input))
        next_state = state + gate * (candidate - state)
        normalized = self.norm(next_state)
        learned_confidence = torch.sigmoid(self.confidence(normalized))
        state_delta = (next_state - state).float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        stability = torch.exp(-state_delta * self.confidence_temperature).clamp(0.0, 1.0)
        agreement = F.cosine_similarity(
            F.normalize(normalized.float(), dim=-1),
            F.normalize(semantic_summary.float(), dim=-1),
            dim=-1,
        ).unsqueeze(-1)
        agreement = ((agreement + 1.0) * 0.5).clamp(0.0, 1.0)
        calibrated_confidence = (0.65 * stability + 0.35 * agreement).type_as(normalized)
        if self.confidence_mode == "learned":
            confidence = learned_confidence
        elif self.confidence_mode == "calibrated":
            confidence = calibrated_confidence
        else:
            confidence = (0.5 * learned_confidence + 0.5 * calibrated_confidence).type_as(normalized)
        return normalized, gate, confidence.unsqueeze(1), state_delta.squeeze(-1), agreement.squeeze(-1)
