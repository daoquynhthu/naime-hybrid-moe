import torch
from torch import nn

from .norm import RMSNorm
from .state_ops import state_softmax_matmul


class LatentFieldCoupler(nn.Module):
    """Bounded interaction between token states and persistent latent slots.

    The persistent slots are treated as regions of the same latent field, not as
    a side-channel memory projected into hidden states. Current-token writes
    still flow through the world/self/memory state modules; this module only
    lets tokens read the currently available field state through a capped update.
    """

    def __init__(
        self,
        d_model: int,
        token_scale: float = 0.02,
        max_ratio: float = 0.05,
        gate_bias: float = -2.5,
    ):
        super().__init__()
        self.token_scale = token_scale
        self.max_ratio = max_ratio
        self.token_norm = RMSNorm(d_model)
        self.slot_norm = RMSNorm(d_model)
        self.token_query = nn.Linear(d_model, d_model, bias=False)
        self.slot_key = nn.Linear(d_model, d_model, bias=False)
        self.slot_value = nn.Linear(d_model, d_model, bias=False)
        self.token_update = nn.Linear(d_model, d_model)
        self.token_gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.token_gate.bias, gate_bias)

    @staticmethod
    def _final_or_trace(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        if state.ndim not in (3, 4):
            raise ValueError("latent field state must be [B,S,D] or [B,T,S,D]")
        return state

    def _collect_bank(
        self,
        *,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, int]:
        banks: list[torch.Tensor] = []
        causal_trace_slots = 0
        for state in (
            self._final_or_trace(world_state),
            self._final_or_trace(self_state),
        ):
            if state is None:
                continue
            if state.ndim == 4:
                batch, trace_len, slots, dim = state.shape
                banks.append(state.reshape(batch, trace_len * slots, dim))
                causal_trace_slots += trace_len * slots
            else:
                banks.append(state)
        if memory is not None:
            banks.append(memory)
        if not banks:
            return None, 0
        return torch.cat(banks, dim=1), causal_trace_slots

    def _causal_trace_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        causal_trace_slots: int,
        block_size: int,
        treat_trace_as_past: bool,
    ) -> torch.Tensor | None:
        if causal_trace_slots <= 0 or block_size <= 0 or treat_trace_as_past:
            return None
        seq_len = hidden_states.size(1)
        slots_per_block = causal_trace_slots
        # When both world and self traces are present, they have the same trace
        # length in normal V6 flow but not necessarily the same slot count. The
        # safest generic mask is to expose a monotonic prefix of the flattened
        # trace bank per token block, never future trace slots.
        block_count = max(1, (seq_len + block_size - 1) // block_size)
        slots_per_block = max(1, causal_trace_slots // block_count)
        token_blocks = torch.arange(seq_len, device=hidden_states.device) // block_size
        readable = (token_blocks * slots_per_block).clamp(max=causal_trace_slots)
        slot_idx = torch.arange(causal_trace_slots, device=hidden_states.device)
        return slot_idx.unsqueeze(0) >= readable.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        world_state: torch.Tensor | None = None,
        self_state: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        block_size: int = 0,
        treat_trace_as_past: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = hidden_states.new_tensor(0.0)
        if self.token_scale <= 0.0 or self.max_ratio <= 0.0:
            return hidden_states, {
                "latent_field_token_delta_norm": zero,
                "latent_field_token_delta_ratio": zero,
                "latent_field_read_entropy": zero,
                "latent_field_read_max": zero,
                "latent_field_gate": zero,
            }

        bank, causal_trace_slots = self._collect_bank(
            world_state=world_state,
            self_state=self_state,
            memory=memory,
        )
        if bank is None or bank.size(1) == 0:
            return hidden_states, {
                "latent_field_token_delta_norm": zero,
                "latent_field_token_delta_ratio": zero,
                "latent_field_read_entropy": zero,
                "latent_field_read_max": zero,
                "latent_field_gate": zero,
            }

        normed_tokens = self.token_norm(hidden_states)
        query = self.token_query(normed_tokens)
        key = self.slot_key(self.slot_norm(bank))
        value = self.slot_value(bank)
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)

        causal_mask = self._causal_trace_mask(
            hidden_states=hidden_states,
            causal_trace_slots=causal_trace_slots,
            block_size=block_size,
            treat_trace_as_past=treat_trace_as_past,
        )
        full_mask = None
        if causal_mask is not None:
            full_mask = torch.zeros(
                scores.shape[1:],
                device=scores.device,
                dtype=torch.bool,
            )
            full_mask[:, :causal_trace_slots] = causal_mask

        context, weights = state_softmax_matmul(
            scores,
            value,
            mask=full_mask.unsqueeze(0) if full_mask is not None else None,
            zero_invalid=full_mask is not None and causal_trace_slots == bank.size(1),
            out_dtype=hidden_states.dtype,
        )
        raw_delta = self.token_update(context)
        gate = torch.sigmoid(self.token_gate(normed_tokens)).type_as(hidden_states)
        delta = raw_delta * gate * self.token_scale

        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        hidden_norm = hidden_states.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cap = hidden_norm * self.max_ratio
        delta = delta * (cap / delta_norm.clamp_min(1e-6)).clamp(max=1.0).type_as(delta)

        if attention_mask is not None:
            delta = delta * attention_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)

        updated = hidden_states + delta
        with torch.no_grad():
            telemetry_weights = weights.float()
            probs = telemetry_weights.clamp_min(1e-6)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().type_as(hidden_states)
            read_max = telemetry_weights.max(dim=-1).values.mean().type_as(hidden_states)
            telemetry_delta_norm = delta.float().norm(dim=-1)
            token_delta_norm = telemetry_delta_norm.mean().type_as(hidden_states)
            token_delta_ratio = (
                telemetry_delta_norm / hidden_states.float().norm(dim=-1).clamp_min(1e-6)
            ).mean().type_as(hidden_states)
            gate_mean = gate.float().mean().type_as(hidden_states)
        return updated, {
            "latent_field_token_delta_norm": token_delta_norm,
            "latent_field_token_delta_ratio": token_delta_ratio,
            "latent_field_read_entropy": entropy,
            "latent_field_read_max": read_max,
            "latent_field_gate": gate_mean,
        }
