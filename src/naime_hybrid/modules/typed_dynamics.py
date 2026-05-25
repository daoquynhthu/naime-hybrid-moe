import torch
from torch import nn

from .norm import RMSNorm
from .state_ops import state_softmax_matmul


class TypedLatentDynamics(nn.Module):
    """V7 typed latent dynamics core.

    The first implementation is deliberately conservative. Current-token hidden
    states may read only the incoming latent field, so logits stay causal. The
    updated latent field is returned as the next state packet and is not reused
    to rewrite the same sequence.
    """

    def __init__(
        self,
        d_model: int,
        *,
        latent_slots: int,
        latent_write_scale: float = 0.03,
        hidden_write_scale: float = 0.01,
        max_hidden_write_ratio: float = 0.05,
        state_write_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if latent_slots <= 0:
            raise ValueError("V7 latent slots must be positive")
        self.latent_slots = latent_slots
        self.latent_write_scale = max(latent_write_scale, 0.0)
        self.hidden_write_scale = max(hidden_write_scale, 0.0)
        self.max_hidden_write_ratio = max(max_hidden_write_ratio, 0.0)
        self.state_write_scale = max(state_write_scale, 0.0)

        self.initial = nn.Parameter(torch.zeros(latent_slots, d_model))
        self.hidden_norm = RMSNorm(d_model)
        self.world_norm = RMSNorm(d_model)
        self.self_norm = RMSNorm(d_model)
        self.latent_norm = RMSNorm(d_model)
        self.token_query = nn.Linear(d_model, d_model, bias=False)
        self.latent_key = nn.Linear(d_model, d_model, bias=False)
        self.latent_value = nn.Linear(d_model, d_model, bias=False)
        self.hidden_update = nn.Linear(d_model, d_model)
        self.hidden_gate = nn.Linear(d_model, 1)
        self.latent_update = nn.Linear(d_model * 4, d_model)
        self.latent_gate = nn.Linear(d_model * 4, d_model)
        self.world_update = nn.Linear(d_model * 4, d_model)
        self.world_gate = nn.Linear(d_model * 4, d_model)
        self.self_update = nn.Linear(d_model * 4, d_model)
        self.self_gate = nn.Linear(d_model * 4, d_model)
        nn.init.normal_(self.initial, mean=0.0, std=0.005)
        nn.init.constant_(self.hidden_gate.bias, -3.0)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    @staticmethod
    def _final_slots(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        if state.ndim == 4:
            return state[:, -1, :, :]
        return state

    def _summary(
        self,
        state: torch.Tensor | None,
        norm: nn.Module,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        state = self._final_slots(state)
        if state is None:
            return torch.zeros_like(fallback)
        return norm(state).mean(dim=1).to(dtype=fallback.dtype)

    def _update_typed_state(
        self,
        state: torch.Tensor | None,
        *,
        hidden_summary: torch.Tensor,
        other_summary: torch.Tensor,
        latent_summary: torch.Tensor,
        update: nn.Linear,
        gate: nn.Linear,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        zero = hidden_summary.new_tensor(0.0)
        if state is None or self.state_write_scale <= 0.0:
            return state, zero, zero
        slots = self._final_slots(state)
        if slots is None:
            return state, zero, zero
        hidden_slots = hidden_summary.unsqueeze(1).expand_as(slots)
        other_slots = other_summary.unsqueeze(1).expand_as(slots)
        latent_slots = latent_summary.unsqueeze(1).expand_as(slots)
        update_input = torch.cat([slots, hidden_slots, other_slots, latent_slots], dim=-1)
        candidate = torch.tanh(update(update_input))
        write_gate = torch.sigmoid(gate(update_input)) * self.state_write_scale
        next_slots = slots + write_gate * (candidate - slots)
        with torch.no_grad():
            delta = (next_slots - slots).float().pow(2).mean(dim=-1).sqrt().mean().type_as(hidden_summary)
            write_gate_mean = write_gate.float().mean().type_as(hidden_summary)
        return next_slots, delta, write_gate_mean

    def _masked_mean(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.hidden_norm(hidden_states)
        if attention_mask is None:
            return normed.mean(dim=1)
        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (normed * mask).sum(dim=1) / denom

    def _hidden_read(
        self,
        hidden_states: torch.Tensor,
        latent_field: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = hidden_states.new_tensor(0.0)
        if self.hidden_write_scale <= 0.0 or self.max_hidden_write_ratio <= 0.0:
            return hidden_states, {
                "v7_hidden_delta": zero,
                "v7_latent_hidden_write_norm": zero,
                "v7_hidden_write_ratio": zero,
                "v7_latent_read_entropy": zero,
                "v7_latent_read_max": zero,
                "v7_hidden_write_gate": zero,
            }

        normed_hidden = self.hidden_norm(hidden_states)
        normed_latent = self.latent_norm(latent_field)
        query = self.token_query(normed_hidden)
        key = self.latent_key(normed_latent)
        # Read normalized latent content; raw latent slots start near zero and
        # otherwise make the V7 path effectively inert during early training.
        value = self.latent_value(normed_latent)
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        context, weights = state_softmax_matmul(scores, value, out_dtype=hidden_states.dtype)

        raw_delta = self.hidden_update(context)
        gate = torch.sigmoid(self.hidden_gate(normed_hidden)).type_as(hidden_states)
        delta = raw_delta * gate * self.hidden_write_scale
        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        hidden_norm = hidden_states.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cap = hidden_norm * self.max_hidden_write_ratio
        delta = delta * (cap / delta_norm.clamp_min(1e-6)).clamp(max=1.0).type_as(delta)
        if attention_mask is not None:
            delta = delta * attention_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)

        updated = hidden_states + delta
        with torch.no_grad():
            telemetry_delta = delta.float()
            telemetry_weights = weights.float()
            probs = telemetry_weights.clamp_min(1e-6)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().type_as(hidden_states)
            read_max = telemetry_weights.max(dim=-1).values.mean().type_as(hidden_states)
            delta_norm = telemetry_delta.norm(dim=-1)
            hidden_delta = delta_norm.mean().type_as(hidden_states)
            hidden_delta_by_sample = delta_norm.mean(dim=1).type_as(hidden_states)
            hidden_ratio = (
                delta_norm / hidden_states.float().norm(dim=-1).clamp_min(1e-6)
            ).mean().type_as(hidden_states)
            hidden_gate_mean = gate.float().mean().type_as(hidden_states)
        return updated, {
            "v7_hidden_delta": hidden_delta,
            "v7_hidden_delta_by_sample": hidden_delta_by_sample,
            "v7_latent_hidden_write_norm": hidden_delta,
            "v7_hidden_write_ratio": hidden_ratio,
            "v7_latent_read_entropy": entropy,
            "v7_latent_read_max": read_max,
            "v7_hidden_write_gate": hidden_gate_mean,
        }

    def _zero_hidden_read_metrics(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = hidden_states.new_tensor(0.0)
        return {
            "v7_hidden_delta": zero,
            "v7_hidden_delta_by_sample": torch.zeros(hidden_states.size(0), device=hidden_states.device),
            "v7_latent_hidden_write_norm": zero,
            "v7_hidden_write_ratio": zero,
            "v7_latent_read_entropy": zero,
            "v7_latent_read_max": zero,
            "v7_hidden_write_gate": zero,
        }

    def _update_latent(
        self,
        *,
        hidden_summary: torch.Tensor,
        world_summary: torch.Tensor,
        self_summary: torch.Tensor,
        latent_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        identity = self.initial.to(device=latent_field.device, dtype=latent_field.dtype)
        identity = identity.unsqueeze(0).expand_as(latent_field)
        latent_summary = self.latent_norm(latent_field).mean(dim=1)
        typed_summary = (hidden_summary + world_summary + self_summary + latent_summary) / 4.0
        typed_slots = typed_summary.unsqueeze(1).expand_as(latent_field)
        update_input = torch.cat([latent_field, typed_slots, identity, latent_field - identity], dim=-1)
        candidate = torch.tanh(self.latent_update(update_input))
        gate = torch.sigmoid(self.latent_gate(update_input)) * self.latent_write_scale
        next_latent = latent_field + gate * (candidate - latent_field)
        with torch.no_grad():
            delta_by_sample = (next_latent - latent_field).float().pow(2).mean(dim=-1).sqrt().mean(dim=1)
            delta = delta_by_sample.mean()
        return next_latent, delta, delta_by_sample.type_as(latent_field)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
        latent_field: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        steps: int,
        dynamic_depth: bool = False,
        min_steps: int = 1,
        convergence_threshold: float = 0.0,
        past_latent_field: bool = False,
        past_latent_adapt_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
        zero = hidden_states.new_tensor(0.0)
        if latent_field is None:
            latent_field = self.initial_state(hidden_states.size(0), hidden_states.device, hidden_states.dtype)
        if steps <= 0:
            return hidden_states, world_state, self_state, latent_field, {
                "v7_thought_steps": zero,
                "v7_latent_delta": zero,
                "v7_latent_velocity": zero,
                "v7_latent_acceleration": zero,
                "v7_hidden_delta": zero,
                "v7_hidden_delta_by_sample": torch.zeros(hidden_states.size(0), device=hidden_states.device),
                "v7_latent_hidden_write_norm": zero,
                "v7_hidden_write_ratio": zero,
                "v7_hidden_write_gate": zero,
                "v7_latent_read_entropy": zero,
                "v7_latent_read_max": zero,
                "v7_latent_state_norm": latent_field.detach().float().norm(dim=-1).mean().type_as(hidden_states),
                "v7_world_state_norm": (
                    self._final_slots(world_state).detach().float().norm(dim=-1).mean().type_as(hidden_states)
                    if self._final_slots(world_state) is not None
                    else zero
                ),
                "v7_self_state_norm": (
                    self._final_slots(self_state).detach().float().norm(dim=-1).mean().type_as(hidden_states)
                    if self._final_slots(self_state) is not None
                    else zero
                ),
                "v7_world_delta": zero,
                "v7_self_delta": zero,
                "v7_world_write_gate": zero,
                "v7_self_write_gate": zero,
                "v7_dynamic_depth_enabled": zero,
                "v7_dynamic_depth_mean": zero,
                "v7_dynamic_halt_fraction": zero,
                "v7_dynamic_continue_score": zero,
                "v7_dynamic_convergence_threshold": zero,
                "v7_past_latent_adapt_steps": zero,
                "v7_past_latent_read_suppressed": zero,
            }

        total_latent_delta = zero
        total_hidden_delta = zero
        total_world_delta = zero
        total_self_delta = zero
        previous_latent_delta = zero
        acceleration = zero
        read_metrics: dict[str, torch.Tensor] = {}
        current_latent = latent_field
        current_world = self._final_slots(world_state)
        current_self = self._final_slots(self_state)
        world_gate = zero
        self_gate = zero
        dynamic_depth = bool(dynamic_depth)
        min_steps = max(1, min(int(min_steps), int(steps)))
        adapt_steps = max(0, int(past_latent_adapt_steps)) if past_latent_field else 0
        suppressed_reads = 0
        threshold = max(float(convergence_threshold), 0.0)
        active = torch.ones(hidden_states.size(0), device=hidden_states.device, dtype=torch.bool)
        actual_steps = torch.zeros(hidden_states.size(0), device=hidden_states.device, dtype=torch.float32)
        final_continue_score = torch.zeros_like(actual_steps)
        for step_index in range(steps):
            previous_hidden = hidden_states
            previous_latent = current_latent
            suppress_past_read = step_index < adapt_steps
            if suppress_past_read:
                # Carried latent slots are a prior from the previous chunk. They
                # must reconcile with the current input before writing token hidden states.
                updated_hidden = hidden_states
                read_metrics = self._zero_hidden_read_metrics(hidden_states)
                suppressed_reads += 1
            else:
                updated_hidden, read_metrics = self._hidden_read(hidden_states, current_latent, attention_mask)
            if dynamic_depth:
                active_f = active.to(dtype=hidden_states.dtype).view(-1, 1, 1)
                hidden_for_summary = torch.where(active_f.bool(), updated_hidden, previous_hidden)
            else:
                hidden_for_summary = updated_hidden
            hidden_summary = self._masked_mean(hidden_for_summary, attention_mask)
            world_summary = self._summary(current_world, self.world_norm, hidden_summary)
            self_summary = self._summary(current_self, self.self_norm, hidden_summary)
            next_latent, latent_delta, latent_delta_by_sample = self._update_latent(
                hidden_summary=hidden_summary,
                world_summary=world_summary,
                self_summary=self_summary,
                latent_field=current_latent,
            )
            latent_summary = self.latent_norm(next_latent).mean(dim=1).to(dtype=hidden_summary.dtype)
            next_world, world_delta, world_gate = self._update_typed_state(
                current_world,
                hidden_summary=hidden_summary,
                other_summary=self_summary,
                latent_summary=latent_summary,
                update=self.world_update,
                gate=self.world_gate,
            )
            next_self, self_delta, self_gate = self._update_typed_state(
                current_self,
                hidden_summary=hidden_summary,
                other_summary=world_summary,
                latent_summary=latent_summary,
                update=self.self_update,
                gate=self.self_gate,
            )
            hidden_delta_by_sample = read_metrics.get(
                "v7_hidden_delta_by_sample",
                torch.zeros_like(latent_delta_by_sample),
            ).to(device=hidden_states.device, dtype=latent_delta_by_sample.dtype)
            continue_score = latent_delta_by_sample.detach().float() + hidden_delta_by_sample.detach().float()
            if dynamic_depth:
                hidden_states = torch.where(active_f.bool(), updated_hidden, previous_hidden)
                current_latent = torch.where(active_f.bool(), next_latent, previous_latent)
                if current_world is not None and next_world is not None:
                    current_world = torch.where(active_f.bool(), next_world, current_world)
                if current_self is not None and next_self is not None:
                    current_self = torch.where(active_f.bool(), next_self, current_self)
                with torch.no_grad():
                    active_count = active.float().sum().clamp_min(1.0)
                    latent_delta = (latent_delta_by_sample.float() * active.float()).sum() / active_count
                    hidden_delta = (hidden_delta_by_sample.float() * active.float()).sum() / active_count
                read_metrics["v7_hidden_delta"] = hidden_delta.type_as(hidden_states)
                read_metrics["v7_latent_hidden_write_norm"] = hidden_delta.type_as(hidden_states)
                actual_steps = actual_steps + active.float()
                final_continue_score = torch.where(active, continue_score, final_continue_score)
                if threshold > 0.0 and step_index + 1 >= min_steps:
                    active = active & (continue_score > threshold)
                if not bool(active.any()):
                    total_hidden_delta = total_hidden_delta + hidden_delta.type_as(hidden_states)
                    total_latent_delta = total_latent_delta + latent_delta.type_as(hidden_states)
                    total_world_delta = total_world_delta + world_delta.type_as(hidden_states)
                    total_self_delta = total_self_delta + self_delta.type_as(hidden_states)
                    acceleration = acceleration + (latent_delta.type_as(hidden_states) - previous_latent_delta).abs()
                    previous_latent_delta = latent_delta.detach().type_as(hidden_states)
                    break
            else:
                hidden_states = updated_hidden
                current_latent = next_latent
                current_world = next_world
                current_self = next_self
                actual_steps = actual_steps + 1.0
                final_continue_score = continue_score
                hidden_delta = read_metrics["v7_hidden_delta"]

            total_hidden_delta = total_hidden_delta + hidden_delta.type_as(hidden_states)
            total_latent_delta = total_latent_delta + latent_delta.type_as(hidden_states)
            total_world_delta = total_world_delta + world_delta.type_as(hidden_states)
            total_self_delta = total_self_delta + self_delta.type_as(hidden_states)
            acceleration = acceleration + (latent_delta.type_as(hidden_states) - previous_latent_delta).abs()
            previous_latent_delta = latent_delta.detach()

        actual_steps = actual_steps.clamp_min(1.0)
        steps_t = actual_steps.mean().type_as(hidden_states)
        latent_delta_avg = total_latent_delta / steps_t.clamp_min(1.0)
        hidden_delta_avg = total_hidden_delta / steps_t.clamp_min(1.0)
        world_delta_avg = total_world_delta / steps_t.clamp_min(1.0)
        self_delta_avg = total_self_delta / steps_t.clamp_min(1.0)
        halt_fraction = (actual_steps < float(steps)).float().mean().type_as(hidden_states) if dynamic_depth else zero
        world_final = current_world
        self_final = current_self
        metrics = {
            "v7_thought_steps": steps_t,
            "v7_latent_delta": latent_delta_avg,
            "v7_latent_velocity": latent_delta_avg.float().sqrt().type_as(hidden_states),
            "v7_latent_acceleration": acceleration / steps_t.clamp_min(1.0),
            "v7_hidden_delta": hidden_delta_avg,
            "v7_latent_hidden_write_norm": read_metrics.get("v7_latent_hidden_write_norm", zero),
            "v7_hidden_write_ratio": read_metrics.get("v7_hidden_write_ratio", zero),
            "v7_hidden_write_gate": read_metrics.get("v7_hidden_write_gate", zero),
            "v7_latent_read_entropy": read_metrics.get("v7_latent_read_entropy", zero),
            "v7_latent_read_max": read_metrics.get("v7_latent_read_max", zero),
            "v7_latent_state_norm": current_latent.detach().float().norm(dim=-1).mean().type_as(hidden_states),
            "v7_world_state_norm": (
                world_final.detach().float().norm(dim=-1).mean().type_as(hidden_states)
                if world_final is not None
                else zero
            ),
            "v7_self_state_norm": (
                self_final.detach().float().norm(dim=-1).mean().type_as(hidden_states)
                if self_final is not None
                else zero
            ),
            "v7_world_delta": world_delta_avg,
            "v7_self_delta": self_delta_avg,
            "v7_world_write_gate": world_gate,
            "v7_self_write_gate": self_gate,
            "v7_dynamic_depth_enabled": hidden_states.new_tensor(1.0 if dynamic_depth else 0.0),
            "v7_dynamic_depth_mean": steps_t,
            "v7_dynamic_halt_fraction": halt_fraction,
            "v7_dynamic_continue_score": final_continue_score.mean().type_as(hidden_states),
            "v7_dynamic_convergence_threshold": hidden_states.new_tensor(threshold),
            "v7_past_latent_adapt_steps": hidden_states.new_tensor(float(adapt_steps)),
            "v7_past_latent_read_suppressed": hidden_states.new_tensor(float(suppressed_reads)),
        }
        return hidden_states, world_final, self_final, current_latent, metrics
