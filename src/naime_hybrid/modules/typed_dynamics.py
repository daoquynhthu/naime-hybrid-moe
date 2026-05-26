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
        controller_slots: int = 1,
        latent_write_scale: float = 0.03,
        hidden_write_scale: float = 0.01,
        max_hidden_write_ratio: float = 0.05,
        state_write_scale: float = 0.02,
        controller_write_scale: float = 0.02,
        world_state_write_scale: float | None = None,
        self_state_write_scale: float | None = None,
        latent_timescale: float = 1.0,
        world_timescale: float = 1.0,
        self_timescale: float = 1.0,
        controller_mode: str = "fixed",
        homeostatic_control: bool = False,
        homeostatic_strength: float = 0.25,
        homeostatic_min_scale: float = 0.5,
        homeostatic_max_scale: float = 1.5,
        state_compatibility_gate: bool = False,
        state_compatibility_strength: float = 1.0,
        state_compatibility_min: float = 0.0,
        adaptive_tau: bool = False,
        adaptive_tau_min: float = 0.5,
        adaptive_tau_max: float = 1.5,
        hyperspherical_state: bool = True,
        causal_summary: bool = True,
        causal_summary_decay: float = 0.98,
    ) -> None:
        super().__init__()
        if latent_slots <= 0:
            raise ValueError("V7 latent slots must be positive")
        self.latent_slots = latent_slots
        self.controller_slots = max(int(controller_slots), 0)
        self.latent_timescale = max(latent_timescale, 0.0)
        self.world_timescale = max(world_timescale, 0.0)
        self.self_timescale = max(self_timescale, 0.0)
        self.controller_mode = controller_mode
        self.homeostatic_control = bool(homeostatic_control)
        self.homeostatic_strength = max(float(homeostatic_strength), 0.0)
        self.homeostatic_min_scale = max(float(homeostatic_min_scale), 1e-3)
        self.homeostatic_max_scale = max(float(homeostatic_max_scale), self.homeostatic_min_scale)
        self.state_compatibility_gate = bool(state_compatibility_gate)
        self.state_compatibility_strength = max(float(state_compatibility_strength), 0.0)
        self.state_compatibility_min = min(max(float(state_compatibility_min), 0.0), 1.0)
        self.adaptive_tau = bool(adaptive_tau)
        self.adaptive_tau_min = max(float(adaptive_tau_min), 0.0)
        self.adaptive_tau_max = max(float(adaptive_tau_max), self.adaptive_tau_min)
        self.hyperspherical_state = bool(hyperspherical_state)
        self.causal_summary = bool(causal_summary)
        self.causal_summary_decay = min(max(float(causal_summary_decay), 0.0), 0.9999)
        self.latent_write_scale = max(latent_write_scale, 0.0) * self.latent_timescale
        self.hidden_write_scale = max(hidden_write_scale, 0.0)
        self.max_hidden_write_ratio = max(max_hidden_write_ratio, 0.0)
        self.state_write_scale = max(state_write_scale, 0.0)
        world_scale = state_write_scale if world_state_write_scale is None else world_state_write_scale
        self_scale = state_write_scale if self_state_write_scale is None else self_state_write_scale
        self.world_state_write_scale = max(world_scale, 0.0) * self.world_timescale
        self.self_state_write_scale = max(self_scale, 0.0) * self.self_timescale
        self.controller_write_scale = max(controller_write_scale, 0.0)

        self.initial = nn.Parameter(torch.zeros(latent_slots, d_model))
        self.controller_initial = nn.Parameter(torch.zeros(max(self.controller_slots, 1), d_model))
        self.hidden_norm = RMSNorm(d_model)
        self.world_norm = RMSNorm(d_model)
        self.self_norm = RMSNorm(d_model)
        self.latent_norm = RMSNorm(d_model)
        self.controller_norm = RMSNorm(d_model)
        self.token_query = nn.Linear(d_model, d_model, bias=False)
        self.latent_key = nn.Linear(d_model, d_model, bias=False)
        self.latent_value = nn.Linear(d_model, d_model, bias=False)
        self.hidden_update = nn.Linear(d_model, d_model)
        self.hidden_gate = nn.Linear(d_model, 1)
        self.latent_update = nn.Linear(d_model * 4, d_model)
        self.latent_gate = nn.Linear(d_model * 4, d_model)
        self.latent_tau = nn.Linear(d_model * 4, d_model)
        self.world_update = nn.Linear(d_model * 4, d_model)
        self.world_gate = nn.Linear(d_model * 4, d_model)
        self.world_tau = nn.Linear(d_model * 4, d_model)
        self.self_update = nn.Linear(d_model * 4, d_model)
        self.self_gate = nn.Linear(d_model * 4, d_model)
        self.self_tau = nn.Linear(d_model * 4, d_model)
        self.controller_update = nn.Linear(d_model * 4, d_model)
        self.controller_gate = nn.Linear(d_model * 4, d_model)
        self.controller_tau = nn.Linear(d_model * 4, d_model)
        self.state_compatibility = nn.Linear(d_model * 4, 5)
        nn.init.normal_(self.initial, mean=0.0, std=0.005)
        nn.init.normal_(self.controller_initial, mean=0.0, std=0.005)
        nn.init.constant_(self.hidden_gate.bias, -3.0)

    def reset_protocol_biases(self) -> None:
        """Keep optional V7 controllers neutral after external init passes."""

        nn.init.zeros_(self.state_compatibility.weight)
        nn.init.constant_(self.state_compatibility.bias, 2.0)
        for layer in (self.latent_tau, self.world_tau, self.self_tau, self.controller_tau):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        initial = self.initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        if self.hyperspherical_state:
            initial = self.latent_norm(initial).to(dtype=dtype)
        return initial

    def initial_controller_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.controller_slots <= 0:
            return None
        initial = self.controller_initial[: self.controller_slots]
        initial = initial.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        if self.hyperspherical_state:
            initial = self.controller_norm(initial).to(dtype=dtype)
        return initial

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

    def _state_candidate(
        self,
        layer: nn.Linear,
        norm: nn.Module,
        update_input: torch.Tensor,
    ) -> torch.Tensor:
        raw = layer(update_input)
        if not self.hyperspherical_state:
            return torch.tanh(raw)
        return norm(raw).to(dtype=update_input.dtype)

    def _state_update(
        self,
        slots: torch.Tensor,
        candidate: torch.Tensor,
        write_gate: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:
        next_slots = slots + write_gate * (candidate - slots)
        if not self.hyperspherical_state:
            return next_slots
        return norm(next_slots).to(dtype=slots.dtype)

    def _adaptive_tau(
        self,
        layer: nn.Linear,
        update_input: torch.Tensor,
        *,
        dtype_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one = dtype_ref.new_tensor(1.0)
        if not self.adaptive_tau:
            return one, one
        tau = torch.sigmoid(layer(update_input))
        tau = self.adaptive_tau_min + (self.adaptive_tau_max - self.adaptive_tau_min) * tau
        tau = tau.to(dtype=dtype_ref.dtype)
        with torch.no_grad():
            tau_mean = tau.float().mean().type_as(dtype_ref)
        return tau, tau_mean

    def _apply_state_compatibility(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        latent_field: torch.Tensor,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
        controller_state: torch.Tensor | None,
        enabled: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        zero = hidden_states.new_tensor(0.0)
        one = hidden_states.new_tensor(1.0)
        metrics = {
            "v7_state_compatibility_enabled": hidden_states.new_tensor(
                1.0 if self.state_compatibility_gate else 0.0
            ),
            "v7_carry_compatibility": one,
            "v7_carry_latent_gate": one,
            "v7_carry_world_gate": one,
            "v7_carry_self_gate": one,
            "v7_carry_controller_gate": one,
            "v7_carry_memory_gate": one,
            "v7_carry_blend_delta": zero,
        }
        if (
            not enabled
            or not self.state_compatibility_gate
            or self.state_compatibility_strength <= 0.0
        ):
            return latent_field, world_state, self_state, controller_state, metrics

        hidden_summary = self._sequence_summary(hidden_states, attention_mask)
        latent_summary = self.latent_norm(latent_field).mean(dim=1).to(dtype=hidden_summary.dtype)
        world_summary = self._summary(world_state, self.world_norm, hidden_summary)
        self_summary = self._summary(self_state, self.self_norm, hidden_summary)
        compat_input = torch.cat([hidden_summary, latent_summary, world_summary, self_summary], dim=-1)
        raw_gates = torch.sigmoid(self.state_compatibility(compat_input) * self.state_compatibility_strength)
        gates = self.state_compatibility_min + (1.0 - self.state_compatibility_min) * raw_gates
        gates = gates.to(dtype=latent_field.dtype)
        latent_gate = gates[:, 0].view(-1, 1, 1)
        latent_prior = self.initial_state(latent_field.size(0), latent_field.device, latent_field.dtype)
        next_latent = latent_gate * latent_field + (1.0 - latent_gate) * latent_prior

        next_world = world_state
        if world_state is not None:
            world_gate = gates[:, 1].view(-1, 1, 1).to(dtype=world_state.dtype)
            compact_world = self._final_slots(world_state)
            world_prior = torch.zeros_like(compact_world)
            if self.hyperspherical_state:
                world_prior = self.world_norm(world_prior).to(dtype=world_state.dtype)
                compact_world = self.world_norm(compact_world).to(dtype=world_state.dtype)
            next_world = world_gate * compact_world + (1.0 - world_gate) * world_prior

        next_self = self_state
        if self_state is not None:
            self_gate = gates[:, 2].view(-1, 1, 1).to(dtype=self_state.dtype)
            compact_self = self._final_slots(self_state)
            self_prior = torch.zeros_like(compact_self)
            if self.hyperspherical_state:
                self_prior = self.self_norm(self_prior).to(dtype=self_state.dtype)
                compact_self = self.self_norm(compact_self).to(dtype=self_state.dtype)
            next_self = self_gate * compact_self + (1.0 - self_gate) * self_prior

        next_controller = controller_state
        controller_gate_mean = one
        if controller_state is not None and self.controller_slots > 0:
            controller_gate = gates[:, 3].view(-1, 1, 1).to(dtype=controller_state.dtype)
            controller_prior = self.initial_controller_state(
                controller_state.size(0),
                controller_state.device,
                controller_state.dtype,
            )
            if controller_prior is not None:
                next_controller = controller_gate * controller_state + (1.0 - controller_gate) * controller_prior
                controller_gate_mean = controller_gate.float().mean().type_as(hidden_states)

        with torch.no_grad():
            blend_delta = (next_latent - latent_field).float().pow(2).mean(dim=-1).sqrt().mean().type_as(hidden_states)
            metrics = {
                "v7_state_compatibility_enabled": hidden_states.new_tensor(1.0),
                "v7_carry_compatibility": gates.float().mean().type_as(hidden_states),
                "v7_carry_latent_gate": gates[:, 0].float().mean().type_as(hidden_states),
                "v7_carry_world_gate": gates[:, 1].float().mean().type_as(hidden_states),
                "v7_carry_self_gate": gates[:, 2].float().mean().type_as(hidden_states),
                "v7_carry_controller_gate": controller_gate_mean,
                "v7_carry_memory_gate": gates[:, 4].float().mean().type_as(hidden_states),
                "v7_carry_blend_delta": blend_delta,
            }
        return next_latent, next_world, next_self, next_controller, metrics

    def _update_typed_state(
        self,
        state: torch.Tensor | None,
        *,
        hidden_summary: torch.Tensor,
        other_summary: torch.Tensor,
        latent_summary: torch.Tensor,
        update: nn.Linear,
        gate: nn.Linear,
        tau_layer: nn.Linear,
        write_scale: float,
        rate_scale: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = hidden_summary.new_tensor(0.0)
        one = hidden_summary.new_tensor(1.0)
        zero_by_sample = torch.zeros(hidden_summary.size(0), device=hidden_summary.device, dtype=hidden_summary.dtype)
        if state is None or write_scale <= 0.0:
            return state, zero, zero_by_sample, zero, one
        slots = self._final_slots(state)
        if slots is None:
            return state, zero, zero_by_sample, zero, one
        norm = self.world_norm if tau_layer is self.world_tau else self.self_norm
        if self.hyperspherical_state:
            slots = norm(slots).to(dtype=hidden_summary.dtype)
        hidden_slots = hidden_summary.unsqueeze(1).expand_as(slots)
        other_slots = other_summary.unsqueeze(1).expand_as(slots)
        latent_slots = latent_summary.unsqueeze(1).expand_as(slots)
        update_input = torch.cat([slots, hidden_slots, other_slots, latent_slots], dim=-1)
        candidate = self._state_candidate(update, norm, update_input)
        rate = torch.as_tensor(rate_scale, device=hidden_summary.device, dtype=hidden_summary.dtype)
        tau, tau_mean = self._adaptive_tau(tau_layer, update_input, dtype_ref=hidden_summary)
        write_gate = torch.sigmoid(gate(update_input)) * (write_scale * rate) * tau
        next_slots = self._state_update(slots, candidate, write_gate, norm)
        with torch.no_grad():
            delta_by_sample = (next_slots - slots).float().pow(2).mean(dim=-1).sqrt().mean(dim=1)
            delta = delta_by_sample.mean().type_as(hidden_summary)
            write_gate_mean = write_gate.float().mean().type_as(hidden_summary)
        return next_slots, delta, delta_by_sample.type_as(hidden_summary), write_gate_mean, tau_mean

    def _masked_mean(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.hidden_norm(hidden_states)
        if attention_mask is None:
            return normed.mean(dim=1)
        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (normed * mask).sum(dim=1) / denom

    def _causal_ema_summary(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.hidden_norm(hidden_states)
        batch, seq_len, _ = normed.shape
        decay = hidden_states.new_tensor(float(self.causal_summary_decay))
        positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32)
        weights = torch.pow(decay.float(), (seq_len - 1 - positions).clamp_min(0.0))
        weights = weights.to(dtype=hidden_states.dtype).view(1, seq_len, 1)
        if attention_mask is not None:
            weights = weights * attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).view(
                batch, seq_len, 1
            )
        denom = weights.sum(dim=1).clamp_min(1e-6)
        return (normed * weights).sum(dim=1) / denom

    def _sequence_summary(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if self.causal_summary:
            return self._causal_ema_summary(hidden_states, attention_mask)
        return self._masked_mean(hidden_states, attention_mask)

    def _hidden_read(
        self,
        hidden_states: torch.Tensor,
        latent_field: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rate_scale: torch.Tensor | float = 1.0,
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
        rate = torch.as_tensor(rate_scale, device=hidden_states.device, dtype=hidden_states.dtype)
        delta = raw_delta * gate * (self.hidden_write_scale * rate)
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

    def _homeostatic_scales(
        self,
        *,
        latent_delta: torch.Tensor,
        world_delta: torch.Tensor,
        self_delta: torch.Tensor,
        hidden_delta: torch.Tensor,
        previous_latent_delta: torch.Tensor,
        previous_world_delta: torch.Tensor,
        previous_self_delta: torch.Tensor,
        previous_hidden_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute next-step rate scales from relative dynamic homeostasis.

        The controller deliberately avoids absolute "fast/slow" thresholds.
        It compares each typed state's motion against the current typed-state
        ensemble and damps abrupt acceleration against the previous dynamics
        step. The returned scales are detached control signals, not learned
        gradients.
        """

        one = latent_delta.new_tensor(1.0)
        zero = latent_delta.new_tensor(0.0)
        if not self.homeostatic_control or self.homeostatic_strength <= 0.0:
            return {
                "latent": one,
                "world": one,
                "self": one,
                "hidden": one,
                "balance_pressure": zero,
                "accel_pressure": zero,
                "dhi": one,
            }

        eps = latent_delta.new_tensor(1e-6)
        strength = latent_delta.new_tensor(float(self.homeostatic_strength))
        min_scale = float(self.homeostatic_min_scale)
        max_scale = float(self.homeostatic_max_scale)
        motions = torch.stack(
            [
                latent_delta.detach().float().clamp_min(0.0),
                world_delta.detach().float().clamp_min(0.0),
                self_delta.detach().float().clamp_min(0.0),
            ]
        )
        mean_motion = motions.mean().clamp_min(eps.float())
        log_pressure = torch.log((motions + eps.float()) / mean_motion)
        balance_scales = torch.exp(-strength.float() * log_pressure).clamp(min_scale, max_scale)

        previous = torch.stack(
            [
                previous_latent_delta.detach().float().clamp_min(0.0),
                previous_world_delta.detach().float().clamp_min(0.0),
                previous_self_delta.detach().float().clamp_min(0.0),
            ]
        )
        accel = torch.where(
            previous > eps.float(),
            (motions - previous).abs() / (motions + previous + eps.float()),
            torch.zeros_like(motions),
        )
        accel_scales = torch.exp(-strength.float() * accel).clamp(min_scale, max_scale)
        typed_scales = (balance_scales * accel_scales).clamp(min_scale, max_scale).to(dtype=latent_delta.dtype)

        hidden_prev = previous_hidden_delta.detach().float().clamp_min(0.0)
        hidden_now = hidden_delta.detach().float().clamp_min(0.0)
        hidden_accel = torch.where(
            hidden_prev > eps.float(),
            (hidden_now - hidden_prev).abs() / (hidden_now + hidden_prev + eps.float()),
            hidden_now.new_tensor(0.0),
        )
        hidden_scale = torch.exp(-strength.float() * hidden_accel).clamp(min_scale, max_scale).to(dtype=latent_delta.dtype)
        balance_pressure = log_pressure.abs().mean().to(dtype=latent_delta.dtype)
        accel_pressure = torch.cat([accel, hidden_accel.view(1)]).mean().to(dtype=latent_delta.dtype)
        dhi = torch.exp(-(balance_pressure.float() + accel_pressure.float())).to(dtype=latent_delta.dtype)
        return {
            "latent": typed_scales[0],
            "world": typed_scales[1],
            "self": typed_scales[2],
            "hidden": hidden_scale,
            "balance_pressure": balance_pressure,
            "accel_pressure": accel_pressure,
            "dhi": dhi,
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
        rate_scale: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        identity = self.initial.to(device=latent_field.device, dtype=latent_field.dtype)
        identity = identity.unsqueeze(0).expand_as(latent_field)
        if self.hyperspherical_state:
            latent_field = self.latent_norm(latent_field).to(dtype=latent_field.dtype)
            identity = self.latent_norm(identity).to(dtype=latent_field.dtype)
        latent_summary = self.latent_norm(latent_field).mean(dim=1)
        typed_summary = (hidden_summary + world_summary + self_summary + latent_summary) / 4.0
        typed_slots = typed_summary.unsqueeze(1).expand_as(latent_field)
        update_input = torch.cat([latent_field, typed_slots, identity, latent_field - identity], dim=-1)
        candidate = self._state_candidate(self.latent_update, self.latent_norm, update_input)
        rate = torch.as_tensor(rate_scale, device=latent_field.device, dtype=latent_field.dtype)
        tau, tau_mean = self._adaptive_tau(self.latent_tau, update_input, dtype_ref=latent_field)
        gate = torch.sigmoid(self.latent_gate(update_input)) * (self.latent_write_scale * rate) * tau
        next_latent = self._state_update(latent_field, candidate, gate, self.latent_norm)
        with torch.no_grad():
            delta_by_sample = (next_latent - latent_field).float().pow(2).mean(dim=-1).sqrt().mean(dim=1)
            delta = delta_by_sample.mean()
        return next_latent, delta, delta_by_sample.type_as(latent_field), tau_mean

    def _update_controller_state(
        self,
        controller_state: torch.Tensor | None,
        *,
        hidden_summary: torch.Tensor,
        world_summary: torch.Tensor,
        self_summary: torch.Tensor,
        latent_summary: torch.Tensor,
        rate_scale: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = hidden_summary.new_tensor(0.0)
        one = hidden_summary.new_tensor(1.0)
        zero_by_sample = torch.zeros(hidden_summary.size(0), device=hidden_summary.device, dtype=hidden_summary.dtype)
        if controller_state is None or self.controller_slots <= 0 or self.controller_write_scale <= 0.0:
            return controller_state, zero, zero_by_sample, zero, one
        slots = self._final_slots(controller_state)
        if slots is None:
            return controller_state, zero, zero_by_sample, zero, one
        if self.hyperspherical_state:
            slots = self.controller_norm(slots).to(dtype=hidden_summary.dtype)
        typed_summary = (hidden_summary + world_summary + self_summary + latent_summary) / 4.0
        hidden_slots = hidden_summary.unsqueeze(1).expand_as(slots)
        typed_slots = typed_summary.unsqueeze(1).expand_as(slots)
        latent_slots = latent_summary.unsqueeze(1).expand_as(slots)
        current_slots = self.controller_norm(slots)
        update_input = torch.cat([current_slots, hidden_slots, typed_slots, latent_slots], dim=-1)
        candidate = self._state_candidate(self.controller_update, self.controller_norm, update_input)
        rate = torch.as_tensor(rate_scale, device=hidden_summary.device, dtype=hidden_summary.dtype)
        tau, tau_mean = self._adaptive_tau(self.controller_tau, update_input, dtype_ref=hidden_summary)
        write_gate = torch.sigmoid(self.controller_gate(update_input)) * (self.controller_write_scale * rate) * tau
        next_slots = self._state_update(slots, candidate, write_gate, self.controller_norm)
        with torch.no_grad():
            delta_by_sample = (next_slots - slots).float().pow(2).mean(dim=-1).sqrt().mean(dim=1)
            delta = delta_by_sample.mean().type_as(hidden_summary)
            write_gate_mean = write_gate.float().mean().type_as(hidden_summary)
        return next_slots, delta, delta_by_sample.type_as(hidden_summary), write_gate_mean, tau_mean

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
        controller_state: torch.Tensor | None,
        latent_field: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        steps: int,
        dynamic_depth: bool = False,
        min_steps: int = 1,
        convergence_threshold: float = 0.0,
        past_latent_field: bool = False,
        past_latent_adapt_steps: int = 0,
        apply_state_compatibility: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        zero = hidden_states.new_tensor(0.0)
        if latent_field is None:
            latent_field = self.initial_state(hidden_states.size(0), hidden_states.device, hidden_states.dtype)
        default_compat_metrics = {
            "v7_state_compatibility_enabled": hidden_states.new_tensor(
                1.0 if self.state_compatibility_gate else 0.0
            ),
            "v7_carry_compatibility": hidden_states.new_tensor(1.0),
            "v7_carry_latent_gate": hidden_states.new_tensor(1.0),
            "v7_carry_world_gate": hidden_states.new_tensor(1.0),
            "v7_carry_self_gate": hidden_states.new_tensor(1.0),
            "v7_carry_controller_gate": hidden_states.new_tensor(1.0),
            "v7_carry_memory_gate": hidden_states.new_tensor(1.0),
            "v7_carry_blend_delta": zero,
        }
        if steps <= 0:
            metrics = {
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
                "v7_controller_state_norm": (
                    self._final_slots(controller_state).detach().float().norm(dim=-1).mean().type_as(hidden_states)
                    if self._final_slots(controller_state) is not None
                    else zero
                ),
                "v7_world_delta": zero,
                "v7_self_delta": zero,
                "v7_controller_delta": zero,
                "v7_world_write_gate": zero,
                "v7_self_write_gate": zero,
                "v7_controller_write_gate": zero,
                "v7_dynamic_depth_enabled": zero,
                "v7_dynamic_depth_mean": zero,
                "v7_dynamic_halt_fraction": zero,
                "v7_dynamic_continue_score": zero,
                "v7_dynamic_convergence_threshold": zero,
                "v7_past_latent_adapt_steps": zero,
                "v7_past_latent_read_suppressed": zero,
                "v7_latent_timescale": hidden_states.new_tensor(float(self.latent_timescale)),
                "v7_world_timescale": hidden_states.new_tensor(float(self.world_timescale)),
                "v7_self_timescale": hidden_states.new_tensor(float(self.self_timescale)),
                "v7_controller_fixed": hidden_states.new_tensor(1.0 if self.controller_mode == "fixed" else 0.0),
                "v7_homeostatic_control_enabled": hidden_states.new_tensor(1.0 if self.homeostatic_control else 0.0),
                "v7_homeostatic_dhi": hidden_states.new_tensor(1.0),
                "v7_homeostatic_balance_pressure": zero,
                "v7_homeostatic_accel_pressure": zero,
                "v7_latent_rate_scale": hidden_states.new_tensor(1.0),
                "v7_world_rate_scale": hidden_states.new_tensor(1.0),
                "v7_self_rate_scale": hidden_states.new_tensor(1.0),
                "v7_hidden_read_rate_scale": hidden_states.new_tensor(1.0),
                "v7_hyperspherical_state_enabled": hidden_states.new_tensor(
                    1.0 if self.hyperspherical_state else 0.0
                ),
                "v7_causal_summary_enabled": hidden_states.new_tensor(1.0 if self.causal_summary else 0.0),
                "v7_causal_summary_decay": hidden_states.new_tensor(float(self.causal_summary_decay)),
                "v7_adaptive_tau_enabled": hidden_states.new_tensor(1.0 if self.adaptive_tau else 0.0),
                "v7_latent_tau": hidden_states.new_tensor(1.0),
                "v7_world_tau": hidden_states.new_tensor(1.0),
                "v7_self_tau": hidden_states.new_tensor(1.0),
                "v7_controller_tau": hidden_states.new_tensor(1.0),
                "v7_effective_latent_write_scale": hidden_states.new_tensor(float(self.latent_write_scale)),
                "v7_effective_world_write_scale": hidden_states.new_tensor(float(self.world_state_write_scale)),
                "v7_effective_self_write_scale": hidden_states.new_tensor(float(self.self_state_write_scale)),
                "v7_effective_controller_write_scale": hidden_states.new_tensor(float(self.controller_write_scale)),
            }
            metrics.update(default_compat_metrics)
            return hidden_states, world_state, self_state, latent_field, controller_state, metrics

        total_latent_delta = zero
        total_hidden_delta = zero
        total_world_delta = zero
        total_self_delta = zero
        total_controller_delta = zero
        previous_latent_delta = zero
        previous_world_delta = zero
        previous_self_delta = zero
        previous_hidden_delta = zero
        acceleration = zero
        read_metrics: dict[str, torch.Tensor] = {}
        rate_scales = {
            "latent": hidden_states.new_tensor(1.0),
            "world": hidden_states.new_tensor(1.0),
            "self": hidden_states.new_tensor(1.0),
            "hidden": hidden_states.new_tensor(1.0),
            "balance_pressure": zero,
            "accel_pressure": zero,
            "dhi": hidden_states.new_tensor(1.0),
        }
        total_latent_rate = zero
        total_world_rate = zero
        total_self_rate = zero
        total_hidden_rate = zero
        total_homeostasis_balance = zero
        total_homeostasis_accel = zero
        total_homeostasis_dhi = zero
        total_latent_tau = zero
        total_world_tau = zero
        total_self_tau = zero
        total_controller_tau = zero
        latent_field, world_state, self_state, controller_state, compat_metrics = self._apply_state_compatibility(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            latent_field=latent_field,
            world_state=world_state,
            self_state=self_state,
            controller_state=controller_state,
            enabled=past_latent_field and apply_state_compatibility,
        )
        current_latent = latent_field
        # Causal state protocol: current logits may read only the latent field
        # that entered this forward call. The newly updated latent is an
        # outgoing state packet for future chunks/sessions, not a same-sequence
        # hidden patch. When no past latent is provided, V7 only writes state.
        readable_latent = latent_field if past_latent_field else None
        current_world = self._final_slots(world_state)
        current_self = self._final_slots(self_state)
        current_controller = self._final_slots(controller_state)
        world_gate = zero
        self_gate = zero
        controller_gate = zero
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
            if suppress_past_read or readable_latent is None:
                # Carried latent slots are a prior from the previous chunk. They
                # must reconcile with the current input before writing token hidden states.
                updated_hidden = hidden_states
                read_metrics = self._zero_hidden_read_metrics(hidden_states)
                suppressed_reads += 1
            else:
                updated_hidden, read_metrics = self._hidden_read(
                    hidden_states,
                    readable_latent,
                    attention_mask,
                    rate_scale=rate_scales["hidden"],
                )
            if dynamic_depth:
                active_f = active.to(dtype=hidden_states.dtype).view(-1, 1, 1)
                hidden_for_summary = torch.where(active_f.bool(), updated_hidden, previous_hidden)
            else:
                hidden_for_summary = updated_hidden
            hidden_summary = self._sequence_summary(hidden_for_summary, attention_mask)
            world_summary = self._summary(current_world, self.world_norm, hidden_summary)
            self_summary = self._summary(current_self, self.self_norm, hidden_summary)
            next_latent, latent_delta, latent_delta_by_sample, latent_tau = self._update_latent(
                hidden_summary=hidden_summary,
                world_summary=world_summary,
                self_summary=self_summary,
                latent_field=current_latent,
                rate_scale=rate_scales["latent"],
            )
            latent_summary = self.latent_norm(next_latent).mean(dim=1).to(dtype=hidden_summary.dtype)
            next_world, world_delta, world_delta_by_sample, world_gate, world_tau = self._update_typed_state(
                current_world,
                hidden_summary=hidden_summary,
                other_summary=self_summary,
                latent_summary=latent_summary,
                update=self.world_update,
                gate=self.world_gate,
                tau_layer=self.world_tau,
                write_scale=self.world_state_write_scale,
                rate_scale=rate_scales["world"],
            )
            next_self, self_delta, self_delta_by_sample, self_gate, self_tau = self._update_typed_state(
                current_self,
                hidden_summary=hidden_summary,
                other_summary=world_summary,
                latent_summary=latent_summary,
                update=self.self_update,
                gate=self.self_gate,
                tau_layer=self.self_tau,
                write_scale=self.self_state_write_scale,
                rate_scale=rate_scales["self"],
            )
            next_controller, controller_delta, controller_delta_by_sample, controller_gate, controller_tau = self._update_controller_state(
                current_controller,
                hidden_summary=hidden_summary,
                world_summary=world_summary,
                self_summary=self_summary,
                latent_summary=latent_summary,
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
                if current_controller is not None and next_controller is not None:
                    current_controller = torch.where(active_f.bool(), next_controller, current_controller)
                with torch.no_grad():
                    active_count = active.float().sum().clamp_min(1.0)
                    latent_delta = (latent_delta_by_sample.float() * active.float()).sum() / active_count
                    hidden_delta = (hidden_delta_by_sample.float() * active.float()).sum() / active_count
                    world_delta = (world_delta_by_sample.float() * active.float()).sum() / active_count
                    self_delta = (self_delta_by_sample.float() * active.float()).sum() / active_count
                    controller_delta = (
                        controller_delta_by_sample.float() * active.float()
                    ).sum() / active_count
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
                    total_controller_delta = total_controller_delta + controller_delta.type_as(hidden_states)
                    acceleration = acceleration + (latent_delta.type_as(hidden_states) - previous_latent_delta).abs()
                    total_latent_rate = total_latent_rate + rate_scales["latent"].type_as(hidden_states)
                    total_world_rate = total_world_rate + rate_scales["world"].type_as(hidden_states)
                    total_self_rate = total_self_rate + rate_scales["self"].type_as(hidden_states)
                    total_hidden_rate = total_hidden_rate + rate_scales["hidden"].type_as(hidden_states)
                    total_latent_tau = total_latent_tau + latent_tau.type_as(hidden_states)
                    total_world_tau = total_world_tau + world_tau.type_as(hidden_states)
                    total_self_tau = total_self_tau + self_tau.type_as(hidden_states)
                    total_controller_tau = total_controller_tau + controller_tau.type_as(hidden_states)
                    total_homeostasis_balance = total_homeostasis_balance + rate_scales["balance_pressure"].type_as(
                        hidden_states
                    )
                    total_homeostasis_accel = total_homeostasis_accel + rate_scales["accel_pressure"].type_as(
                        hidden_states
                    )
                    total_homeostasis_dhi = total_homeostasis_dhi + rate_scales["dhi"].type_as(hidden_states)
                    previous_latent_delta = latent_delta.detach().type_as(hidden_states)
                    break
            else:
                hidden_states = updated_hidden
                current_latent = next_latent
                current_world = next_world
                current_self = next_self
                current_controller = next_controller
                actual_steps = actual_steps + 1.0
                final_continue_score = continue_score
                hidden_delta = read_metrics["v7_hidden_delta"]

            total_hidden_delta = total_hidden_delta + hidden_delta.type_as(hidden_states)
            total_latent_delta = total_latent_delta + latent_delta.type_as(hidden_states)
            total_world_delta = total_world_delta + world_delta.type_as(hidden_states)
            total_self_delta = total_self_delta + self_delta.type_as(hidden_states)
            total_controller_delta = total_controller_delta + controller_delta.type_as(hidden_states)
            acceleration = acceleration + (latent_delta.type_as(hidden_states) - previous_latent_delta).abs()
            total_latent_rate = total_latent_rate + rate_scales["latent"].type_as(hidden_states)
            total_world_rate = total_world_rate + rate_scales["world"].type_as(hidden_states)
            total_self_rate = total_self_rate + rate_scales["self"].type_as(hidden_states)
            total_hidden_rate = total_hidden_rate + rate_scales["hidden"].type_as(hidden_states)
            total_latent_tau = total_latent_tau + latent_tau.type_as(hidden_states)
            total_world_tau = total_world_tau + world_tau.type_as(hidden_states)
            total_self_tau = total_self_tau + self_tau.type_as(hidden_states)
            total_controller_tau = total_controller_tau + controller_tau.type_as(hidden_states)
            total_homeostasis_balance = total_homeostasis_balance + rate_scales["balance_pressure"].type_as(hidden_states)
            total_homeostasis_accel = total_homeostasis_accel + rate_scales["accel_pressure"].type_as(hidden_states)
            total_homeostasis_dhi = total_homeostasis_dhi + rate_scales["dhi"].type_as(hidden_states)
            rate_scales = self._homeostatic_scales(
                latent_delta=latent_delta,
                world_delta=world_delta,
                self_delta=self_delta,
                hidden_delta=hidden_delta,
                previous_latent_delta=previous_latent_delta,
                previous_world_delta=previous_world_delta,
                previous_self_delta=previous_self_delta,
                previous_hidden_delta=previous_hidden_delta,
            )
            previous_latent_delta = latent_delta.detach()
            previous_world_delta = world_delta.detach()
            previous_self_delta = self_delta.detach()
            previous_hidden_delta = hidden_delta.detach()

        actual_steps = actual_steps.clamp_min(1.0)
        steps_t = actual_steps.mean().type_as(hidden_states)
        latent_delta_avg = total_latent_delta / steps_t.clamp_min(1.0)
        hidden_delta_avg = total_hidden_delta / steps_t.clamp_min(1.0)
        world_delta_avg = total_world_delta / steps_t.clamp_min(1.0)
        self_delta_avg = total_self_delta / steps_t.clamp_min(1.0)
        controller_delta_avg = total_controller_delta / steps_t.clamp_min(1.0)
        latent_rate_avg = total_latent_rate / steps_t.clamp_min(1.0)
        world_rate_avg = total_world_rate / steps_t.clamp_min(1.0)
        self_rate_avg = total_self_rate / steps_t.clamp_min(1.0)
        hidden_rate_avg = total_hidden_rate / steps_t.clamp_min(1.0)
        latent_tau_avg = total_latent_tau / steps_t.clamp_min(1.0)
        world_tau_avg = total_world_tau / steps_t.clamp_min(1.0)
        self_tau_avg = total_self_tau / steps_t.clamp_min(1.0)
        controller_tau_avg = total_controller_tau / steps_t.clamp_min(1.0)
        homeostasis_balance = total_homeostasis_balance / steps_t.clamp_min(1.0)
        homeostasis_accel = total_homeostasis_accel / steps_t.clamp_min(1.0)
        homeostasis_dhi = total_homeostasis_dhi / steps_t.clamp_min(1.0)
        halt_fraction = (actual_steps < float(steps)).float().mean().type_as(hidden_states) if dynamic_depth else zero
        world_final = current_world
        self_final = current_self
        controller_final = current_controller
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
            "v7_controller_state_norm": (
                controller_final.detach().float().norm(dim=-1).mean().type_as(hidden_states)
                if controller_final is not None
                else zero
            ),
            "v7_world_delta": world_delta_avg,
            "v7_self_delta": self_delta_avg,
            "v7_controller_delta": controller_delta_avg,
            "v7_world_write_gate": world_gate,
            "v7_self_write_gate": self_gate,
            "v7_controller_write_gate": controller_gate,
            "v7_dynamic_depth_enabled": hidden_states.new_tensor(1.0 if dynamic_depth else 0.0),
            "v7_dynamic_depth_mean": steps_t,
            "v7_dynamic_halt_fraction": halt_fraction,
            "v7_dynamic_continue_score": final_continue_score.mean().type_as(hidden_states),
            "v7_dynamic_convergence_threshold": hidden_states.new_tensor(threshold),
            "v7_past_latent_adapt_steps": hidden_states.new_tensor(float(adapt_steps)),
            "v7_past_latent_read_suppressed": hidden_states.new_tensor(float(suppressed_reads)),
            "v7_latent_timescale": hidden_states.new_tensor(float(self.latent_timescale)),
            "v7_world_timescale": hidden_states.new_tensor(float(self.world_timescale)),
            "v7_self_timescale": hidden_states.new_tensor(float(self.self_timescale)),
            "v7_controller_fixed": hidden_states.new_tensor(1.0 if self.controller_mode == "fixed" else 0.0),
            "v7_homeostatic_control_enabled": hidden_states.new_tensor(1.0 if self.homeostatic_control else 0.0),
            "v7_homeostatic_dhi": homeostasis_dhi,
            "v7_homeostatic_balance_pressure": homeostasis_balance,
            "v7_homeostatic_accel_pressure": homeostasis_accel,
            "v7_latent_rate_scale": latent_rate_avg,
            "v7_world_rate_scale": world_rate_avg,
            "v7_self_rate_scale": self_rate_avg,
            "v7_hidden_read_rate_scale": hidden_rate_avg,
            "v7_hyperspherical_state_enabled": hidden_states.new_tensor(
                1.0 if self.hyperspherical_state else 0.0
            ),
            "v7_causal_summary_enabled": hidden_states.new_tensor(1.0 if self.causal_summary else 0.0),
            "v7_causal_summary_decay": hidden_states.new_tensor(float(self.causal_summary_decay)),
            "v7_adaptive_tau_enabled": hidden_states.new_tensor(1.0 if self.adaptive_tau else 0.0),
            "v7_latent_tau": latent_tau_avg,
            "v7_world_tau": world_tau_avg,
            "v7_self_tau": self_tau_avg,
            "v7_controller_tau": controller_tau_avg,
            "v7_effective_latent_write_scale": hidden_states.new_tensor(float(self.latent_write_scale)),
            "v7_effective_world_write_scale": hidden_states.new_tensor(float(self.world_state_write_scale)),
            "v7_effective_self_write_scale": hidden_states.new_tensor(float(self.self_state_write_scale)),
            "v7_effective_controller_write_scale": hidden_states.new_tensor(float(self.controller_write_scale)),
        }
        metrics.update(compat_metrics)
        return hidden_states, world_final, self_final, current_latent, controller_final, metrics
