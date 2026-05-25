import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm
from .state_ops import state_softmax, state_softmax_matmul


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
        world_gate: bool = True,
        world_gate_min: float = 0.10,
        world_gate_scale: float = 1.0,
        latent_thought_steps: int = 0,
        latent_thought_write_mode: str = "state_only",
        latent_thought_hidden_scale: float = 0.0,
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
        self.world_gate = world_gate
        self.world_gate_min = min(max(world_gate_min, 0.0), 1.0)
        self.world_gate_scale = max(world_gate_scale, 1e-6)
        if latent_thought_steps < 0:
            raise ValueError("latent_thought_steps must be non-negative")
        if latent_thought_write_mode not in {"state_only", "final_hidden"}:
            raise ValueError("latent_thought_write_mode must be state_only or final_hidden")
        self.latent_thought_steps = latent_thought_steps
        self.latent_thought_write_mode = latent_thought_write_mode
        self.latent_thought_hidden_scale = max(latent_thought_hidden_scale, 0.0)

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

    def _hidden_write_gate(self, world_summary: torch.Tensor) -> torch.Tensor:
        if not self.world_gate:
            return torch.ones(world_summary.size(0), 1, device=world_summary.device, dtype=world_summary.dtype)
        norm = world_summary.float().norm(dim=-1, keepdim=True) / (world_summary.size(-1) ** 0.5)
        gate = torch.tanh(norm * self.world_gate_scale).to(dtype=world_summary.dtype)
        if self.world_gate_min > 0:
            floor = torch.full_like(gate, self.world_gate_min)
            gate = torch.maximum(gate, floor)
        return gate.clamp(0.0, 1.0)

    def _world_residual(self, hidden_summary: torch.Tensor, world_summary: torch.Tensor) -> torch.Tensor:
        # Protocol rule: self-state should explain what world-state did not
        # already explain.  The summaries already share d_model and are RMS
        # normalized before this point, so identity projection is the safest
        # first implementation.
        return hidden_summary - world_summary.to(dtype=hidden_summary.dtype)

    def _world_summary_from_state(
        self,
        world_state: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        dim: int,
    ) -> torch.Tensor:
        if world_state is None:
            return torch.zeros(batch_size, dim, device=device, dtype=dtype)
        state = world_state[:, -1, :, :] if world_state.ndim == 4 else world_state
        return self.world_norm(state).mean(dim=1)

    def _apply_latent_thought(
        self,
        hidden_states: torch.Tensor,
        *,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor,
        causal_safe: bool,
        steps: int | None = None,
        metric_prefix: str = "latent_thought",
        allow_hidden_write: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        thought_steps = self.latent_thought_steps if steps is None else steps
        if thought_steps <= 0:
            zero = hidden_states.new_tensor(0.0)
            return hidden_states, self_state, {
                f"{metric_prefix}_delta": zero,
                f"{metric_prefix}_velocity": zero,
                f"{metric_prefix}_write_norm": zero,
                f"{metric_prefix}_steps": zero,
            }

        batch_size, _, dim = hidden_states.shape
        world_summary = self._world_summary_from_state(
            world_state,
            batch_size,
            hidden_states.device,
            hidden_states.dtype,
            dim,
        )
        current = self_state[:, -1, :, :] if self_state.ndim == 4 else self_state
        identity = self.slot_identity.to(device=hidden_states.device, dtype=hidden_states.dtype)
        identity = identity.unsqueeze(0).expand_as(current)
        total_delta = hidden_states.new_tensor(0.0)
        reflection = torch.zeros_like(world_summary)

        for _ in range(thought_steps):
            pooled_self = self.self_norm(current).mean(dim=1)
            internal_residual = self._world_residual(pooled_self, world_summary)
            reflection_input = torch.cat([internal_residual, pooled_self, world_summary, pooled_self], dim=-1)
            reflection = torch.tanh(self.reflect(reflection_input))
            reflection_slots = reflection.unsqueeze(1).expand_as(current)
            update_input = torch.cat([current, reflection_slots + identity, identity], dim=-1)
            candidate = torch.tanh(self.update(update_input))
            gate = torch.sigmoid(self.update_gate(update_input)) * self.write_scale
            next_state = current + gate * (candidate - current)
            total_delta = total_delta + (next_state - current).float().pow(2).mean()
            current = next_state

        write_norm = hidden_states.new_tensor(0.0)
        if allow_hidden_write is None:
            allow_hidden_write = not causal_safe
        if self.latent_thought_write_mode == "final_hidden" and allow_hidden_write and self.latent_thought_hidden_scale > 0:
            hidden_write_gate = self._hidden_write_gate(world_summary)
            hidden_write = self.hidden_modulation(reflection) * (self.latent_thought_hidden_scale * hidden_write_gate)
            hidden_states = hidden_states + hidden_write.unsqueeze(1)
            write_norm = hidden_write.detach().float().norm(dim=-1).mean()

        steps_t = hidden_states.new_tensor(float(thought_steps))
        metrics = {
            f"{metric_prefix}_delta": total_delta / steps_t.clamp_min(1.0),
            f"{metric_prefix}_velocity": (total_delta / steps_t.clamp_min(1.0)).float().sqrt(),
            f"{metric_prefix}_write_norm": write_norm,
            f"{metric_prefix}_steps": steps_t,
        }
        return hidden_states, current, metrics

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
        causal_safe: bool = True,
        block_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden_states.size(0)
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
        if self_state is None:
            self_state = self.initial_state(batch_size, hidden_states.device, hidden_states.dtype)
        if causal_safe:
            return self._forward_causal(
                hidden_states,
                attention_mask=attention_mask,
                world_state=world_state,
                self_state=self_state,
                block_size=block_size or hidden_states.size(1),
            )

        normed_hidden = self.hidden_norm(hidden_states)
        boundary_logits = self.boundary(normed_hidden.float()) / self.boundary_temperature
        boundary_probs = state_softmax(boundary_logits, dim=-1).to(dtype=hidden_states.dtype)

        hidden_summary = self._masked_mean(normed_hidden, attention_mask)
        if world_state is None:
            world_summary = torch.zeros_like(hidden_summary)
        else:
            world_summary = self.world_norm(world_state).mean(dim=1)
        residual_hidden = normed_hidden - world_summary.unsqueeze(1)
        hidden_residual_summary = self._masked_mean(residual_hidden, attention_mask)
        self_summary = self._masked_mean(residual_hidden, attention_mask, boundary_probs[..., 0:1])

        current = self_state
        reflection = torch.zeros_like(hidden_summary)
        recursion_delta = hidden_summary.new_tensor(0.0)
        identity = self.slot_identity.to(device=hidden_states.device, dtype=hidden_states.dtype)
        identity = identity.unsqueeze(0).expand_as(current)
        slot_queries = F.normalize((current + identity).float(), dim=-1).to(dtype=hidden_states.dtype)
        slot_scores = torch.einsum("bsd,btd->bst", slot_queries, residual_hidden) * self.context_score_scale
        slot_context, _slot_weights = state_softmax_matmul(
            slot_scores,
            residual_hidden,
            mask=~attention_mask.unsqueeze(1),
            zero_invalid=True,
            out_dtype=hidden_states.dtype,
        )

        for _ in range(self.recursion_depth):
            pooled_self = self.self_norm(current).mean(dim=1)
            reflection_input = torch.cat([hidden_residual_summary, self_summary, world_summary, pooled_self], dim=-1)
            reflection = torch.tanh(self.reflect(reflection_input))
            reflection_slots = reflection.unsqueeze(1).expand_as(current)
            update_input = torch.cat([current, reflection_slots + slot_context, identity], dim=-1)
            candidate = torch.tanh(self.update(update_input))
            gate = torch.sigmoid(self.update_gate(update_input)) * self.write_scale
            next_state = current + gate * (candidate - current)
            recursion_delta = recursion_delta + (next_state - current).float().pow(2).mean()
            current = next_state

        hidden_write_gate = self._hidden_write_gate(world_summary)
        hidden_write = self.hidden_modulation(reflection) * (self.hidden_scale * hidden_write_gate)
        if not causal_safe:
            hidden_states = hidden_states + hidden_write.unsqueeze(1)

        previous_summary = self.self_norm(self_state).mean(dim=1)
        pred_summary = self.transition(previous_summary)
        pred_target = self_summary.detach() if self.pred_detach_target else self_summary
        self_pred_loss = F.smooth_l1_loss(pred_summary.float(), pred_target.float())

        normalized = F.normalize(current.detach().float(), dim=-1)
        cosine = torch.bmm(normalized, normalized.transpose(1, 2))
        context_normalized = F.normalize(slot_context.detach().float(), dim=-1)
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

        probs = boundary_probs.detach().float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        mask_f = attention_mask.float()
        boundary_entropy = (entropy * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        boundary_means = self._masked_mean(boundary_probs, attention_mask)

        metrics = {
            "self_pred": self_pred_loss,
            "slot_diversity": slot_diversity,
            "slot_cosine": slot_cosine,
            "slot_context_cosine": slot_context_cosine,
            "state_delta": (recursion_delta / self.recursion_depth).detach(),
            "state_norm": current.detach().float().norm(dim=-1).mean(),
            "reflection_norm": reflection.detach().float().norm(dim=-1).mean(),
            "world_explained_norm": world_summary.detach().float().norm(dim=-1).mean(),
            "hidden_residual_norm": hidden_residual_summary.detach().float().norm(dim=-1).mean(),
            "world_residual_ratio": hidden_residual_summary.detach().float().norm(dim=-1).mean()
            / (hidden_summary.detach().float().norm(dim=-1).mean() + hidden_summary.new_tensor(1e-6)),
            "hidden_write_gate": hidden_write_gate.detach().float().mean(),
            "hidden_write_norm": hidden_write.detach().float().norm(dim=-1).mean(),
            "hidden_write_scale": hidden_states.new_tensor(self.hidden_scale),
            "boundary_entropy": boundary_entropy,
            "boundary_self": boundary_means[..., 0].mean(),
            "boundary_world": boundary_means[..., 1].mean(),
            "boundary_other": boundary_means[..., 2].mean(),
            "boundary_unknown": boundary_means[..., 3].mean(),
        }
        hidden_states, current, thought_metrics = self._apply_latent_thought(
            hidden_states,
            world_state=world_state,
            self_state=current,
            causal_safe=causal_safe,
        )
        metrics.update(thought_metrics)
        metrics["state_norm"] = current.detach().float().norm(dim=-1).mean()
        return hidden_states, current, metrics

    def _forward_causal(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        block_size = max(1, int(block_size))
        incoming_self_trace = self_state if self_state.ndim == 4 else None
        current = (
            self.initial_state(batch_size, hidden_states.device, hidden_states.dtype)
            if incoming_self_trace is not None
            else self_state
        )
        identity = self.slot_identity.to(device=hidden_states.device, dtype=hidden_states.dtype)
        identity = identity.unsqueeze(0).expand_as(current)
        zero_summary = hidden_states.new_zeros(batch_size, hidden_states.size(-1))
        incoming_world_trace = world_state if world_state is not None and world_state.ndim == 4 else None

        block_count = (seq_len + block_size - 1) // block_size

        # Precompute static, compile-friendly history summaries outside the loop
        precomputed_self_summaries = zero_summary.unsqueeze(1).expand(-1, block_count, -1).clone()
        if incoming_self_trace is not None and block_count > 1:
            self_block_summaries = self.self_norm(incoming_self_trace).mean(dim=2)
            self_cumsum = torch.cumsum(self_block_summaries, dim=1)
            steps = torch.arange(1, block_count, device=hidden_states.device, dtype=hidden_states.dtype).view(1, -1, 1)
            precomputed_self_summaries[:, 1:, :] = self_cumsum[:, : block_count - 1, :] / steps

        precomputed_world_summaries = zero_summary.unsqueeze(1).expand(-1, block_count, -1).clone()
        if incoming_world_trace is not None and block_count > 1:
            world_block_summaries = self.world_norm(incoming_world_trace).mean(dim=2)
            world_cumsum = torch.cumsum(world_block_summaries, dim=1)
            steps = torch.arange(1, block_count, device=hidden_states.device, dtype=hidden_states.dtype).view(1, -1, 1)
            precomputed_world_summaries[:, 1:, :] = world_cumsum[:, : block_count - 1, :] / steps

        # Compute per-token norm + boundary once for the full sequence.
        normed_hidden = self.hidden_norm(hidden_states)
        boundary_logits = self.boundary(normed_hidden.float()) / self.boundary_temperature
        boundary_probs = state_softmax(boundary_logits, dim=-1).to(dtype=hidden_states.dtype)

        # Pre-allocate output buffer so each block writes in-place.
        output = torch.empty_like(hidden_states)
        state_trace: list[torch.Tensor] = []

        metric_values: dict[str, list[torch.Tensor]] = {
            "self_pred": [],
            "slot_diversity": [],
            "slot_cosine": [],
            "slot_context_cosine": [],
            "state_delta": [],
            "state_velocity": [],
            "state_acceleration": [],
            "reflection_norm": [],
            "world_explained_norm": [],
            "hidden_residual_norm": [],
            "world_residual_ratio": [],
            "hidden_write_gate": [],
            "hidden_write_norm": [],
            "hidden_write_scale": [],
            "boundary_entropy": [],
            "boundary_self": [],
            "boundary_world": [],
            "boundary_other": [],
            "boundary_unknown": [],
            "history_self_norm": [],
            "history_world_norm": [],
            "latent_thought_delta": [],
            "latent_thought_velocity": [],
            "latent_thought_write_norm": [],
            "latent_thought_steps": [],
        }
        previous_delta = hidden_states.new_tensor(0.0)

        for block_idx, start in enumerate(range(0, seq_len, block_size)):
            end = min(seq_len, start + block_size)
            block = hidden_states[:, start:end, :]
            block_mask = attention_mask[:, start:end]
            normed_block = normed_hidden[:, start:end, :]
            block_boundary = boundary_probs[:, start:end, :]
            state_before_block = current
            history_self_summary = precomputed_self_summaries[:, block_idx, :]
            world_summary = precomputed_world_summaries[:, block_idx, :]
            history_residual_summary = self._world_residual(history_self_summary, world_summary)

            if self.latent_thought_steps > 0 and self.latent_thought_write_mode == "final_hidden":
                block, current, thought_metrics = self._apply_latent_thought(
                    block,
                    world_state=world_summary.unsqueeze(1),
                    self_state=current,
                    causal_safe=True,
                    allow_hidden_write=True,
                )
                for key in (
                    "latent_thought_delta",
                    "latent_thought_velocity",
                    "latent_thought_write_norm",
                    "latent_thought_steps",
                ):
                    metric_values[key].append(thought_metrics[key])

            pooled_self = self.self_norm(current).mean(dim=1)
            # The modulation applied to the current block is computed before
            # seeing the current block summary. This keeps the self-state path
            # strictly prefix-causal.
            reflection_input = torch.cat([pooled_self, history_residual_summary, world_summary, pooled_self], dim=-1)
            reflection = torch.tanh(self.reflect(reflection_input))
            hidden_write_gate = self._hidden_write_gate(world_summary)
            hidden_write = self.hidden_modulation(reflection) * (self.hidden_scale * hidden_write_gate)
            output[:, start:end, :] = block + hidden_write.unsqueeze(1)

            residual_block = normed_block - world_summary.unsqueeze(1)
            hidden_summary = self._masked_mean(normed_block, block_mask)
            hidden_residual_summary = self._masked_mean(residual_block, block_mask)
            self_summary = self._masked_mean(residual_block, block_mask, block_boundary[..., 0:1])
            slot_queries = F.normalize((current + identity).float(), dim=-1).to(dtype=hidden_states.dtype)
            slot_scores = torch.einsum("bsd,btd->bst", slot_queries, residual_block) * self.context_score_scale
            slot_context, _slot_weights = state_softmax_matmul(
                slot_scores,
                residual_block,
                mask=~block_mask.unsqueeze(1),
                zero_invalid=True,
                out_dtype=hidden_states.dtype,
            )

            block_delta = hidden_states.new_tensor(0.0)
            for _ in range(self.recursion_depth):
                pooled_current = self.self_norm(current).mean(dim=1)
                update_reflection_input = torch.cat(
                    [hidden_residual_summary, self_summary, world_summary, pooled_current],
                    dim=-1,
                )
                update_reflection = torch.tanh(self.reflect(update_reflection_input))
                reflection_slots = update_reflection.unsqueeze(1).expand_as(current)
                update_input = torch.cat([current, reflection_slots + slot_context, identity], dim=-1)
                candidate = torch.tanh(self.update(update_input))
                gate = torch.sigmoid(self.update_gate(update_input)) * self.write_scale
                next_state = current + gate * (candidate - current)
                with torch.no_grad():
                    block_delta = block_delta + (next_state - current).float().pow(2).mean()
                current = next_state

            previous_summary = self.self_norm(state_before_block).mean(dim=1)
            pred_summary = self.transition(previous_summary)
            pred_target = self_summary.detach() if self.pred_detach_target else self_summary
            self_pred_loss = F.smooth_l1_loss(pred_summary.float(), pred_target.float())

            normalized = F.normalize(current.detach().float(), dim=-1)
            cosine = torch.bmm(normalized, normalized.transpose(1, 2))
            context_normalized = F.normalize(slot_context.detach().float(), dim=-1)
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

            with torch.no_grad():
                probs = block_boundary.float().clamp_min(1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1)
                mask_f = block_mask.float()
                boundary_entropy = (entropy * mask_f).sum() / mask_f.sum().clamp_min(1.0)
                boundary_means = self._masked_mean(block_boundary.detach(), block_mask)
                state_delta = block_delta / self.recursion_depth
                state_velocity = state_delta.float().sqrt()
                state_acceleration = (state_velocity - previous_delta).abs()
                previous_delta = state_velocity
            state_trace.append(current)

            metric_values["self_pred"].append(self_pred_loss)
            metric_values["slot_diversity"].append(slot_diversity)
            metric_values["slot_cosine"].append(slot_cosine)
            metric_values["slot_context_cosine"].append(slot_context_cosine)
            metric_values["state_delta"].append(state_delta.detach())
            metric_values["state_velocity"].append(state_velocity.detach())
            metric_values["state_acceleration"].append(state_acceleration.detach())
            metric_values["reflection_norm"].append(reflection.detach().float().norm(dim=-1).mean())
            world_explained_norm = world_summary.detach().float().norm(dim=-1).mean()
            hidden_residual_norm = hidden_residual_summary.detach().float().norm(dim=-1).mean()
            hidden_summary_norm = hidden_summary.detach().float().norm(dim=-1).mean()
            metric_values["world_explained_norm"].append(world_explained_norm)
            metric_values["hidden_residual_norm"].append(hidden_residual_norm)
            metric_values["world_residual_ratio"].append(
                hidden_residual_norm / (hidden_summary_norm + hidden_states.new_tensor(1e-6))
            )
            metric_values["hidden_write_gate"].append(hidden_write_gate.detach().float().mean())
            metric_values["hidden_write_norm"].append(hidden_write.detach().float().norm(dim=-1).mean())
            metric_values["hidden_write_scale"].append(hidden_states.new_tensor(self.hidden_scale))
            metric_values["boundary_entropy"].append(boundary_entropy)
            metric_values["boundary_self"].append(boundary_means[..., 0].mean())
            metric_values["boundary_world"].append(boundary_means[..., 1].mean())
            metric_values["boundary_other"].append(boundary_means[..., 2].mean())
            metric_values["boundary_unknown"].append(boundary_means[..., 3].mean())
            metric_values["history_self_norm"].append(history_self_summary.detach().float().norm(dim=-1).mean())
            metric_values["history_world_norm"].append(world_summary.detach().float().norm(dim=-1).mean())

        metrics = {key: torch.stack(values).mean() for key, values in metric_values.items() if values}
        if self.latent_thought_steps > 0 and self.latent_thought_write_mode == "final_hidden":
            for key in (
                "latent_thought_delta",
                "latent_thought_velocity",
                "latent_thought_write_norm",
                "latent_thought_steps",
            ):
                if metric_values[key]:
                    metrics[key] = torch.stack(metric_values[key]).mean()
                else:
                    metrics[key] = hidden_states.new_tensor(0.0)
        else:
            output, current, thought_metrics = self._apply_latent_thought(
                output,
                world_state=world_state,
                self_state=current,
                causal_safe=True,
            )
            metrics.update(thought_metrics)
        metrics["state_norm"] = current.detach().float().norm(dim=-1).mean()
        if state_trace:
            state_trace[-1] = current
        traced_state = torch.stack(state_trace, dim=1) if state_trace else current.unsqueeze(1)
        return output, traced_state, metrics
