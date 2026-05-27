from dataclasses import dataclass

import torch

from .typed_dynamics import TypedLatentDynamics


@dataclass(frozen=True)
class TypedThoughtStepOutput:
    """Single committed thought-dynamics step output.

    V8 places hidden-state thought inside decoder blocks, not only after the
    full block stack. This dataclass carries the one-step result needed by a
    future ThoughtCommitBlock while keeping StatePacket/KV-cache concerns
    separate.
    """

    hidden_states: torch.Tensor
    world_state: torch.Tensor | None
    self_state: torch.Tensor | None
    controller_state: torch.Tensor | None
    latent_field: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _prefixed_read_metrics(
    read_metrics: dict[str, torch.Tensor],
    *,
    prefix: str,
    zero: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    hidden_delta_by_sample = read_metrics.get(
        "v7_hidden_delta_by_sample",
        torch.zeros(batch_size, device=zero.device, dtype=zero.dtype),
    )
    return {
        f"{prefix}_hidden_delta": read_metrics.get("v7_hidden_delta", zero),
        f"{prefix}_hidden_delta_by_sample": hidden_delta_by_sample,
        f"{prefix}_latent_hidden_write_norm": read_metrics.get("v7_latent_hidden_write_norm", zero),
        f"{prefix}_hidden_write_ratio": read_metrics.get("v7_hidden_write_ratio", zero),
        f"{prefix}_hidden_write_gate": read_metrics.get("v7_hidden_write_gate", zero),
        f"{prefix}_latent_read_entropy": read_metrics.get("v7_latent_read_entropy", zero),
        f"{prefix}_latent_read_max": read_metrics.get("v7_latent_read_max", zero),
    }


def typed_thought_step(
    dynamics: TypedLatentDynamics,
    hidden_states: torch.Tensor,
    *,
    world_state: torch.Tensor | None,
    self_state: torch.Tensor | None,
    controller_state: torch.Tensor | None,
    latent_field: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    readable_latent: torch.Tensor | None = None,
    apply_hidden_read: bool = True,
    latent_rate_scale: torch.Tensor | float = 1.0,
    world_rate_scale: torch.Tensor | float = 1.0,
    self_rate_scale: torch.Tensor | float = 1.0,
    hidden_rate_scale: torch.Tensor | float = 1.0,
    controller_rate_scale: torch.Tensor | float = 1.0,
    metric_prefix: str = "v8_step",
) -> TypedThoughtStepOutput:
    """Run one typed hidden/latent/state dynamics step.

    This is the V8 block-level primitive extracted from the V7 post-stack
    dynamics loop. It deliberately reuses ``TypedLatentDynamics`` internals so
    that V8 does not fork a second thought mechanism.

    ``readable_latent`` controls whether current hidden states may read a prior
    latent field. Passing ``None`` keeps the causal V7 behavior where same-call
    newly written latent state is not reused to patch current hidden states.
    """

    zero = hidden_states.new_tensor(0.0)
    if latent_field is None:
        latent_field = dynamics.initial_state(hidden_states.size(0), hidden_states.device, hidden_states.dtype)

    current_world = dynamics._final_slots(world_state)
    current_self = dynamics._final_slots(self_state)
    current_controller = dynamics._final_slots(controller_state)

    if apply_hidden_read and readable_latent is not None:
        updated_hidden, read_metrics = dynamics._hidden_read(
            hidden_states,
            readable_latent,
            attention_mask,
            rate_scale=hidden_rate_scale,
        )
    else:
        updated_hidden = hidden_states
        read_metrics = dynamics._zero_hidden_read_metrics(hidden_states)

    hidden_summary = dynamics._sequence_summary(updated_hidden, attention_mask)
    world_summary = dynamics._summary(current_world, dynamics.world_norm, hidden_summary)
    self_summary = dynamics._summary(current_self, dynamics.self_norm, hidden_summary)

    next_latent, latent_delta, latent_delta_by_sample, latent_tau = dynamics._update_latent(
        hidden_summary=hidden_summary,
        world_summary=world_summary,
        self_summary=self_summary,
        latent_field=latent_field,
        rate_scale=latent_rate_scale,
    )
    latent_summary = dynamics.latent_norm(next_latent).mean(dim=1).to(dtype=hidden_summary.dtype)

    next_world, world_delta, world_delta_by_sample, world_gate, world_tau = dynamics._update_typed_state(
        current_world,
        hidden_summary=hidden_summary,
        other_summary=self_summary,
        latent_summary=latent_summary,
        update=dynamics.world_update,
        gate=dynamics.world_gate,
        tau_layer=dynamics.world_tau,
        write_scale=dynamics.world_state_write_scale,
        rate_scale=world_rate_scale,
    )
    next_self, self_delta, self_delta_by_sample, self_gate, self_tau = dynamics._update_typed_state(
        current_self,
        hidden_summary=hidden_summary,
        other_summary=world_summary,
        latent_summary=latent_summary,
        update=dynamics.self_update,
        gate=dynamics.self_gate,
        tau_layer=dynamics.self_tau,
        write_scale=dynamics.self_state_write_scale,
        rate_scale=self_rate_scale,
    )
    next_controller, controller_delta, controller_delta_by_sample, controller_gate, controller_tau = (
        dynamics._update_controller_state(
            current_controller,
            hidden_summary=hidden_summary,
            world_summary=world_summary,
            self_summary=self_summary,
            latent_summary=latent_summary,
            rate_scale=controller_rate_scale,
        )
    )

    metrics = _prefixed_read_metrics(
        read_metrics,
        prefix=metric_prefix,
        zero=zero,
        batch_size=hidden_states.size(0),
    )
    metrics.update(
        {
            f"{metric_prefix}_thought_steps": hidden_states.new_tensor(1.0),
            f"{metric_prefix}_latent_delta": latent_delta.type_as(hidden_states),
            f"{metric_prefix}_latent_delta_by_sample": latent_delta_by_sample.type_as(hidden_states),
            f"{metric_prefix}_world_delta": world_delta.type_as(hidden_states),
            f"{metric_prefix}_world_delta_by_sample": world_delta_by_sample.type_as(hidden_states),
            f"{metric_prefix}_self_delta": self_delta.type_as(hidden_states),
            f"{metric_prefix}_self_delta_by_sample": self_delta_by_sample.type_as(hidden_states),
            f"{metric_prefix}_controller_delta": controller_delta.type_as(hidden_states),
            f"{metric_prefix}_controller_delta_by_sample": controller_delta_by_sample.type_as(hidden_states),
            f"{metric_prefix}_world_write_gate": world_gate.type_as(hidden_states),
            f"{metric_prefix}_self_write_gate": self_gate.type_as(hidden_states),
            f"{metric_prefix}_controller_write_gate": controller_gate.type_as(hidden_states),
            f"{metric_prefix}_latent_tau": latent_tau.type_as(hidden_states),
            f"{metric_prefix}_world_tau": world_tau.type_as(hidden_states),
            f"{metric_prefix}_self_tau": self_tau.type_as(hidden_states),
            f"{metric_prefix}_controller_tau": controller_tau.type_as(hidden_states),
        }
    )

    return TypedThoughtStepOutput(
        hidden_states=updated_hidden,
        world_state=next_world,
        self_state=next_self,
        controller_state=next_controller,
        latent_field=next_latent,
        metrics=metrics,
    )
