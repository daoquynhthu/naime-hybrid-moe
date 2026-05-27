import torch

from naime_hybrid.modules.thought_commit import typed_thought_step
from naime_hybrid.modules.typed_dynamics import TypedLatentDynamics


def test_v8_typed_thought_step_matches_v7_single_step_without_hidden_read():
    torch.manual_seed(2060)
    dynamics = TypedLatentDynamics(
        d_model=32,
        latent_slots=4,
        controller_slots=2,
        latent_write_scale=0.03,
        hidden_write_scale=0.01,
        max_hidden_write_ratio=0.05,
        state_write_scale=0.02,
        controller_write_scale=0.015,
        hyperspherical_state=True,
        causal_summary=True,
    ).eval()
    hidden = torch.randn(2, 11, 32)
    attention_mask = torch.ones(2, 11, dtype=torch.bool)
    world_state = torch.randn(2, 3, 32)
    self_state = torch.randn(2, 5, 32)
    controller_state = torch.randn(2, 2, 32)
    latent_field = torch.randn(2, 4, 32)

    with torch.no_grad():
        v7_hidden, v7_world, v7_self, v7_latent, v7_controller, v7_metrics = dynamics(
            hidden,
            world_state=world_state,
            self_state=self_state,
            controller_state=controller_state,
            latent_field=latent_field,
            attention_mask=attention_mask,
            steps=1,
            past_latent_field=False,
            apply_state_compatibility=False,
        )
        step = typed_thought_step(
            dynamics,
            hidden,
            world_state=world_state,
            self_state=self_state,
            controller_state=controller_state,
            latent_field=latent_field,
            attention_mask=attention_mask,
            readable_latent=None,
            apply_hidden_read=True,
            metric_prefix="v8_step",
        )

    assert torch.allclose(step.hidden_states, v7_hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(step.latent_field, v7_latent, atol=1e-6, rtol=1e-6)
    assert step.world_state is not None
    assert v7_world is not None
    assert torch.allclose(step.world_state, v7_world, atol=1e-6, rtol=1e-6)
    assert step.self_state is not None
    assert v7_self is not None
    assert torch.allclose(step.self_state, v7_self, atol=1e-6, rtol=1e-6)
    assert step.controller_state is not None
    assert v7_controller is not None
    assert torch.allclose(step.controller_state, v7_controller, atol=1e-6, rtol=1e-6)

    assert step.metrics["v8_step_thought_steps"].item() == 1.0
    assert torch.allclose(step.metrics["v8_step_latent_delta"], v7_metrics["v7_latent_delta"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(step.metrics["v8_step_world_delta"], v7_metrics["v7_world_delta"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(step.metrics["v8_step_self_delta"], v7_metrics["v7_self_delta"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        step.metrics["v8_step_controller_delta"],
        v7_metrics["v7_controller_delta"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_v8_typed_thought_step_can_read_prior_latent_into_hidden():
    torch.manual_seed(2061)
    dynamics = TypedLatentDynamics(
        d_model=32,
        latent_slots=4,
        controller_slots=1,
        latent_write_scale=0.03,
        hidden_write_scale=0.02,
        max_hidden_write_ratio=0.08,
        state_write_scale=0.02,
    ).eval()
    hidden = torch.randn(2, 7, 32)
    prior_latent = torch.randn(2, 4, 32)

    with torch.no_grad():
        step = typed_thought_step(
            dynamics,
            hidden,
            world_state=None,
            self_state=None,
            controller_state=None,
            latent_field=prior_latent,
            attention_mask=None,
            readable_latent=prior_latent,
            apply_hidden_read=True,
            metric_prefix="v8_step",
        )

    assert step.hidden_states.shape == hidden.shape
    assert not torch.allclose(step.hidden_states, hidden)
    assert step.metrics["v8_step_hidden_delta"].item() > 0.0
    assert step.metrics["v8_step_hidden_write_ratio"].item() <= dynamics.max_hidden_write_ratio + 1e-4
    assert step.latent_field.shape == prior_latent.shape
