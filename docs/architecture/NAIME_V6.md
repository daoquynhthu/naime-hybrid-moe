# NAIME V6: Recursive Self-State MoE

Status: **active architecture family, protocol-aligned work in progress**

V6 extends the V5 world-state model with recursive self-state slots. Current V6
work is governed by `docs/architecture/STATE_PROTOCOL.md`; clean claims must use
causal-integrity-v2 checkpoints and clean runs only.

Older V6 continuation runs that reported ultra-low validation perplexity are
legacy/contaminated evidence. They must not be cited as proof of model quality or
architecture superiority.

## Core Mechanism

- `naime_v6_recursive_self_moe` keeps V5 world-state slots and adds recursive
  self-state slots.
- V5 world state contributes an explicitly measured world component to the MoE
  router bus.
- V6 self state receives world-conditioned residual summaries. In protocol
  terms, self state should explain what world state did not already explain,
  rather than absorbing the full hidden stream as unconstrained self evidence.
- Self-state hidden modulation is gated by world signal strength and logs its
  effective gate, norm, and scale.
- Boundary metrics split state signal into `self`, `world`, `other`, and
  `unknown` components.

## Protocol-Aligned Metrics

Router bus:

```text
v5_router_semantic_norm
v5_router_world_norm
v5_router_world_ratio
v5_router_world_cosine
v5_router_world_gate
v5_router_memory_norm
v5_router_memory_ratio
v5_router_effective_norm
```

Hidden writes:

```text
v5_semantic_hidden_write_norm
v5_semantic_hidden_write_scale
v5_memory_hidden_write_norm
v5_memory_hidden_write_scale
v6_hidden_write_gate
v6_hidden_write_norm
v6_hidden_write_scale
```

World-residual self path:

```text
v6_world_explained_norm
v6_hidden_residual_norm
v6_world_residual_ratio
```

Recursive self-state:

```text
v6_self_pred
v6_slot_cosine
v6_slot_context_cosine
v6_state_delta
v6_state_norm
v6_reflection_norm
v6_boundary_self
v6_boundary_world
v6_boundary_other
v6_boundary_unknown
```

## Current Interpretation

V6 is promising, but the current project standard is strict:

- Do not resume from old checkpoints unless intentionally running a legacy
  forensic baseline.
- Do not use pre-protocol ultra-low perplexity runs as validation evidence.
- Prefer training templates over manual command construction.
- Treat LM loss, router-bus metrics, hidden-write metrics, and boundary metrics
  as a coupled health surface.

## Training Policy

Use template launches:

```powershell
.\scripts\train_template.ps1 -List
.\scripts\train_template.ps1 -Template v6_local_smoke
.\scripts\train_template.ps1 -Template v6_local_64m_probe
```

Remote large segments should use the remote templates as command sources and the
hidden/background remote workflow documented in `docs/REMOTE_4090_OPERATIONS.md`.

## Remaining Work

- Verify that world-residual self input improves boundary allocation under clean
  causal training.
- Promote selected warning metrics into control mechanisms only after the metric
  behavior is stable.
- Keep memory contribution explicit: in causal V6, memory hidden/router influence
  is currently dormant unless the architecture is deliberately changed.
- Continue improving generation quality evaluation; validation loss alone is not
  sufficient evidence of model usefulness.
