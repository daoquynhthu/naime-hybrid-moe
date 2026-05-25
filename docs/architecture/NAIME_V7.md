# NAIME V7: Typed Internal Dynamics

Status: **V7.0-V7.6 engineering path landed locally, mechanism unverified**

V7 is the proposed successor to the V6.5 stateful architecture. It is not a
claim of validated superiority. It is a design contract for the next
architecture step: moving from a Transformer with state modules to a model whose
prediction is mediated by a typed internal dynamics process.

V7 is governed by `docs/architecture/STATE_PROTOCOL.md`. Any implementation
must preserve causal integrity, named router-bus components, bounded hidden
writes, and clean checkpoint lineage.

## 1. Motivation

V6.5 proves that world slots, recursive self slots, latent field coupling, and
latent thought can be made stable enough to train. Early V7 made hidden and
latent field co-evolve, but still left `world_state` and `self_state` mostly as
conditioning context. The final V7 line closes that gap: all four streams
participate in the same bounded typed dynamics loop.

However, V7 remains unvalidated as a mechanism claim:

- `latent_thought_gain` is positive but very small in current clean runs.
- `state_carry_gain` is near zero in current evaluation.
- Internal dynamics must still prove that typed state evolution improves
  prediction, continuity, and stability rather than merely adding capacity.
- The architecture has several state carriers, and further module stacking would
  make attribution worse.

V7 should therefore not add another independent "thinking module". It should
turn the existing state system into a typed, continuous, causal dynamics core.

## 2. North Star

V7 should implement this computation pattern:

```text
token hidden
  + typed prior internal state
      -> repeated typed dynamics steps
      -> final hidden/state readout
      -> next-token logits
      -> updated typed internal state packet
```

The model should not merely attach memory to a Transformer. It should treat
internal state as the evolving endpoint of prior computation and the starting
condition for future computation.

The core claim to test is:

```text
Correct typed internal state + internal dynamics
should improve prediction, stability, and continuity
relative to fresh, erased, swapped, or zero-thought states.
```

## 3. Non-Goals

V7 must avoid these failure modes:

- Do not merge `world_state`, `self_state`, and `latent_field` into one
  undifferentiated vector bank.
- Do not introduce a text CoT objective or require supervised reasoning traces.
- Do not store KV cache as "thought"; KV cache is an inference cache, not the
  persistent internal state target.
- Do not let `self_state` absorb all unexplained text heterogeneity.
- Do not let `latent_field` become an unbounded hidden-state write shortcut.
- Do not treat lower validation loss alone as proof that the intended mechanism
  works.

## 4. State Types

V7 keeps state types separate. They share a dynamics scheduler, not a common
identity.

### 4.1 `world_state`

`world_state` represents outside-facing text-world structure:

- topic and discourse continuity;
- entity-like and event-like latent structure;
- local context that belongs to the described world;
- long-range semantic relations available from the prefix.

World state may influence routing and hidden states only through named,
bounded world channels.

### 4.2 `self_state`

`self_state` represents the model's internal processing boundary:

- uncertainty and conflict;
- what the model has already absorbed or reflected on;
- self/world/other/unknown boundary allocation;
- internal prediction of the next processing state.

Self state should primarily read world-conditioned residual evidence:

```text
self_input = hidden_residual_after_world + prior_self + prior_world_summary
```

It should not read full hidden summaries without an explicit anti-absorption
control.

### 4.3 `latent_field`

`latent_field` represents the continuous internal reasoning trajectory:

- direction and momentum of current internal computation;
- unfinished latent constraints;
- recurrent refinement state across thought steps;
- compact continuation state across chunks or turns when enabled.

`latent_field` is not a second world model and not a second self model. It is the
dynamics carrier that evolves between typed world/self interpretations and the
main hidden stream.

## 5. Typed State Packet

V7 should pass and return a typed packet, not raw hidden or KV cache:

```python
@dataclass
class NAIMEV7StatePacket:
    world_state: Tensor | None
    self_state: Tensor | None
    latent_field: Tensor | None
    memory: Tensor | None
    state_version: int
    protocol_version: str
    causal_integrity_version: str
    architecture_id: str
    tokenizer_hash: str | None
    created_step: int | None
    confidence: float | None
```

Rules:

- Packet tensors are detached by default when passed across chunks, batches, or
  sessions.
- Packet tensors must be batch-aligned; cross-sample leakage is forbidden.
- Packet metadata must prevent accidental use across incompatible tokenizer,
  architecture, or causal-integrity versions.
- Packet serialization must be optional. Training should not depend on writing
  a packet every step.

## 6. Typed Dynamics Core

V7 introduces a core interface:

```python
class TypedLatentDynamics(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        state: NAIMEV7StatePacket,
        *,
        attention_mask: Tensor | None,
        step_index: int,
        causal_safe: bool,
        return_aux: bool,
    ) -> tuple[Tensor, NAIMEV7StatePacket, dict[str, Tensor]]:
        ...
```

The implementation may be fused or optimized, but conceptually each dynamics
step follows a typed sequence:

```text
1. read hidden, world, self, and latent summaries causally
2. evolve latent_field from hidden + world + self typed summaries
3. update world_state from hidden + self + latent evidence
4. update self_state from hidden + world + latent evidence
5. apply bounded hidden refinement from the evolved latent field
6. emit metrics for every active typed channel
```

The order matters. `latent_field` should evolve from typed evidence, while
`world_state` and `self_state` must also be updated inside the loop. This is the
line between true internal dynamics and a hidden-write add-on.

## 7. Dynamics Step

The basic V7 thought step is:

```text
hidden_t, world_t, self_t, latent_t
    -> dynamics_step
hidden_t+1, world_t+1, self_t+1, latent_t+1
```

For `thought_steps = k`:

```text
for t in range(k):
    hidden, state = dynamics_step(hidden, state, step_index=t)

logits = lm_head(final_norm(hidden))
```

This differs from V6.5:

```text
V6.5: latent thought produces a small final hidden write.
V7: hidden and typed state co-evolve for one or more internal dynamics steps.
```

The hidden stream is not merely patched after thought. It participates in the
dynamics process. The packet returned by V7 should represent the endpoint of
the internal dynamics trajectory, so the next call can continue from that state
without storing hidden activations or KV cache.

## 8. Cross-State Communication Contract

All cross-state communication must be named and typed.

Allowed channels:

```text
world -> self:
  world_summary
  world_confidence
  world_explained_projection

self -> world:
  uncertainty_signal
  contradiction_signal
  unresolved_boundary_signal

world/self -> latent_field:
  typed_condition_summary
  confidence-weighted constraints

latent_field -> hidden:
  bounded_refinement
  convergence-conditioned write

latent_field -> router_bus:
  optional low-scale typed routing hint
```

Forbidden pattern:

```text
mixed_state = world + self + latent
router_semantic = semantic + mixed_state
hidden = hidden + project(mixed_state)
```

Required pattern:

```text
router_bus =
  semantic_base
  + world_gate  * world_component
  + self_gate   * self_component
  + latent_gate * latent_component
```

Each component must expose norm, ratio, cosine, gate, cap, and entropy impact.

## 9. Hidden Write Policy

V7 hidden writes must remain bounded:

```text
hidden = hidden + write_gate * write_scale * projected_typed_signal
```

Required write sources:

- `world_hidden_write`, if world writes to hidden;
- `self_hidden_write`, if self writes to hidden;
- `latent_hidden_write`, if latent field writes to hidden.

Required metrics:

```text
v7_world_hidden_write_norm
v7_self_hidden_write_norm
v7_latent_hidden_write_norm
v7_total_hidden_write_norm
v7_hidden_write_ratio
v7_hidden_write_gate
```

Hard rule:

```text
latent_hidden_write_norm must not be the only evidence that latent dynamics is
working.
```

It is an output channel, not the reasoning process itself.

## 10. Training Philosophy

V7 should remain compatible with standard causal LM pretraining. It should not
require supervised text CoT or fully continuous document streams.

The model is still a next-token predictor, but the predictor now receives typed
internal state as part of its causal prefix condition:

```text
P(next_token | visible_prefix, typed_internal_state)
```

Training should mostly use normal shuffled causal LM batches. Additional probes
should test whether the internal state is useful, not force every batch into a
long session.

## 11. Self-Supervised Mechanism Probes

V7 must include direct mechanism probes. These probes can run during validation
or low-frequency diagnostic steps.

### 11.1 Thought-depth gain

Compare zero-step and k-step dynamics:

```text
thought_gain_lm = lm_loss(k=0) - lm_loss(k=k)
```

Expected result:

```text
thought_gain_lm > 0
```

The gain should be measured per step and per compute cost:

```text
thought_gain_per_step
thought_gain_per_ms
thought_gain_per_flop_estimate
```

### 11.2 State swap penalty

Compare correct state against mismatched state:

```text
state_swap_delta = lm_loss(wrong_state) - lm_loss(correct_state)
```

Expected result:

```text
state_swap_delta > 0
```

If wrong state and correct state perform the same, the state is not being used in
a meaningful way.

### 11.3 State erase sensitivity

Compare full state against erased typed fields:

```text
world_erase_delta  = lm_loss(no_world)  - lm_loss(full_state)
self_erase_delta   = lm_loss(no_self)   - lm_loss(full_state)
latent_erase_delta = lm_loss(no_latent) - lm_loss(full_state)
```

Expected result:

```text
Each active state type should have nonzero sensitivity on tasks where it should
matter.
```

### 11.4 Dynamics convergence

Measure whether repeated thought steps stabilize:

```text
v7_latent_velocity_t
v7_latent_acceleration_t
v7_hidden_delta_t
v7_state_cosine_t_to_t_minus_1
```

Healthy dynamics should usually show bounded movement and partial convergence,
not explosive drift or immediate zero delta.

## 12. Metrics

V7 must add metrics in five groups.

### Dynamics health

```text
v7_thought_steps
v7_thought_gain_lm
v7_thought_gain_per_step
v7_dynamics_convergence
v7_latent_velocity
v7_latent_acceleration
v7_hidden_delta
v7_world_delta
v7_self_delta
v7_latent_delta
v7_world_write_gate
v7_self_write_gate
```

### State usefulness

```text
v7_state_swap_delta
v7_world_erase_delta
v7_self_erase_delta
v7_latent_erase_delta
v7_state_carry_gain_lm
```

### Type boundaries

```text
v7_boundary_self
v7_boundary_world
v7_boundary_other
v7_boundary_unknown
v7_world_to_self_residual_ratio
v7_self_absorption_score
v7_latent_type_leakage
```

### Router contributions

```text
v7_router_semantic_ratio
v7_router_world_ratio
v7_router_self_ratio
v7_router_latent_ratio
v7_router_component_entropy_impact
```

### Hidden writes

```text
v7_world_hidden_write_norm
v7_self_hidden_write_norm
v7_latent_hidden_write_norm
v7_total_hidden_write_norm
```

## 13. Acceptance Criteria

V7 is not considered successful merely because training loss decreases.

Minimum acceptance criteria for a clean small-scale run:

- LM loss improves normally under clean causal evaluation.
- No recurring bad-gradient clusters correlate with dynamics deltas.
- `thought_gain_lm` is positive beyond noise for at least one useful
  `thought_steps` setting.
- `state_swap_delta` is positive on diagnostic batches.
- `latent_erase_delta` is positive on complex-context diagnostics.
- Router components remain bounded and non-opaque.
- Hidden write norms remain bounded relative to hidden norm.
- `world_state`, `self_state`, and `latent_field` do not collapse into a single
  dominant all-purpose path.

Acceptance criteria for scale-up:

- The best fixed thought depth is known from ablation.
- Compute-adjusted gain justifies the additional dynamics cost.
- Generation quality improves or at least does not regress.
- Stateful inference can load and continue from a `NAIMEV7StatePacket`.
- Clean lineage prevents V6 or contaminated checkpoints from being mistaken for
  V7 evidence.

## 14. Implementation Plan

### V7.0: Interfaces and Packet

Deliverables:

- `NAIMEV7StatePacket`
- typed state metadata and compatibility checks
- forward API accepting and returning typed state
- diagnostics that verify no cross-sample state leakage

No new mechanism claims are allowed at this stage.

### V7.1: Typed Dynamics Core

Deliverables:

- `TypedLatentDynamics`
- one shared `dynamics_step` interface
- separate world, self, and latent update paths
- named cross-state channels
- hidden write metrics by source

The first implementation may reuse V6.5 submodules internally.

### V7.2: Hidden-State Co-Evolution

Deliverables:

- hidden participates in each dynamics step;
- logits read from final evolved hidden;
- zero-step path remains available as baseline;
- `thought_steps=0/1/2/3` can be selected without changing model weights.

### V7.3: Mechanism Probes

Deliverables:

- thought-depth gain;
- state swap penalty;
- typed erase sensitivity;
- convergence metrics;
- compact validation summary.

Current implementation status:

- validation can compute `val_v7_dynamics_gain_lm` by comparing configured V7
  dynamics depth against `v7_dynamics_steps=0`;
- validation can compute `val_v7_state_swap_delta_lm` by rolling state packets
  across batch entries before the second chunk;
- validation can compute typed erase deltas for world, self, and latent state;
- probes are opt-in CLI/template switches so normal training does not pay the
  extra double-forward cost.

### V7.4: Clean Ablation

Required matrix:

```text
V7 thought_steps=0
V7 thought_steps=1
V7 thought_steps=2
V7 thought_steps=3
V7 no latent_field
V7 no world-to-self residual
V7 wrong-state diagnostic
V7 erased-state diagnostic
```

This matrix is an engineering validation requirement, not part of the protocol.

Current implementation status:

- `scripts/run_v7_local_ablation_matrix.ps1` runs a local CPU-safe smoke matrix;
- the script writes full per-run logs to disk and prints only a compact summary;
- the local matrix is only a crash/chain test, not evidence of architectural
  superiority.

### V7.5: Dynamic Thought Depth

Only after fixed thought depth is validated:

- add a halting or depth controller;
- log controller confidence and expected cost;
- cap maximum thought steps;
- maintain deterministic eval modes.

Current implementation status:

- V7.5 uses convergence-based dynamic depth rather than an unsupervised learned
  depth head;
- `v7_dynamic_depth` enables per-sample halting inside a batch while preserving
  a shared Python loop up to `v7_max_dynamics_steps`;
- samples that meet the convergence threshold after `v7_min_dynamics_steps` keep
  their current hidden and latent field while the rest continue evolving;
- metrics include `v7_dynamic_depth_mean`, `v7_dynamic_halt_fraction`,
  `v7_dynamic_continue_score`, and `v7_dynamic_convergence_threshold`;
- fixed-depth behavior remains the default path.

Local ablation note:

- early V7.5 ablation showed that reading raw near-zero latent slots made the V7
  path almost inert (`v7_hidden_write_ratio` near zero);
- V7 hidden read now reads normalized latent content while retaining gate, scale,
  and max-ratio caps;
- stronger hidden-write authority is useful as an ablation branch, but overdrive
  can increase probe gain without clearly improving validation loss.

### V7.6: Full Typed State Co-Evolution

Deliverables:

- `world_state`, `self_state`, `latent_field`, and hidden all evolve inside the
  same V7 dynamics loop;
- world/self updates use separate typed slot updates rather than a shared
  undifferentiated state bank;
- state writes have an explicit `v7_state_write_scale` and bounded gates;
- validation and ablation summaries expose world/self deltas and write gates;
- zero-step behavior remains available as the clean baseline.

Current implementation status:

- `TypedLatentDynamics` now returns updated world, self, and latent state
  alongside evolved hidden;
- state packets carry the final typed state endpoint for continuation;
- metrics include `v7_world_delta`, `v7_self_delta`,
  `v7_world_write_gate`, and `v7_self_write_gate`;
- this makes the architecture match the intended philosophy structurally, but
  it still requires real ablation evidence before it can be called validated.

### V7.7: Kernel and Runtime Optimization

Only after the mechanism is stable:

- fuse safe softmax/matmul/update paths where profiling proves value;
- reduce per-step kernel launch overhead in dynamics loops;
- keep eager fallback for correctness;
- test `torch.compile` and CUDA extensions separately.

## 15. Risks

### Risk: Extra compute without reasoning

Mitigation:

- require thought-depth gain and state-swap diagnostics;
- report compute-adjusted gain;
- keep zero-step baseline always available.

### Risk: Type collapse

Mitigation:

- log type boundaries;
- require state erase sensitivity;
- cap latent hidden writes;
- keep world-to-self residual path explicit.

### Risk: Self-state absorption

Mitigation:

- self reads residual after world;
- self write authority depends on world confidence or residual ratio;
- log self absorption score.

### Risk: Training instability

Mitigation:

- start with `thought_steps=1`;
- clamp hidden write norms;
- track bad-gradient correlation with dynamics metrics;
- do not enable dynamic depth until fixed depth is stable.

### Risk: Throughput collapse

Mitigation:

- profile before scale-up;
- keep thought-depth configurable;
- implement fused kernels after stable math is confirmed;
- avoid diagnostic double-runs during every training step.

## 16. Default V7 Starting Configuration

Initial local/remote smoke configuration:

```text
thought_steps = 1
max_thought_steps = 3
v7_state_write_scale = 0.02
latent_hidden_write_scale = 0.01
self_hidden_write_scale = current V6.5 default or lower
world_router_cap = current V6.5 default
state_packet_detach = true
state_swap_probe_every = eval only
erase_probe_every = eval only
dynamic_halting = false
```

The first V7 run should prioritize legibility and stability over raw loss.

## 17. Summary

V7 should make internal reasoning a typed dynamics process:

```text
world_state explains the text-world.
self_state explains internal processing boundaries.
latent_field carries continuous internal computation.
hidden_states co-evolve with all three through bounded, named channels.
```

The architecture is successful only if the internal state becomes measurably
useful: correct state must beat wrong state, more thought must beat no thought
within a bounded compute budget, and typed state erasure must hurt where that
state type should matter.
