# NAIME State Protocol

Status: **active engineering protocol**

This document defines the architectural contract for NAIME V6 and later stateful
architectures. It is not an ablation plan, experiment matrix, training recipe, or
claim of validated superiority. Any future architecture work that changes
semantic routing, world state, self state, memory, or hidden-state modulation
must either follow this protocol or explicitly introduce a new protocol version.

## 1. Purpose

NAIME has moved from a Transformer-plus-MoE stack into a multi-state feedback
system. The main engineering risk is no longer one isolated module behaving
incorrectly. The main risk is that several state and control signals rewrite one
another until attribution, stability, and scaling behavior become opaque.

The protocol therefore exists to keep the architecture legible:

- Every state subsystem must have a defined responsibility.
- Every signal that can affect routing or hidden states must have an explicit
  permission boundary.
- Metrics must distinguish observation from control.
- Causal integrity must be preserved by construction.
- Clean baselines must not reuse contaminated or pre-protocol checkpoints.

## 2. Core Terms

`hidden_states`

The main representation stream used by attention, MoE, normalization, and the LM
head. This is the highest-permission path in the model. Any module that writes to
`hidden_states` is part of the core architecture, not a passive diagnostic.

`semantic_base`

The token-level semantic signal produced by the semantic compressor and its
downstream alpha/gate logic. This is the default semantic routing input.

`router_bus`

The full control signal passed to MoE routing. It may include semantic, world,
memory, or other state-conditioned components, but each component must be named,
scaled, logged, and bounded.

`world_state`

Structured slots intended to represent external context: topic, object-like
structure, discourse state, latent entities, and other text-world regularities.
World state explains the outside-facing structure of the sequence.

`self_state`

Recursive slots intended to represent the model's internal processing boundary:
what has been absorbed, reflected, predicted, or treated as self-relevant during
processing. Self state must not become an unbounded fallback for everything that
world state fails to explain.

`memory`

A lower-permission working-memory carrier for temporary summaries or retrieval
context. Memory must not silently duplicate world state or self state without an
explicit read/write contract.

`reflection`

The vector produced by self-state processing before it is projected back into
hidden space. Reflection is allowed to modulate hidden states only through a
bounded write path.

## 3. Responsibility Boundaries

### Semantic Compressor

The semantic compressor is responsible for token-local and block-local semantic
compression. It may produce semantic embeddings, alpha/gate values, KL/prediction
metrics, and semantic summaries.

It must not silently become a cross-layer memory system. If semantic summaries
are used across layers, that usage belongs to an explicit state subsystem.

### World State

World state is responsible for external structure. It may read semantic or hidden
summaries and produce world context for the router bus.

World state should be the primary owner of:

- topic and discourse continuity;
- entity-like latent slots;
- external context structure;
- cross-block context that describes the sequence rather than the model itself.

World state must expose confidence or utilization metrics whenever its output is
used to influence routing or hidden states.

### Self State

Self state is responsible for recursive internal reflection. It may read prior
self state, prior world state, and causal hidden summaries. It may write back to
hidden states only through a bounded modulation path.

Self state should be the primary owner of:

- internal processing continuity;
- boundary between self/world/other/unknown signals;
- reflection over prior internal state;
- prediction of the next internal self summary.

Self state must not be allowed to absorb arbitrary text heterogeneity simply
because world state is weak. Its hidden-state write strength must be constrained
by world-state confidence, residual-explained ratio, or another explicit control
signal.

### Memory

Memory is responsible for temporary working context. It may support local
read/write behavior, but it must not be treated as a hidden third state system
with undefined authority.

If memory affects `hidden_states` or `router_bus`, the code must log its
contribution norm and scaling factor. If memory is dormant in causal mode, logs
and documentation must state that clearly.

## 4. Permission Levels

Every new module or signal must declare the highest permission level it uses.

Level 0: Observation only

The module computes metrics but cannot affect logits, loss, router decisions, or
hidden states.

Level 1: Auxiliary loss only

The module affects training through an explicitly weighted objective but does not
change forward activations used by the LM head.

Level 2: Router influence

The module contributes to `router_bus` and can affect expert selection or expert
weights. Level 2 modules must log their routing contribution.

Level 3: Hidden modulation

The module writes to `hidden_states` through a bounded residual/modulation path.
Level 3 modules must expose scale, norm, and gating metrics.

Level 4: State mutation

The module mutates persistent per-forward state that future blocks will read.
Level 4 modules must be causal-safe and must expose state delta/velocity metrics.

Level 5: Architecture controller

The module dynamically controls another module's permission, scale, or write
strength. Level 5 changes require an architecture ID or protocol-version update.

## 5. Router Bus Contract

The MoE router must not receive an opaque mixed control field. The effective
router input must be conceptually decomposable as:

```text
router_bus =
  semantic_base
  + world_gate  * world_component
  + memory_gate * memory_component
  + other_named_components
```

The implementation may fuse these operations for performance, but metrics must
preserve this decomposition.

Required router-bus metrics for any active component:

- component norm;
- component contribution ratio;
- cosine similarity against `semantic_base`;
- effective gate or scale;
- router entropy after mixing;
- alpha/downstream alpha after mixing.

No component may be added to router control without a name, scale, and metric
surface. If a component is inactive in a mode such as causal training, that must
be explicit in logs or docs.

## 6. Hidden-State Write Contract

Any write into `hidden_states` must be treated as an architecture-level behavior.
It must follow this form:

```text
hidden_states = hidden_states + write_gate * write_scale * projected_signal
```

The exact implementation may differ, but the following quantities must be
recoverable from metrics:

- source subsystem;
- raw signal norm;
- projected signal norm;
- effective write scale;
- effective write gate;
- resulting hidden delta norm when practical.

Self-state hidden modulation must be bounded by at least one explicit control
signal. Preferred controls are:

- world confidence;
- residual unexplained-by-world ratio;
- scheduled cap on self write strength;
- reflection norm clamp or normalized write.

Semantic residual writes and memory hidden writes must follow the same reporting
rule when active.

## 7. World-to-Self Contract

World state may condition self state, but self state must not treat world state as
optional decoration. If world state is weak, uncertain, or inactive, self-state
write authority must be reduced or redirected to observation-only behavior.

The preferred self update input is:

```text
self_input = hidden_residual_after_world + prior_self + prior_world_summary
```

where:

```text
hidden_residual_after_world = hidden_summary - project(world_summary)
```

This is a design preference, not a mandatory implementation detail. If full
hidden summaries are used instead, the architecture must include another
explicit mechanism preventing self state from becoming an all-purpose structure
absorber.

## 8. Causal Integrity

State updates must be prefix-causal. A signal may affect token or block `t` only
if it was computed from information available before or at the permitted causal
boundary.

For block-causal state paths:

- the modulation applied to the current block must be computed before reading the
  current block summary, or must use a strictly causal token-level implementation;
- current block summaries may update state for future blocks;
- future block summaries must never influence current block routing, hidden
  modulation, labels, or auxiliary targets.

Checkpoints produced before the current causal-integrity version must not be used
for clean training unless explicitly marked as legacy forensic baselines.

## 9. State Boundary Rules

The model may expose `self`, `world`, `other`, and `unknown` boundary metrics, but
these metrics are not automatically control signals.

If a boundary metric is used to control training or forward behavior, the
implementation must define:

- the target behavior;
- the allowed range;
- the response when the metric leaves range;
- whether the response affects loss, router control, hidden writes, or scheduler
  behavior.

Self dominance is not forbidden. Unbounded self dominance is forbidden. A high
self boundary value is acceptable only when world utilization and generation
quality remain healthy under clean causal evaluation.

## 10. Metrics: Observation vs Control

Metrics must be classified as one of:

- observational: used for diagnosis only;
- warning: may trigger logs, alerts, or structural-stop decisions;
- control: directly changes loss weights, gates, schedules, or module behavior.

Adding a control metric requires documenting:

- the source metric;
- the control target;
- the update rule;
- the minimum warmup before activation;
- the fallback behavior if the metric becomes NaN, zero, or unavailable.

Bad-gradient handling must not be treated as sufficient evidence that the
architecture is stable. If bad-gradient windows correlate with state deltas,
reflection norms, router entropy jumps, or boundary drift, the instability should
be considered structural until disproven.

## 11. Loss and Objective Contract

The primary LM loss must remain separately logged from total objective loss.
Every auxiliary objective must log:

- raw value;
- effective weight;
- contribution to total loss;
- whether it is active in train, eval, or both;
- whether its target is detached.

Auxiliary losses may shape state systems, but they must not obscure language
modeling progress. Training and validation logs must keep comparable LM-loss
fields.

## 12. Checkpoint and Lineage Contract

Clean architecture validation must start from one of:

- random initialization;
- a checkpoint produced by the same causal-integrity version and same protocol
  family;
- an explicitly approved clean base checkpoint.

Legacy checkpoints may be loaded only with an explicit opt-in flag and must be
treated as forensic or contaminated baselines unless proven otherwise.

Run names and metadata must expose:

- architecture ID;
- protocol version if not default;
- causal-integrity version;
- dataset identity;
- resume source or `resume=none`;
- effective token budget.

## 13. Scale-Up Contract

Scaling is allowed only when the state system remains legible. Larger models must
not rely on parameter count to hide unclear state boundaries.

Before scaling an architecture family, the latest clean run should show:

- LM loss improving under clean causal evaluation;
- no unexplained validation/metric mismatch;
- router entropy in a non-collapsed range for the chosen expert count;
- bounded hidden write norms;
- world-state utilization not collapsing to zero;
- self-state reflection active but not unconstrained;
- no recurring bad-gradient clusters tied to state jumps.

This protocol does not specify ablation matrices. Ablations are an engineering
requirement for validation, but they are intentionally outside this protocol.

## 14. Implementation Checklist

Before merging a stateful architecture change, confirm:

- The highest permission level of each new signal is known.
- Router-bus components are named and measurable.
- Hidden-state writes are gated, scaled, and logged.
- World-to-self influence is constrained or justified.
- Causal integrity is preserved for train and eval.
- LM loss and total loss remain separately logged.
- Old contaminated checkpoints cannot be used accidentally.
- Documentation states which mode-specific paths are active or dormant.

## 15. Current V6 Interpretation

Current V6 should be interpreted as a strong but not yet fully governed
multi-state architecture. The clean causal baseline is the only valid baseline
for future claims. Older ultra-low perplexity runs and old self/world boundary
ratios must not be used as evidence unless they are explicitly labeled as legacy
or contaminated.

The next architecture step should not merely add more recursion. It should harden
the state protocol: gated self modulation, decomposed router bus metrics,
world-conditioned self authority, and clearer distinction between state
observation and state control.
