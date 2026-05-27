# NAIME V8 Thought-Commit Upgrade Guide

Status: design guide / implementation contract  
Scope: architecture migration from V7 post-stack typed dynamics to V8 thought-committed attention dynamics

---

## 1. Why V8 exists

V7 demonstrates that typed latent dynamics can improve the current segment. It evolves `hidden_states`, `world_state`, `self_state`, `latent_field`, and `controller_state` after the main decoder block stack, before `norm` and `lm_head`.

That is useful, but it leaves a structural limitation:

```text
embedding -> decoder blocks -> typed dynamics -> logits
```

The typed dynamics can refine the final hidden state for the current output, but the refinement is not part of the attention substrate that later layers or later cached tokens read. Philosophically, this makes V7 closer to post-stack hidden refinement than fully continuous thought.

V8 exists to make thought a committed part of the model's computation history:

```text
raw hidden -> thought-refined hidden -> attention-readable committed memory
```

The goal is not to add a larger thinking module. The goal is to change where thinking lives.

---

## 2. Core principle

V8 is built around one rule:

> A hidden state that has not passed through the model's implicit thought dynamics should not be treated as the final committed representation for future attention.

In V7, hidden thought occurs after the decoder stack. In V8, hidden thought must occur inside the stack, especially between attention and transformation phases.

The target block-level pattern is:

```text
H_l
  -> self-attention
  -> typed thought commit
  -> MoE / FFN / semantic transformation
  -> typed thought commit
  -> H_{l+1}
```

This means the next layer receives thought-refined hidden states. Therefore, the next layer's `Q/K/V`, router signals, and state summaries are computed from hidden states that have already undergone thought dynamics.

---

## 3. What V8 is not

V8 must avoid four failure modes.

### 3.1 V8 is not a bigger V7 tail module

Do not simply increase `v7_dynamics_steps` or make the post-stack typed dynamics more aggressive. That can strengthen current-segment refinement, but it does not solve the architectural problem that thought does not shape later attention.

### 3.2 V8 is not arbitrary KV rewriting

V8 should not retroactively rewrite all past KV cache entries after future thought. That would blur causal semantics, make training/inference equivalence difficult, and turn attention history into an unstable self-editing memory.

The correct invariant is:

```text
Past committed memory is immutable.
Current hidden must be thought-refined before commit.
Slow StatePacket evolves separately.
```

### 3.3 V8 is not a second thought model

Do not fork a new thought mechanism independent of `TypedLatentDynamics`. V8 should reuse and refactor the existing typed hidden/latent/state dynamics. If a new block-level helper is introduced, it must be behaviorally anchored to the existing V7 dynamics.

### 3.4 V8 is not a semantic hard-coding of world/self

`world_state` and `self_state` are stable state channels, not guaranteed human-readable world/self databases. Their names are design intentions, not ontological guarantees. V8 should not directly write object tables, visual semantics, or explicit self-model assertions into these channels.

---

## 4. Three memory scales

V8 should make the memory hierarchy explicit.

### 4.1 Fast memory: attention KV

Fast memory is token-level, layer-level, attention-readable history. In generation, this becomes KV cache. In full-sequence training, it is the hidden representation from which a layer computes `Q/K/V`.

V8 requirement:

```text
Committed fast memory should be derived from thought-refined hidden states.
```

### 4.2 Working thought: hidden dynamics

Working thought is the continuous update of hidden states:

```text
H_0 -> H_1 -> H_2 -> ...
```

This is the true carrier of implicit thinking. Typed state channels modulate it, but they are not a replacement for hidden-state thought.

### 4.3 Slow memory: StatePacket

`NAIMEStatePacket` remains the compact model-owned continuation state. It is not KV cache, not hidden activations, and not a replayable computation graph.

It carries compact slow state across chunks, sessions, and later multimodal streams:

```text
world_state
self_state
latent_field
memory
controller_state
```

V8 must preserve this distinction.

---

## 5. Thought commit semantics

A thought commit is the point at which a hidden representation becomes eligible as future computation substrate.

Minimal definition:

```text
H_raw -> TypedDynamicsStep -> H_thought -> commit
```

For full-sequence training, commit means:

```text
H_thought becomes the input to later layer attention / MoE / state updates.
```

For autoregressive generation, commit means:

```text
H_thought is projected into current-token K/V and appended to KV cache.
```

The model must never commit raw hidden when a thought step is configured for that location.

---

## 6. Placement strategy

V8 thought should not be moved blindly to the start of the model. Token embeddings alone have insufficient context. The first strong placement should be after information has been read.

Recommended placement order:

### Phase A: post-attention thought

```text
H -> self-attention -> H_attn -> thought -> H_thought
```

This lets the model first read local/history context, then integrate it through typed dynamics.

### Phase B: post-MoE / post-FFN thought

```text
H_thought -> MoE/FFN -> H_moe -> thought -> H_out
```

This makes expert/transformation output also pass through thought before being handed to the next layer.

### Phase C: optional pre-attention thought

```text
H -> thought -> self-attention
```

This is stronger and riskier because it directly alters attention query formation before a layer reads context. It should be introduced only after Phase A/B are stable.

---

## 7. Implementation sequence

The implementation should be staged, but each stage must be tied to this document. Avoid standalone orphan modules.

### Stage 0: preserve V7

V7 behavior must remain available and unchanged unless explicitly selecting a V8 architecture.

Requirements:

```text
architecture="naime_v7_typed_dynamics" remains behavior-compatible.
Existing V7 tests remain valid.
Existing StatePacket semantics remain valid.
```

### Stage 1: attention KV protocol

Prepare attention modules for committed fast memory.

Required changes:

```text
GQAAttention.forward(..., past_key_value=None, use_cache=False, position_offset=0)
MLAAttention.forward(..., past_key_value=None, use_cache=False, position_offset=0)
AttentionKVCache(key, value)
```

Invariant:

```text
use_cache=False returns the same tensor path as before.
cache-enabled segmented attention matches full-sequence attention.
```

This stage does not make V8 complete. It only prepares the fast-memory substrate.

### Stage 2: refactor TypedLatentDynamics into a step API

Do not create a second thought mechanism. Refactor the existing V7 typed dynamics so one internal step can be called from block-level code.

Target API shape:

```python
def step(
    self,
    hidden_states,
    *,
    world_state,
    self_state,
    controller_state,
    latent_field,
    attention_mask,
    readable_latent=None,
    apply_hidden_read=True,
    rate_scales=None,
    metric_prefix="v8",
):
    ...
```

Requirements:

```text
step(..., no hidden read) must match V7 forward(..., steps=1, past_latent_field=False).
step(..., readable_latent=prior) must produce bounded hidden writes.
V7 forward should be reimplemented using this step where practical, or tested against it.
```

Important: this refactor should live in `typed_dynamics.py` if possible, not as an unrelated permanent helper file. A helper module is acceptable only if it is clearly part of V8 and immediately consumed by block integration.

### Stage 3: ThoughtCommitBlock

Introduce a V8 block that embeds typed thought into the layer computation.

Target structure:

```text
input H_l, state S_l

1. self attention
2. post-attention typed thought step
3. semantic/router/world modulation as appropriate
4. MoE or FFN
5. post-transform typed thought step
6. output H_{l+1}, state S_{l+1}, metrics
```

The block must not merely call a tail dynamics function after all layers.

### Stage 4: V8 decoder

Introduce a new decoder class and architecture ID.

Suggested IDs:

```text
naime_v8_thought_commit
naime_v8
```

The V8 decoder should:

```text
initialize hidden from embeddings
initialize / resolve slow state from StatePacket
iterate ThoughtCommitBlock layers
return logits, aux, and optional StatePacket
```

It should not replace V7 in-place.

### Stage 5: thought-aware generation cache

After V8 block-level training path exists, implement generation semantics:

```text
current token -> layer attention reads committed past KV
current hidden -> thought step
thought hidden -> project current K/V
append current K/V to cache
```

Invariant:

```text
Past cache entries are immutable.
Only the current token/segment is committed after thought.
```

### Stage 6: observation attention / future multimodal path

Only after thought commit is stable, introduce observation reads.

Observation attention should follow the same abstraction as text attention:

```text
Q = hidden-derived query
K/V = observation substrate
read = attention(Q, K, V)
```

The observation side must not become an independent vision model with its own semantic closure.

---

## 8. File-level design

### 8.1 `src/naime_hybrid/modules/attention.py`

Responsibilities:

```text
AttentionKVCache
cache-compatible GQA
cache-compatible MLA
causal mask handling for past + current segment
```

No thought semantics should live here. Attention should only expose the fast-memory substrate.

### 8.2 `src/naime_hybrid/modules/typed_dynamics.py`

Responsibilities:

```text
V7/V8 typed hidden-latent-state dynamics
single-step API
multi-step V7-compatible forward
metrics for hidden, latent, world, self, controller movement
```

This is the right place for the thought dynamics core.

### 8.3 `src/naime_hybrid/modules/v8_blocks.py`

Responsibilities:

```text
ThoughtCommitBlock
block-level order of attention -> thought -> transform -> thought
state threading across layers
per-layer commit metrics
```

### 8.4 `src/naime_hybrid/models/decoder.py`

Responsibilities:

```text
NAIMEV8ThoughtCommitDecoder
StatePacket resolution and return
architecture-specific forward path
```

Avoid turning the existing decoder file into a large if/else tangle. If needed, split V8 decoder into a separate file and expose it through factory imports.

### 8.5 `src/naime_hybrid/config.py`

Add V8-specific config fields. Suggested fields:

```python
v8_thought_commit: bool = False
v8_thought_placement: str = "post_attention_post_moe"
v8_thought_steps_per_block: int = 1
v8_pre_attention_thought: bool = False
v8_commit_hidden_to_attention: bool = True
v8_commit_residual_gate: bool = True
v8_commit_max_delta_ratio: float = 0.10
v8_state_commit_mode: str = "layerwise"
v8_preserve_v7_tail_dynamics: bool = False
v8_observation_attention: bool = False
v8_observation_write_scale: float = 0.02
v8_observation_max_ratio: float = 0.05
```

V8 config must not silently change V7 behavior.

### 8.6 `src/naime_hybrid/models/factory.py`

Register V8 explicitly:

```python
"naime_v8_thought_commit" -> NAIMEV8ThoughtCommitDecoder
```

---

## 9. Metrics and validation

V8 must not be judged only by lower validation loss.

Required mechanism metrics:

```text
v8_commit_delta_norm
v8_commit_delta_ratio
v8_post_attention_thought_delta
v8_post_moe_thought_delta
v8_layerwise_commit_gain
v8_interlayer_thought_gain_lm
v8_commit_enabled_lm
v8_commit_disabled_lm
v8_poststack_vs_interlayer_delta_lm
```

State carry metrics must continue:

```text
state_carry_gain_lm
v7/v8_state_swap_delta_lm
world_erase_delta_lm
self_erase_delta_lm
latent_erase_delta_lm
```

Acceptance standard:

```text
1. V8 does not break V7 paths.
2. V8 commit enabled beats or matches commit disabled after compute adjustment.
3. Inter-layer thought is better than tail-only thought, or the result is explicitly negative.
4. State carry gain becomes less negative / non-negative over training.
5. Hidden write ratios stay bounded.
6. No causal leakage in prefix tests.
7. No NaN/Inf or recurrent bad-gradient windows.
```

---

## 10. Causal invariants

V8 must preserve causal semantics.

### Full-sequence training

A token may attend only to previous/current positions allowed by the causal mask. Thought steps may refine hidden states, but they must not introduce future information through summaries or state updates.

### State summaries

If a summary is used inside a causal path, it must be causal or restricted to the current segment protocol. Avoid global unmasked sequence summaries that let future tokens influence earlier token hidden states.

### KV generation

During generation:

```text
past committed KV is read-only
current hidden is refined
current refined hidden is committed
future tokens read the committed representation
```

No future token may rewrite previous committed cache.

---

## 11. Multimodal extension: visual attention without a second model

V8 should prepare for multimodality, but should not begin by adding an external vision model.

The correct abstraction is:

```text
ObservationPacket -> low-level K/V substrate
main decoder hidden -> Q
observation attention -> evidence
hidden thought dynamics -> integration
```

The visual side should not independently produce a complete semantic interpretation. The main decoder should form interpretation through hidden-query reads and thought commits.

This keeps perception inside the model's thought process:

```text
not: image -> vision model -> language model
but: hidden dynamics <-> observation field
```

---

## 12. Recommended PR plan

Do not submit isolated files without reference to this guide. The PR chain should be:

### PR A: V8 guide

This document only.

### PR B: attention KV substrate

Includes `AttentionKVCache`, GQA/MLA cache support, and V8-focused cache tests.

### PR C: typed dynamics step refactor

Refactor `TypedLatentDynamics` itself to expose a reusable step API, with equivalence tests against V7 single-step behavior.

### PR D: ThoughtCommitBlock

Add the block-level integration. This is the first PR where V8 becomes an actual model computation path.

### PR E: V8 decoder + factory + config

Register a selectable V8 architecture.

### PR F: validation probes

Add commit-enabled/disabled comparisons and interlayer-vs-poststack diagnostics.

### PR G: generation KV commit

Implement thought-aware cached generation semantics.

### PR H: observation attention substrate

Add low-level observation attention after text V8 is stable.

---

## 13. Rollback strategy

Each stage must be reversible.

```text
PR B can be reverted without changing V7.
PR C must keep V7 forward behavior equivalent.
PR D/E add new architecture IDs, not replace V7.
PR F adds diagnostics only.
PR G is generation-path specific.
PR H is optional multimodal extension.
```

If V8 fails mechanism probes, the failure should be documented. Do not hide a negative result by adding auxiliary losses or stronger gates.

---

## 14. Final design statement

V8 is the move from post-hoc thought to committed thought.

V7:

```text
blocks -> typed dynamics -> logits / StatePacket
```

V8:

```text
for each layer:
    attention
    thought commit
    transformation
    thought commit

final hidden -> logits
final slow state -> StatePacket
thought hidden -> future attention-readable memory
```

This is the architectural line V8 must hold. If an implementation does not make thought-refined hidden become the substrate of later computation, it is not V8; it is only a stronger V7 tail module.
