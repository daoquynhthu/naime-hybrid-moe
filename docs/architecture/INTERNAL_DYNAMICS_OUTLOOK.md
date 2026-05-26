# NAIME Internal Dynamics Outlook

Status: **architecture outlook, reserved design direction**

This document records the architectural watershed behind V7 and later work.
It is not an implementation claim and not evidence of model capability. It
defines the direction that future text, image, video, audio, memory, and
controller work must preserve.

## 1. Watershed

NAIME should no longer be understood as a static Transformer with attached state
modules. The long-term target is an internal continuous dynamics system:

```text
observation
  -> typed observation encoder
  -> incoming internal state
  -> bounded internal dynamics
  -> outgoing internal state
  -> answer / action / next observation request
```

The backbone, MoE experts, attention, projectors, and multimodal encoders are
not the whole architecture. They are local perception and execution machinery
around a persistent typed internal state.

## 2. Core Claim

The central hypothesis is:

```text
Useful understanding is the evolution of internal state under new observations,
not a one-shot embedding of the current frame, prompt, or token window.
```

For text, this means the model should continue from prior internal state rather
than merely re-read visible context. For image and video, this means perception
should update an internal world rather than become a static captioning prefix.

## 3. Non-Negotiable Causal Rule

The same rule applies to text chunks, image frames, video segments, audio spans,
and tool observations:

```text
current readout may read incoming_state
current observation may write outgoing_state
current readout must not read outgoing_state from the same causal segment
```

This rule prevents latent state from becoming a same-segment leakage path. It
also keeps stateful inference meaningful: the packet returned by one call is the
starting condition for a later call, not a hidden rewrite shortcut for the same
call.

## 4. State Is Not KV Cache

KV cache is an inference optimization tied to tokens, positions, layers, and a
specific forward implementation. It should not be treated as thought or memory.

NAIME internal state should be compact, typed, and versioned:

```text
world_state      external world, scene, entities, events, temporal structure
self_state       model-side processing boundary, uncertainty, reflection
latent_field     continuous internal computation endpoint
memory_state     slower useful records, if enabled
controller_state compute budget, halting, and mode-selection context
```

The state packet is the endpoint of internal evolution. It may be passed across
chunks, turns, frames, or sessions when policy allows.

## 5. Multimodal Implication

Multimodality must not be implemented as "image tokens appended to text". That
is an acceptable bootstrap interface, but not the architecture philosophy.

Future multimodal NAIME should introduce `ObservationPacket`:

```python
class ObservationPacket:
    modality: str              # text, image, video, audio, tool, sensor
    embeddings: Tensor
    time_index: Tensor | None
    spatial_anchors: Tensor | None
    confidence: Tensor | None
    provenance: str
    causal_segment_id: str
```

Each observation type writes typed evidence into the same internal state system:

```text
text span     -> discourse/world/self evidence
image region  -> object/scene/spatial evidence
video segment -> event/motion/temporal evidence
audio span    -> speaker/sound/prosody evidence
tool result   -> external factual/procedural evidence
```

The model should not "understand a frame" in isolation. It should update an
internal scene/world trajectory and decide whether more internal evolution is
needed before answering.

## 6. Multi-Timescale Dynamics

Fixed `thought_steps` and fixed write scales are only a transitional mechanism.
The final architecture needs typed timescales, but the baseline rate of each
state family is an empirical question. We should not hard-code a scientific
claim such as "self is slower than world" until ablations show that the claim
improves prediction, state usefulness, stability, and compute efficiency.

The intended hypothesis space is:

```text
latent_field: local constraints and immediate internal computation
world_state: scene, discourse, entities, events, temporal relations
self_state: task stance, uncertainty, reflection, boundary allocation
memory_state: durable useful summaries and revisions
controller_state: compute budget, halting reason, mode choice
```

Current defaults remain unbiased (`latent/world/self = 1/1/1`). Non-uniform
rates are experimental variables measured by the V7 timescale ablation harness,
not architecture defaults.

The current V7 controller state is intentionally observability-only: it is
carried across state packets and logs its own norm/delta/write gate, but it does
not yet decide depth, timescale, routing, or hidden write authority. That keeps
the protocol inspectable before we grant it Level 5/6 control.

Future controllers should decide update intensity from measurable signals:

- novelty;
- uncertainty;
- contradiction;
- router entropy;
- state delta and acceleration;
- expected dynamics gain;
- available compute budget.

## 7. Autonomy Without Losing Control

The long-term goal is not an opaque learned switch that can do anything. The
goal is autonomous deliberation inside a protocol:

```text
controller proposes: continue / halt / observe more / retrieve / answer
protocol enforces: causal boundary, max cost, write caps, named channels
metrics expose: reason, expected gain, actual gain, instability
```

Early versions may use deterministic controllers. Learned controllers should
arrive only after fixed-depth dynamics and deterministic adaptive depth are
legible.

## 8. Component Cooperation

The intended cooperation pattern is:

- Backbone and MoE produce local working representations.
- Semantic compressor produces compact local evidence, not hidden memory.
- World state owns external structure.
- Self state owns internal processing boundary and uncertainty.
- Latent field owns the continuous internal computation endpoint.
- Memory owns slower durable summaries, when enabled.
- Controller owns compute allocation and halting, not factual content.
- Multimodal encoders produce observations, not separate minds.

No component should become an all-purpose sink for unexplained information.

## 9. Research Milestones

This outlook reserves the following path:

```text
V7.6  causal internal state protocol and typed dynamics
V7.7  kernel/runtime optimization after mechanism stability
V7.8  typed multi-timescale dynamics
V7.9  deterministic adaptive deliberation controller
V8.0  ObservationPacket and text-first multimodal state interface
V8.x  image/video/audio state grounding after text-state validation
```

The order matters. Multimodal capability should extend the internal dynamics
system; it should not replace it with a pile of modality-specific adapters.

## 10. Success Criteria

Future NAIME should be judged by:

- correct state beating erased or swapped state;
- incoming state improving continuity without hidden leakage;
- internal dynamics gain exceeding compute cost on suitable tasks;
- world/self/latent boundaries remaining legible;
- video/image observations producing measurable state updates;
- generation and reasoning improving without relying on textual CoT traces;
- stateful inference continuing from a compact packet rather than replaying all
  previous tokens.

This is the architectural north star. Implementations may be staged and modest,
but they must not violate this direction.
