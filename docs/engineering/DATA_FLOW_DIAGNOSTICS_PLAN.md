# Data Flow Diagnostics Plan

Date: 2026-05-26

Status: V1 implementation in progress

Current implemented entry point:

```powershell
.\scripts\run_packet_diagnostics.ps1 `
  -RunDir <run_dir> `
  -Checkpoint models/model_best.pt `
  -DataPath <dataset_path> `
  -DataFormat hf_disk `
  -DataSplit validation `
  -OutputDir analysis\data_flow_diagnostics\<run_id>
```

Equivalent Python module:

```powershell
.\venv312\Scripts\python.exe -m naime_hybrid.diagnostics.run_packet_diagnostics `
  --run-dir <run_dir> `
  --checkpoint models/model_best.pt `
  --data-path <dataset_path> `
  --data-format hf_disk `
  --data-split validation `
  --output-dir analysis\data_flow_diagnostics\<run_id>
```

V1 currently writes:

- `manifest.json`
- `trace_events.jsonl`
- `summary.json`

The current report includes full-packet stateful-vs-fresh gain plus
field-level `erase` and `swap` interventions for:

- `world_state`
- `self_state`
- `latent_field`
- `controller_state`
- `memory`

## 1. Goal

This document defines a dedicated diagnostics framework for NAIME data-path tracing.

The framework is explicitly **not** a normal training feature:

- it is disabled by default;
- it does not need to be fast;
- it is allowed to add heavy logging, caching, intervention, and repeated forward passes;
- it is intended to answer mechanism questions, not to improve throughput.

The target problem is no longer "do we have enough metrics", but:

- which internal path actually carries useful information;
- whether a path is read, written, or silently ignored;
- where carried state is amplified, attenuated, gated, overwritten, or bypassed;
- which mechanism changes logits/loss, and which only changes internal telemetry;
- how chunk boundary state affects early continuation tokens before local context takes over.

## 2. Why This Is Needed

The current system already contains:

- compact `NAIMEStatePacket` ingress/egress;
- V5 world-state read/write;
- V6 recursive self-state;
- V7 typed latent dynamics with world/self/latent/controller paths;
- continuation training through `chunk1 -> packet -> chunk2`;
- state carry, swap, erase, and doc continuity probes.

This means failure modes can now hide in several places:

- a mechanism computes values but is never consumed downstream;
- a state path is written, but later gates collapse it;
- a feature looks useful only because evaluation changed;
- a carry objective is numerically present, but its effect is localized to a narrow boundary window;
- a route writes into hidden state while packet state remains decorative;
- one path dominates while another path stays permanently inert.

The existing training metrics and validation probes are necessary but no longer sufficient. A dedicated diagnostics mode is needed to make the system auditable.

## 3. Design Principles

The diagnostics framework should obey the following rules.

### 3.1 Separate Diagnostics From Training

Normal training should not pay for this framework. Diagnostics should run in a separate mode, ideally through:

- a dedicated script;
- a validation-only diagnostics path;
- or an offline replay against cached mini-batches / checkpoints.

### 3.2 Track Mechanisms, Not Everything

The framework should not attempt to dump every tensor in the model. It should instead trace a defined set of mechanism nodes and edges.

This keeps the output interpretable and avoids turning the framework into a raw tensor archive.

### 3.3 Support Causal Intervention

Observing a path is not enough. The framework must also support interventions such as:

- zero;
- swap across batch;
- erase selected packet fields;
- detach;
- clamp or rescale;
- replace with cached reference values.

Without interventions, the framework will show correlation but not mechanism importance.

### 3.4 Produce Structured Reports

The main output should be a report with mechanism-level summaries, not only raw traces.

The framework should answer:

- was the path active;
- how strong was it;
- what did disabling it change;
- which tokens were affected;
- whether the effect was boundary-local or persistent.

## 4. Existing Code Touchpoints

The current codebase already contains the right hooks for a first implementation.

### 4.1 Protocol Boundary

`src/naime_hybrid/models/state_packet.py`

- `NAIMEStatePacket` already gives a compact public packet contract.
- This is the natural trace unit for cross-chunk diagnostics.
- Every diagnostics report should treat packet ingress and egress as explicit graph boundaries.

### 4.2 Decoder Ingress/Egress

`src/naime_hybrid/models/decoder.py`

Relevant functions and regions:

- `_resolve_state_packet()`
- V6/V7 `forward(..., past_state=..., return_state=True)`
- V7 `_apply_ingress_state_compatibility()`
- V7 `_run_typed_dynamics()`
- final `output["state_packet"] = NAIMEStatePacket(...)`

These are the best locations to trace:

- packet arrival;
- detach vs non-detach behavior;
- prior vs carried-state blending;
- outgoing packet composition;
- per-family field availability:
  - `world_state`
  - `self_state`
  - `latent_field`
  - `controller_state`
  - `memory`

### 4.3 V5 World-State Read/Write Path

`src/naime_hybrid/modules/world_state.py`

Key functions:

- `read()`
- `read_update_sequence()`
- `update_slots()`

These functions already expose a meaningful mechanism boundary:

- token query -> slot read;
- slot confidence;
- slot write weights;
- state update gate;
- transition prediction;
- traced state sequence in causal mode.

This is the right place to diagnose whether world-state:

- is being read;
- is trusted by downstream gates;
- is being updated meaningfully;
- affects only local routed semantics or later continuation.

### 4.4 V6 Self-State Path

`src/naime_hybrid/modules/self_state.py`

Key functions:

- `forward()`
- `_forward_causal()`
- `_apply_latent_thought()`
- `_hidden_write_gate()`

This module already contains internal structure worth tracing:

- hidden summary;
- world summary;
- residual-to-world;
- reflection state;
- slot context read;
- recursive update;
- optional hidden write.

This is the right place to answer whether self-state:

- is actually explaining residual information after world-state;
- is feeding hidden-state modulation;
- is only moving internally or changing outputs.

### 4.5 V7 Typed Dynamics Path

`src/naime_hybrid/modules/typed_dynamics.py`

Key areas:

- typed state summaries;
- `_apply_state_compatibility()`;
- latent/world/self/controller updates;
- tau / rate scaling;
- hidden write gate;
- causal summary behavior.

This is the core region for answering:

- which typed state path matters;
- whether ingress compatibility is suppressing carry;
- whether latent dynamics are useful only inside the segment;
- whether controller state is decorative or causally active.

### 4.6 Block-Level Aggregation

`src/naime_hybrid/modules/blocks.py`

`NAIMEV5WorldStateMoEBlock` already aggregates useful telemetry into `aux["v4"]` and `aux["v5"]`.

This is the best place to add a block-local diagnostics emission layer because:

- routed semantic tensor exists here;
- world router component exists here;
- state/memory read-write decisions are local here;
- current telemetry names already map closely to mechanism questions.

### 4.7 Validation Probes

`src/naime_hybrid/training/validation.py`

The existing validation probes already provide seed implementations for intervention:

- state carry;
- state swap;
- state erase;
- doc continuity;
- boundary vs tail views.

The diagnostics framework should reuse this structure instead of inventing a second unrelated intervention system.

## 5. What The Framework Should Diagnose

The first version should focus on a small number of critical path questions.

### 5.1 Carry Boundary Question

Question:

- does `chunk1 -> packet -> chunk2` change early continuation tokens;
- if yes, for how long before local context takes over.

Required outputs:

- boundary 16/32/64/128 token effect;
- tail effect;
- token-level delta curve for chunk2;
- optional per-field packet ablations.

### 5.2 Packet Field Usefulness

Question:

- among `world/self/latent/controller/memory`, which fields are causally consumed.

Required outputs:

- full packet baseline;
- one-field erase deltas;
- one-field swap deltas;
- optional field-only carry tests.

### 5.3 Hidden Shortcut vs Persistent State

Question:

- is the gain coming from persistent packet state, or only from within-segment hidden routing.

Required outputs:

- packet-preserved vs packet-erased comparisons;
- same-segment mechanism diagnostics;
- boundary-only effect curves.

### 5.4 Gate Saturation and Dead Paths

Question:

- which gates are informative, and which are effectively constant.

Required outputs:

- gate mean / std / min / max;
- activation fraction;
- downstream sensitivity when the gate is clamped;
- identification of always-open or always-closed paths.

## 6. Proposed Architecture

The framework should be implemented as five layers.

### 6.1 Diagnostics Config

Add a separate config object, not mixed into normal training flags:

```text
TraceConfig
  enabled
  record_tokens
  record_hidden_norms
  record_packet_fields
  record_block_events
  intervention_specs
  output_dir
  sample_limit
  batch_limit
```

This should live separately from `TrainConfig` to avoid polluting the training CLI.

### 6.2 Trace Context

Introduce a runtime context object:

```text
TraceContext
  run_id
  sample_ids
  segment_ids
  active_interventions
  event_buffer
  tensor_store
```

This context is passed explicitly through diagnostics runs and never implicitly enabled in normal training.

### 6.3 Trace Emitter

Add a lightweight emitter API used by modules:

```python
emit_trace(
    name="v7.ingress.world_blend",
    kind="state_edge",
    tensors={...},
    stats={...},
    tags={...},
)
```

Important rule:

- modules should emit structured diagnostics events, not directly write files.

This keeps storage and formatting logic centralized.

### 6.4 Intervention Engine

Create a small intervention layer capable of applying operations at named points:

```text
zero
swap_batch
erase_field
scale
detach
replace_from_cache
```

Interventions should be declared by symbolic names such as:

```text
packet.world_state
packet.self_state
v7.ingress.latent
v6.hidden_write
v5.world_router_component
```

### 6.5 Report Builder

Convert trace events into human-readable and machine-readable summaries:

- JSON summary;
- token-effect tables;
- mechanism graph report;
- optional plots.

The report builder should be a separate layer so the same trace capture can support multiple analyses.

## 7. Minimal Node And Edge Graph

Version 1 should trace only the following nodes.

### 7.1 Nodes

- `input.hidden.embed`
- `v5.world_state.in`
- `v6.self_state.in`
- `v7.latent_field.in`
- `v7.controller_state.in`
- `packet.ingress`
- `v5.world_read.context`
- `v5.world_write.next`
- `v6.self_reflection`
- `v6.self_hidden_write`
- `v7.ingress_blended.world`
- `v7.ingress_blended.self`
- `v7.ingress_blended.latent`
- `v7.ingress_blended.controller`
- `v7.hidden_after_dynamics`
- `packet.egress`
- `logits`
- `token_loss`

### 7.2 Edges

- `chunk1 -> packet.egress`
- `packet.ingress -> v7.ingress_blended.*`
- `world_state.in -> v5.world_read.context`
- `v5.world_read.context -> router_semantic`
- `self_state.in -> v6.self_reflection`
- `v6.self_reflection -> v6.self_hidden_write`
- `latent_field.in -> v7.hidden_after_dynamics`
- `controller_state.in -> v7.hidden_after_dynamics`
- `packet.ingress -> chunk2.boundary_logits`
- `packet.ingress -> chunk2.tail_logits`

This graph is intentionally small. It is enough to answer the current carry and mechanism questions without trying to model the full transformer graph.

## 8. Minimal Interventions For Version 1

Version 1 should support only the following interventions.

### 8.1 Packet-Level

- erase all packet fields;
- erase one packet field;
- swap one packet field across batch;
- replace packet with cached reference packet.

### 8.2 Module-Level

- zero V5 world read context;
- zero V6 self hidden write;
- zero V7 latent hidden contribution;
- zero V7 controller contribution;
- clamp selected ingress compatibility gate to `0` or `1`.

### 8.3 Loss-View

- report token loss delta for:
  - full chunk2;
  - boundary 16/32/64/128;
  - tail;
  - per-token boundary curve.

## 9. Storage And Artifact Layout

Diagnostics output should be written under a separate root, for example:

```text
analysis/data_flow_diagnostics/<run_id>/
  manifest.json
  trace_events.jsonl
  interventions.json
  summary.json
  token_delta_curves.csv
  packet_field_report.json
  mechanism_graph.md
```

This directory should stay outside normal training logs and checkpoints.

## 10. Integration Strategy

The framework should be introduced in phases.

### Phase 0: Planning And Naming

- define node names;
- define event schema;
- define intervention schema;
- define artifact directory layout.

Deliverable:

- this plan document;
- a code-facing naming spec.

### Phase 1: Passive Trace Capture

Implement diagnostics-only event emission at:

- packet ingress/egress in `decoder.py`;
- V5 world-state read/write in `world_state.py`;
- V6 self-state reflection/hidden write in `self_state.py`;
- V7 ingress compatibility and typed dynamics in `decoder.py` and `typed_dynamics.py`.

Deliverable:

- one offline diagnostics run that records structured events without interventions.

### Phase 2: Packet Interventions

Reuse the validation probe style to add:

- packet field erase;
- packet field swap;
- packet replacement.

Deliverable:

- report showing per-field causal effect on boundary tokens.

### Phase 3: Module Interventions

Add named intervention points for:

- V5 world read context;
- V6 self hidden write;
- V7 latent hidden contribution;
- V7 controller contribution;
- ingress compatibility gates.

Deliverable:

- mechanism report comparing disabled vs enabled module paths.

### Phase 4: Boundary Continuation Report

Add token-level loss and logit-delta visualization for:

- boundary windows;
- tail;
- local-context takeover point.

Deliverable:

- a single report that answers whether carried state helps boundary tokens and how quickly the effect decays.

## 11. Proposed First Deliverable

The first useful implementation should be intentionally narrow.

### Scope

- V7 only;
- checkpoint-based offline diagnostics only;
- one batch at a time;
- no distributed support;
- no live training integration.

### Required questions it must answer

- does packet carry affect chunk2 boundary logits;
- which packet field matters most;
- does world/self/latent/controller have separable causal effect;
- does local context quickly absorb the advantage.

### Not required in version 1

- generic transformer-wide graph tracing;
- UI dashboards;
- distributed runs;
- streaming visualization;
- full tensor persistence for every layer.

## 12. Risks

### 12.1 Too Much Raw Data

If the framework logs arbitrary tensors without a schema, it will become unusable.

Mitigation:

- log only named nodes and edges;
- summarize by default;
- persist full tensors only when explicitly requested.

### 12.2 Hook Proliferation

If diagnostics hooks are scattered ad hoc across modules, maintenance cost will grow quickly.

Mitigation:

- centralize through a trace emitter API and symbolic node names.

### 12.3 Diagnostics Drift From Evaluation

If diagnostics invent a separate notion of carry that differs from validation, conclusions will split.

Mitigation:

- reuse `validation.py` probe logic and boundary/tail views wherever possible.

### 12.4 Framework Becomes Architecture

The diagnostics layer should not become a hidden control path or a second training system.

Mitigation:

- no diagnostics state in normal forward unless explicitly enabled;
- no training dependence on diagnostics artifacts.

## 13. Acceptance Criteria

The first milestone is successful only if all of the following are true.

- A single offline diagnostics run can produce a packet-field causal report.
- The report can distinguish boundary and tail effects.
- The report can show whether `world/self/latent/controller` matter differently.
- The implementation uses explicit symbolic trace points, not ad hoc print statements.
- Normal training remains unaffected when diagnostics are disabled.

## 14. Recommended Next Step

Implementation should begin with a small diagnostics package, for example:

```text
src/naime_hybrid/diagnostics/
  trace_config.py
  trace_context.py
  emitter.py
  interventions.py
  report_builder.py
  packet_diagnostics.py
```

Then wire version-1 trace points into:

- `models/decoder.py`
- `modules/world_state.py`
- `modules/self_state.py`
- `modules/typed_dynamics.py`
- `training/validation.py`

This provides a concrete path to a diagnostics-only mechanism framework without contaminating the normal training path.
