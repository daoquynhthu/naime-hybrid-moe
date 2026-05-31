# NAIME Progress

This file is the persistent project memory for phase-level changes. Update it
after each meaningful phase so later agents do not drift or repeat old mistakes.

## Project Arc

- **V0-V3:** Started from semantic-state influenced MoE routing. The early goal
  was to test whether compressed semantic state could improve expert selection
  over dense and token-only baselines.
- **V4-V5:** Added explicit memory and world-state slots. This shifted the
  project from "semantic routing" toward a stateful model with observable
  world-like latent structure.
- **V6:** Added recursive self-state and latent thought/state evolution. The
  system began to expose internal self/world boundaries, but state interaction
  and attribution became hard to reason about.
- **V7:** Moved toward typed internal dynamics: latent/world/self/controller
  state, ingress compatibility, hyperspherical state updates, adaptive tau,
  causal summaries, and packet-based continuation. V7 is now best treated as a
  diagnosable baseline rather than the final philosophical form.
- **V8 direction:** The next architecture target is thought-commit dynamics:
  thought-refined hidden state should become the substrate of later computation,
  not just a post-stack refinement. Open PRs contain the V8 design guide and
  attention KV substrate.

## Current Mainline

The immediate priority is the diagnostics system. Before further architecture
or loss expansion, we need a reliable way to answer:

- which state paths are active;
- which state paths are causally consumed;
- whether packet carry affects boundary tokens or only telemetry;
- whether world/self/latent/controller/memory have separable effects;
- whether improvements are real mechanism gains or hidden shortcuts.

## Current Phase: Data-Flow Diagnostics V1

Implemented / in progress:

- passive `TraceContext` and structured `emit_trace_event`;
- V7 ingress/egress trace points;
- V5 world-state, V6 self-state, V7 typed dynamics trace hooks;
- offline packet diagnostics comparing `stateful` vs `fresh`;
- field-level packet `erase` and `swap` interventions for world/self/latent/
  controller/memory;
- boundary/full/tail loss views;
- CLI module and PowerShell wrapper for reproducible diagnostics runs.
- explicit training-time diagnostics mode, disabled by default;
- `--diagnostics-mode` + `--diagnostics-every N` training loop integration;
- per-step training diagnostics artifacts under `training_diagnostics/step_*`;
- scalar training diagnostics summary fields in `metrics.jsonl` without
  dumping full token curves into the main metric stream.

Phase result on 2026-05-31:

- Local diagnostics check pipeline added: `scripts/check_diagnostics.ps1`.
- Local gate passed: architecture tests `63 passed, 3 skipped`; diagnostics
  modules compile; PowerShell wrapper parses; `git diff --check` passes.
- Remote real diagnostics run completed on:
  `L:\NAIME_REMOTE\runs\v7_64m_ablate_boundary_20260526_211921`.
- Remote artifacts:
  `L:\NAIME_REMOTE\runs\v7_64m_ablate_boundary_20260526_211921\diagnostics\packet_flow_v1`
- Remote report contained 324 structured events and field-level packet
  interventions for world/self/latent/controller/memory.
- Observed remote full-packet carry gains on that diagnostic batch:
  `full_gain=-0.0011735`, `boundary_gain=-0.0020905`,
  `tail_gain=-0.0008678`.

Interpretation: the diagnostic system is operational, and this specific V7
checkpoint shows slightly negative packet carry on the sampled validation
batch. This is not yet an architecture verdict; it proves the tool can expose
field-level packet effects.

Training-time diagnostics update on 2026-05-31:

- Added `TrainConfig` and CLI controls for dedicated diagnostics mode:
  `--diagnostics-mode`, `--diagnostics-every`, `--diagnostics-output-dir`,
  `--diagnostics-chunk-len`, `--diagnostics-boundary-tokens`,
  `--diagnostics-max-batch`, and `--diagnostics-no-tensor-stats`.
- Normal training remains unchanged unless both `--diagnostics-mode` and a
  positive `--diagnostics-every` are supplied.
- Diagnostics run on the current micro-batch after the optimizer step, under
  `no_grad`/eval mode, using the uncompiled model when `torch.compile` wraps
  training.
- Local 2-step CPU smoke verified that step artifacts are written and
  diagnostics scalars appear in `metrics.jsonl`; the smoke run was removed.

Training dynamics trace update on 2026-05-31:

- Added `training_diagnostics/dynamics_events.jsonl` as a dedicated
  training-time event stream, separate from normal `metrics.jsonl`.
- Each diagnostics step now binds core loss/LR/grad fields, router fields,
  V5/V6/V7 state fields, and packet carry diagnostics into one
  `post_optimizer` event.
- Bad-gradient skips in diagnostics mode now emit `bad_grad_skip` events before
  the optimizer step is skipped, preserving the state/route/grad context that
  would otherwise be lost.
- Local 2-step CPU smoke verified that `dynamics_events.jsonl` is produced and
  contains packet-linked dynamics events; the smoke run was removed.

Gradient/component diagnostics update on 2026-05-31:

- Added diagnostics-only per-module gradient component grouping. Events now
  include grouped total norm, max absolute gradient, and parameter count for
  broad components such as attention, router/gate, MoE experts, world state,
  self state, typed dynamics, embeddings, norms, and LM head.
- Gradient component stats are collected after AMP unscale and before clipping,
  so the values describe the actual step gradient pressure rather than clipped
  artifacts.
- Added recent-event bad-gradient window snapshots:
  `training_diagnostics/window_bad_grad_step_*.json`.
- Local 2-step CPU smoke verified that `dynamics_events.jsonl` includes
  gradient component diagnostics; the smoke run was removed.

Loss-component gradient attribution update on 2026-05-31:

- Added diagnostics-only loss-component gradient probes. In explicit
  diagnostics mode, selected loss groups can be replayed on the retained graph
  of the final microbatch and attributed to broad parameter groups.
- Supported probe selectors: `lm`, `router`, `state`, `self`, `carry`, and
  `all`, controlled by `--diagnostics-loss-grad-components`.
- The probe restores the original accumulated gradients before the real
  optimizer path continues. It is skipped when AMP `GradScaler` is enabled, so
  the normal mixed-precision training path is not perturbed.
- Events now include `loss_grad_component_norm` and
  `loss_grad_component_cosine`, making it possible to see whether LM/router/
  state/self/carry objectives push the same component in aligned or conflicting
  directions.
- Local 2-step CPU smoke verified that loss-component gradient attribution is
  present in `training_diagnostics/dynamics_events.jsonl`; the smoke run was
  removed.

Training diagnostics report update on 2026-05-31:

- Added an automatic report builder for `training_diagnostics/dynamics_events.jsonl`.
- New outputs:
  `training_diagnostics/diagnostics_report.json` and
  `training_diagnostics/diagnostics_report.md`.
- The report summarizes phase counts, key metric series, packet carry trend,
  gradient component peaks, loss-component gradient peaks, low cosine alignment,
  and warning conditions such as bad-gradient skips or negative boundary carry
  gain.
- New script:
  `scripts/summarize_training_diagnostics.ps1 -Path <training_diagnostics_dir>`.
- Local 2-step CPU smoke verified end-to-end training diagnostics plus report
  generation; the smoke run was removed.

Important command:

```powershell
.\scripts\run_packet_diagnostics.ps1 `
  -RunDir <run_dir> `
  -Checkpoint models/model_best.pt `
  -DataPath <dataset_path> `
  -DataFormat hf_disk `
  -DataSplit validation `
  -OutputDir analysis\data_flow_diagnostics\<run_id>
```

## Known Pitfalls

- Do not treat a lower LM loss as proof that state mechanisms are working.
- Do not resume polluted checkpoints across causal-integrity or state-protocol
  changes.
- Do not let diagnostics become part of normal training behavior.
- Training-time diagnostics are allowed only in explicit diagnostics mode; they
  must remain outside the loss/objective path.
- `metrics.jsonl` is the normal training metric stream; diagnostic event
  causality should be read from `training_diagnostics/dynamics_events.jsonl`.
- Use `diagnostics_report.md` for quick human triage and
  `diagnostics_report.json` for automated comparison.
- Gradient component diagnostics are for attribution and anomaly localization;
  they must not become optimization targets.
- Loss-component gradient probes are diagnostic replays only. They must not
  change accumulated gradients, LR policy, optimizer state, or checkpoint
  contents.
- Do not create one-off monitoring scripts when a unified probe already exists.
- Keep remote commands windowless and use the established `remote.ps1` path.
- Keep `pony_remote/`, `analysis/`, `bin/`, checkpoints, datasets, and local
  credentials out of git.

## Next Required Gates

- Local tests must pass after diagnostics changes.
- Training-time diagnostics must be verified by a real training-loop smoke, not
  only by offline checkpoint diagnostics.
- A diagnostic smoke must prove both per-step packet artifacts and
  `dynamics_events.jsonl` are written.
- A diagnostic smoke must prove gradient component diagnostics are present when
  not disabled.
- A diagnostic smoke must prove loss-component gradient attribution is present
  when enabled and AMP scaler is disabled.
- A diagnostic smoke must prove `diagnostics_report.json` and
  `diagnostics_report.md` can be generated from the training-time event stream.
- A real remote diagnostics run must produce `manifest.json`, `summary.json`,
  and `trace_events.jsonl`.
- The remote report must include packet field interventions with finite
  full/boundary/tail deltas.
