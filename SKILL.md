# NAIME Hybrid MoE Project Skill

Use this skill when onboarding to this repository, continuing architecture work, debugging training, or preparing a new experiment for the NAIME Hybrid MoE project.

## Project Posture

This is an experimental language-model architecture project, not a static library. The goal is to build a practical, influential architecture that combines Transformer language modeling with semantic compression, state-aware routing, memory, world modeling, recursive self-state, and MoE specialization.

Default behavior:

- Inspect local code and logs before proposing changes.
- Preserve working baselines and add new architecture IDs for incompatible behavior.
- Prefer measurable architecture upgrades over rewrites.
- Treat training stability, checkpoint I/O, remote launch behavior, and disk pressure as first-class constraints.
- Do not revert user or previous-agent changes unless explicitly asked.

## Repository Map

- `src/naime_hybrid/modules/`: reusable neural modules such as MoE, semantic compressor, state, memory, and blocks.
- `src/naime_hybrid/models/`: decoder assembly and model factory.
- `src/naime_hybrid/training/`: training loop, CLI config, runtime, validation, losses, checkpoint policy, logging.
- `scripts/train_model.ps1`: primary local training launcher.
- `scripts/remote_ctl.py`, `scripts/watch_remote.ps1`, `scripts/sync_to_remote.ps1`: remote 4090 workflow helpers.
- `scripts/env.ps1`: environment setup helper.
- `tests/`: architecture and training smoke tests.
- `docs/`: architecture, training, environment, and remote-operation docs.
- `experiments/runs/`: local outputs, logs, metrics, and checkpoints. Keep out of git.

## Environment

Use Python 3.12 unless there is a clear reason not to.

Preferred local setup:

```powershell
cd <PROJECT_ROOT>
.\scripts\env.ps1
```

Direct commands usually use:

```powershell
.\<VENV>\Scripts\python.exe
```

Required checks before handing off code:

```powershell
.\scripts\check_style.ps1
.\scripts\run_tests.ps1
```

## Current Architecture Context

Important architecture IDs:

- `dense`: baseline dense Transformer.
- `token_moe`: baseline token-level MoE.
- `naime_state_moe`: early NAIME semantic MoE.
- `naime_v4_state_moe`: V4 state/memory/gate-mixer architecture.
- `naime_v41_state_moe`: V4.1 anti-degeneration refinement.
- `naime_v42_state_moe`: V4.2 accountable semantic influence refinement.
- `naime_v5_world_state_moe`: structured world-state model.
- `naime_v6_recursive_self_moe`: active recursive self-state architecture.

V6 concepts:

- V5 world-state slots remain present.
- Recursive self-state slots predict and update an internal state.
- Boundary metrics split state into self/world/other/unknown.
- V6 structure metrics must be read together with LM loss; low loss alone is not enough.

Latest completed V6 validation:

```text
run      naime_v6_100m_1b_conservative_logfix_add40m_20260517_1645
step     32958
val_lm   0.7033
val_ppl  2.020
```

Healthy late V6 behavior so far:

- `alpha` around `0.64-0.66`, not pinned at 0 or 1.
- router entropy around `1.2-1.3`.
- self boundary strongest but not exclusive.
- world boundary still weak and worth improving.
- `v6_self_pred` very small without obvious structure collapse.

## Training Defaults

Use `scripts/train_model.ps1` locally. Use the remote workflow docs for 4090 runs.

Local V6 probe:

```powershell
.\scripts\train_model.ps1 -Model naime_v6_recursive_self_moe -RunName v6_local_probe -DataPath <LOCAL_FINEWEB_50M> -TargetTokens 3000000 -EvalEvery 500 -SaveEvery 5000 -LatestEvery 2500
```

Current remote continuation baseline:

```text
resume          previous validated model_best.pt
target mode     additional
segment size    100M tokens when GPU is available
vram fraction   0.80
learning rate   2.5e-5
warmup steps    500
min lr ratio    0.03
grad clip       0.8
eval every      5000
save every      10000
latest every    5000
```

Checkpoint policy is intentionally conservative. Saving too often can become the throughput bottleneck and can aggravate Windows/native-extension instability.

## Metrics To Watch

Primary language metrics:

- `loss_lm`
- `val_lm_loss`
- `ppl_lm`
- `val_ppl`

Stability:

- `grad_norm`
- `bad_grad_window_count`
- `lr_safety_factor`
- throughput (`tok/s`)

Routing and semantic health:

- `alpha_downstream_mean`
- `val_alpha_downstream_mean`
- `router_entropy`
- `val_router_entropy`
- `loss_sparse_contrib`
- `loss_kl_contrib`

V6 internals:

- `v6_self_pred`
- `v6_slot_cosine`
- `v6_slot_context_cosine`
- `v6_boundary_self`
- `v6_boundary_world`
- `v6_boundary_other`
- `v6_boundary_unknown`
- `v6_reflection_norm`

## Remote 4090 Rules

Remote paths are read from `configs/workspace.local.json`, which must not be committed.

```text
repo     remote.repo
venv     remote.venv
dataset  remote.datasets
runs     remote.runs
```

Rules:

- Never leave visible windows on the remote desktop.
- Prefer hidden/background launch helpers.
- Use a `STOP` file for graceful shutdown.
- Do not kill unknown processes.
- Sample GPU use before launch on the shared server.
- Keep special remote one-off scripts out of git unless they are generalized.

## Disk Hygiene

G drive and remote run disks can fill quickly. Before long runs:

```powershell
Get-PSDrive G,E | Select-Object Name,Free,Used
```

Checkpoint files dominate disk usage. Safe cleanup usually includes:

- `step_*.pt`
- `model_step_*.pt`
- `failed.pt`
- stale test runs
- temporary dump/log analysis files

Do not delete `latest.pt`, `model_latest.pt`, `model_best.pt`, or the latest validated run unless the user explicitly approves or the run is known obsolete.

## Recent Failure Patterns

Windows/native failures seen during this project include:

- PyArrow access violation after heavy I/O/checkpoint pressure.
- Intel graphics driver blue screen unrelated to model divergence.
- Remote training interruption caused by visible window/session handling.
- Late-run bad-gradient spikes during long continuation.

Response:

- Inspect `train.log`, `metrics.jsonl`, launcher logs, and Windows event logs.
- Verify checkpoint artifacts before restarting.
- Reduce save frequency if I/O dominates.
- Use adaptive LR safety and shorter continuation segments for late instability.

## Architecture Upgrade Direction

Near-term priority is not blind scaling. The next valuable upgrades are:

- strengthen world-state utilization relative to self-state;
- reduce bad-gradient spike rate without suppressing useful learning;
- make sparse dispatch materially faster for larger expert counts;
- add generation quality evaluation beside validation loss;
- preserve measurable self/world boundary metrics as model scale grows.

Follow `docs/CODING_STANDARDS.md` for naming, style, and experiment hygiene.
