# Coding Standards

This project is moving quickly, so consistency matters more than perfect abstractions. New code should make the next experiment easier to reason about, compare, and roll back.

## Formatting

- Use Python 3.12 as the development baseline.
- Use Ruff for linting and formatting. The canonical settings live in `pyproject.toml`.
- Keep lines near 120 characters. Long metric/logging format strings are allowed when wrapping would make the metric groups harder to read.
- Prefer explicit imports from package modules. Avoid hidden wildcard exports.

## Architecture Names

- External architecture IDs use lowercase snake case: `dense`, `token_moe`, `naime_state_moe`, `naime_v4_state_moe`, `naime_v41_state_moe`, `naime_v42_state_moe`.
- Version IDs do not use dots in code or run names. Use `v41`, not `v4.1`.
- Python classes use PascalCase and may group compatible versions when implementation is shared: `NAIMEV4StateMoEDecoder` may serve V4-compatible IDs such as `naime_v4_state_moe`, `naime_v41_state_moe`, and `naime_v42_state_moe`.
- New architecture variants must be added consistently in `models/factory.py`, `training/train.py`, `scripts/train_model.ps1`, smoke/preflight checks, and tests.

## Config Naming

- Config fields are snake case and grouped by subsystem.
- Semantic compressor fields start with `semantic_` unless they are legacy core fields such as `stride`, `window`, or `z_dim`.
- Router-specific fields start with `semantic_router_`.
- Gate mixer fields start with `semantic_gate_mixer_`.
- Cross-layer state fields start with `semantic_state_`.
- Memory fields start with `semantic_memory_`.
- Training objective weights use `lambda_`.
- Runtime/checkpoint cadence uses `*_every`, `*_patience`, or `keep_*`.

## Metrics Naming

- Training metrics use plain subsystem names: `loss_lm`, `alpha_downstream_mean`, `v4_state_confidence`.
- Validation metrics must use the same base name with a `val_` prefix: `val_lm_loss`, `val_alpha_downstream_mean`, `val_v4_state_confidence`.
- Loss components should log both raw loss and contribution when weighted: `loss_sparse` and `loss_sparse_contrib`.
- V4/V4.1 structure metrics keep the `v4_` prefix unless the metric belongs to a more specific subsystem such as `gate_mix_*`.

## Module Boundaries

- `modules/` contains reusable neural building blocks only.
- `models/` assembles blocks into decoders and exposes architecture-level behavior.
- `training/` owns training loops, losses, checkpointing, logging, preflight, and evaluation utilities.
- `scripts/` owns user-facing entrypoints and PowerShell orchestration.
- Avoid adding more responsibilities to `training/train.py` unless they are genuinely part of the training loop. New reusable logic should move into a focused module.

## Experiment Hygiene

- Never overwrite an existing architecture ID with incompatible behavior. Add a new ID instead.
- Keep old baselines runnable for fair comparisons.
- Run names should include architecture, dataset, and intent, for example `naime_v41_fineweb_edu_50m_v1`.
- Model checkpoints and generated datasets stay out of git.

## Required Checks

Before committing architecture or training changes:

```powershell
.\scripts\check_style.ps1
.\.venv312\Scripts\python.exe -m pytest tests -q
```

If Ruff is not installed:

```powershell
.\.venv312\Scripts\python.exe -m pip install ruff
```
