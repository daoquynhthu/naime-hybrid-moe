# NAIME Hybrid MoE

Working repository for a practical language-model architecture that combines:

- NAIME-style selective semantic compression;
- context/state-aware MoE routing through hybrid gate mixing;
- cross-layer semantic memory with confidence-gated read/write;
- structured world-state and recursive self-state slots;
- optional GQA/MLA attention variants;
- sparse MoE dispatch work for scaling expert count and throughput.

## Current Status (2026-05-17)

V6 (`naime_v6_recursive_self_moe`) is the active architecture. It has moved past the first 1B-corpus smoke validation and is now being trained in segmented continuation runs on the remote RTX 4090 machine.

License: source code is released under AGPL-3.0. See `LICENSE` and `NOTICE`.

| Checkpoint / Run | Step | val_lm | val_ppl | Notes |
|------------------|------|--------|---------|-------|
| `naime_v6_100m_1b_conservative_logfix_add40m_20260517_1645` | 32958 | 0.7033 | 2.020 | Best validated V6 checkpoint so far; completed cleanly. |
| Same run | 32500 | 0.7183 | 2.051 | Stable trend before final eval. |
| `naime_v6_100m_1b_formal_20260517_013635` | 10000 | 2.0106 | 7.47 | First credible remote V6 validation. |
| V5 reference | 200M tokens | 1.263 | 3.5 | Previous world-state baseline. |

Key findings:

- V6 is no longer merely a speculative path: recursive self-state metrics are active while validation loss improves strongly.
- The latest completed V6 segment reached `val_ppl ~= 2.02` on the FineWeb-Edu ctx1024 validation split.
- Healthy late-run structure is more balanced than the early self-dominated V6 run: `alpha ~= 0.65`, router entropy `~= 1.25`, self boundary `~= 0.71`, world boundary `~= 0.07`.
- Bad-gradient spikes remain the main training risk. The trainer now logs bad-gradient window counts and applies an adaptive LR safety factor when spikes cluster.
- Current 1B-token strategy is segmented continuation with non-replaying data flow, conservative checkpoint frequency, and GPU-aware auto-batch probing.

Current active remote run:

```text
naime_v6_100m_1b_add100m_autobatch_stable_20260517_1900
```

It resumes from the completed `...add40m_20260517_1645\models\model_best.pt` checkpoint and adds another 100M tokens.

## Workspace Layout

Machine-specific paths are not stored in tracked code. Copy
`configs/workspace.example.json` to `configs/workspace.local.json` and fill in
local/remote paths for your workstation or server.

| Config key | Purpose |
|------------|---------|
| `local.data_root` | Local data root. |
| `local.run_root` | Local training outputs. |
| `local.hf_home` | Local HuggingFace cache. |
| `remote.repo` | Remote deployed worktree. |
| `remote.datasets` | Remote datasets. |
| `remote.runs` | Remote run outputs. |
| `remote.venv` | Remote Python environment. |

## Directory Map

- `docs/architecture/` - architecture specs, design decisions, and validation results.
- `docs/CODING_STANDARDS.md` - naming, style, metrics, and experiment hygiene rules.
- `docs/ENVIRONMENT.md` - virtual environment setup and cross-machine migration guide.
- `docs/TRAINING.md` - training commands, data preparation, and run structure.
- `docs/REMOTE_4090_OPERATIONS.md` - shared remote 4090 operating rules.
- `docs/research-notes/` - notes distilled from frontier repositories and local projects.
- `src/naime_hybrid/` - model, modules, training, data, and eval code.
- `configs/` - experiment/model configuration files.
- `scripts/` - launch, sync, monitoring, and utility scripts.
- `tests/` - unit and smoke tests.

## Datasets

| Name | Local / Remote Path | Size | Blocks |
|------|---------------------|------|--------|
| FineWeb-Edu 1B ctx1024 | `local.fineweb_edu_1b` / remote dataset config | 1B train tokens + validation | HF disk / Arrow |
| FineWeb-Edu 50M | `local.fineweb_edu_50m` | 50M tokens | HF disk / Arrow |
| WikiText GPT-2 | `data\naime\gpt2` | GPT-2 tokenizer | legacy |
| WikiText processed | `data\naime\wikitext_processed` | legacy | legacy |

The 1B corpus is prebuilt before training. Training reads from disk; it is not intended to download or tokenize data online during a run.

## Quick Start

```powershell
# Setup/check local environment
.\scripts\setup_env.ps1
.\scripts\run_tests.ps1

# Local V6 smoke run
.\scripts\train_model.ps1 -Model naime_v6_recursive_self_moe -RunName v6_smoke -DataPath <LOCAL_FINEWEB_50M> -TargetTokens 3000000 -EvalEvery 500 -SaveEvery 5000 -LatestEvery 2500

# Generate text
.\scripts\infer.ps1 "<RUN_ROOT>\<run>\models\model_best.pt" "prompt text"
```

For remote 4090 training, follow `docs/REMOTE_4090_OPERATIONS.md` instead of launching visible foreground PowerShell windows.
