# NAIME Hybrid MoE

Working repository for a practical language-model architecture that combines:

- NAIME-style selective semantic compression;
- context/state-aware MoE routing through hybrid gate mixing;
- cross-layer semantic memory with confidence-gated read/write;
- structured world-state and recursive self-state slots;
- V7 typed internal dynamics and state packets;
- optional GQA/MLA attention variants;
- sparse MoE dispatch work for scaling expert count and throughput.

## Current Status (2026-05-25)

V7 (`naime_v7_typed_dynamics`) is the active experimental architecture family.
V6 (`naime_v6_recursive_self_moe`) remains the last mature stateful baseline.
Only clean causal-integrity-v2 runs should be used for current claims. Older
continuation runs that produced ultra-low perplexity are treated as
legacy/contaminated evidence and must not be used as proof of architecture
quality.

License: source code is released under AGPL-3.0. See `LICENSE` and `NOTICE`.

Current engineering baseline:

| Item | Status |
|------|--------|
| Clean resume default | `resume=none`; legacy checkpoints require explicit opt-in. |
| Causal integrity | Current checkpoint marker is causal-integrity version 2. |
| State protocol | `docs/architecture/STATE_PROTOCOL.md` is the active architecture contract. |
| Internal dynamics outlook | `docs/architecture/INTERNAL_DYNAMICS_OUTLOOK.md` defines the V7+ north star. |
| Training entry | Use templates through `scripts/train_template.ps1` for normal runs. |

Key findings:

- V7 introduces typed internal dynamics, but mechanism claims still require
  clean ablation and state-usefulness evidence.
- V6 is no longer merely a speculative path, but its claims must come from clean causal runs only.
- Self-state hidden writes are now world-gated and logged.
- Self-state now consumes world-residual summaries instead of treating full hidden summaries as unconstrained self evidence.
- Router-bus contributions are decomposed into semantic/world/memory metrics instead of remaining an opaque control field.
- V7 state packets distinguish incoming readable state from outgoing updated
  state; current-segment outgoing state must not become a hidden leakage path.
- Current 1B-token strategy remains segmented continuation with non-replaying data flow, conservative checkpoint frequency, GPU-aware auto-batch probing, and deterministic random validation windows.

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
- `docs/architecture/INTERNAL_DYNAMICS_OUTLOOK.md` - V7+ internal dynamics and multimodal-state north star.
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

# Local V7 smoke run
.\scripts\train_template.ps1 -Template v7_local_smoke

# Generate text
.\scripts\infer.ps1 "<RUN_ROOT>\<run>\models\model_best.pt" "prompt text"
```

For remote 4090 training, follow `docs/REMOTE_4090_OPERATIONS.md` instead of launching visible foreground PowerShell windows.
