# Training

Date: 2026-05-25

## Entry Point

Use `scripts/train_template.ps1` for normal experiments. It expands a named JSON
template into the full training command and keeps the long argument surface out
of day-to-day work:

```powershell
.\scripts\train_template.ps1 -List
.\scripts\train_template.ps1 -Template v7_local_smoke
.\scripts\train_template.ps1 -Template v7_local_smoke -PrintArgs
```

See `docs/TRAINING_TEMPLATES.md` for template rules and safe overrides.

`scripts/train_model.ps1` and `python -m naime_hybrid.training.train` remain
expert-level interfaces for debugging and one-off research runs.

Supported daily architecture IDs:

```text
dense
token_moe
naime_state_moe
naime_v4_state_moe
naime_v5_world_state_moe
naime_v6_recursive_self_moe
naime_v7_typed_dynamics
```

Older V1/V2/V3 and V4.1/V4.2 aliases are legacy/forensic only. Do not launch
them through `scripts/train_model.ps1`; use raw Python only if deliberately
reproducing an old run.

All model and training parameters should pass through CLI/config. Avoid hard-coding experiment-specific values in Python.

## V7 Training (Experimental Recommended Path)

V7 is the active experimental path. The current large-run policy is segmented
continuation on the prebuilt FineWeb-Edu 1B ctx1024 corpus, with clean lineage
and the incoming/outgoing state protocol preserved.

Local quick probe:

```powershell
.\scripts\train_template.ps1 -Template v7_local_smoke -RunName v7_local_probe
```

Remote 4090 runs should be launched through the hidden/background remote workflow documented in `docs/REMOTE_4090_OPERATIONS.md`, not from a visible foreground PowerShell window.

Current remote continuation baseline:

```text
model              naime_v7_typed_dynamics
dataset            <REMOTE_DATASETS>\fineweb_edu_1b_ctx1024
resume checkpoint  previous validated models\model_best.pt
target mode        additional
segment size       500M tokens when the GPU is free
vram fraction      0.90-0.95 when dedicated, 0.70-0.80 when shared
learning rate      4e-6 for conservative continuation
warmup steps       2000
min lr ratio       0.08
grad clip          0.5
eval every         5000
eval batches       40
eval sampling      random
save every         10000
latest every       5000
best mode          model
```

V6 remains the mature stateful baseline and should be kept runnable for fair
comparison. V7 claims require mechanism probes such as state swap, state erase,
and dynamics gain; lower loss alone is not enough.

V7 timescales are not yet scientific priors. Clean templates keep
`latent/world/self = 1/1/1`; use `scripts/run_v7_timescale_ablation.ps1` when we
need to measure whether one state family should move faster or slower.

V7 typed dynamics should be tested with causal state chunks enabled. Without
`--v7-state-chunk-size`, a no-past single chunk can only write outgoing state,
so `v7_dynamics_gain` may correctly remain zero even when state motion exists.
With causal chunks, earlier chunks may write state that later chunks read,
preserving causal direction while giving the dynamics path an LM training signal.

Use smaller fixed batches only when another GPU job is active. When the GPU is
free, prefer auto-batch with conservative prediction/headroom. Serious remote
runs should first print the resolved template arguments, then launch through
`scripts/launch_train_detached.py` so the process survives SSH disconnects and
does not create a visible remote window.

## LR Schedule

The scheduler supports several resume policies:

- `checkpoint`: keep the scheduler state loaded from a full checkpoint.
- `absolute`: align scheduler step to the resumed global step.
- `progress`: remap progress when changing total step budget.
- `reset`: restart the LR schedule after loading weights/optimizer state.

For segmented continuation from model-only checkpoints, use an explicit low LR and short warmup. Long single schedules have produced late gradient instability, so prefer shorter continuation segments instead of one huge uninterrupted schedule.

Bad-gradient protection now adds a runtime safety layer:

- isolated non-finite or very large gradients are skipped;
- a rolling bad-gradient window is logged as `bad_grad_window_count`;
- clustered bad gradients reduce `lr_safety_factor` without changing the scheduler shape;
- repeated bad gradients can reload the last stable checkpoint and continue with a lower effective LR.

## Run Directory

Each run writes to `--output-dir\<run_name>\` (default: `experiments\runs\<run_name>` locally).

Files:

- `config.json`: full training/model config.
- `train.log`: persistent console log.
- `metrics.jsonl`: one JSON metrics row per step/eval.
- `metrics.csv`: CSV export generated during and at the end of training.
- `latest.pt`: latest resumable full checkpoint.
- `step_XXXXXXXX.pt`: periodic full snapshots when enabled.
- `models/model_best.pt`: best model-only weights.
- `models/model_latest.pt`: latest model-only weights.
- `interrupted.pt` / `model_interrupted.pt`: saved on Ctrl+C or STOP.
- `failed.pt` / `model_failed.pt`: saved on exception.

New runs should keep model-only weights in `models\` and full checkpoints at the run root.

## Resume And Stop

Default resume mode:

```text
--resume none
```

Resume is intentionally opt-in. A clean checkpoint must carry
`causal_integrity_version >= 2`, which marks it as produced after the current
causal semantic/state path fixes. Checkpoints without this marker are refused by
default because older runs may include non-causal leakage or other contaminated
training paths.

Use `--allow-legacy-resume` only for forensic analysis or deliberately
contaminated baselines. Do not use it for clean architecture validation or
future foundation runs.

If `--resume auto` is explicitly requested, stable auto-resume priority is:

```text
latest.pt -> interrupted.pt -> best.pt -> model_latest.pt -> model_interrupted.pt -> model_best.pt
```

`failed.pt` is not used by `--resume auto` unless `--resume-allow-failed` is explicitly set.

For additional non-replaying training:

```powershell
--target-tokens 100000000 --target-tokens-mode additional
```

Training uses a resumable shuffled sampler. Check for a log line like:

```text
train sampler resumed stream seed=1234 resume_step=<step> offset_batches=<offset>/<epoch_batches>
```

If this line is missing during segmented continuation, treat token-accounting as suspect.

To stop safely, create `STOP` in the run directory. The trainer finishes the current optimizer step, saves stable artifacts, writes `metrics.csv`, and exits.

## Logged Metrics

Core:

- `loss_total`, `loss_lm`, `ppl`, `lr`, `grad_norm`, `tokens`, `tok/s`
- `lr_safety_factor`, `bad_grad_window_count`

MoE / router:

- `router_entropy`, `semantic_prior_entropy`, `alpha_mean`, `alpha_*`
- `dispatch_dense`
- `lambda_sparse_effective`, `lambda_kl_effective`

V5 world state:

- `v5_slot_*`
- `v5_state_pred`
- `gate_mix_alpha_weight`, `gate_mix_clean_weight`, `gate_mix_state_weight`

V6 recursive self-state:

- `v6_self_pred`
- `v6_slot_cosine`
- `v6_slot_context_cosine`
- `v6_boundary_self`, `v6_boundary_world`, `v6_boundary_other`, `v6_boundary_unknown`
- `v6_reflection_norm`

V7 typed dynamics:

- `v7_latent_delta`, `v7_world_delta`, `v7_self_delta`, `v7_controller_delta`
- `v7_hidden_write_ratio`, `v7_latent_read_entropy`, `v7_latent_read_max`
- `v7_*_timescale`, `v7_effective_*_write_scale`
- `v7_dynamic_depth_mean`, `v7_dynamic_halt_fraction`
- validation probes: `val_v7_dynamics_gain_lm`, `val_v7_state_swap_delta_lm`, `val_v7_*_erase_delta_lm`

## Validation Sampling

By default validation now uses a deterministic random window when
`--eval-max-batches > 0`:

```powershell
--eval-sampling random --eval-seed 4321
```

This avoids repeatedly measuring only the prefix of the validation split. Use
`--eval-sampling sequential` only when comparing against older runs that used
legacy prefix validation. Use `--eval-max-batches 0` for full sequential
validation.

## Robustness Features

- full checkpoints include model, optimizer, scheduler, AMP scaler, config, metrics, and RNG state;
- model-only weights are saved separately under `models\`;
- checkpoint writes use temporary files and replacement;
- checkpoint frequency is intentionally conservative to reduce I/O stalls;
- async checkpoint writer is available where safe;
- non-finite loss and bad-gradient detection skip unsafe updates;
- adaptive LR safety factor responds to clustered gradient spikes;
- Ctrl+C and STOP request graceful checkpoint saving;
- console output is compact, while full logs remain persisted.

## `torch.compile` Optimization & Safety

Using `torch.compile` (`-CompileModel`) is a powerful throughput optimization that fuses operators and minimizes GPU pipeline stalls. It is highly integrated with the NAIME Hybrid MoE V6 training engine but must be configured carefully depending on your platform.

### Compilation Parameters

When launching training via `scripts/train_template.ps1` or `scripts/train_model.ps1`, you can configure compilation using the following parameters:

- `-CompileModel`: Switch to enable compilation (mapped to `"CompileModel": true` in templates).
- `-CompileScope <full | dense>`: Specifies the extent of model compilation.
  - **`full` (Default)**: Compiles the entire model graph, including standard attention layers, the MoE routing path, and recursive self/world state structures.
  - **`dense` (Recommended for Windows / Local debug)**: Compiles standard dense Transformer blocks using PyTorch Inductor (which yields over 90% of standard operator fusion speedup) while keeping complex recursive MoE dispatcher and state networks eager.
- `-CompileBackend <inductor | eager | aot_eager>`: JIT compilation backend. Defaults to `inductor`.

---

### Platform Constraints & Windows Stack Workaround

> [!WARNING]
> **Windows JIT Stack Crash**: Under `--compile-scope full`, compiling the highly recursive MoE + multi-state control-flow graphs triggers a native C-stack overflow inside Windows CPython (`python312.dll`, Access Violation `0xC0000005`) due to the default 1MB thread stack size limit on Windows (Linux defaults to 8MB, which compiles without issues).

If training on **Windows/Local environment**:
- **Always set `-CompileScope dense`**.
- This perfectly bypasses the Windows CPython thread stack limit by keeping the recursive structures eager, while still optimizing the heavy compute blocks (standard multihead attention and feedforward layers).
- Local training on an RTX 5060 Laptop GPU under `dense` compile scope hits an exceptional **`~2500 tokens/second`** throughput.

If training on **Linux/Remote clusters**:
- Both `dense` and `full` scopes are fully supported due to Linux's larger default stack size.
- Note that even on Linux, **`dense` scope remains highly recommended** since it reduces JIT compilation startup time from several minutes to just ~30 seconds, avoids recompilation storms when dynamic routing shapes shift, and provides nearly identical final throughput (< 2% speed difference).

---

### Safe Compilation Checklist

1. **Checkpoints**: Checkpoints are automatically saved from the unwrapped, eager model so state dict keys do not acquire the `_orig_mod.` prefix. Compiled and eager checkpoints can be seamlessly loaded across each other.
2. **MoE Dispatch**: Stochastic semantic gates and sparse MoE dispatch are kept eager inside compiler boundaries to reduce graph breaks.
3. **Smoke Probing**: Never launch a multi-million token run with compilation enabled without first verifying a 5-to-10 step local smoke test to confirm:
   - Normal learning rate and finite losses (no NaNs or divergence).
   - Expected `tok/s` throughput improvements (standard first step JIT compilation takes 1-2 minutes; subsequent steps must run at full speed).
   - Perfect validation matching against the eager baseline.

## Data Preparation

For large raw corpora, especially 300GB+ multi-source builds, follow the full
data engineering requirements in [DATA_PIPELINE_SPEC.md](DATA_PIPELINE_SPEC.md).
The commands below are convenience wrappers for already-defined corpus builds;
they are not a substitute for source manifests, deduplication, data cards, and
pre-training quality gates.

```powershell
# 1B-token FineWeb-Edu corpus (ctx1024, GPT-2 tokenized, HF disk format)
.\scripts\prepare_fineweb_edu_1b.ps1 -Output <LOCAL_FINEWEB_1B>

# Small 50M corpus for quick experiments
.\scripts\prepare_fineweb_edu_1b.ps1 -Output <LOCAL_FINEWEB_50M> -TrainTokens 50000000 -BlockSize 513
```

Parameters for `prepare_fineweb_edu_1b.ps1`:

| Param | Default | Description |
|-------|---------|-------------|
| `-Output` | from `configs/workspace.local.json` | Output directory |
| `-TrainTokens` | `1000000000` | Target training tokens |
| `-ValidationTokens` | `10000000` | Validation tokens |
| `-BlockSize` | `1025` | Use seq_len + 1 for causal shift |
| `-TokenizerPath` | `data\naime\gpt2` | Local GPT-2 tokenizer |
| `-DatasetName` | `HuggingFaceFW/fineweb-edu` | HF dataset |
| `-DatasetConfig` | `sample-10BT` | Dataset config variant |
| `-MinScore` | `3.0` | Minimum FineWeb-Edu quality score |
| `-MinTextChars` | `256` | Minimum document length |

The 1B corpus should be prepared before large training. Do not rely on downloading/tokenizing inside the training process.

## Performance Notes

- `torch.compile` can improve throughput but increases first-step compilation cost.
- Async prefetch overlaps CPU-to-GPU transfer with GPU compute.
- `collate_fn` performs batch causal shift in DataLoader workers.
- `persistent_workers` and `prefetch_factor` reduce DataLoader churn.
- `HFDiskCausalDataset.set_format(type="torch")` reduces Python overhead.
- `auto-batch` probes VRAM and now avoids obviously doomed higher batches by prediction.
