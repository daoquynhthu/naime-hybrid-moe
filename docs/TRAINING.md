# Training

Date: 2026-05-17

## Entry Point

Use `scripts/train_model.ps1` for normal experiments:

```powershell
.\scripts\train_model.ps1 -Model <name> -RunName <name> -DataPath <path> [args]
```

Available architecture IDs:

```text
dense
token_moe
naime_state_moe
naime_v4_state_moe
naime_v41_state_moe
naime_v42_state_moe
naime_v5_world_state_moe
naime_v6_recursive_self_moe
```

All model and training parameters should pass through CLI/config. Avoid hard-coding experiment-specific values in Python.

## V6 Training (Recommended)

V6 is the active path. The current large-run policy is segmented continuation on the prebuilt FineWeb-Edu 1B ctx1024 corpus.

Local quick probe:

```powershell
.\scripts\train_model.ps1 -Model naime_v6_recursive_self_moe -RunName v6_local_probe -DataPath <LOCAL_FINEWEB_50M> -TargetTokens 3000000 -EvalEvery 500 -SaveEvery 5000 -LatestEvery 2500
```

Remote 4090 runs should be launched through the hidden/background remote workflow documented in `docs/REMOTE_4090_OPERATIONS.md`, not from a visible foreground PowerShell window.

Current remote continuation baseline:

```text
model              naime_v6_recursive_self_moe
dataset            <REMOTE_DATASETS>\fineweb_edu_1b_ctx1024
resume checkpoint  previous validated models\model_best.pt
target mode        additional
segment size       100M tokens when the GPU is available
vram fraction      0.80
learning rate      2.5e-5
warmup steps       500
min lr ratio       0.03
grad clip          0.8
eval every         5000
eval batches       40
save every         10000
latest every       5000
best mode          model
```

Use smaller fixed batches only when another GPU job is active. When the GPU is free, prefer auto-batch with conservative prediction/headroom.

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

Legacy checkpoints are still loadable, but new runs should keep model-only weights in `models\` and full checkpoints at the run root.

## Resume And Stop

Default resume mode:

```text
--resume auto
```

Stable auto-resume priority:

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

## Data Preparation

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
