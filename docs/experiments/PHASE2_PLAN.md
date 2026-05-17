# Phase 2 Plan: Adaptive NAIME-State MoE

## Goal

Phase 1 showed that NAIME-State MoE can beat dense and token-only MoE baselines under a comparable token budget. Phase 2 should test whether that gain survives a less hand-tuned setting.

The main question is:

Can semantic-state routing stay useful when the semantic gate is controlled by feedback instead of manual sparse-weight tuning?

## What Changes

### 1. Adaptive sparse control

Training now supports:

- `--adaptive-sparse-control`
- `--sparse-control-gain`
- `--sparse-control-ema`
- `--lambda-sparse-min`
- `--lambda-sparse-max`

The controller tracks `alpha_mean` with an EMA. If semantic writes are above `target_sparsity`, it increases the effective sparse penalty. If writes are below target, it relaxes the penalty.

This avoids treating one small-run value of `lambda_sparse` as an architectural truth.

### 2. Gate eval alignment

The semantic gate now defaults to `gate_eval_mode=prob`. Validation and inference use sigmoid probabilities as the semantic write strength instead of hard `logits > 0` thresholding.

This avoids the Phase 2 failure mode where training-time Gumbel gates hovered near target sparsity, but eval-time hard gates collapsed the semantic path.

### 3. Formal evaluation

Evaluation now has a dedicated command:

```powershell
.\scripts\eval.ps1 --run-dir experiments\runs\<run_name> --data-split validation --max-batches 0
```

It writes `<split>_summary.json` into the run directory and reports:

- validation loss and perplexity;
- semantic gate activity;
- router entropy;
- KL, load-balance, and sparse penalties;
- evaluation throughput when CUDA timing is available.

### 4. Best checkpointing

Training supports periodic validation:

- `--eval-every`
- `--eval-split`
- `--eval-max-batches`

When enabled, training saves:

- `best.pt`
- `models/model_best.pt`

These are selected by validation LM loss, so an over-controlled final checkpoint cannot silently replace the best model.

## First Phase 2 Run

```powershell
cd <PROJECT_ROOT>
.\scripts\train_model.ps1 -Model naime_v1 -RunName phase2_naime_adaptive_v1 -TargetTokens 6144000 -SeqLen 256 -DModel 256 -Layers 6 -Dff 1024 -ExpertHidden 512 -Stride 8 -Window 12 -ZDim 64
.\scripts\eval.ps1 --run-dir experiments\runs\phase2_naime_adaptive_v1 --data-split validation --max-batches 0
```

Default token budget is `6,144,000`, twice the Phase 1 fair budget.

## Success Criteria

- Validation PPL remains clearly below the Phase 1 fair dense/token MoE baselines.
- `alpha_mean` trends closer to `target_sparsity` than the Phase 1 fair run.
- Router entropy stays non-collapsed.
- KL stays nonzero but does not explode.
- Auto-batch and checkpointing remain stable.
- `best.pt` beats or matches the final checkpoint if late training degrades.

## Do Not Over-Interpret Yet

If adaptive control improves PPL or sparsity on this small setup, it is promising but not proof of scale. The next serious test after this is a larger-context or larger-model run with the same controller unchanged.
