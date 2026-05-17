# Scale-Up Pilot

## Purpose

This is the bridge between the successful small architecture validation and any future large-scale run.

The goal is not to chase a final model yet. The goal is to answer:

Does NAIME-State MoE still beat a token-only MoE baseline when width, depth, and context length are all increased?

## Default Configuration

Use the generic model launcher:

```powershell
.\scripts\train_model.ps1 -Model token_moe -RunName scale_token_moe_pilot_v1
.\scripts\train_model.ps1 -Model naime_v1 -RunName scale_naime_state_moe_pilot_v1
```

Optionally run a dense Transformer baseline:

```powershell
.\scripts\train_model.ps1 -Model dense -RunName scale_dense_pilot_v1
```

## Model Size Step-Up

Compared with Phase 1/2:

- `d_model`: `256 -> 384`
- `n_layers`: `6 -> 8`
- `seq_len`: `256 -> 512`
- `d_ff`: `1024 -> 1536`
- MoE expert hidden size: `512 -> 768`
- NAIME `z_dim`: `64 -> 96`
- NAIME compressor: `stride=16`, `window=24`

This is deliberately a medium pilot, not a full large-scale training job.

## Training Defaults

- `TargetTokens`: `12,288,000`
- auto-batch enabled
- `VramFraction`: `0.90`
- periodic validation every `100` steps
- `best.pt` and `models/model_best.pt` are saved by validation loss

## Decision Rule

Proceed toward larger training only if:

- NAIME `best.pt` beats token MoE `best.pt` on validation PPL.
- NAIME does not win only by over-opening semantic gates.
- Router entropy remains non-collapsed.
- The best checkpoint appears before severe late-run degradation.

If NAIME does not beat token MoE at this scale, pause architecture work and inspect:

- semantic gate schedule;
- KL pressure;
- expert routing distribution;
- whether `seq_len=512` exposes data scarcity rather than architecture weakness.

## Evaluation Commands

After training:

```powershell
.\scripts\eval.ps1 --run-dir experiments\runs\scale_token_moe_pilot_v1 --checkpoint best.pt --data-split validation --max-batches 0
.\scripts\eval.ps1 --run-dir experiments\runs\scale_naime_state_moe_pilot_v1 --checkpoint best.pt --data-split validation --max-batches 0
```
