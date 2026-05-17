# NAIME 3.0: Semantic Operating Router

NAIME v3 adds several architecture switches at once while keeping v1/v2 reproducible.

## Added Switches

- `--semantic-scales local|local_mid|local_mid_global`
- `--mid-stride`
- `--mid-window`
- `--global-semantic`
- `--semantic-fusion local|gated_sum|concat`
- `--semantic-residual-write`
- `--semantic-write-scale`
- `--semantic-pred-horizon`
- `--lambda-semantic-pred`

## Presets

### naime_v3_safe

```powershell
.\scripts\train_model.ps1 -Model naime_v3_safe -RunName naime_v3_safe_v1
```

Uses:

- hybrid semantic router;
- local + mid semantic states;
- gated-sum semantic fusion;
- small residual semantic write;
- light next-block semantic prediction loss.

### naime_v3_aggressive

```powershell
.\scripts\train_model.ps1 -Model naime_v3_aggressive -RunName naime_v3_aggressive_v1
```

Uses:

- hybrid semantic router;
- local + mid + global semantic states;
- concat semantic fusion;
- stronger residual semantic write;
- larger semantic prediction loss.

## Comparison

Recommended first run:

```powershell
.\scripts\train_model.ps1 -Model naime_v2 -RunName naime_v2_reference_v1
.\scripts\train_model.ps1 -Model naime_v3_safe -RunName naime_v3_safe_v1
.\scripts\train_model.ps1 -Model naime_v3_aggressive -RunName naime_v3_aggressive_v1
```

Use `best.pt` for comparison.
