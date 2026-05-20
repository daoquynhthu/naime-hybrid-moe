# Training Templates

Use templates for normal training. The raw `train.py` CLI and the legacy
`train_model.ps1` parameter surface are now treated as expert-level interfaces.

Templates live in:

```text
configs/training_templates/
```

List templates:

```powershell
.\scripts\train_template.ps1 -List
```

Print the resolved command without launching training:

```powershell
.\scripts\train_template.ps1 -Template v6_local_smoke -PrintArgs
```

Start a local smoke run:

```powershell
.\scripts\train_template.ps1 -Template v6_local_smoke
```

Start a local 64M-class architecture probe:

```powershell
.\scripts\train_template.ps1 -Template v6_local_64m_probe
```

Prepare a remote 100M-class 250M-token segment command:

```powershell
.\scripts\train_template.ps1 -Template v6_remote_100m_250m_segment -PrintArgs
```

Continue from a clean checkpoint:

```powershell
.\scripts\train_template.ps1 `
  -Template v6_remote_100m_continue_100m `
  -Resume "<clean checkpoint path>"
```

## Safe Overrides

Prefer only these overrides during normal use:

- `-RunName`
- `-DataPath`
- `-OutputDir`
- `-Resume`
- `-TargetTokens`
- `-TargetTokensMode`
- `-LearningRate`
- `-WarmupSteps`
- `-MinLrRatio`
- `-GradClip`
- `-VramFraction`
- `-BatchSize`
- `-NoAutoBatch`
- `-EvalEvery`
- `-SaveEvery`
- `-LatestEvery`

If a run needs many more overrides, create a new template instead of manually
assembling a long command.

## Template Rules

- A template is the source of truth. Template launches disable hidden adaptive
  defaults in `train_model.ps1`.
- Workspace-specific paths should use variables such as `${local.run_root}` or
  `${remote.datasets}` from `configs/workspace.local.json`.
- Keep templates conservative. Architecture changes should happen in code or in
  a new named template, not in one-off command tails.
- Never set `AllowLegacyResume` in a clean training template.
