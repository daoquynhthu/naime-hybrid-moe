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
.\scripts\train_template.ps1 -Template v7_local_smoke -PrintArgs
```

Start a local smoke run:

```powershell
.\scripts\train_template.ps1 -Template v7_local_smoke
```

Start a local V7 64M-class architecture probe when available:

```powershell
.\scripts\train_template.ps1 -Template v7_remote_64m_probe -PrintArgs
```

Prepare a remote V7 full-data command:

```powershell
.\scripts\train_template.ps1 -Template v7_remote_508m_fineweb_full -PrintArgs
```

Continue from a clean checkpoint:

```powershell
.\scripts\train_template.ps1 `
  -Template v7_remote_508m_fineweb_full `
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

## Ablation Harnesses

Use dedicated ablation scripts when the question is architectural rather than
"start the next normal run." These scripts keep run naming, logs, metrics, and
summary CSVs consistent.

V7 timescale measurement:

```powershell
.\scripts\run_v7_timescale_ablation.ps1 -PrintOnly
```

This harness varies `V7LatentTimescale`, `V7WorldTimescale`, and
`V7SelfTimescale`. The default architecture templates keep all three at `1.0`;
non-uniform rates must be earned by the ablation results, not assumed.

## Template Rules

- A template is the source of truth. Template launches disable hidden adaptive
  defaults in `train_model.ps1`.
- Workspace-specific paths should use variables such as `${local.run_root}` or
  `${remote.datasets}` from `configs/workspace.local.json`.
- Keep templates conservative. Architecture changes should happen in code or in
  a new named template, not in one-off command tails.
- Never set `AllowLegacyResume` in a clean training template.
- V7 templates must preserve the incoming/outgoing state protocol. Do not add
  overrides that turn outgoing same-segment state into current hidden readout
  unless the architecture ID and causal tests are updated.
