# Remote GPU Training Operations

This document defines the safe operating procedure for running NAIME training
on a shared remote Windows GPU machine. Keep this file machine-agnostic: do not
commit real host names, user names, IP addresses, passwords, private run names,
or private absolute paths.

## 1. Local Configuration

Remote-specific values belong in the ignored workspace config:

```powershell
Copy-Item configs\workspace.example.json configs\workspace.local.json
```

Required keys:

```text
remote.user
remote.host
remote.ssh
remote.root
remote.repo
remote.runs
remote.datasets
remote.venv
remote.python
```

Scripts may also read environment overrides such as:

```text
NAIME_REMOTE_HOST
NAIME_REMOTE_SSH
NAIME_REMOTE_ROOT
NAIME_REMOTE_RUNS
NAIME_REMOTE_DATASETS
NAIME_REMOTE_PYTHON
```

The remote virtual environment should be the same project environment used for
local compatibility checks. Do not silently switch Python versions or package
sets between local and remote runs.

## 2. Shared Machine Rules

- Check GPU and process ownership before every launch.
- Do not kill unknown processes.
- Do not use `taskkill` unless the PID is confirmed to belong to our run and
  graceful shutdown has failed.
- Do not leave visible PowerShell, CMD, or bash windows on the remote desktop.
- Launch training only through the detached launcher or remote control server.
- Prefer `STOP`-file graceful shutdown over keyboard interrupts.
- Keep checkpoint writes sparse enough to avoid I/O stalls.
- Treat the remote machine as shared infrastructure, not as an exclusive node.

## 3. Preflight Checklist

Run these checks before starting a large segment.

GPU snapshot:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
}
```

Training-related processes:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    Get-CimInstance Win32_Process |
      Where-Object {
        $_.CommandLine -like "*naime_hybrid.training.train*" -or
        $_.CommandLine -like "*naime_guardian*" -or
        $_.CommandLine -like "*NAIME_REMOTE*"
      } |
      Select-Object ProcessId,ParentProcessId,Name,CommandLine
}
```

Disk space:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    Get-PSDrive -PSProvider FileSystem |
      Select-Object Name,Used,Free,Root
}
```

A launch is considered safe only when:

```text
GPU memory is mostly free, or the other user's usage has been sampled.
No previous NAIME trainer is active unless this is a deliberate continuation.
The target run disk has enough free space for logs and checkpoints.
The resume checkpoint is the intended clean model.
```

## 4. Sync Before Launch

Before a serious remote run, sync code and verify the remote import path:

```powershell
.\scripts\sync_to_remote.ps1

.\scripts\ssh_cmd.ps1 -ScriptBlock {
    Set-Location $env:NAIME_REMOTE_REPO
    & $env:NAIME_REMOTE_PYTHON -m compileall src
}
```

If the remote environment variables are not configured, use the values from
`configs/workspace.local.json` or explicit script parameters.

## 5. Template-First Launch

Normal training must use `scripts/train_template.ps1`. Print the resolved
arguments first:

```powershell
.\scripts\train_template.ps1 `
  -Template v6_remote_100m_250m_segment `
  -RunName <RUN_NAME> `
  -Resume <REMOTE_MODEL_BEST_PT> `
  -ResumeLrPolicy reset `
  -TargetTokens 500000000 `
  -TargetTokensMode additional `
  -LearningRate 0.000004 `
  -WarmupSteps 2000 `
  -MinLrRatio 0.08 `
  -GradClip 0.5 `
  -VramFraction 0.95 `
  -AutoBatchMax 64 `
  -SaveEvery 10000 `
  -LatestEvery 5000 `
  -EvalEvery 5000 `
  -EvalMaxBatches 40 `
  -NumWorkers 4 `
  -PrintArgs
```

Check the printed arguments for:

```text
--target-tokens-mode additional
--target-tokens 500000000
--resume <intended model_best.pt>
--resume-lr-policy reset
--learning-rate 4E-06
--warmup-steps 2000
--min-lr-ratio 0.08
--vram-fraction 0.95
```

Do not start if any of these values are wrong.

## 6. Detached Background Launch

Use `scripts/launch_train_detached.py` on the remote machine. It starts
`naime_guardian.exe`, redirects output to files, and avoids visible windows.

Example local-to-remote launch pattern:

```powershell
$runName = "<RUN_NAME>"
$runDir = "<REMOTE_RUNS>/$runName"
$resume = "<REMOTE_RUNS>/<PREVIOUS_RUN>/models/model_best.pt"

$remote = @"
`$ErrorActionPreference='Stop'
Set-Location '<REMOTE_REPO>'
`$env:NAIME_HYBRID_PYTHON='<REMOTE_PYTHON>'
`$env:PYTHONPATH='<REMOTE_REPO>/src'
& '<REMOTE_PYTHON>' '<REMOTE_REPO>/scripts/launch_train_detached.py' `
  --repo '<REMOTE_REPO>' `
  --python '<REMOTE_PYTHON>' `
  --run-dir '$runDir' `
  -- `
  -Template v6_remote_100m_250m_segment `
  -RunName '$runName' `
  -Resume '$resume' `
  -ResumeLrPolicy reset `
  -TargetTokens 500000000 `
  -TargetTokensMode additional `
  -LearningRate 0.000004 `
  -WarmupSteps 2000 `
  -MinLrRatio 0.08 `
  -GradClip 0.5 `
  -VramFraction 0.95 `
  -AutoBatchMax 64 `
  -SaveEvery 10000 `
  -LatestEvery 5000 `
  -EvalEvery 5000 `
  -EvalMaxBatches 40 `
  -NumWorkers 4
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remote))
ssh <REMOTE_SSH> "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded"
```

The command should print a guardian PID and return. The training process must
continue after SSH disconnects.

## 7. Immediate Post-Launch Verification

Wait until auto-batch finishes, then inspect the run:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    $run = "<REMOTE_RUNS>\<RUN_NAME>"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    Get-CimInstance Win32_Process |
      Where-Object { $_.CommandLine -like "*<RUN_NAME>*" } |
      Select-Object ProcessId,ParentProcessId,Name,CommandLine
    Get-Content "$run\train.log" -Tail 80
    Get-Content "$run\trainer.stderr.log" -Tail 20
}
```

Expected log lines:

```text
primed DataLoader workers before CUDA init
auto-batch selected batch_size=<N>
target_tokens=<segment> mode=additional resume_step=<step>
resuming from <model_best.pt>
resume lr policy reset: overriding checkpoint base_lr with configured learning_rate=<lr>
resume lr policy reset: scheduler restarted after loading step=<step>
tr <step>/<max_steps> | lm ... | alpha ... | ent ... | grad ...
```

If `resume lr policy reset` is missing, stop and investigate before the run
consumes meaningful tokens.

## 8. VRAM Occupancy Policy

Use the following policy:

```text
GPU free and dedicated:       VramFraction 0.90-0.95
Shared but stable load:       VramFraction 0.70-0.80
Unknown or volatile load:     wait, sample, or use fixed conservative batch
```

`auto-batch` probes actual peak memory and predicts obviously unsafe batches.
High `VramFraction` is allowed only when the machine is truly idle. A few
hundred MiB of remaining free VRAM is normal at `0.95`, but it leaves almost no
room for another user to allocate memory.

If another process appears after launch, prefer graceful STOP and restart with
lower `VramFraction` rather than fighting for memory.

## 9. Learning Rate Policy

Segmented continuation should not inherit stale checkpoint scheduler state
unless explicitly intended. Use:

```text
ResumeLrPolicy reset
```

Current conservative V6 continuation defaults:

```text
segment size        500M additional tokens when GPU is free
learning rate       4e-6
warmup steps        2000
min lr ratio        0.08
grad clip           0.5
eval every          5000
eval batches        40
save every          10000
latest every        5000
best mode           model
```

Use lower LR if:

```text
bad_grad_window_count rises repeatedly;
grad_norm spikes above the normal band several times per segment;
validation loss plateaus or worsens for multiple evals;
state metrics jump at the same time as gradient spikes.
```

Do not resume from contaminated or legacy checkpoints for clean architecture
validation. Prefer the latest validated `models/model_best.pt`.

## 10. Monitoring

Incremental log watch:

```powershell
.\scripts\watch_remote.ps1 -RunName <RUN_NAME>
```

Snapshot:

```powershell
.\scripts\watch_remote.ps1 -RunName <RUN_NAME> -TailLines 80 -Follow:$false
```

Direct GPU check:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv
}
```

Key metrics to watch:

```text
lm / val_lm
ppl / val_ppl
grad_norm
bad_grad_window_count
lr and lr_safety_factor
alpha_mean
router_entropy
v5_router_world_ratio
v6_boundary_self/world/other/unknown
v6_reflection_norm
tokens_per_second
```

Healthy V6 continuation generally shows:

```text
alpha_mean around the configured sparsity band, not collapsing high;
router_entropy not collapsing toward zero;
grad_norm mostly stable with rare skipped spikes;
self/world boundary not dominated by self alone;
validation improving across the segment, even if slowly.
```

## 11. Graceful Stop

Create a `STOP` file in the run directory:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    New-Item -ItemType File -Force "<REMOTE_RUNS>\<RUN_NAME>\STOP" | Out-Null
}
```

Expected behavior:

```text
trainer observes STOP;
current optimizer step completes;
latest/stopped checkpoints are written;
metrics.csv is exported;
guardian exits with code 0.
```

Only after this fails should you consider terminating the known guardian/trainer
PIDs manually.

## 12. Run Artifacts

Each remote run should contain:

```text
config.json
train_args.txt
wrapper_args.txt
launch_cmd.txt
daemon.pid
guardian.log
trainer.pid
train.log
trainer.stdout.log
trainer.stderr.log
metrics.jsonl
metrics.csv
latest.pt
models/model_best.pt
models/model_latest.pt
```

`trainer.stderr.log` should normally be empty except for harmless PowerShell
CLIXML noise from startup. A Python traceback or CUDA error there requires
triage.

## 13. Failure Triage

If training stops unexpectedly:

```text
1. Check guardian.log for trainer exit code.
2. Check train.log tail for STOP, traceback, bad-gradient reload, or OOM.
3. Check trainer.stderr.log.
4. Check metrics.jsonl for the last successful step and LR.
5. Check GPU/process state with nvidia-smi and Win32_Process.
6. Check Windows event logs only if Python disappears without traceback.
7. Confirm latest.pt or stopped.pt exists before restarting.
```

Do not immediately reuse a failed checkpoint. Prefer:

```text
models/model_best.pt
latest.pt only if it is known clean and complete
stopped.pt after graceful STOP
```

## 14. Cleanup

After a completed or abandoned run:

```powershell
.\scripts\clean_checkpoints.ps1 -RunsRoot <REMOTE_RUNS> -KeepLastN 2
```

Keep:

```text
models/model_best.pt
models/model_latest.pt
latest.pt for the active run
metrics.jsonl
metrics.csv
train.log
config.json
```

Remove obsolete smoke runs, duplicate failed attempts, and old full
checkpoints only after confirming the best model for the stage has been
preserved.
