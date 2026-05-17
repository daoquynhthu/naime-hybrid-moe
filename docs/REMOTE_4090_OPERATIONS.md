# Remote GPU Operations

This document records the safe operating procedure for a shared remote GPU
machine. Do not store real host names, user names, IP addresses, passwords, or
private paths in tracked documentation.

## Configuration

Copy the template and fill local values:

```powershell
Copy-Item configs\workspace.example.json configs\workspace.local.json
```

`configs/workspace.local.json` is ignored by git. Relevant keys:

- `remote.user`
- `remote.host`
- `remote.ssh`
- `remote.root`
- `remote.repo`
- `remote.runs`
- `remote.datasets`
- `remote.venv`
- `remote.python`

Most remote scripts also accept explicit parameters and environment variable
overrides such as `NAIME_REMOTE_HOST`, `NAIME_REMOTE_SSH`, `NAIME_REMOTE_ROOT`,
and `NAIME_REMOTE_RUNS`.

## Shared Machine Rules

- Do not kill unknown processes.
- Do not use `taskkill` unless the PID is confirmed to belong to our run and
  graceful shutdown failed.
- Do not leave visible PowerShell windows on the remote desktop.
- Always check GPU users before launch.
- If another training process is active, sample its GPU use over a short window
  before choosing batch size.
- Keep checkpoints infrequent enough to avoid disk and I/O contention.

## Safe Launch Policy

Remote training must be launched as a hidden/background process through the
remote control scripts. Avoid foreground SSH commands that tie process lifetime
to the SSH session or open a visible server-side window.

Preferred local workflow:

```powershell
.\scripts\sync_to_remote.ps1
.\scripts\remote_ctl.py launch --run-name <RUN_NAME> -- <train args>
```

Expected launcher/run files:

- `launcher.stdout.log`
- `launcher.stderr.log`
- `daemon.pid` or equivalent PID marker
- `train.log`
- `metrics.jsonl`
- `metrics.csv`

## Graceful Stop

Training watches for a `STOP` file in the run directory. Create that file
instead of sending repeated keyboard interrupts:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    New-Item -ItemType File -Force "$env:NAIME_REMOTE_RUNS\<RUN_NAME>\STOP" | Out-Null
}
```

The trainer finishes the current optimizer step, saves stable artifacts,
exports metrics, and exits cleanly.

## Monitoring

Use the project watch script for live logs:

```powershell
.\scripts\watch_remote.ps1 -RunName <RUN_NAME>
```

Snapshot without follow:

```powershell
.\scripts\watch_remote.ps1 -RunName <RUN_NAME> -TailLines 40 -Follow:$false
```

GPU status:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv
}
```

## Current Large-Run Baseline

The current large-run policy is intentionally machine-agnostic:

```text
resume          previous validated model_best.pt
target mode     additional
segment size    100M tokens when GPU is available
vram fraction   0.80
learning rate   2.5e-5
warmup steps    500
min lr ratio    0.03
grad clip       0.8
eval every      5000
save every      10000
latest every    5000
```

Use conservative batch sizing when another GPU process is present. When the GPU
is free, use auto-batch and check that the selected batch leaves headroom.

## Failure Triage

If a run stops unexpectedly:

- check `launcher.stderr.log` and `train.log`;
- check `metrics.jsonl` for the last successful step;
- check OS event logs only if Python exited without a handled exception;
- inspect GPU contention history if another process was active;
- avoid restarting aggressively until checkpoint artifacts are confirmed.
