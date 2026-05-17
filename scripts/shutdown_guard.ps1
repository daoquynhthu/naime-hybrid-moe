<#
.SYNOPSIS
Gracefully stop NAIME training before the OS kills it during shutdown/logoff.

.DESCRIPTION
Signals the training run via a STOP file, then waits for the guardian/trainer
to exit cleanly after checkpointing. This script deliberately never force-kills
the trainer; if the OS or an administrator hard-kills processes, user-space
scripts cannot guarantee another save opportunity.

Register (admin PowerShell):
  gpedit.msc → Computer Config → Windows Settings → Scripts → Shutdown

.INPUTS
  -RunName   Training run name (matches the run directory under RemoteRunRoot)
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$RunName,

    [string]$RunRoot = "",
    [string]$WorkspaceConfig = ""
)

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $RunRoot) { $RunRoot = Resolve-NaimeConfigValue $workspace "remote.runs" "NAIME_REMOTE_RUNS" }
$runDir = Join-Path $RunRoot $RunName

if (-not (Test-Path $runDir)) {
    Write-Warning "Run directory not found: $runDir"
    exit 0
}

$pidFile = Join-Path $runDir "daemon.pid"

if (-not (Test-Path $pidFile)) {
    Write-Warning "No daemon.pid in $runDir; creating STOP file anyway"
    "" | Out-File -FilePath (Join-Path $runDir "STOP") -Encoding utf8 -NoNewline
    exit 0
}

$pid = Get-Content $pidFile -Raw
$pid = $pid.Trim()

try {
    $process = Get-Process -Id $pid -ErrorAction Stop
    Write-Host "Training process found: PID=$pid Name=$($process.ProcessName) StartTime=$($process.StartTime)"
} catch {
    Write-Warning "Process $pid not running; skipping"
    exit 0
}

$stopFile = Join-Path $runDir "STOP"
"" | Out-File -FilePath $stopFile -Encoding utf8 -NoNewline
Write-Host "STOP file created: $stopFile"

$trainerPidFile = Join-Path $runDir "trainer.pid"
$pidsToWait = @($pid)
if (Test-Path $trainerPidFile) {
    $tpid = (Get-Content $trainerPidFile -Raw).Trim()
    try {
        $tproc = Get-Process -Id $tpid -ErrorAction Stop
        $pidsToWait += $tpid
        Write-Host "Trainer process found: PID=$tpid"
    } catch {}
}

$timeout = 30
$deadline = (Get-Date).AddSeconds($timeout)

while ((Get-Date) -lt $deadline) {
    $allDead = $true
    foreach ($waitPid in $pidsToWait) {
        try {
            $proc = Get-Process -Id $waitPid -ErrorAction Stop
            $allDead = $false
        } catch {}
    }
    if ($allDead) {
        Write-Host "All processes exited gracefully"
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Warning "Some processes still alive after ${timeout}s; leaving them to finish checkpointing"
foreach ($waitPid in $pidsToWait) {
    try {
        $proc = Get-Process -Id $waitPid -ErrorAction Stop
        Write-Host "Still running: PID=$waitPid Name=$($proc.ProcessName)"
    } catch {
        Write-Host "Process $waitPid already exited"
    }
}
exit 0
