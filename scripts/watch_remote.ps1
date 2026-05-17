param(
    [string]$RemoteHost = "",
    [string]$RemoteRunRoot = "",
    [string]$WorkspaceConfig = "",

    [Parameter(Mandatory = $true)]
    [string]$RunName,

    [ValidateSet("train", "launcher", "stderr")]
    [string]$Log = "train",

    [int]$TailLines = 50,
    [int]$PollSeconds = 5,
    [switch]$Follow = $true,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $RemoteHost) { $RemoteHost = Resolve-NaimeConfigValue $workspace "remote.ssh" "NAIME_REMOTE_SSH" }
if (-not $RemoteRunRoot) { $RemoteRunRoot = Resolve-NaimeConfigValue $workspace "remote.runs" "NAIME_REMOTE_RUNS" }
$runDir = "$RemoteRunRoot\$RunName"

if ($Check) {
    Write-Host "=== Checking daemon status: $RunName ===" -ForegroundColor Cyan

    $pidCommand = "if (Test-Path '$runDir\daemon.pid') { Get-Content '$runDir\daemon.pid' -Raw } else { echo 'NO_PID_FILE' }"
    $pidResult = ssh $RemoteHost "powershell -NoProfile -Command `"$pidCommand`""
    $pidResult = $pidResult.Trim()

    if ($pidResult -and $pidResult -ne "NO_PID_FILE") {
        $daemonPid = $pidResult.Trim()
        Write-Host "  PID file found: $daemonPid" -ForegroundColor Yellow

        $procCommand = "`$p = Get-Process -Id $daemonPid -ErrorAction SilentlyContinue; if (`$p) { Write-Host ('RUNNING|' + `$p.StartTime.ToString('yyyy-MM-dd HH:mm:ss')) } else { Write-Host 'NOT_RUNNING' }"
        $procInfo = ssh $RemoteHost "powershell -NoProfile -Command `"$procCommand`""
        if ($procInfo -match "RUNNING\|(.+)") {
            Write-Host "  Daemon is RUNNING" -ForegroundColor Green
            Write-Host "  Started: $($matches[1])" -ForegroundColor Green
        } else {
            Write-Host "  Daemon is NOT RUNNING (process exited)" -ForegroundColor Red
        }
    } else {
        Write-Host "  No PID file found" -ForegroundColor Yellow
    }

    Write-Host "  Run directory:" -ForegroundColor Cyan
    $listCommand = "if (Test-Path '$runDir') { Get-ChildItem '$runDir' | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize } else { Write-Host 'MISSING_RUN_DIR' }"
    ssh $RemoteHost "powershell -NoProfile -Command `"$listCommand`""

    Write-Host "  GPU status:" -ForegroundColor Cyan
    ssh $RemoteHost "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"

    exit 0
}

$logPath = switch ($Log) {
    "train" { "$runDir\train.log" }
    "launcher" { "$runDir\launcher.stdout.log" }
    "stderr" { "$runDir\launcher.stderr.log" }
}

Write-Host "=== Monitoring $Log log: $RunName ===" -ForegroundColor Cyan
Write-Host "  $logPath" -ForegroundColor Gray
Write-Host "  Press Ctrl+C to stop local polling; no remote -Wait process is created" -ForegroundColor Yellow
Write-Host ""

if ($Follow) {
    $lastText = ""
    while ($true) {
        $tailCommand = "if (Test-Path '$logPath') { Get-Content -Tail $TailLines '$logPath' -Encoding utf8 }"
        $text = ssh $RemoteHost "powershell -NoProfile -Command `"$tailCommand`""
        $joined = ($text -join "`n")
        if ($joined -and $joined -ne $lastText) {
            Clear-Host
            Write-Host "=== Monitoring $Log log: $RunName ===" -ForegroundColor Cyan
            Write-Host "  $logPath" -ForegroundColor Gray
            Write-Host "  Local polling every $PollSeconds seconds; no remote persistent process" -ForegroundColor Yellow
            Write-Host ""
            $text
            $lastText = $joined
        }
        Start-Sleep -Seconds $PollSeconds
    }
} else {
    $tailCommand = "Get-Content -Tail $TailLines '$logPath' -Encoding utf8"
    ssh $RemoteHost "powershell -NoProfile -Command `"$tailCommand`""
}
