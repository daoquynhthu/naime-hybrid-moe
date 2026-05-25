param(
    [Parameter(Position = 0)]
    [ValidateSet("status", "log", "eval", "stop", "ckpt", "cmd", "push", "pull", "save")]
    [string]$Command = "status",
    [Parameter(Position = 1)]
    [string]$Arg1 = "",
    [Parameter(Position = 2)]
    [string]$Arg2 = "",
    [string]$RemoteHost = "",
    [string]$RunsRoot = "",
    [switch]$Follow,
    [int]$Tail = 20
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath

# ── Resolve remote host ────────────────────────────────────────────
$sshHost = $RemoteHost
if (-not $sshHost) { $sshHost = $env:NAIME_REMOTE_SSH }
if (-not $sshHost) {
    . "$ScriptDir\load_workspace_config.ps1"
    $ws = Get-NaimeWorkspaceConfig -AllowMissing
    $sshHost = Resolve-NaimeConfigValue $ws "remote.ssh" "" ""
}
if (-not $sshHost) {
    Write-Error "No remote host. Run: .\scripts\remote.ps1 save -RemoteHost user@host"
    exit 1
}

# ── Resolve runs root ──────────────────────────────────────────────
$rr = $RunsRoot
if (-not $rr) { $rr = $env:NAIME_REMOTE_RUNS }
if (-not $rr) {
    . "$ScriptDir\load_workspace_config.ps1"
    $ws = Get-NaimeWorkspaceConfig -AllowMissing
    $rr = Resolve-NaimeConfigValue $ws "remote.runs" "" ""
}
if (-not $rr) {
    $rr = 'L:\NAIME_REMOTE\runs'
    Write-Output "Default runs: $rr"
    Write-Output "Override via: `$env:NAIME_REMOTE_RUNS = 'L:\path'"
}

# ── Helper: run on remote ──────────────────────────────────────────
function Run-R {
    param([string]$Cmd)
    & "$ScriptDir\ssh_cmd.ps1" -RemoteHost $sshHost -ScriptBlock $Cmd
}

# ── Helper: get latest run name ────────────────────────────────────
function Get-LatestRun {
    $r = & "$ScriptDir\ssh_cmd.ps1" -RemoteHost $sshHost -ScriptBlock `
        ('Get-ChildItem "{0}" -Directory | Where-Object {{ (Test-Path (Join-Path $_.FullName "train.log")) -or (Test-Path (Join-Path $_.FullName "metrics.jsonl")) }} | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name' -f $rr)
    return $r[-1].Trim()
}

# ── Resolve run name ───────────────────────────────────────────────
$runName = $Arg1
if (-not $runName -and $Command -in @("log","eval","stop","ckpt")) {
    $runName = Get-LatestRun
    Write-Output "Auto-detected: $runName"
}

# ── Commands ───────────────────────────────────────────────────────
switch ($Command) {

    "save" {
        $env:NAIME_REMOTE_SSH = $sshHost
        $env:NAIME_REMOTE_RUNS = $rr
        Write-Output "Saved: NAIME_REMOTE_SSH=$sshHost  NAIME_REMOTE_RUNS=$rr"
        exit 0
    }

    "status" {
        Write-Output "========== GPU =========="
        Run-R 'nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'
        Write-Output ""
        Write-Output "========== Training Processes =========="
        Run-R 'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "naime_hybrid|python.*train" } | Select-Object ProcessId,Name,CommandLine | Format-Table -AutoSize'
        Write-Output ""
        Write-Output "========== L: Disk =========="
        Run-R '$f=[math]::Round((Get-PSDrive L).Free/1GB,1); $u=[math]::Round(((Get-PSDrive L).Used)/1GB,1); Write-Output "Used: ${u}GB  Free: ${f}GB"'
        Write-Output ""
        Write-Output "========== Recent Runs =========="
        Run-R ('Get-ChildItem "{0}" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name,LastWriteTime | Format-Table -AutoSize' -f $rr)
        Write-Output ""
        Write-Output "========== Latest Log (tail 5) =========="
        Run-R ('$r=Get-ChildItem "{0}" -Directory | Where-Object {{ (Test-Path (Join-Path $_.FullName "train.log")) -or (Test-Path (Join-Path $_.FullName "metrics.jsonl")) }} | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name; if ($r) {{ Write-Output "${{r}}:"; Get-Content "{0}\$r\train.log" -Tail 5 -ErrorAction SilentlyContinue }} else {{ Write-Output "no run logs found" }}' -f $rr)
    }

    "log" {
        if ($Follow) {
            Write-Output "Following (Ctrl+C to stop)..."
            ssh $sshHost ('powershell -NoProfile -Command Get-Content "{0}\{1}\train.log" -Wait' -f $rr, $runName)
        } else {
            Run-R ('Get-Content "{0}\{1}\train.log" -Tail {2}' -f $rr, $runName, $Tail)
        }
    }

    "eval" {
        $p = "$rr\$runName"
        $cmd = '$last = Get-Content "' + $p + '\metrics.jsonl" -Tail 1 -ErrorAction SilentlyContinue | ConvertFrom-Json; '
        $cmd += 'if ($last) { '
        $cmd += 'Write-Output "train step=$($last.step) loss=$($last.loss_lm) ppl=$($last.ppl_lm) tok/s=$($last.tokens_per_second) grad=$($last.grad_norm) alpha=$($last.alpha_mean) ent=$($last.router_entropy)"; '
        $cmd += '} else { Write-Output "no train metrics" }; '
        $cmd += '$log = Get-Content "' + $p + '\train.log" -ErrorAction SilentlyContinue; '
        $cmd += '$valLines = $log | Select-String "val \d+" | Select-Object -Last 1; '
        $cmd += 'if ($valLines) { $valLines.Line } else { Write-Output "no validation" }'
        Run-R $cmd
    }

    "stop" {
        Run-R ('New-Item -ItemType File -Force "{0}\{1}\STOP" | Out-Null; Write-Output "STOP created"' -f $rr, $runName)
    }

    "ckpt" {
        Run-R ('Write-Output "=== {1} ==="; Get-ChildItem "{0}\{1}" -File -Filter "*.pt" | Select-Object Name,Length | Format-Table -AutoSize; Write-Output "-- models --"; Get-ChildItem "{0}\{1}\models" -File -Filter "*.pt" -ErrorAction SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize' -f $rr, $runName)
    }

    "cmd" {
        if (-not $Arg1) { Write-Error "Usage: .\scripts\remote.ps1 cmd { your_powershell_code }"; exit 1 }
        $allArgs = $MyInvocation.Line
        $m = [regex]::Match($allArgs, 'cmd\s+\{(.*)\}')
        if ($m.Success) {
            Run-R $m.Groups[1].Value.Trim()
        } else {
            Run-R $Arg1
        }
    }

    "push" {
        if (-not $Arg1 -or -not $Arg2) { Write-Error "Usage: .\scripts\remote.ps1 push <local> <remote>"; exit 1 }
        if (-not (Test-Path -LiteralPath $Arg1)) { Write-Error "Not found: $Arg1"; exit 1 }
        scp $Arg1 "${sshHost}:$Arg2"
        if ($LASTEXITCODE -eq 0) { Write-Output "Uploaded to $Arg2" } else { Write-Error "scp failed" }
    }

    "pull" {
        if (-not $Arg1 -or -not $Arg2) { Write-Error "Usage: .\scripts\remote.ps1 pull <remote> <local>"; exit 1 }
        scp "${sshHost}:$Arg1" $Arg2
        if ($LASTEXITCODE -eq 0) { Write-Output "Downloaded to $Arg2" } else { Write-Error "scp failed" }
    }
}
