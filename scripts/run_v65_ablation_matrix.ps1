param(
    [string]$RunPrefix = "v65_ablation",
    [string]$Template = "v6_remote_64m_v65_probe",
    [string]$WorkspaceConfig = "",
    [string]$DataPath = "",
    [string]$OutputDir = "",
    [int]$MaxSteps = 3000,
    [int]$EvalEvery = 500,
    [int]$EvalMaxBatches = 20,
    [int]$SaveEvery = 3000,
    [int]$LatestEvery = 0,
    [int]$MetricsFlushEvery = 100,
    [int]$MetricsFsyncEvery = 0,
    [int]$LogEvery = 10,
    [int]$IdleBatchSize = 8,
    [int]$SharedBatchSize = 4,
    [int]$BusyBatchSize = 2,
    [double]$AggressiveFreeFraction = 0.82,
    [double]$SharedFreeFraction = 0.55,
    [switch]$AllowBusy,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig -AllowMissing
$remotePython = Resolve-NaimeConfigValue $workspace "remote.python" "NAIME_REMOTE_PYTHON" ""
if (-not $env:NAIME_HYBRID_PYTHON -and $remotePython -and (Test-Path -LiteralPath $remotePython)) {
    $env:NAIME_HYBRID_PYTHON = $remotePython
}
if (-not $env:NAIME_HYBRID_PYTHON) {
    $localVenv = Join-Path $repo ".venv312\Scripts\python.exe"
    $env:NAIME_HYBRID_PYTHON = if (Test-Path -LiteralPath $localVenv) { $localVenv } else { "python" }
}
$srcPath = Join-Path $repo "src"
if ($env:PYTHONPATH) {
    if ($env:PYTHONPATH -notlike "*$srcPath*") {
        $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
    }
} else {
    $env:PYTHONPATH = $srcPath
}

if (-not $OutputDir) {
    $OutputDir = Resolve-NaimeConfigValue $workspace "remote.runs" "NAIME_REMOTE_RUNS" "experiments\runs"
}
if (-not $DataPath) {
    $datasets = Resolve-NaimeConfigValue $workspace "remote.datasets" "NAIME_REMOTE_DATASETS" "data"
    $DataPath = Join-Path $datasets "fineweb_edu_50m"
}

function Get-GpuMemoryInfo {
    $query = & nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $query) {
        throw "nvidia-smi failed; cannot choose a safe ablation batch size"
    }
    $first = @($query)[0]
    $parts = $first -split "," | ForEach-Object { $_.Trim() }
    [pscustomobject]@{
        Index = [int]$parts[0]
        Name = $parts[1]
        TotalMiB = [int]$parts[2]
        UsedMiB = [int]$parts[3]
        FreeMiB = [int]$parts[4]
        FreeFraction = ([double]$parts[4]) / [Math]::Max(1.0, [double]$parts[2])
    }
}

$gpu = Get-GpuMemoryInfo
if ($gpu.FreeFraction -ge $AggressiveFreeFraction) {
    $batchSize = $IdleBatchSize
    $mode = "aggressive"
} elseif ($gpu.FreeFraction -ge $SharedFreeFraction) {
    $batchSize = $SharedBatchSize
    $mode = "shared"
} elseif ($AllowBusy) {
    $batchSize = $BusyBatchSize
    $mode = "busy-allowed"
} else {
    throw ("GPU is busy: {0} free {1}/{2} MiB ({3:P1}). Re-run with -AllowBusy only if this is intentional." -f `
        $gpu.Name, $gpu.FreeMiB, $gpu.TotalMiB, $gpu.FreeFraction)
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$matrixRoot = Join-Path $OutputDir "$RunPrefix`_$stamp"
New-Item -ItemType Directory -Force -Path $matrixRoot | Out-Null
$matrixLog = Join-Path $matrixRoot "matrix.log"
$summaryPath = Join-Path $matrixRoot "matrix_summary.csv"

function Write-MatrixLog {
    param([string]$Message)
    $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $matrixLog -Value $line -Encoding UTF8
}

function Read-ResolvedTrainArgs {
    param([string]$Path)
    $lines = Get-Content -LiteralPath $Path
    $start = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^Resolved training arguments:") {
            $start = $i + 1
            break
        }
    }
    if ($start -lt 0) {
        throw "could not find resolved training arguments in $Path"
    }
    $args = @()
    for ($i = $start; $i -lt $lines.Count; $i++) {
        $line = $lines[$i].Trim()
        if ($line) {
            $args += $line
        }
    }
    if ($args.Count -eq 0) {
        throw "resolved training argument list is empty: $Path"
    }
    return $args
}

function Invoke-TrainingProcess {
    param(
        [string[]]$TrainArgs,
        [string]$RunDir,
        [string]$StdoutPath,
        [string]$StderrPath
    )
    $argsFile = Join-Path $RunDir "train_args.txt"
    $TrainArgs | Set-Content -LiteralPath $argsFile -Encoding ASCII

    $stdoutStream = [System.IO.File]::Open($StdoutPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    $stderrStream = [System.IO.File]::Open($StderrPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $env:NAIME_HYBRID_PYTHON
        $psi.WorkingDirectory = $repo
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.Environment["PYTHONPATH"] = $env:PYTHONPATH
        $psi.Environment["NAIME_HYBRID_PYTHON"] = $env:NAIME_HYBRID_PYTHON
        $psi.Environment["TORCHINDUCTOR_CACHE_DIR"] = $env:TORCHINDUCTOR_CACHE_DIR
        $psi.Environment["TRITON_CACHE_DIR"] = $env:TRITON_CACHE_DIR
        $escapedArgsFile = $argsFile.Replace('"', '\"')
        $psi.Arguments = "-m naime_hybrid.training.train `"@$escapedArgsFile`""

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        if (-not $process.Start()) {
            throw "failed to start training process"
        }
        $stdoutTask = $process.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
        $stderrTask = $process.StandardError.BaseStream.CopyToAsync($stderrStream)
        $process.WaitForExit()
        $stdoutTask.Wait()
        $stderrTask.Wait()
        return $process.ExitCode
    } finally {
        $stdoutStream.Dispose()
        $stderrStream.Dispose()
    }
}

Write-MatrixLog "repo=$repo"
Write-MatrixLog ("gpu={0} free={1}/{2}MiB free_fraction={3:N3} mode={4} batch={5}" -f `
    $gpu.Name, $gpu.FreeMiB, $gpu.TotalMiB, $gpu.FreeFraction, $mode, $batchSize)
Write-MatrixLog "matrix_root=$matrixRoot"

$runs = @(
    [pscustomobject]@{ Name = "baseline"; StateCarry = $false; ThoughtSteps = 0; ThoughtGain = $false },
    [pscustomobject]@{ Name = "state_carry"; StateCarry = $true; ThoughtSteps = 0; ThoughtGain = $false },
    [pscustomobject]@{ Name = "latent_thought"; StateCarry = $false; ThoughtSteps = 1; ThoughtGain = $true },
    [pscustomobject]@{ Name = "state_carry_latent_thought"; StateCarry = $true; ThoughtSteps = 1; ThoughtGain = $true }
)

$summaryRows = @()
foreach ($run in $runs) {
    $runName = "$RunPrefix`_$($run.Name)_$stamp"
    $runDir = Join-Path $matrixRoot $runName
    $stdoutLog = Join-Path $runDir "matrix.stdout.log"
    $stderrLog = Join-Path $runDir "matrix.stderr.log"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $paramsForRun = @{
        Template = $Template
        RunName = $runName
        OutputDir = $matrixRoot
        DataPath = $DataPath
        NoAutoBatch = $true
        BatchSize = $batchSize
        MaxSteps = $MaxSteps
        EvalEvery = $EvalEvery
        EvalMaxBatches = $EvalMaxBatches
        SaveEvery = $SaveEvery
        LatestEvery = $LatestEvery
        MetricsFlushEvery = $MetricsFlushEvery
        MetricsFsyncEvery = $MetricsFsyncEvery
        NumWorkers = 4
        LatentThoughtSteps = $run.ThoughtSteps
        LatentThoughtWriteMode = "state_only"
        LatentThoughtHiddenScale = 0.0
        Resume = "none"
        ResumeLrPolicy = "reset"
    }
    if ($run.StateCarry) {
        $paramsForRun["EvalStateCarry"] = $true
    }
    if ($run.ThoughtGain) {
        $paramsForRun["EvalLatentThoughtGain"] = $true
    }
    $paramsForRun["CompileScope"] = "dense"
    $paramsForRun["CompileBackend"] = "inductor"

    if ($DryRun) {
        $argPreview = ($paramsForRun.GetEnumerator() | Sort-Object Name | ForEach-Object { "-$($_.Name) $($_.Value)" }) -join " "
        Write-MatrixLog "ARGS $runName $argPreview"
    }

    $resolvedPath = Join-Path $runDir "resolved_args.txt"
    & "$PSScriptRoot\train_template.ps1" @paramsForRun -PrintArgs *> $resolvedPath
    if ($LASTEXITCODE -ne 0) {
        throw "failed to resolve args for $runName; see $resolvedPath"
    }
    $resolvedTrainArgs = Read-ResolvedTrainArgs -Path $resolvedPath

    if ($DryRun) {
        Write-MatrixLog "dry-run resolved $runName"
        continue
    }

    $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $runDir "torchinductor_cache"
    $env:TRITON_CACHE_DIR = Join-Path $runDir "triton_cache"
    Remove-Item -LiteralPath $env:TORCHINDUCTOR_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $env:TRITON_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue

    Write-MatrixLog "START $runName state_carry=$($run.StateCarry) thought_steps=$($run.ThoughtSteps)"
    $start = Get-Date
    $exitCode = Invoke-TrainingProcess `
        -TrainArgs $resolvedTrainArgs `
        -RunDir $runDir `
        -StdoutPath $stdoutLog `
        -StderrPath $stderrLog
    $end = Get-Date
    $durationMin = [Math]::Round(($end - $start).TotalMinutes, 2)
    Write-MatrixLog "END $runName exit=$exitCode duration_min=$durationMin"

    $metricsPath = Join-Path $runDir "metrics.jsonl"
    $failedPath = Join-Path $runDir "failed.pt"
    $trainLogPath = Join-Path $runDir "train.log"
    $logLooksFailed = $false
    if (Test-Path -LiteralPath $trainLogPath) {
        $logText = Get-Content -LiteralPath $trainLogPath -Raw
        $logLooksFailed = $logText -match "training failed|Traceback|InductorError"
    }
    if ((Test-Path -LiteralPath $failedPath) -or $logLooksFailed) {
        $exitCode = if ($exitCode -ne 0) { $exitCode } else { 1 }
        Write-MatrixLog "FAIL $runName failed_pt=$(Test-Path -LiteralPath $failedPath) log_failed=$logLooksFailed"
    }
    $lastMetric = $null
    if (Test-Path -LiteralPath $metricsPath) {
        $lastLine = Get-Content -LiteralPath $metricsPath -Tail 1
        if ($lastLine) {
            $lastMetric = $lastLine | ConvertFrom-Json
        }
    }
    $summaryRows += [pscustomobject]@{
        run_name = $runName
        exit_code = $exitCode
        duration_min = $durationMin
        state_carry = $run.StateCarry
        thought_steps = $run.ThoughtSteps
        batch_size = $batchSize
        max_steps = $MaxSteps
        last_step = if ($lastMetric) { $lastMetric.step } else { "" }
        last_lm = if ($lastMetric) { $lastMetric.loss_lm } else { "" }
        last_val_lm = if ($lastMetric -and $lastMetric.PSObject.Properties["val_lm_loss"]) { $lastMetric.val_lm_loss } else { "" }
        grad_norm = if ($lastMetric -and $lastMetric.PSObject.Properties["grad_norm"]) { $lastMetric.grad_norm } else { "" }
        bad_grad_window = if ($lastMetric -and $lastMetric.PSObject.Properties["bad_grad_window_count"]) { $lastMetric.bad_grad_window_count } else { "" }
        state_carry_gain = if ($lastMetric -and $lastMetric.PSObject.Properties["val_state_carry_gain_lm"]) { $lastMetric.val_state_carry_gain_lm } else { "" }
        thought_gain = if ($lastMetric -and $lastMetric.PSObject.Properties["val_latent_thought_gain_lm"]) { $lastMetric.val_latent_thought_gain_lm } else { "" }
    }
    $summaryRows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

    if ($exitCode -ne 0) {
        throw "ablation run failed: $runName exit=$exitCode"
    }
}

Write-MatrixLog "summary=$summaryPath"
if (Test-Path -LiteralPath $summaryPath) {
    Import-Csv -LiteralPath $summaryPath | Format-Table -AutoSize
}
