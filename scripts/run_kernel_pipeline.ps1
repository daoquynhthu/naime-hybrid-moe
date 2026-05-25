param(
    [string]$RemoteHost = "",
    [string]$RemoteProjectRoot = "",
    [string]$RemotePython = "",
    [string]$RemoteRunsRoot = "",
    [string]$Template = "v7_remote_64m_probe",
    [int]$BatchSize = 29,
    [int]$SeqLen = 512,
    [int]$ProfileWarmup = 3,
    [int]$ProfileSteps = 5,
    [int]$SmokeSteps = 50,
    [string]$BaselineBackend = "torch",
    [string]$CandidateBackend = "cuda_ext_ce",
    [bool]$UseFusedStateAttention = $true,
    [string[]]$StylePaths = @(
        "scripts/profile_v7_full_step.py",
        "src/naime_hybrid/kernels/cross_entropy.py",
        "src/naime_hybrid/kernels/fused_lm_ce.py",
        "src/naime_hybrid/kernels/cuda_ext.py",
        "src/naime_hybrid/training/cli.py",
        "src/naime_hybrid/training/config.py",
        "src/naime_hybrid/training/train.py",
        "tests/test_kernels.py"
    ),
    [switch]$SkipSync,
    [switch]$KeepSmokeRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent $ScriptDir
. "$ScriptDir\load_workspace_config.ps1"
$workspace = Get-NaimeWorkspaceConfig -AllowMissing

function Resolve-Value {
    param([string]$Value, [string]$ConfigPath, [string]$EnvName, [string]$Default)
    if ($Value) { return $Value }
    return Resolve-NaimeConfigValue $workspace $ConfigPath $EnvName $Default
}

$RemoteHost = Resolve-Value $RemoteHost "remote.ssh" "NAIME_REMOTE_SSH" ""
$RemoteProjectRoot = Resolve-Value $RemoteProjectRoot "remote.repo" "NAIME_REMOTE_REPO" "L:/NAIME_REMOTE/naime-hybrid-moe"
$RemotePython = Resolve-Value $RemotePython "remote.python" "NAIME_HYBRID_REMOTE_PYTHON" "L:/NAIME_REMOTE/envs/.venv312/Scripts/python.exe"
$RemoteRunsRoot = Resolve-Value $RemoteRunsRoot "remote.runs" "NAIME_REMOTE_RUNS" "L:/NAIME_REMOTE/runs"
if (-not $RemoteHost) {
    throw "Remote host is not configured. Set remote.ssh in configs/workspace.local.json or NAIME_REMOTE_SSH."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$profileRoot = "$RemoteRunsRoot/profiles/kernel_pipeline_$stamp"
$smokeRun = "kernel_${CandidateBackend}_smoke_$stamp"
$extBuild = "L:/NAIME_REMOTE/build/torch_extensions_kernel_pipeline"

function Invoke-Remote {
    param([string]$Script)
    $prefix = "`$ProgressPreference='SilentlyContinue'; `$InformationPreference='SilentlyContinue'; `$ErrorActionPreference='Stop'; "
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($prefix + $Script))
    ssh -o BatchMode=yes $RemoteHost "powershell -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -EncodedCommand $encoded"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Read-RemoteJson {
    param([string]$Path)
    $script = "`$ProgressPreference='SilentlyContinue'; Get-Content '$Path' -Raw"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $json = ssh -o BatchMode=yes $RemoteHost "powershell -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -EncodedCommand $encoded"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read remote JSON: $Path"
    }
    return ($json -join "`n") | ConvertFrom-Json
}

Write-Host "== NAIME Kernel Pipeline =="
Write-Host "Repo       : $RepoRoot"
Write-Host "Remote     : $RemoteHost"
Write-Host "Remote repo: $RemoteProjectRoot"
Write-Host "Template   : $Template"
Write-Host "Backends   : $BaselineBackend -> $CandidateBackend"

Write-Host "`n[1/6] Local kernel-path checks"
$python = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$styleAbs = @()
foreach ($path in $StylePaths) {
    $resolved = Join-Path $RepoRoot $path
    if (Test-Path -LiteralPath $resolved) {
        $styleAbs += $resolved
    }
}
if ($styleAbs.Count -gt 0) {
    & $python -m ruff check @styleAbs
    if ($LASTEXITCODE -ne 0) { throw "Local ruff check failed" }
    & $python -m ruff format --check @styleAbs
    if ($LASTEXITCODE -ne 0) { throw "Local ruff format check failed" }
    & $python -m py_compile @styleAbs
    if ($LASTEXITCODE -ne 0) { throw "Local py_compile failed" }
}

if (-not $SkipSync) {
    Write-Host "`n[2/6] Sync code to remote"
    & "$ScriptDir\sync_to_remote.ps1" -RemoteHost ($RemoteHost -replace "^([^@]+)@(.+)$", '$2') -RemoteUser ($RemoteHost -replace "^([^@]+)@(.+)$", '$1') -RemoteProjectRoot $RemoteProjectRoot
    if ($LASTEXITCODE -ne 0) { throw "Remote sync failed" }
} else {
    Write-Host "`n[2/6] Sync skipped"
}

Write-Host "`n[3/6] Remote CUDA kernel tests"
Invoke-Remote @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RemoteProjectRoot'
`$env:PYTHONPATH = '$RemoteProjectRoot/src'
`$env:NAIME_HYBRID_PYTHON = '$RemotePython'
`$env:NAIME_EXT_BUILD_DIR = '$extBuild'
& '$RemotePython' -m pytest tests/test_kernels.py::test_cuda_ext_cross_entropy_matches_torch_forward_and_backward -q
"@

Write-Host "`n[4/6] Remote profiler baseline: $BaselineBackend"
$baselineOut = "$profileRoot/$BaselineBackend"
Invoke-Remote @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RemoteProjectRoot'
`$env:PYTHONPATH = '$RemoteProjectRoot/src'
`$env:NAIME_HYBRID_PYTHON = '$RemotePython'
`$env:NAIME_EXT_BUILD_DIR = '$extBuild'
& '$RemotePython' scripts/profile_v7_full_step.py --template configs/training_templates/$Template.json --out-dir '$baselineOut' --batch $BatchSize --seq-len $SeqLen --warmup $ProfileWarmup --steps $ProfileSteps --compile --lm-loss-backend $BaselineBackend --row-limit 40 --quiet
"@

Write-Host "`n[5/6] Remote profiler candidate: $CandidateBackend"
$candidateOut = "$profileRoot/$CandidateBackend"
Invoke-Remote @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RemoteProjectRoot'
`$env:PYTHONPATH = '$RemoteProjectRoot/src'
`$env:NAIME_HYBRID_PYTHON = '$RemotePython'
`$env:NAIME_EXT_BUILD_DIR = '$extBuild'
`$stateFlag = if ('$UseFusedStateAttention' -eq 'True') { '--use-fused-state-attention' } else { '' }
& '$RemotePython' scripts/profile_v7_full_step.py --template configs/training_templates/$Template.json --out-dir '$candidateOut' --batch $BatchSize --seq-len $SeqLen --warmup $ProfileWarmup --steps $ProfileSteps --compile --lm-loss-backend $CandidateBackend --row-limit 40 --quiet `$stateFlag
"@

Write-Host "`n[6/6] Remote real-train smoke"
Invoke-Remote @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RemoteProjectRoot'
`$env:NAIME_HYBRID_PYTHON = '$RemotePython'
`$env:NAIME_EXT_BUILD_DIR = '$extBuild'
if ('$UseFusedStateAttention' -eq 'True') {
  & .\scripts\train_template.ps1 -Template $Template -RunName $smokeRun -MaxSteps $SmokeSteps -BatchSize $BatchSize -NoAutoBatch -SeqLen $SeqLen -EvalEvery 0 -SaveEvery 0 -LatestEvery 0 -MetricsFlushEvery 100 -MetricsFsyncEvery 0 -NumWorkers 0 -LmLossBackend $CandidateBackend -UseFusedStateAttention
} else {
  & .\scripts\train_template.ps1 -Template $Template -RunName $smokeRun -MaxSteps $SmokeSteps -BatchSize $BatchSize -NoAutoBatch -SeqLen $SeqLen -EvalEvery 0 -SaveEvery 0 -LatestEvery 0 -MetricsFlushEvery 100 -MetricsFsyncEvery 0 -NumWorkers 0 -LmLossBackend $CandidateBackend
}
"@

$baseline = Read-RemoteJson "$baselineOut/summary.json"
$candidate = Read-RemoteJson "$candidateOut/summary.json"
$baseTok = [double]$baseline.profiled_tokens_per_second
$candTok = [double]$candidate.profiled_tokens_per_second
$baseMem = [double]$baseline.peak_memory_mb
$candMem = [double]$candidate.peak_memory_mb
$speedup = if ($baseTok -gt 0) { 100.0 * ($candTok / $baseTok - 1.0) } else { 0.0 }
$memDelta = $candMem - $baseMem

Write-Host "`n== Kernel Pipeline Summary =="
Write-Host ("Baseline {0}: {1:n0} tok/s, peak {2:n1} MB" -f $BaselineBackend, $baseTok, $baseMem)
Write-Host ("Candidate {0}: {1:n0} tok/s, peak {2:n1} MB" -f $CandidateBackend, $candTok, $candMem)
Write-Host ("Speed delta : {0:n1}%" -f $speedup)
Write-Host ("Memory delta: {0:n1} MB" -f $memDelta)
Write-Host "Profiles    : $profileRoot"
Write-Host "Smoke run   : $RemoteRunsRoot/$smokeRun"

if (-not $KeepSmokeRun) {
    Write-Host "`nCleaning temporary smoke run..."
    Invoke-Remote @"
`$runsRoot = (Resolve-Path '$RemoteRunsRoot').Path
`$run = Resolve-Path '$RemoteRunsRoot/$smokeRun' -ErrorAction SilentlyContinue
if (`$run) {
  `$rootNorm = `$runsRoot.TrimEnd('\', '/').ToLowerInvariant()
  `$runNorm = `$run.Path.ToLowerInvariant()
  if (`$runNorm.StartsWith(`$rootNorm)) {
    Remove-Item -LiteralPath `$run.Path -Recurse -Force
    Write-Output "removed `$(`$run.Path)"
  } else {
    throw "refusing to remove unexpected path: `$(`$run.Path)"
  }
}
"@
}
