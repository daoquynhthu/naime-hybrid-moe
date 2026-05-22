param(
    [string]$RunDir = "",
    [string]$Device = "auto",
    [int]$Steps = 2,
    [int]$BatchSize = 2,
    [int]$SeqLen = 32,
    [int]$DModel = 32,
    [switch]$IncludeAutoBatch,
    [switch]$CompileSmoke,
    [string]$CompileBackend = "inductor",
    [ValidateSet("full", "dense")]
    [string]$CompileScope = "dense",
    [switch]$NoAmp
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

$python = $env:NAIME_HYBRID_PYTHON
if (-not $python) {
    $localPython = Join-Path $repoRoot ".venv312\Scripts\python.exe"
    if (Test-Path $localPython) {
        $python = $localPython
    } else {
        $python = "python"
    }
}

$env:PYTHONPATH = Join-Path $repoRoot "src"

$argsList = @(
    "-m", "naime_hybrid.training.debug_suite",
    "--device", $Device,
    "--steps", "$Steps",
    "--batch-size", "$BatchSize",
    "--seq-len", "$SeqLen",
    "--d-model", "$DModel"
)

if ($RunDir) {
    $argsList += @("--run-dir", $RunDir)
}
if ($IncludeAutoBatch) {
    $argsList += "--include-auto-batch"
}
if ($CompileSmoke) {
    $argsList += @("--compile-smoke", "--compile-backend", $CompileBackend, "--compile-scope", $CompileScope)
}
if ($NoAmp) {
    $argsList += "--no-amp"
}

Write-Host "NAIME component debug suite"
Write-Host "Repo: $repoRoot"
Write-Host "Python: $python"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
& $python @argsList
exit $LASTEXITCODE
