param(
    [Parameter(Mandatory=$true)]
    [string]$Path,

    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $repoRoot "venv312\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Push-Location $repoRoot
try {
    $env:PYTHONPATH = (Resolve-Path "src").Path
    $argsList = @("-m", "naime_hybrid.diagnostics.summarize_training_diagnostics", $Path)
    if ($OutputDir -ne "") {
        $argsList += @("--output-dir", $OutputDir)
    }
    & $python @argsList
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
