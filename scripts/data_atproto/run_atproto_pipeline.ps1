param(
    [string]$Python = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$Pipeline = Join-Path $ScriptDir "atproto_pipeline.py"

if (-not $Python) {
    if ($env:NAIME_PYTHON) {
        $Python = $env:NAIME_PYTHON
    } else {
        $LocalVenv = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
        if (Test-Path $LocalVenv) {
            $Python = $LocalVenv
        } else {
            $Python = "python"
        }
    }
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"

Write-Host "NAIME ATProto data pipeline"
Write-Host "Repo:   $RepoRoot"
Write-Host "Python: $Python"
Write-Host "Args:   $($PipelineArgs -join ' ')"

& $Python $Pipeline @PipelineArgs
exit $LASTEXITCODE
