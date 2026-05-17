param(
    [switch]$Fix
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Push-Location $repoRoot
try {
    & $python -m ruff --version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ruff is not installed"
    }

    if ($Fix) {
        & $python -m ruff check . --fix
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python -m ruff format .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        & $python -m ruff check .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python -m ruff format . --check
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} catch {
    Write-Error "Style check failed: $($_.Exception.Message). Install dev tools with: $python -m pip install -e .[dev]"
    exit 1
} finally {
    Pop-Location
}
