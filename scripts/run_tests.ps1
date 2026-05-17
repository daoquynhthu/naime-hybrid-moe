param(
    [switch]$UseVoice
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

$hasPytest = & $env:NAIME_HYBRID_PYTHON -c "import importlib.util as u; raise SystemExit(0 if u.find_spec('pytest') else 1)"
if ($LASTEXITCODE -eq 0) {
    & $env:NAIME_HYBRID_PYTHON -m pytest -q "$PSScriptRoot\..\tests"
    exit $LASTEXITCODE
}

Write-Host "pytest is not installed in the selected environment; running smoke checks instead."
& "$PSScriptRoot\run_smoke.ps1" -UseVoice:$UseVoice
exit $LASTEXITCODE
