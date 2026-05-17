param(
    [switch]$UseVoice
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

& $env:NAIME_HYBRID_PYTHON "$PSScriptRoot\smoke_forward.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $env:NAIME_HYBRID_PYTHON -m compileall -q "$PSScriptRoot\..\src" "$PSScriptRoot\..\tests" "$PSScriptRoot"
exit $LASTEXITCODE
