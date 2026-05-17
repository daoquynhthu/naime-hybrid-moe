param(
    [switch]$UseVoice
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

& $env:NAIME_HYBRID_PYTHON -m naime_hybrid.training.preflight
exit $LASTEXITCODE
