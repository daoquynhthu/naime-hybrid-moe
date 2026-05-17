param(
    [switch]$UseVoice,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvalArgs
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

& $env:NAIME_HYBRID_PYTHON -m naime_hybrid.training.eval @EvalArgs
exit $LASTEXITCODE
