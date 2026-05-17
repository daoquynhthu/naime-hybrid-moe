param(
    [switch]$UseVoice,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$InferArgs
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

& $env:NAIME_HYBRID_PYTHON -m naime_hybrid.infer @InferArgs
exit $LASTEXITCODE
