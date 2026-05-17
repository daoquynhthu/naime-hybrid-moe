param(
    [switch]$UseVoice,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PrepareArgs
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

& $env:NAIME_HYBRID_PYTHON "$PSScriptRoot\prepare_pretrain_corpus.py" @PrepareArgs
exit $LASTEXITCODE
