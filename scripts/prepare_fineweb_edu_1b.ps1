param(
    [switch]$UseVoice,
    [string]$Output = "",
    [string]$WorkspaceConfig = "",
    [string]$TokenizerPath = "data\naime\gpt2",
    [string]$DatasetName = "HuggingFaceFW/fineweb-edu",
    [string]$DatasetConfig = "sample-10BT",
    [string]$DatasetSplit = "train",
    [int64]$TrainTokens = 1000000000,
    [int64]$ValidationTokens = 10000000,
    [int]$BlockSize = 1025,
    [int]$ValidationFirstDocs = 20000,
    [int]$MinTextChars = 256,
    [double]$MinScore = 3.0,
    [switch]$PrintArgs,
    [switch]$Overwrite
)

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $Output) { $Output = Resolve-NaimeConfigValue $workspace "local.fineweb_edu_1b" "NAIME_FINEWEB_EDU_1B" }

$argsList = @(
    "--dataset-name", $DatasetName,
    "--dataset-config", $DatasetConfig,
    "--dataset-split", $DatasetSplit,
    "--tokenizer-path", $TokenizerPath,
    "--output", $Output,
    "--block-size", "$BlockSize",
    "--train-tokens", "$TrainTokens",
    "--validation-tokens", "$ValidationTokens",
    "--validation-first-docs", "$ValidationFirstDocs",
    "--min-text-chars", "$MinTextChars",
    "--min-score", "$MinScore"
)

if ($Overwrite) {
    $argsList += "--overwrite"
}

Write-Host "Preparing FineWeb-Edu corpus:"
Write-Host "  output=$Output"
Write-Host "  train_tokens=$TrainTokens validation_tokens=$ValidationTokens block_size=$BlockSize"
Write-Host "  dataset=$DatasetName config=$DatasetConfig split=$DatasetSplit"

if ($PrintArgs) {
    Write-Host "Resolved preparation arguments:"
    $argsList | ForEach-Object { Write-Host $_ }
    exit 0
}

& "$PSScriptRoot\prepare_pretrain_corpus.ps1" -UseVoice:$UseVoice @argsList
exit $LASTEXITCODE
