param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [string]$Checkpoint = "models/model_best.pt",
    [string]$DataPath = "",
    [ValidateSet("byte", "hf_disk", "auto")]
    [string]$DataFormat = "auto",
    [string]$DataSplit = "validation",
    [int]$BatchSize = 2,
    [int]$BatchIndex = 0,
    [int]$ChunkLen = 0,
    [int]$BoundaryTokens = 64,
    [string]$OutputDir = "",
    [string]$Device = "auto",
    [switch]$NoAmp,
    [switch]$RecordFullTensors,
    [switch]$UseVoice
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

if ($UseVoice) {
    $python = Join-Path (Split-Path -Parent $repoRoot) "voice\.venv\Scripts\python.exe"
} else {
    $python = Join-Path $repoRoot "venv312\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$env:PYTHONPATH = Join-Path $repoRoot "src"

$argsList = @(
    "-m", "naime_hybrid.diagnostics.run_packet_diagnostics",
    "--run-dir", $RunDir,
    "--checkpoint", $Checkpoint,
    "--data-format", $DataFormat,
    "--data-split", $DataSplit,
    "--batch-size", [string]$BatchSize,
    "--batch-index", [string]$BatchIndex,
    "--boundary-tokens", [string]$BoundaryTokens,
    "--device", $Device
)

if (-not [string]::IsNullOrWhiteSpace($DataPath)) {
    $argsList += @("--data-path", $DataPath)
}
if ($ChunkLen -gt 0) {
    $argsList += @("--chunk-len", [string]$ChunkLen)
}
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $argsList += @("--output-dir", $OutputDir)
}
if ($NoAmp) {
    $argsList += "--no-amp"
}
if ($RecordFullTensors) {
    $argsList += "--record-full-tensors"
}

& $python @argsList
