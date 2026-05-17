param(
    [string]$Python = "",
    [string]$RunName = "",
    [string]$OutputDir = "experiments\runs",
    [string]$WorkspaceConfig = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainModelArgs
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig

if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvName = Resolve-NaimeConfigValue $workspace "local.venv" "NAIME_LOCAL_VENV" ".venv312"
    $candidates = @(
        (Join-Path $repo "$venvName\Scripts\python.exe"),
        (Join-Path $repo ".venv\Scripts\python.exe"),
        "python"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -eq "python" -or (Test-Path -LiteralPath $candidate)) {
            $Python = $candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "detached_train_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

$outputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $repo $OutputDir
}
$runDir = Join-Path $outputPath $RunName

$argsForWrapper = @($TrainModelArgs)
$hasRunName = $false
$hasOutputDir = $false
for ($i = 0; $i -lt $argsForWrapper.Count; $i++) {
    if ($argsForWrapper[$i] -eq "-RunName") { $hasRunName = $true }
    if ($argsForWrapper[$i] -eq "-OutputDir") { $hasOutputDir = $true }
}
if (-not $hasRunName) { $argsForWrapper += @("-RunName", $RunName) }
if (-not $hasOutputDir) { $argsForWrapper += @("-OutputDir", $outputPath) }

New-Item -ItemType Directory -Path $runDir -Force | Out-Null
& $Python (Join-Path $PSScriptRoot "launch_train_detached.py") `
    --repo $repo `
    --python $Python `
    --run-dir $runDir `
    -- @argsForWrapper

Write-Host "Detached training launched"
Write-Host "RunDir: $runDir"
Write-Host "PID: $(Get-Content (Join-Path $runDir 'daemon.pid'))"
