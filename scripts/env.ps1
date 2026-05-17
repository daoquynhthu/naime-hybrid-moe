param(
    [switch]$UseVoice,
    [string]$WorkspaceConfig = ""
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig -AllowMissing
$venvName = Resolve-NaimeConfigValue $workspace "local.venv" "NAIME_LOCAL_VENV" ".venv312"
$ProjectPython = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"
$FallbackTrainPython = Join-Path $ProjectRoot ".venv-train\Scripts\python.exe"
$FallbackProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ConfiguredPython = Join-Path $ProjectRoot "$venvName\Scripts\python.exe"
$ExternalPython = [Environment]::GetEnvironmentVariable("NAIME_EXTERNAL_PYTHON")

if (-not [string]::IsNullOrWhiteSpace($env:NAIME_HYBRID_PYTHON) -and (Test-Path $env:NAIME_HYBRID_PYTHON)) {
    $env:NAIME_HYBRID_PYTHON = $env:NAIME_HYBRID_PYTHON
} elseif ($UseVoice) {
    if ([string]::IsNullOrWhiteSpace($ExternalPython) -or -not (Test-Path $ExternalPython)) {
        Write-Error "External Python requested but NAIME_EXTERNAL_PYTHON is not set to an existing interpreter."
        exit 1
    }
    $env:NAIME_HYBRID_PYTHON = $ExternalPython
} elseif (Test-Path $ConfiguredPython) {
    $env:NAIME_HYBRID_PYTHON = $ConfiguredPython
} elseif (Test-Path $ProjectPython) {
    $env:NAIME_HYBRID_PYTHON = $ProjectPython
} elseif (Test-Path $FallbackTrainPython) {
    $env:NAIME_HYBRID_PYTHON = $FallbackTrainPython
} elseif (Test-Path $FallbackProjectPython) {
    $env:NAIME_HYBRID_PYTHON = $FallbackProjectPython
} else {
    $env:NAIME_HYBRID_PYTHON = "python"
}

$srcPath = Join-Path $ProjectRoot "src"
if ($env:PYTHONPATH) {
    if ($env:PYTHONPATH -notlike "*$srcPath*") {
        $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
    }
} else {
    $env:PYTHONPATH = $srcPath
}

$pythonDir = Split-Path -Parent $env:NAIME_HYBRID_PYTHON
if ((-not [string]::IsNullOrWhiteSpace($pythonDir)) -and (Test-Path -LiteralPath $pythonDir) -and ($env:PATH -notlike "$pythonDir*")) {
    $env:PATH = "$pythonDir;$env:PATH"
}

Write-Host "NAIME Hybrid environment configured."
Write-Host "Python: $env:NAIME_HYBRID_PYTHON"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
