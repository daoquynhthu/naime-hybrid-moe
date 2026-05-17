param(
    [string]$ConfigPath = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\workspace.local.json"
}

function Get-NaimeWorkspaceConfig {
    param(
        [string]$Path = $ConfigPath,
        [switch]$AllowMissing
    )

    if (Test-Path -LiteralPath $Path) {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }

    if ($AllowMissing) {
        return [pscustomobject]@{}
    }

    $example = Join-Path $repoRoot "configs\workspace.example.json"
    throw "Missing workspace config: $Path. Copy $example to configs\workspace.local.json and fill local values."
}

function Resolve-NaimeConfigValue {
    param(
        [object]$Config,
        [string]$Path,
        [string]$EnvName = "",
        [string]$Fallback = ""
    )

    if ($EnvName) {
        $value = [Environment]::GetEnvironmentVariable($EnvName)
        if ($value) { return $value }
    }

    $current = $Config
    foreach ($part in $Path.Split(".")) {
        if ($null -eq $current) { break }
        $prop = $current.PSObject.Properties[$part]
        if ($null -eq $prop) {
            $current = $null
            break
        }
        $current = $prop.Value
    }
    if ($current) { return [string]$current }
    if ($Fallback) { return $Fallback }
    throw "Missing workspace config value: $Path"
}
