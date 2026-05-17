param(
    [string]$RemoteUser = "",
    [string]$RemoteHost = "",
    [string]$RemoteRoot = "",
    [string]$WorkspaceConfig = ""
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $RemoteUser) { $RemoteUser = Resolve-NaimeConfigValue $workspace "remote.user" "NAIME_REMOTE_USER" }
if (-not $RemoteHost) { $RemoteHost = Resolve-NaimeConfigValue $workspace "remote.host" "NAIME_REMOTE_HOST" }
if (-not $RemoteRoot) { $RemoteRoot = Resolve-NaimeConfigValue $workspace "remote.root" "NAIME_REMOTE_ROOT" }

$remoteScript = @"
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteRoot'
if (-not (Test-Path -LiteralPath `$root)) {
    throw "Missing remote root: `$root"
}

`$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
`$paths = @(
    `$root,
    (Join-Path `$root 'naime-hybrid-moe'),
    (Join-Path `$root 'runs'),
    (Join-Path `$root 'datasets'),
    (Join-Path `$root 'envs')
)

Write-Host "Hardening NAIME ACL under `$root for `$currentUser"
foreach (`$path in `$paths) {
    if (-not (Test-Path -LiteralPath `$path)) {
        Write-Host "skip missing: `$path"
        continue
    }

    Write-Host "harden: `$path"
    icacls `$path /inheritance:r | Out-Host
    icacls `$path /grant:r "`${currentUser}:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" | Out-Host
    icacls `$path /remove:g "*S-1-5-11" "*S-1-5-32-545" "*S-1-1-0" | Out-Host
}

Write-Host "--- effective roots"
foreach (`$path in `$paths) {
    if (Test-Path -LiteralPath `$path) {
        Write-Host "--- `$path"
        icacls `$path | Out-Host
    }
}
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
ssh "$RemoteUser@$RemoteHost" "powershell -NoProfile -EncodedCommand $encoded"
