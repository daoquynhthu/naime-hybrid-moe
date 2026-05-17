<#
.SYNOPSIS
Execute a PowerShell script block on a remote host via SSH, cleanly extracting
output from CLIXML without all the XML noise.

.DESCRIPTION
Encodes the script as Base64 UTF-16 (PowerShell's -EncodedCommand format),
sends it via SSH, then decodes the CLIXML result back to clean text.

.PARAMETER Host
  SSH host string. If omitted, it is read from configs/workspace.local.json
  or NAIME_REMOTE_SSH.

.PARAMETER ScriptBlock
  PowerShell commands to execute on the remote host.
  Can be a script block { ... } or a plain string.

.PARAMETER Raw
  If set, return raw output without CLIXML stripping.

.EXAMPLE
  .\scripts\ssh_cmd.ps1 -ScriptBlock { Get-Process python* -ErrorAction SilentlyContinue }

.EXAMPLE
  .\scripts\ssh_cmd.ps1 {
      $runs = Get-ChildItem $env:NAIME_REMOTE_RUNS -Directory
      $runs | ForEach-Object { $_.Name }
  }
#>

param(
    [string]$RemoteHost = "",
    [string]$WorkspaceConfig = "",
    [Parameter(Mandatory = $true, Position = 0)]
    $ScriptBlock,
    [switch]$Raw,
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Continue"

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $RemoteHost) { $RemoteHost = Resolve-NaimeConfigValue $workspace "remote.ssh" "NAIME_REMOTE_SSH" }

# Convert script block or string to PowerShell script text
$scriptText = if ($ScriptBlock -is [scriptblock]) {
    $ScriptBlock.ToString()
} else {
    "$ScriptBlock"
}

# Build a wrapper that outputs clean text (no host info noise)
$wrapped = @"
`$ErrorActionPreference = "Continue"
`$WarningPreference = "SilentlyContinue"
$scriptText
"@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($wrapped)
$b64 = [Convert]::ToBase64String($bytes)

$rawOutput = ssh $RemoteHost "powershell -NoProfile -EncodedCommand $b64" 2>$null
$exitCode = $LASTEXITCODE

if ($Raw) {
    $rawOutput
    exit $exitCode
}

# Parse CLIXML: extract only information records (Write-Host / Write-Output text)
$cleanLines = [System.Collections.Generic.List[string]]::new()
$inXml = $false
foreach ($line in $rawOutput) {
    if ($line -match '^#< CLIXML') {
        $inXml = $true
        continue
    }
    if (-not $inXml) {
        # Lines before CLIXML header are clean (shouldn't happen with EncodedCommand but handle it)
        if ($line -match '^\s*$') { continue }
        $cleanLines.Add($line)
        continue
    }
    # Inside CLIXML, extract <S S="N">text</S> or <ToString>text</ToString>
    if ($line -match '<S\s+S="N">(.*?)</S>') {
        $text = $matches[1]
        # XML unescape
        $text = $text -replace '&lt;', '<' -replace '&gt;', '>' -replace '&amp;', '&' -replace '&apos;', "'" -replace '&quot;', '"'
        # Decode _xHHHH_ unicode escapes from CLIXML
        $text = [Regex]::Replace($text, '_x([0-9A-Fa-f]{4})_', { param($m); [char][int]::Parse($m.Groups[1].Value, [System.Globalization.NumberStyles]::HexNumber) })
        if ($text.Trim() -ne '') {
            $cleanLines.Add($text)
        }
    }
}

$cleanLines

# If exit code is non-zero, write to stderr for visibility
if ($exitCode -ne 0) {
    Write-Warning "SSH command exited with code $exitCode"
}

exit $exitCode
