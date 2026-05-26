param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("ls", "push", "pull", "sync", "rm")]
    [string]$Command,

    [Parameter(Position = 1)]
    [string]$Path1,

    [Parameter(Position = 2)]
    [string]$Path2,

    [string]$Bucket = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath

# ── Load Config ────────────────────────────────────────────────────
. "$ScriptDir\load_workspace_config.ps1"
$ws = Get-NaimeWorkspaceConfig -AllowMissing

$endpoint = Resolve-NaimeConfigValue $ws "storage.contabo.endpoint" "CONTABO_ENDPOINT"
$accessKey = Resolve-NaimeConfigValue $ws "storage.contabo.access_key" "CONTABO_ACCESS_KEY"
$secretKey = Resolve-NaimeConfigValue $ws "storage.contabo.secret_key" "CONTABO_SECRET_KEY"
$defaultBucket = Resolve-NaimeConfigValue $ws "storage.contabo.bucket" "CONTABO_BUCKET" ""
$region = Resolve-NaimeConfigValue $ws "storage.contabo.region" "CONTABO_REGION" "default"

if (-not $Bucket) { $Bucket = $defaultBucket }
if (-not $Bucket -and $Command -ne "ls") {
    Write-Error "Bucket name is required. Set it in configs\workspace.local.json or pass -Bucket"
    exit 1
}

# ── Setup Rclone Env ───────────────────────────────────────────────
# We use environment variables to avoid needing a persistent rclone.conf
$env:RCLONE_CONFIG_CONTABO_TYPE = "s3"
$env:RCLONE_CONFIG_CONTABO_PROVIDER = "other"
$env:RCLONE_CONFIG_CONTABO_ACCESS_KEY_ID = $accessKey
$env:RCLONE_CONFIG_CONTABO_SECRET_ACCESS_KEY = $secretKey
$env:RCLONE_CONFIG_CONTABO_ENDPOINT = $endpoint
$env:RCLONE_CONFIG_CONTABO_REGION = $region

# ── Check if rclone is installed ───────────────────────────────────
$rcloneCmd = "rclone"
if (-not (Get-Command $rcloneCmd -ErrorAction SilentlyContinue)) {
    $localRclone = Join-Path (Split-Path $ScriptDir -Parent) "bin\rclone.exe"
    if (Test-Path $localRclone) {
        $rcloneCmd = $localRclone
    } else {
        Write-Error "rclone is not installed. Please install it first: https://rclone.org/downloads/"
        Write-Output "On Windows (PowerShell): iwr https://rclone.org/install.ps1 | iex"
        exit 1
    }
}

# ── Execute Command ────────────────────────────────────────────────
switch ($Command) {
    "ls" {
        if (-not $Bucket) {
            Write-Output "Listing all buckets:"
            & $rcloneCmd lsd contabo:
        } else {
            Write-Output "Listing files in bucket [$Bucket]:"
            & $rcloneCmd ls "contabo:$Bucket/$Path1"
        }
    }

    "push" {
        if (-not $Path1 -or -not $Path2) { Write-Error "Usage: push <local_file> <remote_path>"; exit 1 }
        Write-Output "Uploading $Path1 to contabo:$Bucket/$Path2 ..."
        & $rcloneCmd copyto "$Path1" "contabo:$Bucket/$Path2" -P
    }

    "pull" {
        if (-not $Path1 -or -not $Path2) { Write-Error "Usage: pull <remote_path> <local_file>"; exit 1 }
        Write-Output "Downloading contabo:$Bucket/$Path1 to $Path2 ..."
        & $rcloneCmd copyto "contabo:$Bucket/$Path1" "$Path2" -P
    }

    "sync" {
        if (-not $Path1 -or -not $Path2) { Write-Error "Usage: sync <local_dir> <remote_dir>"; exit 1 }
        Write-Output "Syncing $Path1 -> contabo:$Bucket/$Path2 ..."
        & $rcloneCmd sync "$Path1" "contabo:$Bucket/$Path2" -P
    }

    "rm" {
        if (-not $Path1) { Write-Error "Usage: rm <remote_path>"; exit 1 }
        Write-Output "Deleting contabo:$Bucket/$Path1 ..."
        & $rcloneCmd delete "contabo:$Bucket/$Path1"
    }
}
