param(
    [string]$RemoteUser = "",
    [string]$RemoteHost = "",
    [string]$RemoteRoot = "",
    [string]$LocalDataset = "",
    [string]$RemoteDatasetName = "",
    [string]$WorkspaceConfig = "",
    [string]$ArchivePath = "",
    [switch]$SkipArchive,
    [switch]$SkipUpload,
    [switch]$KeepArchive
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $RemoteUser) { $RemoteUser = Resolve-NaimeConfigValue $workspace "remote.user" "NAIME_REMOTE_USER" }
if (-not $RemoteHost) { $RemoteHost = Resolve-NaimeConfigValue $workspace "remote.host" "NAIME_REMOTE_HOST" }
if (-not $RemoteRoot) { $RemoteRoot = Resolve-NaimeConfigValue $workspace "remote.root" "NAIME_REMOTE_ROOT" }
if (-not $LocalDataset) { $LocalDataset = Resolve-NaimeConfigValue $workspace "local.fineweb_edu_1b" "NAIME_LOCAL_DATASET" }

if (-not (Test-Path -LiteralPath $LocalDataset)) {
    throw "Local dataset not found: $LocalDataset"
}

$localDatasetPath = (Resolve-Path -LiteralPath $LocalDataset).Path
if (-not $RemoteDatasetName) {
    $RemoteDatasetName = Split-Path -Leaf $localDatasetPath
}
if (-not $ArchivePath) {
    $ArchivePath = Join-Path $env:TEMP "$RemoteDatasetName-naime-dataset.zip"
}

$remoteDatasetRoot = "$RemoteRoot\datasets"
$remoteArchive = "$remoteDatasetRoot\$RemoteDatasetName.zip"
$remoteDataset = "$remoteDatasetRoot\$RemoteDatasetName"
$remoteArchiveScp = $remoteArchive -replace "\\", "/"
$python = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Write-Host "Local dataset : $localDatasetPath"
Write-Host "Archive       : $ArchivePath"
Write-Host "Remote archive: ${RemoteUser}@${RemoteHost}:$remoteArchive"
Write-Host "Remote dataset: $remoteDataset"

if (-not $SkipArchive) {
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
    $env:NAIME_DATASET_SRC = $localDatasetPath
    $env:NAIME_DATASET_ZIP = $ArchivePath
    @'
import os
import sys
import time
import zipfile
from pathlib import Path

src = Path(os.environ["NAIME_DATASET_SRC"])
dst = Path(os.environ["NAIME_DATASET_ZIP"])
files = [p for p in src.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
done = 0
last_print = 0.0
start = time.time()

print(f"Packing {len(files)} files, {total / (1024 ** 3):.2f} GiB")
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for idx, path in enumerate(files, 1):
        rel = path.relative_to(src.parent)
        zf.write(path, rel.as_posix())
        done += path.stat().st_size
        now = time.time()
        if now - last_print >= 1.0 or idx == len(files):
            pct = 100.0 * done / max(1, total)
            speed = done / max(1e-6, now - start) / (1024 ** 2)
            sys.stdout.write(
                f"\rarchive {pct:6.2f}%  {done/(1024**3):.2f}/{total/(1024**3):.2f} GiB  {speed:.1f} MiB/s"
            )
            sys.stdout.flush()
            last_print = now
print()
print(f"Archive complete: {dst} ({dst.stat().st_size / (1024 ** 3):.2f} GiB)")
'@ | & $python -
}

$prepareRemote = @"
`$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path '$remoteDatasetRoot' | Out-Null
if (Test-Path -LiteralPath '$remoteDataset') {
    Write-Host 'Remote dataset already exists; keeping it untouched before upload: $remoteDataset'
}
"@
$prepareEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($prepareRemote))
ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $prepareEncoded"

if (-not $SkipUpload) {
    Write-Host "Uploading archive with scp progress..."
    scp -o BatchMode=yes $ArchivePath "${RemoteUser}@${RemoteHost}:$remoteArchiveScp"
}

$remotePs = @"
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'
`$archive = '$remoteArchive'
`$dataset = '$remoteDataset'
if (-not (Test-Path -LiteralPath `$archive)) {
    throw "Remote archive not found: `$archive"
}
if (Test-Path -LiteralPath `$dataset) {
    Remove-Item -LiteralPath `$dataset -Recurse -Force
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent `$dataset) | Out-Null
Expand-Archive -LiteralPath `$archive -DestinationPath (Split-Path -Parent `$dataset) -Force
`$files = Get-ChildItem -LiteralPath `$dataset -Recurse -File
`$size = (`$files | Measure-Object Length -Sum).Sum
[pscustomobject]@{
    Dataset = `$dataset
    Files = `$files.Count
    GiB = [math]::Round(`$size / 1GB, 3)
} | ConvertTo-Json
"@
$remoteEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remotePs))
ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $remoteEncoded"

if (-not $KeepArchive) {
    $cleanupRemote = "Remove-Item -LiteralPath '$remoteArchive' -Force -ErrorAction SilentlyContinue"
    $cleanupEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cleanupRemote))
    ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $cleanupEncoded"
    if (Test-Path -LiteralPath $ArchivePath) {
        Remove-Item -LiteralPath $ArchivePath -Force
    }
}

Write-Host "Remote dataset ready: $remoteDataset"
