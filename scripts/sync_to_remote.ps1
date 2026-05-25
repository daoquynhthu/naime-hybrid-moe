param(
    [string]$RemoteUser = "",
    [string]$RemoteHost = "",
    [string]$RemoteProjectRoot = "",
    [string]$WorkspaceConfig = "",
    [string]$ArchivePath = "",
    [switch]$KeepArchive
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig
if (-not $RemoteUser) { $RemoteUser = Resolve-NaimeConfigValue $workspace "remote.user" "NAIME_REMOTE_USER" }
if (-not $RemoteHost) { $RemoteHost = Resolve-NaimeConfigValue $workspace "remote.host" "NAIME_REMOTE_HOST" }
if (-not $RemoteProjectRoot) { $RemoteProjectRoot = Resolve-NaimeConfigValue $workspace "remote.repo" "NAIME_REMOTE_REPO" }
if (-not $ArchivePath) {
    $ArchivePath = Join-Path $env:TEMP "naime-hybrid-moe-code.zip"
}
$remoteArchive = "$RemoteProjectRoot-code.zip"
$remoteArchiveScp = $remoteArchive -replace "\\", "/"
$python = Join-Path $repoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Write-Host "Repo          : $repoRoot"
Write-Host "Archive       : $ArchivePath"
Write-Host "Remote project: ${RemoteUser}@${RemoteHost}:$RemoteProjectRoot"

if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

$env:NAIME_SYNC_SRC = $repoRoot
$env:NAIME_SYNC_ZIP = $ArchivePath
@'
import os
import sys
import time
import zipfile
from pathlib import Path

src = Path(os.environ["NAIME_SYNC_SRC"])
dst = Path(os.environ["NAIME_SYNC_ZIP"])
include_roots = ["src", "scripts", "configs", "docs", "tests"]
include_files = ["README.md", "SKILL.md", "pyproject.toml", "requirements.txt", ".gitignore"]
excluded_dirs = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv312",
    "__pycache__",
    "experiments",
    "data",
    "assets",
}
excluded_nested_dirs = {"__pycache__"}
excluded_suffixes = {".pyc", ".pt", ".pth", ".zip", ".jsonl", ".log", ".csv"}

def is_allowed(path: Path) -> bool:
    rel = path.relative_to(src)
    if rel.parts and rel.parts[0] in excluded_dirs:
        return False
    if any(part in excluded_nested_dirs for part in rel.parts):
        return False
    if path.suffix.lower() in excluded_suffixes:
        return False
    return rel.parts[0] in include_roots or rel.as_posix() in include_files

files = [p for p in src.rglob("*") if p.is_file() and is_allowed(p)]
total = sum(p.stat().st_size for p in files)
done = 0
last_print = 0.0
start = time.time()

print(f"Packing code {len(files)} files, {total / (1024 ** 2):.1f} MiB")
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
    for idx, path in enumerate(files, 1):
        rel = path.relative_to(src)
        zf.write(path, rel.as_posix())
        done += path.stat().st_size
        now = time.time()
        if now - last_print >= 0.5 or idx == len(files):
            pct = 100.0 * done / max(1, total)
            speed = done / max(1e-6, now - start) / (1024 ** 2)
            sys.stdout.write(f"\rarchive {pct:6.2f}%  {done/(1024**2):.1f}/{total/(1024**2):.1f} MiB  {speed:.1f} MiB/s")
            sys.stdout.flush()
            last_print = now
print()
print(f"Archive complete: {dst} ({dst.stat().st_size / (1024 ** 2):.1f} MiB)")
'@ | & $python -

Write-Host "Uploading code archive with scp progress..."
scp -o BatchMode=yes $ArchivePath "${RemoteUser}@${RemoteHost}:$remoteArchiveScp"

$syncId = Get-Random -Maximum 999999
$remoteManifestPath = "$RemoteProjectRoot-code-manifest-$syncId.txt"

$remotePs = @"
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'
`$archive = '$remoteArchive'
`$project = '$RemoteProjectRoot'
`$manifestRemote = '$remoteManifestPath'

if (-not (Test-Path -LiteralPath `$archive)) {
    throw "Remote archive not found: `$archive"
}

if (Test-Path -LiteralPath `$project) {
    Remove-Item -LiteralPath `$project\src -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath `$project\scripts -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath `$project\configs -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath `$project\docs -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath `$project\tests -Recurse -Force -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Force -Path `$project | Out-Null
}

# Extract zip and capture any errors
`$errors = @()
try {
    Expand-Archive -LiteralPath `$archive -DestinationPath `$project -Force
} catch {
    `$errors += "Expand-Archive failed: $_"
    
    # Fallback: use .NET directly if Expand-Archive fails
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        `$zip = [System.IO.Compression.ZipFile]::OpenRead(`$archive)
        foreach (`$entry in `$zip.Entries) {
            `$dest = Join-Path `$project `$entry.FullName
            `$dir = Split-Path -Parent `$dest
            if (-not (Test-Path -LiteralPath `$dir)) { New-Item -ItemType Directory -Force -Path `$dir | Out-Null }
            try { [System.IO.Compression.ZipFileExtensions]::ExtractToFile(`$entry, `$dest, `$true) } catch { `$errors += "extract `$(`$entry.FullName): $_" }
        }
        `$zip.Dispose()
    } catch {
        `$errors += "Fallback extraction also failed: $_"
    }
}

# Compute SHA256 manifest of extracted files
`$localManifest = @()
Get-ChildItem -LiteralPath `$project -Recurse -File | ForEach-Object {
    `$rel = `$_.FullName.Substring(`$project.Length + 1) -replace '\\', '/'
    `$hash = (Get-FileHash -LiteralPath `$_.FullName -Algorithm SHA256).Hash
    `$localManifest += "`$hash  `$rel"
}
`$localManifest | Sort-Object | Out-File -LiteralPath `$manifestRemote -Encoding ASCII

# Count extracted files and report
`$fileCount = `$localManifest.Count
`$errorCount = `$errors.Count
Write-Output "SYNC_RESULT: files=`$fileCount errors=`$errorCount"
if (`$errorCount -gt 0) { Write-Output ("SYNC_ERRORS: " + (`$errors -join '|')) }
"@
$remoteEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remotePs))
$result = ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -EncodedCommand $remoteEncoded"
Write-Host "Remote extraction result:"
Write-Host $result

# Fetch and verify manifest
$localManifestPath = Join-Path $env:TEMP "naime-sync-manifest-$syncId.txt"
scp -o BatchMode=yes "${RemoteUser}@${RemoteHost}:$remoteManifestPath" $localManifestPath

if (-not (Test-Path -LiteralPath $localManifestPath)) {
    Write-Warning "Could not fetch remote manifest; sync may have failed"
} else {
    $remoteCount = (Get-Content $localManifestPath | Where-Object { $_ -match '^[A-F0-9]{64}' }).Count
    Write-Host "Remote verification: $remoteCount files synced"
    Remove-Item -LiteralPath $localManifestPath -Force -ErrorAction SilentlyContinue
}

if (-not $KeepArchive) {
    $cleanupRemote = "`$ProgressPreference='SilentlyContinue'; Remove-Item -LiteralPath '$remoteArchive' -Force -ErrorAction SilentlyContinue; if (Test-Path '$remoteManifestPath') { Remove-Item -LiteralPath '$remoteManifestPath' -Force -ErrorAction SilentlyContinue }"
    $cleanupEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cleanupRemote))
    ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -EncodedCommand $cleanupEncoded"
    Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
}

Write-Host "Remote code synced: $RemoteProjectRoot ($remoteCount files)"
