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

$prepareRemote = @"
`$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent '$RemoteProjectRoot') | Out-Null
"@
$prepareEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($prepareRemote))
ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $prepareEncoded"

Write-Host "Uploading code archive with scp progress..."
scp -o BatchMode=yes $ArchivePath "${RemoteUser}@${RemoteHost}:$remoteArchiveScp"

$remotePs = @"
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'
`$archive = '$remoteArchive'
`$project = '$RemoteProjectRoot'
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
Expand-Archive -LiteralPath `$archive -DestinationPath `$project -Force
Get-ChildItem -LiteralPath `$project -File | Select-Object Name,Length | ConvertTo-Json
"@
$remoteEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remotePs))
ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $remoteEncoded"

if (-not $KeepArchive) {
    $cleanupRemote = "Remove-Item -LiteralPath '$remoteArchive' -Force -ErrorAction SilentlyContinue"
    $cleanupEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cleanupRemote))
    ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $cleanupEncoded"
    Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
}

Write-Host "Remote code synced: $RemoteProjectRoot"
