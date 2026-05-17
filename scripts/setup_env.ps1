<#
.SYNOPSIS
One-time environment setup when migrating NAIME-Hybrid to a new machine.
Creates venv, installs dependencies, fixes Triton JIT paths.
#>

$script:ProjectRoot = Split-Path -Parent $PSScriptRoot

# 1. Create venv with system Python 3.12
Write-Host "=== Step 1: Create virtual environment ==="
$python312 = Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -match "Python312" }
if (-not $python312) {
    $python312Paths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Python312\python.exe"
    )
    $python312 = $python312Paths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $python312) {
        Write-Error "Python 3.12 not found. Install it first."
        exit 1
    }
} else {
    $python312 = $python312.Source
}

$venvDir = Join-Path $ProjectRoot ".venv312"
if (-not (Test-Path $venvDir)) {
    & $python312 -m venv $venvDir
    Write-Host "  venv created at $venvDir"
}

# 2. Install dependencies
Write-Host "=== Step 2: Install dependencies ==="
$pip = Join-Path $venvDir "Scripts\python.exe"
& $pip -m pip install --upgrade pip -q
& $pip -m pip install -r (Join-Path $ProjectRoot "requirements.txt") --extra-index-url https://download.pytorch.org/whl/cu128

# 3. Set up Triton JIT headers and libs
Write-Host "=== Step 3: Triton JIT setup ==="
$includeSrc = Join-Path $venvDir "Include"
$libsSrc = Join-Path $venvDir "libs"
$includeLink = Join-Path $ProjectRoot "Include"
$libsLink = Join-Path $ProjectRoot "libs"

# Copy from system Python if venv doesn't have them
if (-not (Test-Path (Join-Path $includeSrc "Python.h"))) {
    $sysInclude = (Split-Path $python312) -replace 'Scripts\\?$','' -replace 'python\.exe$',''
    $sysInclude = Join-Path $sysInclude "include"
    if (Test-Path $sysInclude) {
        Copy-Item -Recurse -Force $sysInclude $includeSrc
    }
}
if (-not (Test-Path (Join-Path $libsSrc "python312.lib"))) {
    $sysLibs = (Split-Path $python312) -replace 'Scripts\\?$','' -replace 'python\.exe$',''
    $sysLibs = Join-Path $sysLibs "libs"
    if (Test-Path $sysLibs) {
        Copy-Item -Recurse -Force $sysLibs $libsSrc
    }
}

# Create root junctions (sysconfig expects them at project root)
if (Test-Path $includeLink) { Remove-Item $includeLink -Force -Recurse -ErrorAction SilentlyContinue }
if (Test-Path $libsLink) { Remove-Item $libsLink -Force -Recurse -ErrorAction SilentlyContinue }
New-Item -ItemType Junction -Path $includeLink -Target $includeSrc -Force | Out-Null
New-Item -ItemType Junction -Path $libsLink -Target $libsSrc -Force | Out-Null
Write-Host "  Triton JIT paths ready"

# 4. Verify
Write-Host "=== Step 4: Verify ==="
& $pip -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Error "torch import failed"
    exit 1
}
& $pip -m pytest -q (Join-Path $ProjectRoot "tests") 2>&1 | Select-Object -Last 3
Write-Host "Setup complete."
