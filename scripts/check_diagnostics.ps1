param(
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

function Invoke-Quiet {
    param(
        [string]$Name,
        [string[]]$Command
    )
    $output = & $Command[0] $Command[1..($Command.Count - 1)] 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED: $Name`n$output"
        exit $LASTEXITCODE
    }
    if ($VerboseOutput) {
        Write-Output "OK: $Name"
        Write-Output $output
    }
}

Push-Location $repoRoot
try {
    Invoke-Quiet "diagnostics py_compile" @(
        $python, "-m", "py_compile",
        "src\naime_hybrid\diagnostics\packet_diagnostics.py",
        "src\naime_hybrid\diagnostics\run_packet_diagnostics.py",
        "src\naime_hybrid\diagnostics\report_builder.py",
        "src\naime_hybrid\diagnostics\trace_context.py",
        "src\naime_hybrid\diagnostics\emitter.py",
        "src\naime_hybrid\diagnostics\training_dynamics.py",
        "src\naime_hybrid\diagnostics\summarize_training_diagnostics.py",
        "src\naime_hybrid\training\train.py",
        "src\naime_hybrid\training\config.py",
        "src\naime_hybrid\training\cli.py"
    )
    Invoke-Quiet "architecture tests" @($python, "-m", "pytest", "tests\test_architecture_forward.py", "-q")
    $errs = $null
    $null = [System.Management.Automation.PSParser]::Tokenize(
        (Get-Content "scripts\run_packet_diagnostics.ps1" -Raw),
        [ref]$errs
    )
    if ($errs) {
        Write-Error "FAILED: run_packet_diagnostics.ps1 parse`n$errs"
        exit 1
    }
    $summaryErrs = $null
    $null = [System.Management.Automation.PSParser]::Tokenize(
        (Get-Content "scripts\summarize_training_diagnostics.ps1" -Raw),
        [ref]$summaryErrs
    )
    if ($summaryErrs) {
        Write-Error "FAILED: summarize_training_diagnostics.ps1 parse`n$summaryErrs"
        exit 1
    }
    $diffOutput = git diff --check 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED: git diff --check`n$diffOutput"
        exit $LASTEXITCODE
    }
    Write-Output "diagnostics checks passed"
} finally {
    Pop-Location
}
