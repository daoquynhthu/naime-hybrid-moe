param(
    [string]$OutputDir = "experiments\runs",
    [string]$RunPrefix = "v7_local_ablation",
    [int]$MaxSteps = 20,
    [int]$EvalEvery = 10,
    [int]$EvalMaxBatches = 1,
    [int]$SeqLen = 64,
    [int]$BatchSize = 2,
    [string]$Device = "cpu",
    [switch]$UseVoice
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }
$ablationRoot = Join-Path $runRoot $RunPrefix
$logRoot = Join-Path $ablationRoot "logs"
$dataPath = Join-Path $ablationRoot "tiny_byte_corpus.txt"
New-Item -ItemType Directory -Force -Path $ablationRoot, $logRoot | Out-Null

if (-not (Test-Path -LiteralPath $dataPath)) {
    $line = "NAIME V7 typed dynamics local ablation corpus. Internal state must remain causal, useful, and stable across chunks. "
    [System.IO.File]::WriteAllText($dataPath, ($line * 4096), [System.Text.Encoding]::UTF8)
}

$runs = @(
    [pscustomobject]@{
        Name = "baseline_steps0"
        DynamicsSteps = 0
        DynamicDepth = $false
        MinSteps = 1
        MaxSteps = 0
        Threshold = 0.0
        HiddenScale = 0.01
        ProbeDynamics = $false
        ProbeSwap = $false
        ProbeErase = $false
    },
    [pscustomobject]@{
        Name = "typed_steps1"
        DynamicsSteps = 1
        DynamicDepth = $false
        MinSteps = 1
        MaxSteps = 0
        Threshold = 0.0
        HiddenScale = 0.01
        ProbeDynamics = $true
        ProbeSwap = $true
        ProbeErase = $true
    },
    [pscustomobject]@{
        Name = "typed_steps2"
        DynamicsSteps = 2
        DynamicDepth = $false
        MinSteps = 1
        MaxSteps = 0
        Threshold = 0.0
        HiddenScale = 0.01
        ProbeDynamics = $true
        ProbeSwap = $true
        ProbeErase = $true
    },
    [pscustomobject]@{
        Name = "typed_state_only"
        DynamicsSteps = 1
        DynamicDepth = $false
        MinSteps = 1
        MaxSteps = 0
        Threshold = 0.0
        HiddenScale = 0.0
        ProbeDynamics = $true
        ProbeSwap = $true
        ProbeErase = $true
    },
    [pscustomobject]@{
        Name = "dynamic_depth"
        DynamicsSteps = 1
        DynamicDepth = $true
        MinSteps = 1
        MaxSteps = 3
        Threshold = 0.001
        HiddenScale = 0.01
        ProbeDynamics = $true
        ProbeSwap = $true
        ProbeErase = $true
    }
)

$summary = @()
foreach ($run in $runs) {
    $runName = "$RunPrefix`_$($run.Name)"
    $logPath = Join-Path $logRoot "$runName.log"
    $args = @(
        "--architecture", "naime_v7_typed_dynamics",
        "--run-name", $runName,
        "--output-dir", $ablationRoot,
        "--data-path", $dataPath,
        "--data-format", "byte",
        "--vocab-size", "257",
        "--seq-len", "$SeqLen",
        "--batch-size", "$BatchSize",
        "--max-steps", "$MaxSteps",
        "--eval-every", "$EvalEvery",
        "--eval-max-batches", "$EvalMaxBatches",
        "--eval-sampling", "sequential",
        "--device", $Device,
        "--d-model", "64",
        "--n-layers", "3",
        "--n-dense-layers", "1",
        "--n-heads", "4",
        "--n-kv-heads", "2",
        "--d-ff", "128",
        "--n-experts", "2",
        "--top-k", "1",
        "--expert-hidden-dim", "96",
        "--moe-dispatch-mode", "dense",
        "--stride", "4",
        "--window", "8",
        "--z-dim", "16",
        "--semantic-memory-slots", "2",
        "--world-state-slots", "2",
        "--self-state-slots", "2",
        "--v7-latent-slots", "2",
        "--v7-dynamics-steps", "$($run.DynamicsSteps)",
        "--v7-min-dynamics-steps", "$($run.MinSteps)",
        "--v7-max-dynamics-steps", "$($run.MaxSteps)",
        "--v7-dynamic-convergence-threshold", "$($run.Threshold)",
        "--v7-hidden-write-scale", "$($run.HiddenScale)",
        "--v7-latent-write-scale", "0.03",
        "--v7-max-hidden-write-ratio", "0.05",
        "--causal-state-stride", "$SeqLen",
        "--log-every", "$MaxSteps",
        "--save-every", "0",
        "--latest-every", "0",
        "--keep-last-n", "0",
        "--metrics-flush-every", "1",
        "--metrics-fsync-every", "0",
        "--no-amp",
        "--no-async-checkpoint"
    )
    if ($run.DynamicDepth) { $args += "--v7-dynamic-depth" }
    if ($run.ProbeDynamics) { $args += "--eval-v7-dynamics-gain" }
    if ($run.ProbeSwap) { $args += "--eval-v7-state-swap" }
    if ($run.ProbeErase) { $args += "--eval-v7-state-erase" }

    Write-Host "RUN $runName"
    & "$PSScriptRoot\train.ps1" -UseVoice:$UseVoice @args *> $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Ablation run failed: $runName. See $logPath"
    }

    $metricsPath = Join-Path $ablationRoot "$runName\metrics.jsonl"
    $lastMetric = $null
    if (Test-Path -LiteralPath $metricsPath) {
        $lastMetric = Get-Content -LiteralPath $metricsPath |
            ForEach-Object { $_ | ConvertFrom-Json } |
            Where-Object { $_.PSObject.Properties["val_lm_loss"] } |
            Select-Object -Last 1
        if ($null -eq $lastMetric) {
            $lastMetric = Get-Content -LiteralPath $metricsPath -Tail 1 | ConvertFrom-Json
        }
    }
    $summary += [pscustomobject]@{
        run_name = $runName
        steps = $run.DynamicsSteps
        hidden_scale = $run.HiddenScale
        dynamic = $run.DynamicDepth
        dynamic_depth = if ($lastMetric -and $lastMetric.PSObject.Properties["val_v7_dynamic_depth_mean"]) { $lastMetric.val_v7_dynamic_depth_mean } else { "" }
        halt_fraction = if ($lastMetric -and $lastMetric.PSObject.Properties["val_v7_dynamic_halt_fraction"]) { $lastMetric.val_v7_dynamic_halt_fraction } else { "" }
        lm = if ($lastMetric -and $lastMetric.PSObject.Properties["lm_loss"]) {
            $lastMetric.lm_loss
        } elseif ($lastMetric -and $lastMetric.PSObject.Properties["loss_lm"]) {
            $lastMetric.loss_lm
        } else {
            ""
        }
        val_lm = if ($lastMetric -and $lastMetric.PSObject.Properties["val_lm_loss"]) { $lastMetric.val_lm_loss } else { "" }
        v7_gain = if ($lastMetric -and $lastMetric.PSObject.Properties["val_v7_dynamics_gain_lm"]) { $lastMetric.val_v7_dynamics_gain_lm } else { "" }
        swap_delta = if ($lastMetric -and $lastMetric.PSObject.Properties["val_v7_state_swap_delta_lm"]) { $lastMetric.val_v7_state_swap_delta_lm } else { "" }
        latent_erase = if ($lastMetric -and $lastMetric.PSObject.Properties["val_v7_latent_erase_delta_lm"]) { $lastMetric.val_v7_latent_erase_delta_lm } else { "" }
        log = $logPath
    }
}

$summaryPath = Join-Path $ablationRoot "summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
Write-Host "Summary: $summaryPath"
$summary | Format-Table -AutoSize
