param(
    [string]$WorkspaceConfig = "",
    [string]$DataPath = "",
    [string]$OutputDir = "",
    [string]$RunPrefix = "",
    [int]$MaxSteps = 220,
    [int]$EvalEvery = 55,
    [int]$EvalMaxBatches = 3,
    [int]$SeqLen = 512,
    [int]$BatchSize = 4,
    [string]$Device = "auto",
    [switch]$UseVoice,
    [switch]$ContinueOnFailure
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig -AllowMissing

if ([string]::IsNullOrWhiteSpace($DataPath)) {
    $DataPath = Resolve-NaimeConfigValue $workspace "local.fineweb_edu_50m" "NAIME_LOCAL_ABLATION_DATA" ""
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Resolve-NaimeConfigValue $workspace "local.run_root" "NAIME_RUN_ROOT" "experiments\runs"
}
if ([string]::IsNullOrWhiteSpace($RunPrefix)) {
    $RunPrefix = "v7_full_ablation_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "DataPath not found: $DataPath"
}

$runRoot = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }
$ablationRoot = Join-Path $runRoot $RunPrefix
$logRoot = Join-Path $ablationRoot "logs"
New-Item -ItemType Directory -Force -Path $ablationRoot, $logRoot | Out-Null

$commonModelArgs = @(
    "--data-path", $DataPath,
    "--data-format", "hf_disk",
    "--data-split", "train",
    "--vocab-size", "50257",
    "--seq-len", "$SeqLen",
    "--batch-size", "$BatchSize",
    "--num-workers", "0",
    "--grad-accum-steps", "1",
    "--max-steps", "$MaxSteps",
    "--eval-every", "$EvalEvery",
    "--eval-max-batches", "$EvalMaxBatches",
    "--eval-sampling", "random",
    "--eval-seed", "4321",
    "--device", $Device,
    "--d-model", "256",
    "--n-layers", "6",
    "--n-dense-layers", "2",
    "--n-heads", "4",
    "--n-kv-heads", "2",
    "--d-ff", "1024",
    "--dropout", "0.03",
    "--n-experts", "4",
    "--top-k", "2",
    "--expert-hidden-dim", "512",
    "--moe-dispatch-mode", "dense",
    "--stride", "16",
    "--causal-state-stride", "512",
    "--window", "24",
    "--z-dim", "64",
    "--semantic-router-mode", "hybrid",
    "--semantic-scales", "local_mid_global",
    "--mid-stride", "32",
    "--mid-window", "64",
    "--global-semantic",
    "--semantic-fusion", "concat",
    "--semantic-residual-write",
    "--semantic-write-scale", "0.03",
    "--semantic-pred-horizon", "1",
    "--semantic-router-prior-scale", "1.5",
    "--semantic-router-prior-clip", "2.0",
    "--semantic-router-detach",
    "--semantic-gate-downstream", "clean_prob",
    "--semantic-sparse-alpha", "downstream",
    "--semantic-router-alpha-cap", "0.90",
    "--semantic-alpha-cap-mode", "clamp",
    "--semantic-downstream-deterministic",
    "--semantic-gate-mixer",
    "--semantic-gate-mixer-temperature", "2.5",
    "--semantic-gate-mixer-min-weight", "0.08",
    "--semantic-gate-mixer-max-clean-weight", "0.45",
    "--semantic-gate-mixer-max-state-weight", "0.35",
    "--semantic-state-confidence-mode", "hybrid",
    "--semantic-state-confidence-temperature", "3.0",
    "--semantic-state-confidence-gate",
    "--semantic-memory-read-gate",
    "--semantic-memory-slots", "4",
    "--semantic-memory-write-scale", "0.035",
    "--semantic-memory-hidden-scale", "0.035",
    "--semantic-state-write-scale", "0.075",
    "--layerwise-semantic-schedule",
    "--target-sparsity", "0.45",
    "--logvar-clip", "2.0",
    "--lambda-load", "0.01",
    "--lambda-sparse", "0.01",
    "--lambda-kl", "0.003",
    "--kl-warmup-steps", "500",
    "--lambda-semantic-pred", "0.015",
    "--world-state-slots", "4",
    "--world-router-max-ratio", "0.08",
    "--lambda-state-pred", "0.02",
    "--lambda-slot-diversity", "0.01",
    "--self-state-slots", "4",
    "--self-state-recursion-depth", "1",
    "--self-state-write-scale", "0.03",
    "--self-state-hidden-scale", "0.02",
    "--self-state-world-gate-min", "0.10",
    "--self-state-world-gate-scale", "1.0",
    "--lambda-self-pred", "0.01",
    "--lambda-self-slot-diversity", "0.02",
    "--learning-rate", "0.00012",
    "--warmup-steps", "60",
    "--min-lr-ratio", "0.05",
    "--weight-decay", "0.05",
    "--grad-clip", "1.0",
    "--log-every", "25",
    "--save-every", "0",
    "--latest-every", "0",
    "--keep-last-n", "0",
    "--metrics-flush-every", "5",
    "--metrics-fsync-every", "0",
    "--best-checkpoint-mode", "model",
    "--resume", "none",
    "--resume-lr-policy", "reset",
    "--eval-state-carry"
)

$runs = @(
    [pscustomobject]@{
        Name = "v6_control"
        Architecture = "naime_v6_recursive_self_moe"
        V7Args = @()
    },
    [pscustomobject]@{
        Name = "v7_zero_steps"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @("--v7-dynamics-steps", "0", "--v7-latent-slots", "4")
    },
    [pscustomobject]@{
        Name = "v7_fixed_step1"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @("--v7-dynamics-steps", "1", "--v7-latent-slots", "4")
    },
    [pscustomobject]@{
        Name = "v7_fixed_step2"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @("--v7-dynamics-steps", "2", "--v7-latent-slots", "4")
    },
    [pscustomobject]@{
        Name = "v7_state_only"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @("--v7-dynamics-steps", "2", "--v7-latent-slots", "4", "--v7-hidden-write-scale", "0.0")
    },
    [pscustomobject]@{
        Name = "v7_no_latent_update"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @("--v7-dynamics-steps", "2", "--v7-latent-slots", "4", "--v7-latent-write-scale", "0.0")
    },
    [pscustomobject]@{
        Name = "v7_dynamic_depth"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @(
            "--v7-dynamics-steps", "1",
            "--v7-latent-slots", "4",
            "--v7-dynamic-depth",
            "--v7-min-dynamics-steps", "1",
            "--v7-max-dynamics-steps", "3",
            "--v7-dynamic-convergence-threshold", "0.002"
        )
    },
    [pscustomobject]@{
        Name = "v7_strong_hidden"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @(
            "--v7-dynamics-steps", "2",
            "--v7-latent-slots", "4",
            "--v7-hidden-write-scale", "0.05",
            "--v7-max-hidden-write-ratio", "0.10",
            "--v7-state-write-scale", "0.03"
        )
    },
    [pscustomobject]@{
        Name = "v7_strong_latent"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @(
            "--v7-dynamics-steps", "2",
            "--v7-latent-slots", "4",
            "--v7-latent-write-scale", "0.10",
            "--v7-hidden-write-scale", "0.02",
            "--v7-state-write-scale", "0.05"
        )
    },
    [pscustomobject]@{
        Name = "v7_overdrive"
        Architecture = "naime_v7_typed_dynamics"
        V7Args = @(
            "--v7-dynamics-steps", "3",
            "--v7-latent-slots", "4",
            "--v7-latent-write-scale", "0.15",
            "--v7-hidden-write-scale", "0.12",
            "--v7-max-hidden-write-ratio", "0.20",
            "--v7-state-write-scale", "0.08"
        )
    }
)

$manifest = [pscustomobject]@{
    run_prefix = $RunPrefix
    data_path = $DataPath
    output_dir = $ablationRoot
    max_steps = $MaxSteps
    eval_every = $EvalEvery
    eval_max_batches = $EvalMaxBatches
    seq_len = $SeqLen
    batch_size = $BatchSize
    runs = $runs | ForEach-Object { $_.Name }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ablationRoot "manifest.json") -Encoding UTF8

function Read-MetricsJsonl {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $rows = @()
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $rows += ($line | ConvertFrom-Json)
    }
    return $rows
}

function Get-MetricValue {
    param([object]$Row, [string]$Name)
    if ($null -ne $Row -and $Row.PSObject.Properties[$Name]) {
        return $Row.$Name
    }
    return ""
}

$summary = @()
foreach ($run in $runs) {
    $runName = "$RunPrefix`_$($run.Name)"
    $logPath = Join-Path $logRoot "$runName.log"
    $runDir = Join-Path $ablationRoot $runName
    $args = @(
        "--architecture", $run.Architecture,
        "--run-name", $runName,
        "--output-dir", $ablationRoot
    ) + $commonModelArgs
    if ($run.Architecture -eq "naime_v7_typed_dynamics") {
        $args += @(
            "--v7-latent-write-scale", "0.03",
            "--v7-hidden-write-scale", "0.01",
            "--v7-max-hidden-write-ratio", "0.05",
            "--v7-state-write-scale", "0.02",
            "--eval-v7-dynamics-gain",
            "--eval-v7-state-swap",
            "--eval-v7-state-erase"
        )
    }
    $args += $run.V7Args

    Write-Host "START $runName"
    $start = Get-Date
    try {
        & "$PSScriptRoot\train.ps1" -UseVoice:$UseVoice @args *> $logPath
        $exitCode = $LASTEXITCODE
    } catch {
        $exitCode = 1
        Add-Content -LiteralPath $logPath -Value $_.Exception.ToString()
    }
    $end = Get-Date
    if ($exitCode -ne 0 -and -not $ContinueOnFailure) {
        throw "Ablation run failed: $runName. See $logPath"
    }

    $metricsPath = Join-Path $runDir "metrics.jsonl"
    $metrics = @(Read-MetricsJsonl -Path $metricsPath)
    $evalRows = @($metrics | Where-Object { $_.PSObject.Properties["val_lm_loss"] })
    $lastEval = if ($evalRows.Count -gt 0) { $evalRows[-1] } else { $null }
    $firstEval = if ($evalRows.Count -gt 0) { $evalRows[0] } else { $null }
    $bestEval = if ($evalRows.Count -gt 0) { $evalRows | Sort-Object {[double]$_.val_lm_loss} | Select-Object -First 1 } else { $null }
    $lastTrain = if ($metrics.Count -gt 0) { $metrics[-1] } else { $null }
    $finalVal = Get-MetricValue $lastEval "val_lm_loss"
    $firstVal = Get-MetricValue $firstEval "val_lm_loss"
    $bestVal = Get-MetricValue $bestEval "val_lm_loss"
    $deltaVal = if ($firstVal -ne "" -and $finalVal -ne "") { [double]$firstVal - [double]$finalVal } else { "" }

    $summary += [pscustomobject]@{
        run_name = $runName
        family = $run.Name
        architecture = $run.Architecture
        exit_code = $exitCode
        duration_min = [math]::Round(($end - $start).TotalMinutes, 2)
        evals = $evalRows.Count
        first_val_lm = $firstVal
        final_val_lm = $finalVal
        best_val_lm = $bestVal
        val_lm_drop = $deltaVal
        final_train_lm = Get-MetricValue $lastTrain "loss_lm"
        final_grad = Get-MetricValue $lastTrain "grad_norm"
        final_alpha = Get-MetricValue $lastTrain "alpha_downstream_mean"
        final_entropy = Get-MetricValue $lastTrain "router_entropy"
        v7_steps = Get-MetricValue $lastEval "val_v7_thought_steps"
        v7_gain = Get-MetricValue $lastEval "val_v7_dynamics_gain_lm"
        state_swap = Get-MetricValue $lastEval "val_v7_state_swap_delta_lm"
        latent_erase = Get-MetricValue $lastEval "val_v7_latent_erase_delta_lm"
        world_delta = Get-MetricValue $lastEval "val_v7_world_delta"
        self_delta = Get-MetricValue $lastEval "val_v7_self_delta"
        world_gate = Get-MetricValue $lastEval "val_v7_world_write_gate"
        self_gate = Get-MetricValue $lastEval "val_v7_self_write_gate"
        dynamic_depth = Get-MetricValue $lastEval "val_v7_dynamic_depth_mean"
        halt_fraction = Get-MetricValue $lastEval "val_v7_dynamic_halt_fraction"
        carry_gain = Get-MetricValue $lastEval "val_state_carry_gain_lm"
        metrics = $metricsPath
        log = $logPath
    }
    Write-Host "DONE $runName exit=$exitCode duration_min=$([math]::Round(($end - $start).TotalMinutes, 2))"
}

$summaryPath = Join-Path $ablationRoot "summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

$valid = @($summary | Where-Object { $_.exit_code -eq 0 -and $_.final_val_lm -ne "" })
$best = if ($valid.Count -gt 0) { $valid | Sort-Object {[double]$_.best_val_lm} | Select-Object -First 1 } else { $null }
$baseline = $valid | Where-Object { $_.family -eq "v7_zero_steps" } | Select-Object -First 1
$analysisPath = Join-Path $ablationRoot "analysis.md"
$lines = @()
$lines += "# V7 Local Full Ablation"
$lines += ""
$lines += "- Data: $DataPath"
$lines += "- Steps: $MaxSteps"
$lines += "- Eval every: $EvalEvery"
$lines += "- Eval max batches: $EvalMaxBatches"
$lines += "- SeqLen/Batch: $SeqLen/$BatchSize"
$lines += ""
if ($best) {
    $lines += "Best run by best validation LM: **$($best.family)** (best_val_lm=$($best.best_val_lm))."
}
if ($baseline) {
    $lines += ""
    $lines += "Delta vs V7 zero-step baseline:"
    foreach ($row in $valid) {
        if ($row.family -eq "v7_zero_steps" -or $row.best_val_lm -eq "") { continue }
        $delta = [double]$baseline.best_val_lm - [double]$row.best_val_lm
        $lines += "- $($row.family): best_val_delta=$([math]::Round($delta, 6)), v7_gain=$($row.v7_gain), swap=$($row.state_swap), latent_erase=$($row.latent_erase), world_delta=$($row.world_delta), self_delta=$($row.self_delta), world_gate=$($row.world_gate), self_gate=$($row.self_gate), dynamic_depth=$($row.dynamic_depth), halt=$($row.halt_fraction)"
    }
}
$lines += ""
$lines += "Summary CSV: $summaryPath"
$lines += ""
$lines += "Interpretation guide:"
$lines += "- v7_gain > 0 means configured dynamics improves LM loss versus zero dynamics on the probe batch."
$lines += "- state_swap > 0 means wrong carried state hurts, so state identity is being used."
$lines += "- latent_erase > 0 means the latent field contributes useful information."
$lines += "- world_delta/self_delta show whether V7 is co-evolving typed state, not only rewriting hidden/latent."
$lines += "- world_gate/self_gate should stay bounded; high gates without validation gain suggest uncontrolled state churn."
$lines += "- dynamic_depth < max_steps with stable validation means dynamic halting is saving compute without immediate collapse."
$lines += ""
$lines += "Current experimental note:"
$lines += "- The strong-hidden branch is intentionally included because low hidden-write authority made V7 nearly inert in early ablations."
$lines += "- The dynamic-depth threshold is deliberately low enough to permit extra steps when continue_score remains active."
$lines | Set-Content -LiteralPath $analysisPath -Encoding UTF8

Write-Host "Summary: $summaryPath"
Write-Host "Analysis: $analysisPath"
$summary | Format-Table -AutoSize
