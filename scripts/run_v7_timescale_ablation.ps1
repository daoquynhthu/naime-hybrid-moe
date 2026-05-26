param(
    [string]$WorkspaceConfig = "",
    [string]$DataPath = "",
    [ValidateSet("byte", "hf_disk")]
    [string]$DataFormat = "byte",
    [string]$OutputDir = "experiments\runs",
    [string]$RunPrefix = "",
    [int]$MaxSteps = 160,
    [int]$EvalEvery = 40,
    [int]$EvalMaxBatches = 2,
    [int]$SeqLen = 128,
    [int]$BatchSize = 2,
    [int]$StateChunkSize = 128,
    [int]$InternalLatentAdaptSteps = 0,
    [string]$Device = "cpu",
    [switch]$UseVoice,
    [switch]$ContinueOnFailure,
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not [string]::IsNullOrWhiteSpace($WorkspaceConfig)) {
    . "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
    $workspace = Get-NaimeWorkspaceConfig -AllowMissing
    if ([string]::IsNullOrWhiteSpace($DataPath)) {
        $DataPath = Resolve-NaimeConfigValue $workspace "local.fineweb_edu_50m" "NAIME_LOCAL_ABLATION_DATA" ""
        if (-not [string]::IsNullOrWhiteSpace($DataPath)) {
            $DataFormat = "hf_disk"
        }
    }
    if ($OutputDir -eq "experiments\runs") {
        $OutputDir = Resolve-NaimeConfigValue $workspace "local.run_root" "NAIME_RUN_ROOT" $OutputDir
    }
}

if ([string]::IsNullOrWhiteSpace($RunPrefix)) {
    $RunPrefix = "v7_timescale_ablation_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$runRoot = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }
$ablationRoot = Join-Path $runRoot $RunPrefix
$logRoot = Join-Path $ablationRoot "logs"
New-Item -ItemType Directory -Force -Path $ablationRoot, $logRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($DataPath)) {
    $DataPath = Join-Path $ablationRoot "tiny_byte_corpus.txt"
    $DataFormat = "byte"
}

if ($DataFormat -eq "byte" -and -not (Test-Path -LiteralPath $DataPath)) {
    $line = "NAIME V7 timescale ablation corpus. Typed latent, world, and self states must expose measurable baseline update rates. "
    [System.IO.File]::WriteAllText($DataPath, ($line * 8192), [System.Text.Encoding]::UTF8)
}

if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "DataPath not found: $DataPath"
}

$dataArgs = @("--data-path", $DataPath, "--data-format", $DataFormat)
if ($DataFormat -eq "byte") {
    $dataArgs += @("--vocab-size", "257")
} else {
    $dataArgs += @("--data-split", "train", "--vocab-size", "50257")
}

$runs = @(
    [pscustomobject]@{ Name = "uniform_1_1_1"; Latent = 1.0; World = 1.0; Self = 1.0 },
    [pscustomobject]@{ Name = "latent_fast"; Latent = 1.5; World = 1.0; Self = 1.0 },
    [pscustomobject]@{ Name = "world_fast"; Latent = 1.0; World = 1.5; Self = 1.0 },
    [pscustomobject]@{ Name = "self_fast"; Latent = 1.0; World = 1.0; Self = 1.5 },
    [pscustomobject]@{ Name = "world_slow"; Latent = 1.0; World = 0.5; Self = 1.0 },
    [pscustomobject]@{ Name = "self_slow"; Latent = 1.0; World = 1.0; Self = 0.5 },
    [pscustomobject]@{ Name = "all_half"; Latent = 0.5; World = 0.5; Self = 0.5 },
    [pscustomobject]@{ Name = "all_high"; Latent = 1.5; World = 1.5; Self = 1.5 }
)

$manifest = [pscustomobject]@{
    run_prefix = $RunPrefix
    purpose = "Measure V7 typed-state baseline update rates. These settings are experimental variables, not architecture priors."
    data_path = $DataPath
    data_format = $DataFormat
    output_dir = $ablationRoot
    max_steps = $MaxSteps
    eval_every = $EvalEvery
    eval_max_batches = $EvalMaxBatches
    seq_len = $SeqLen
    batch_size = $BatchSize
    runs = $runs
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

$commonModelArgs = @(
    "--architecture", "naime_v7_typed_dynamics",
    "--seq-len", "$SeqLen",
    "--batch-size", "$BatchSize",
    "--num-workers", "0",
    "--grad-accum-steps", "1",
    "--max-steps", "$MaxSteps",
    "--eval-every", "$EvalEvery",
    "--eval-max-batches", "$EvalMaxBatches",
    "--eval-sampling", "sequential",
    "--device", $Device,
    "--d-model", "128",
    "--n-layers", "4",
    "--n-dense-layers", "1",
    "--n-heads", "4",
    "--n-kv-heads", "2",
    "--d-ff", "384",
    "--dropout", "0.02",
    "--n-experts", "4",
    "--top-k", "2",
    "--expert-hidden-dim", "256",
    "--moe-dispatch-mode", "dense",
    "--stride", "8",
    "--causal-state-stride", "$SeqLen",
    "--window", "16",
    "--z-dim", "32",
    "--semantic-router-mode", "hybrid",
    "--semantic-memory-slots", "3",
    "--world-state-slots", "3",
    "--self-state-slots", "3",
    "--v7-latent-slots", "3",
    "--v7-dynamics-steps", "2",
    "--v7-hidden-write-scale", "0.01",
    "--v7-latent-write-scale", "0.03",
    "--v7-state-write-scale", "0.02",
    "--v7-controller-mode", "fixed",
    "--v7-state-chunk-size", "$StateChunkSize",
    "--v7-internal-latent-adapt-steps", "$InternalLatentAdaptSteps",
    "--v7-max-hidden-write-ratio", "0.05",
    "--lambda-load", "0.01",
    "--lambda-sparse", "0.01",
    "--lambda-kl", "0.003",
    "--lambda-state-pred", "0.01",
    "--lambda-self-pred", "0.005",
    "--learning-rate", "0.00012",
    "--warmup-steps", "40",
    "--min-lr-ratio", "0.05",
    "--weight-decay", "0.05",
    "--grad-clip", "1.0",
    "--log-every", "$EvalEvery",
    "--save-every", "0",
    "--latest-every", "0",
    "--keep-last-n", "0",
    "--metrics-flush-every", "5",
    "--metrics-fsync-every", "0",
    "--resume", "none",
    "--resume-lr-policy", "reset",
    "--no-async-checkpoint",
    "--eval-state-carry",
    "--eval-v7-dynamics-gain",
    "--eval-v7-state-swap",
    "--eval-v7-state-erase"
)

$summary = @()
foreach ($run in $runs) {
    $runName = "$RunPrefix`_$($run.Name)"
    $logPath = Join-Path $logRoot "$runName.log"
    $runDir = Join-Path $ablationRoot $runName
    $args = @(
        "--run-name", $runName,
        "--output-dir", $ablationRoot
    ) + $dataArgs + $commonModelArgs + @(
        "--v7-latent-timescale", "$($run.Latent)",
        "--v7-world-timescale", "$($run.World)",
        "--v7-self-timescale", "$($run.Self)"
    )

    if ($PrintOnly) {
        $quotedArgs = $args | ForEach-Object {
            if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
        }
        Write-Host "RUN $runName"
        Write-Host "& `"$PSScriptRoot\train.ps1`" -UseVoice:`$$UseVoice $($quotedArgs -join ' ')"
        continue
    }

    Write-Host "START $runName latent=$($run.Latent) world=$($run.World) self=$($run.Self)"
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
        throw "Timescale ablation run failed: $runName. See $logPath"
    }

    $metricsPath = Join-Path $runDir "metrics.jsonl"
    $metrics = @(Read-MetricsJsonl -Path $metricsPath)
    $evalRows = @($metrics | Where-Object { $_.PSObject.Properties["val_lm_loss"] })
    $lastEval = if ($evalRows.Count -gt 0) { $evalRows[-1] } else { $null }
    $bestEval = if ($evalRows.Count -gt 0) { $evalRows | Sort-Object {[double]$_.val_lm_loss} | Select-Object -First 1 } else { $null }
    $lastTrain = if ($metrics.Count -gt 0) { $metrics[-1] } else { $null }

    $summary += [pscustomobject]@{
        run_name = $runName
        exit_code = $exitCode
        duration_min = [math]::Round(($end - $start).TotalMinutes, 2)
        latent_timescale = $run.Latent
        world_timescale = $run.World
        self_timescale = $run.Self
        evals = $evalRows.Count
        best_val_lm = Get-MetricValue $bestEval "val_lm_loss"
        final_val_lm = Get-MetricValue $lastEval "val_lm_loss"
        final_train_lm = Get-MetricValue $lastTrain "loss_lm"
        v7_gain = Get-MetricValue $lastEval "val_v7_dynamics_gain_lm"
        carry_gain = Get-MetricValue $lastEval "val_state_carry_gain_lm"
        state_swap = Get-MetricValue $lastEval "val_v7_state_swap_delta_lm"
        latent_erase = Get-MetricValue $lastEval "val_v7_latent_erase_delta_lm"
        world_erase = Get-MetricValue $lastEval "val_v7_world_erase_delta_lm"
        self_erase = Get-MetricValue $lastEval "val_v7_self_erase_delta_lm"
        world_delta = Get-MetricValue $lastEval "val_v7_world_delta"
        self_delta = Get-MetricValue $lastEval "val_v7_self_delta"
        latent_delta = Get-MetricValue $lastEval "val_v7_latent_delta"
        controller_delta = Get-MetricValue $lastEval "val_v7_controller_delta"
        causal_segments = Get-MetricValue $lastEval "val_v7_causal_segments"
        hidden_ratio = Get-MetricValue $lastEval "val_v7_hidden_write_ratio"
        effective_latent_scale = Get-MetricValue $lastEval "val_v7_effective_latent_write_scale"
        effective_world_scale = Get-MetricValue $lastEval "val_v7_effective_world_write_scale"
        effective_self_scale = Get-MetricValue $lastEval "val_v7_effective_self_write_scale"
        effective_controller_scale = Get-MetricValue $lastEval "val_v7_effective_controller_write_scale"
        log = $logPath
        metrics = $metricsPath
    }
    Write-Host "DONE $runName exit=$exitCode duration_min=$([math]::Round(($end - $start).TotalMinutes, 2))"
}

if ($PrintOnly) {
    Write-Host "PrintOnly complete. Manifest: $(Join-Path $ablationRoot 'manifest.json')"
    exit 0
}

$summaryPath = Join-Path $ablationRoot "summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

$analysisPath = Join-Path $ablationRoot "analysis.md"
$lines = @()
$lines += "# V7 Timescale Ablation"
$lines += ""
$lines += "This experiment measures typed-state baseline rates. It must not be read as a built-in architecture prior."
$lines += ""
$lines += "- Data: $DataPath"
$lines += "- Data format: $DataFormat"
$lines += "- Steps: $MaxSteps"
$lines += "- SeqLen/Batch: $SeqLen/$BatchSize"
$lines += "- Summary CSV: $summaryPath"
$lines += ""
$lines += "Interpretation guide:"
$lines += "- `v7_gain` checks whether two dynamics steps help versus no V7 dynamics on the probe batch."
$lines += "- `state_swap` checks whether state identity matters; near zero means typed state may be decorative."
$lines += "- `latent/world/self/controller_delta` show observed motion, not value by themselves."
$lines += "- `latent/world/self_erase` and `carry_gain` are stronger usefulness signals than raw norms."
$lines += "- A faster timescale is only justified when it improves validation/usefulness without destabilizing gradients or write ratios."
$lines | Set-Content -LiteralPath $analysisPath -Encoding UTF8

Write-Host "Summary: $summaryPath"
Write-Host "Analysis: $analysisPath"
$summary | Format-Table -AutoSize
