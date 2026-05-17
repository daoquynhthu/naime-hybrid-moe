param(
    [switch]$UseVoice,
    [string]$DataPath = "data\naime\wikitext103_processed",
    [string]$Prefix = "tune_gate",
    [int]$MaxSteps = 2000,
    [int]$EvalEvery = 500,
    [int]$EvalMaxBatches = 8,
    [int]$SeqLen = 512,
    [int]$AutoBatchMax = 32,
    [double[]]$AlphaCaps = @(0.80, 0.90, 1.00),
    [double[]]$TargetSparsities = @(0.45, 0.50),
    [double[]]$SparseMaxes = @(0.08, 0.12),
    [double[]]$PriorScales = @(0.5),
    [switch]$KeepCheckpoints,
    [switch]$DryRun
)

. "$PSScriptRoot\env.ps1" -UseVoice:$UseVoice

$projectRoot = Split-Path -Parent $PSScriptRoot
$runsRoot = Join-Path $projectRoot "experiments\runs"
$summaryPath = Join-Path $runsRoot "$Prefix-summary.csv"
$rows = New-Object System.Collections.Generic.List[object]

if (-not $DryRun) {
    $resolvedDataPath = if (Test-Path -LiteralPath $DataPath) {
        (Resolve-Path -LiteralPath $DataPath).Path
    } else {
        $DataPath
    }
    Write-Host "Dataset preflight: $resolvedDataPath"
    $preflightCode = @"
from datasets import load_from_disk

path = r'''$resolvedDataPath'''
dataset = load_from_disk(path)
print("dataset preflight ok:", path, list(dataset.keys()) if hasattr(dataset, "keys") else type(dataset).__name__)
"@
    $preflightCode | & $env:NAIME_HYBRID_PYTHON -
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dataset preflight failed. Check that the selected Python environment has a compatible datasets package."
        exit $LASTEXITCODE
    }
}

function Get-LatestEval {
    param([string]$MetricsPath)
    if (-not (Test-Path -LiteralPath $MetricsPath)) {
        return $null
    }
    $evalRows = Get-Content -LiteralPath $MetricsPath |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.PSObject.Properties.Name -contains "val_lm_loss" }
    if (-not $evalRows) {
        return $null
    }
    return $evalRows | Sort-Object {[int]$_.step} | Select-Object -Last 1
}

function Get-BestEval {
    param([string]$MetricsPath)
    if (-not (Test-Path -LiteralPath $MetricsPath)) {
        return $null
    }
    $evalRows = Get-Content -LiteralPath $MetricsPath |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.PSObject.Properties.Name -contains "val_lm_loss" }
    if (-not $evalRows) {
        return $null
    }
    return $evalRows | Sort-Object {[double]$_.val_lm_loss} | Select-Object -First 1
}

$total = $AlphaCaps.Count * $TargetSparsities.Count * $SparseMaxes.Count * $PriorScales.Count
$index = 0

foreach ($alphaCap in $AlphaCaps) {
    foreach ($targetSparsity in $TargetSparsities) {
        foreach ($sparseMax in $SparseMaxes) {
            foreach ($priorScale in $PriorScales) {
                $index += 1
                $runName = "{0}_cap{1}_ts{2}_sp{3}_prior{4}" -f `
                    $Prefix,
                    ($alphaCap.ToString("0.00") -replace "\.", "p"),
                    ($targetSparsity.ToString("0.00") -replace "\.", "p"),
                    ($sparseMax.ToString("0.00") -replace "\.", "p"),
                    ($priorScale.ToString("0.00") -replace "\.", "p")
                $runDir = Join-Path $runsRoot $runName
                $extra = @(
                    "--max-steps", "$MaxSteps",
                    "--semantic-router-alpha-cap", "$alphaCap",
                    "--target-sparsity", "$targetSparsity",
                    "--lambda-sparse-max", "$sparseMax",
                    "--semantic-router-prior-scale", "$priorScale",
                    "--resume", "none"
                )

                Write-Host "[$index/$total] $runName"
                Write-Host "  alpha_cap=$alphaCap target=$targetSparsity sparse_max=$sparseMax prior=$priorScale"
                if ($DryRun) {
                    continue
                }

                & "$PSScriptRoot\train_model.ps1" `
                    -UseVoice:$UseVoice `
                    -Model naime_v3_repaired `
                    -RunName $runName `
                    -DataPath $DataPath `
                    -TargetTokens 0 `
                    -SeqLen $SeqLen `
                    -EvalEvery $EvalEvery `
                    -EvalMaxBatches $EvalMaxBatches `
                    -SaveEvery $MaxSteps `
                    -EarlyStopPatience 0 `
                    -EarlyStopMinEvals 0 `
                    -EarlyStopMinDelta 0 `
                    -AutoBatchMax $AutoBatchMax `
                    -PriorScale $priorScale `
                    -TargetSparsity $targetSparsity `
                    -ExtraArgs $extra

                $exitCode = $LASTEXITCODE
                $metricsPath = Join-Path $runDir "metrics.jsonl"
                $best = Get-BestEval $metricsPath
                $last = Get-LatestEval $metricsPath

                $rows.Add([PSCustomObject]@{
                    run_name = $runName
                    exit_code = $exitCode
                    alpha_cap = $alphaCap
                    target_sparsity = $targetSparsity
                    lambda_sparse_max = $sparseMax
                    prior_scale = $priorScale
                    best_step = if ($best) { $best.step } else { $null }
                    best_val_lm_loss = if ($best) { $best.val_lm_loss } else { $null }
                    best_val_ppl = if ($best) { $best.val_ppl } else { $null }
                    best_alpha_downstream = if ($best) { $best.val_alpha_downstream_mean } else { $null }
                    best_alpha_clean_prob = if ($best) { $best.val_alpha_clean_prob_mean } else { $null }
                    best_router_entropy = if ($best) { $best.val_router_entropy } else { $null }
                    best_val_kl = if ($best) { $best.val_kl } else { $null }
                    last_step = if ($last) { $last.step } else { $null }
                    last_val_lm_loss = if ($last) { $last.val_lm_loss } else { $null }
                    last_val_ppl = if ($last) { $last.val_ppl } else { $null }
                })

                if (-not $KeepCheckpoints -and (Test-Path -LiteralPath $runDir)) {
                    Get-ChildItem -LiteralPath $runDir -Recurse -File -Include *.pt,*.pth,*.safetensors,*.ckpt |
                        Remove-Item -Force -ErrorAction SilentlyContinue
                    $modelDir = Join-Path $runDir "models"
                    if (Test-Path -LiteralPath $modelDir) {
                        Get-ChildItem -LiteralPath $modelDir -Recurse -File |
                            Remove-Item -Force -ErrorAction SilentlyContinue
                    }
                }
            }
        }
    }
}

if (-not $DryRun) {
    $rows |
        Sort-Object @{Expression = "best_val_lm_loss"; Ascending = $true} |
        Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

    Write-Host "Summary saved: $summaryPath"
    $rows |
        Sort-Object @{Expression = "best_val_lm_loss"; Ascending = $true} |
        Select-Object -First 12 run_name,best_step,best_val_ppl,best_alpha_downstream,best_router_entropy,best_val_kl |
        Format-Table -AutoSize
}
