param(
    [string]$RunName = "",
    [string]$DatasetPath = "",
    [string]$RunsRoot = "",
    [string]$RemoteRepo = "",
    [string]$RemotePython = "",
    [string]$WorkspaceConfig = "",
    [int]$RecentWindow = 200,
    [int]$TailLog = 80,
    [int]$GpuSamples = 3,
    [double]$GpuInterval = 1.0,
    [int]$SeqLen = 0,
    [int]$AutoBatchMax = 64,
    [switch]$ScanAllMetrics,
    [switch]$NoSyncProbe,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent $ScriptDir

. "$ScriptDir\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig -AllowMissing

if (-not $RunsRoot) { $RunsRoot = Resolve-NaimeConfigValue $workspace "remote.runs" "NAIME_REMOTE_RUNS" "L:/NAIME_REMOTE/runs" }
if (-not $RemoteRepo) { $RemoteRepo = Resolve-NaimeConfigValue $workspace "remote.repo" "" "L:/NAIME_REMOTE/naime-hybrid-moe" }
if (-not $RemotePython) { $RemotePython = Resolve-NaimeConfigValue $workspace "remote.python" "" "L:/NAIME_REMOTE/envs/.venv312/Scripts/python.exe" }
if (-not $DatasetPath -and $workspace.PSObject.Properties["remote"]) {
    $remoteFineWeb = $workspace.remote.PSObject.Properties["fineweb_edu_1b"]
    if ($remoteFineWeb) { $DatasetPath = [string]$remoteFineWeb.Value }
}
if (-not $DatasetPath) {
    $DatasetPath = (Resolve-NaimeConfigValue $workspace "remote.datasets" "" "L:/NAIME_REMOTE/datasets") + "/fineweb_edu_1b_ctx1024"
}
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot "analysis\remote_inspections" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$localProbe = Join-Path $ScriptDir "remote_training_probe.py"
$remoteProbe = ($RemoteRepo.TrimEnd("/", "\")) + "/scripts/remote_training_probe.py"
if (-not $NoSyncProbe) {
    & "$ScriptDir\remote.ps1" push $localProbe $remoteProbe | Out-Null
}

$scanFlag = if ($ScanAllMetrics) { "--scan-all-metrics" } else { "" }
$runArg = if ($RunName) { "--run-name `"$RunName`"" } else { "" }
$seqArg = if ($SeqLen -gt 0) { "--seq-len $SeqLen" } else { "" }
$cmd = @"
& '$RemotePython' '$remoteProbe' --runs-root '$RunsRoot' $runArg --dataset-path '$DatasetPath' $seqArg --auto-batch-max $AutoBatchMax --recent-window $RecentWindow --tail-log $TailLog --gpu-samples $GpuSamples --gpu-interval $GpuInterval $scanFlag
"@.Trim()

$raw = & "$ScriptDir\remote.ps1" cmd $cmd
$jsonText = ($raw -join "`n").Trim()
if (-not $jsonText.StartsWith("{")) {
    throw "Remote probe did not return JSON. Output:`n$jsonText"
}

$report = $jsonText | ConvertFrom-Json
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeRun = if ($report.run_name) { [string]$report.run_name } else { "latest" }
$safeRun = $safeRun -replace '[^\w.-]+', '_'
$outPath = Join-Path $OutDir "remote_probe_${safeRun}_${stamp}.json"
$jsonText | Set-Content -LiteralPath $outPath -Encoding UTF8

Write-Output "Report saved: $outPath"
Write-Output ""
Write-Output "========== GPU =========="
foreach ($sample in $report.gpu_samples) {
    foreach ($gpu in $sample.gpus) {
        Write-Output ("sample={0} gpu={1} used={2}MiB free={3}MiB util={4}% temp={5}C power={6}W" -f `
            $sample.sample, $gpu.index, $gpu.'memory.used', $gpu.'memory.free', $gpu.'utilization.gpu', $gpu.'temperature.gpu', $gpu.'power.draw')
    }
}
Write-Output ""
Write-Output "========== Dataset =========="
$train = $report.dataset.splits.train
if ($train) {
    Write-Output ("train_examples={0} estimated_tokens={1} safe_no_repeat_target={2}" -f `
        $train.num_examples, $report.dataset.token_estimate, $report.dataset.safe_no_repeat_target_tokens)
}
$val = $report.dataset.splits.validation
if ($val) {
    Write-Output ("validation_examples={0}" -f $val.num_examples)
}
Write-Output ""
Write-Output "========== Run =========="
$progress = $report.run.progress
if ($progress) {
    Write-Output ("run={0} step={1}/{2} progress={3}% remaining={4}" -f `
        $report.run.name, $progress.step, $progress.max_steps, $progress.percent, $progress.remaining_steps)
} else {
    Write-Output ("run={0}" -f $report.run.name)
}
$latest = $report.run.metrics.latest_train
if ($latest) {
    Write-Output ("latest train: step={0} lm={1:N4} ppl={2:N2} grad={3:N3} lr={4:E3} tok/s={5:N0} alpha={6:N3} ent={7:N3}" -f `
        $latest.step, $latest.loss_lm, $latest.ppl_lm, $latest.grad_norm, $latest.lr, $latest.tokens_per_second, $latest.alpha_mean, $latest.router_entropy)
}
$best = $report.run.metrics.best_val
if ($best) {
    $bestLoss = if ($best.val_lm_loss) { $best.val_lm_loss } elseif ($best.loss_lm) { $best.loss_lm } else { $best.loss }
    $bestPpl = if ($best.val_ppl_lm) { $best.val_ppl_lm } elseif ($best.ppl_lm) { $best.ppl_lm } else { $best.ppl }
    Write-Output ("best val: step={0} lm={1:N4} ppl={2:N2}" -f $best.step, $bestLoss, $bestPpl)
}
if ($report.warnings.Count -gt 0) {
    Write-Output ""
    Write-Output "========== Warnings =========="
    $report.warnings | ForEach-Object { Write-Output "- $_" }
}
