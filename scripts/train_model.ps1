param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dense", "token_moe", "naime_state_moe", "naime_v4_state_moe", "naime_v5_world_state_moe", "naime_v6_recursive_self_moe")]
    [string]$Model,
    [switch]$UseVoice,
    [string]$RunName = "",
    [string]$WorkspaceConfig = "",
    [string]$OutputDir = "",
    [string]$DataPath = "data\naime\wikitext_processed",
    [int64]$TargetTokens = 12288000,
    [int]$SeqLen = 512,
    [int]$BatchSize = 4,
    [int]$DModel = 384,
    [int]$Layers = 8,
    [int]$DenseLayers = 2,
    [int]$Heads = 6,
    [int]$KvHeads = 2,
    [int]$Dff = 1536,
    [double]$Dropout = 0.0,
    [int]$Experts = 4,
    [int]$TopK = 2,
    [int]$ExpertHidden = 768,
    [ValidateSet("auto", "dense", "sparse")]
    [string]$MoeDispatchMode = "auto",
    [int]$Stride = 16,
    [int]$CausalStateStride = 512,
    [int]$Window = 24,
    [int]$ZDim = 96,
    [double]$LogvarClip = 10.0,
    [double]$VramFraction = 0.90,
    [int]$AutoBatchMax = 64,
    [switch]$NoAutoBatch,
    [switch]$NoAdaptiveDefaults,
    [int]$NumWorkers = 4,
    [int]$GradAccumSteps = 1,
    [ValidateSet("auto", "torch", "triton_ce", "cuda_ext_fused_ce")]
    [string]$LmLossBackend = "auto",
    [double]$LearningRate = 0.0003,
    [int]$WarmupSteps = 100,
    [double]$MinLrRatio = 0.1,
    [int]$LrCycleLength = 0,
    [double]$LrRestartRatio = 0.5,
    [int]$LrRestartWarmup = 200,
    [double]$GradClip = 1.0,
    [string]$Device = "auto",
    [switch]$CompileModel,
    [ValidateSet("full", "dense")]
    [string]$CompileScope = "full",
    [ValidateSet("inductor", "eager", "aot_eager")]
    [string]$CompileBackend = "inductor",
    [switch]$DisableFlashSdp,
    [switch]$NoAmp,
    [int]$SaveEvery = 2000,
    [int]$LatestEvery = 1000,
    [int]$KeepLastN = 2,
    [switch]$AsyncLatest,
    [switch]$SyncLatest,
    [int]$MetricsFlushEvery = 1,
    [int]$MetricsFsyncEvery = 100,
    [ValidateSet("full", "model")]
    [string]$BestCheckpointMode = "model",
    [int]$EvalEvery = 100,
    [int]$EvalMaxBatches = 10,
    [ValidateSet("random", "sequential")]
    [string]$EvalSampling = "random",
    [int]$EvalSeed = 4321,
    [switch]$EvalStateCarry,
    [switch]$EvalLatentThoughtGain,
    [int]$EarlyStopPatience = 0,
    [int]$EarlyStopMinEvals = 0,
    [double]$EarlyStopMinDelta = 0.0,
    [double]$PriorScale = 1.0,
    [double]$TargetSparsity = 0.0,
    [int]$KlWarmupSteps = 0,
    [double]$WeightDecay = 0.01,
    [int]$WorldStateSlots = 4,
    [double]$SemanticStateWriteScale = 0.0,
    [double]$SemanticMemoryHiddenScale = 0.0,
    [double]$SemanticGateMixerMaxStateWeight = 0.0,
    [switch]$NoWorldRouterNormalize,
    [switch]$NoWorldRouterConfidenceGate,
    [double]$WorldRouterMaxRatio = 0.08,
    [int]$SelfStateSlots = 4,
    [int]$SelfStateRecursionDepth = 1,
    [double]$SelfStateWriteScale = 0.03,
    [double]$SelfStateHiddenScale = 0.02,
    [double]$SelfStateBoundaryTemperature = 1.0,
    [double]$SelfStateIdentityScale = 0.02,
    [double]$SelfStateContextScoreScale = 4.0,
    [switch]$NoSelfStateWorldGate,
    [double]$SelfStateWorldGateMin = 0.10,
    [double]$SelfStateWorldGateScale = 1.0,
    [int]$LatentThoughtSteps = 0,
    [ValidateSet("state_only", "final_hidden")]
    [string]$LatentThoughtWriteMode = "state_only",
    [double]$LatentThoughtHiddenScale = 0.0,
    [int]$StateEvolutionSteps = 0,
    [switch]$NoStateEvolutionMemory,
    [switch]$LatentFieldCoupling,
    [double]$LatentFieldTokenScale = 0.02,
    [double]$LatentFieldMaxRatio = 0.05,
    [double]$LambdaStatePred = 0.0,
    [double]$LambdaSlotDiversity = 0.0,
    [double]$LambdaSlotStability = 0.0,
    [double]$LambdaSelfPred = 0.0,
    [double]$LambdaSelfSlotDiversity = 0.0,
    [switch]$SemanticRouterPriorGate,
    [double]$WorldStateStabilityThreshold = 1e-3,
    [ValidateSet("gqa", "mla")]
    [string]$AttentionType = "gqa",
    [int]$MlaLatentDim = 128,
    [int]$MlaRopePerHead = 32,
    [string]$Resume = "none",
    [ValidateSet("checkpoint", "absolute", "progress", "reset")]
    [string]$ResumeLrPolicy = "checkpoint",
    [switch]$ResumeAllowFailed,
    [switch]$AllowLegacyResume,
    [string]$StopFile = "",
    [int]$StopCheckEvery = 1,
    [ValidateSet("total", "additional")]
    [string]$TargetTokensMode = "total",
    [string]$ReferenceMetrics = "",
    [switch]$StructuralStop,
    [switch]$PrintArgs,
    [switch]$ResearchUnsafe,
    [int]$MaxSteps = -1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig -AllowMissing
if (-not $OutputDir) { $OutputDir = Resolve-NaimeConfigValue $workspace "local.run_root" "NAIME_RUN_ROOT" "experiments\runs" }

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "${Model}_${timestamp}"
}

$StateModels = @(
    "naime_v4_state_moe",
    "naime_v5_world_state_moe",
    "naime_v6_recursive_self_moe"
)
$HybridRouterModels = @(
    "naime_state_moe"
) + $StateModels
$StructuralStopModels = $StateModels

$isStateModel = $Model -in $StateModels
$isFineWebLike = $DataPath -match "(?i)fineweb|cosmopedia|slimpajama|openwebtext"
$effectiveDropout = $Dropout
$effectiveWeightDecay = $WeightDecay
$effectiveEvalMaxBatches = $EvalMaxBatches
$effectiveTargetTokens = $TargetTokens
$effectiveLatestEvery = $LatestEvery
$effectiveEvalEvery = $EvalEvery
$effectiveSaveEvery = $SaveEvery
$effectiveBestCheckpointMode = $BestCheckpointMode
$effectiveGradClip = $GradClip
if (-not $NoAdaptiveDefaults -and $isStateModel -and $isFineWebLike) {
    if ($Dropout -eq 0.0) {
        $effectiveDropout = 0.05
    }
    if ($WeightDecay -eq 0.01) {
        $effectiveWeightDecay = 0.05
    }
    if ($GradClip -eq 1.0) {
        $effectiveGradClip = 3.0
    }
    if ($EvalMaxBatches -eq 10) {
        $effectiveEvalMaxBatches = 0
    }
    if ($TargetTokens -eq 12288000) {
        $effectiveTargetTokens = 50000000
    }
    if ($LatestEvery -eq 1000) {
        $effectiveLatestEvery = 0
    }
    if ($EvalEvery -eq 100) {
        $effectiveEvalEvery = 5000
    }
    if ($SaveEvery -eq 2000) {
        $effectiveSaveEvery = 10000
    }
}

$architecture = switch ($Model) {
    "dense" { "dense" }
    "token_moe" { "token_moe" }
    "naime_state_moe" { "naime_state_moe" }
    "naime_v4_state_moe" { "naime_v4_state_moe" }
    "naime_v5_world_state_moe" { "naime_v5_world_state_moe" }
    "naime_v6_recursive_self_moe" { "naime_v6_recursive_self_moe" }
    default { "naime_state_moe" }
}

$routerMode = switch ($Model) {
    { $_ -in $HybridRouterModels } { "hybrid" }
    default { "concat" }
}

$common = @(
    "--architecture", $architecture,
    "--run-name", $RunName,
    "--output-dir", $OutputDir,
    "--data-path", $DataPath,
    "--data-format", "hf_disk",
    "--data-split", "train",
    "--vocab-size", "50257",
    "--seq-len", "$SeqLen",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--grad-accum-steps", "$GradAccumSteps",
    "--lm-loss-backend", "$LmLossBackend",
    "--learning-rate", "$LearningRate",
    "--warmup-steps", "$WarmupSteps",
    "--min-lr-ratio", "$MinLrRatio",
    "--grad-clip", "$effectiveGradClip",
    "--device", "$Device",
    "--d-model", "$DModel",
    "--n-layers", "$Layers",
    "--n-heads", "$Heads",
    "--n-kv-heads", "$KvHeads",
    "--d-ff", "$Dff",
    "--dropout", "$effectiveDropout",
    "--weight-decay", "$effectiveWeightDecay",
    "--log-every", "10",
    "--save-every", "$effectiveSaveEvery",
    "--latest-every", "$effectiveLatestEvery",
    "--keep-last-n", "$KeepLastN",
    "--metrics-flush-every", "$MetricsFlushEvery",
    "--metrics-fsync-every", "$MetricsFsyncEvery",
    "--best-checkpoint-mode", "$effectiveBestCheckpointMode",
    "--eval-every", "$effectiveEvalEvery",
    "--eval-split", "validation",
    "--eval-max-batches", "$effectiveEvalMaxBatches",
    "--eval-sampling", $EvalSampling,
    "--eval-seed", "$EvalSeed",
    "--early-stop-patience", "$EarlyStopPatience",
    "--early-stop-min-evals", "$EarlyStopMinEvals",
    "--early-stop-min-delta", "$EarlyStopMinDelta",
    "--resume", $Resume,
    "--resume-lr-policy", $ResumeLrPolicy,
    "--stop-check-every", "$StopCheckEvery",
    "--target-tokens-mode", $TargetTokensMode
)
if ($EvalStateCarry) {
    $common += "--eval-state-carry"
}
if ($EvalLatentThoughtGain) {
    $common += "--eval-latent-thought-gain"
}

if (-not $NoAutoBatch) {
    $common += @(
        "--auto-batch",
        "--vram-fraction", "$VramFraction",
        "--auto-batch-max", "$AutoBatchMax"
    )
}

if (-not [string]::IsNullOrWhiteSpace($StopFile)) {
    $common += @("--stop-file", $StopFile)
}

if ($ResumeAllowFailed) {
    $common += @("--resume-allow-failed")
}
if ($AllowLegacyResume) {
    $common += @("--allow-legacy-resume")
}

if ($LrCycleLength -gt 0) {
    $common += @(
        "--lr-cycle-length", "$LrCycleLength",
        "--lr-restart-ratio", "$LrRestartRatio",
        "--lr-restart-warmup", "$LrRestartWarmup"
    )
}

if ($effectiveTargetTokens -gt 0) {
    $common += @("--target-tokens", "$effectiveTargetTokens")
}

if ($MaxSteps -gt 0) {
    $common += @("--max-steps", "$MaxSteps")
}

if ($AsyncLatest) {
    $common += @("--async-latest")
}

if ($SyncLatest) {
    $common += @("--sync-latest")
}

if ($CompileModel) {
    $common += @("--compile-model", "--compile-scope", $CompileScope, "--compile-backend", $CompileBackend)
}

if ($DisableFlashSdp) {
    $common += @("--disable-flash-sdp")
}

if ($NoAmp) {
    $common += @("--no-amp")
}

if ($ResearchUnsafe) {
    $common += @("--research-unsafe")
}

if ($Model -eq "dense") {
    $common += @("--n-dense-layers", "$Layers")
} else {
    $common += @("--n-dense-layers", "$DenseLayers")
}

if ($Model -ne "dense") {
    $common += @(
        "--n-experts", "$Experts",
        "--top-k", "$TopK",
        "--expert-hidden-dim", "$ExpertHidden",
        "--moe-dispatch-mode", "$MoeDispatchMode"
    )
}

if ($Model -like "naime_*") {
    $common += @(
        "--stride", "$Stride",
        "--causal-state-stride", "$CausalStateStride",
        "--window", "$Window",
        "--z-dim", "$ZDim",
        "--semantic-router-mode", $routerMode,
        "--lambda-load", "0.01",
        "--lambda-sparse", "0.01"
    )
}

if ($Model -eq "naime_state_moe") {
    $common += @(
        "--target-sparsity", $(if ($TargetSparsity -gt 0.0) { "$TargetSparsity" } else { "0.2" }),
        "--logvar-clip", "$LogvarClip",
        "--semantic-router-prior-scale", "$PriorScale",
        "--lambda-kl", "0.001"
    )
}

if ($isStateModel) {
    $common += @(
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
        "--layerwise-semantic-schedule",
        "--semantic-memory-slots", "4",
        "--lambda-kl", "0.003",
        "--kl-warmup-steps", $(if ($KlWarmupSteps -gt 0) { "$KlWarmupSteps" } else { "1500" }),
        "--logvar-clip", $(if ($LogvarClip -ne 10.0) { "$LogvarClip" } else { "2.0" }),
        "--target-sparsity", $(if ($TargetSparsity -gt 0.0) { "$TargetSparsity" } else { "0.45" }),
        "--lambda-semantic-pred", "0.015"
    )
    if ($Model -eq "naime_v4_state_moe") {
        $common += @(
            "--semantic-memory-write-scale", "0.02",
            "--semantic-state-write-scale", "0.03"
        )
    }
    if ($Model -in @("naime_v5_world_state_moe", "naime_v6_recursive_self_moe")) {
        $common += @(
            "--semantic-gate-mixer-temperature", "2.5",
            "--semantic-gate-mixer-min-weight", "0.08",
            "--semantic-gate-mixer-max-clean-weight", "0.45",
            "--semantic-gate-mixer-max-state-weight", $(if ($SemanticGateMixerMaxStateWeight -gt 0.0) { "$SemanticGateMixerMaxStateWeight" } else { "0.35" }),
            "--semantic-state-confidence-mode", "hybrid",
            "--semantic-state-confidence-temperature", "3.0",
            "--semantic-state-confidence-gate",
            "--semantic-memory-read-gate",
            "--semantic-memory-write-scale", "0.035",
            "--semantic-state-write-scale", $(if ($SemanticStateWriteScale -gt 0.0) { "$SemanticStateWriteScale" } else { "0.045" })
        )
    }
    if ($Model -in @("naime_v5_world_state_moe", "naime_v6_recursive_self_moe")) {
        $common += @(
            "--world-state-slots", "$WorldStateSlots",
            "--semantic-memory-hidden-scale", $(if ($SemanticMemoryHiddenScale -gt 0.0) { "$SemanticMemoryHiddenScale" } else { "0.035" }),
            "--lambda-state-pred", $(if ($LambdaStatePred -gt 0.0) { "$LambdaStatePred" } else { "0.02" }),
            "--lambda-slot-diversity", $(if ($LambdaSlotDiversity -gt 0.0) { "$LambdaSlotDiversity" } else { "0.01" }),
            "--lambda-slot-stability", "$LambdaSlotStability",
            "--world-state-stability-threshold", "$WorldStateStabilityThreshold",
            "--world-router-max-ratio", "$WorldRouterMaxRatio",
            "--attention-type", "$AttentionType",
            "--mla-latent-dim", "$MlaLatentDim",
            "--mla-rope-per-head", "$MlaRopePerHead"
        )
        if ($NoWorldRouterNormalize) {
            $common += @("--no-world-router-normalize")
        }
        if ($NoWorldRouterConfidenceGate) {
            $common += @("--no-world-router-confidence-gate")
        }
        if ($SemanticRouterPriorGate) {
            $common += @("--semantic-router-prior-gate")
        }
    }
    if ($Model -eq "naime_v6_recursive_self_moe") {
        $common += @(
            "--self-state-slots", "$SelfStateSlots",
            "--self-state-recursion-depth", "$SelfStateRecursionDepth",
            "--self-state-write-scale", "$SelfStateWriteScale",
            "--self-state-hidden-scale", "$SelfStateHiddenScale",
            "--self-state-boundary-temperature", "$SelfStateBoundaryTemperature",
            "--self-state-identity-scale", "$SelfStateIdentityScale",
            "--self-state-context-score-scale", "$SelfStateContextScoreScale",
            "--self-state-world-gate-min", "$SelfStateWorldGateMin",
            "--self-state-world-gate-scale", "$SelfStateWorldGateScale",
            "--latent-thought-steps", "$LatentThoughtSteps",
            "--latent-thought-write-mode", "$LatentThoughtWriteMode",
            "--latent-thought-hidden-scale", "$LatentThoughtHiddenScale",
            "--state-evolution-steps", "$StateEvolutionSteps",
            "--latent-field-token-scale", "$LatentFieldTokenScale",
            "--latent-field-max-ratio", "$LatentFieldMaxRatio",
            "--lambda-self-pred", $(if ($LambdaSelfPred -gt 0.0) { "$LambdaSelfPred" } else { "0.01" }),
            "--lambda-self-slot-diversity", $(if ($LambdaSelfSlotDiversity -gt 0.0) { "$LambdaSelfSlotDiversity" } else { "0.02" })
        )
        if ($LatentFieldCoupling) {
            $common += "--latent-field-coupling"
        }
        if ($NoStateEvolutionMemory) {
            $common += "--no-state-evolution-memory"
        }
        if ($NoSelfStateWorldGate) {
            $common += "--no-self-state-world-gate"
        }
    }
}

if ($Model -in $StructuralStopModels -and $StructuralStop) {
    $referenceMetricsPath = if ([System.IO.Path]::IsPathRooted($ReferenceMetrics)) {
        $ReferenceMetrics
    } else {
        Join-Path (Split-Path -Parent $PSScriptRoot) $ReferenceMetrics
    }
    if (Test-Path -LiteralPath $referenceMetricsPath) {
        $common += @(
            "--reference-metrics-path", $referenceMetricsPath,
            "--structural-stop",
            "--structural-stop-min-gap", "0.30",
            "--structural-stop-widen-delta", "0.05",
            "--structural-stop-patience", "2",
            "--structural-stop-min-evals", "3",
            "--structural-stop-warmup-steps", "1000"
        )
    } else {
        Write-Warning "Reference metrics not found; structural stop disabled: $referenceMetricsPath"
    }
}

$allArgs = @($common)
if ($ExtraArgs) {
    $allArgs += @($ExtraArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

if ($PrintArgs) {
    Write-Host "Resolved training arguments:"
    $allArgs | ForEach-Object { Write-Host $_ }
    exit 0
}

& "$PSScriptRoot\train.ps1" -UseVoice:$UseVoice @allArgs
exit $LASTEXITCODE
