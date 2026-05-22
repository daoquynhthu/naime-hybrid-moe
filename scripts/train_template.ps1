param(
    [string]$Template = "v6_local_smoke",
    [string]$TemplateDir = "",
    [string]$WorkspaceConfig = "",
    [switch]$List,
    [switch]$PrintArgs,
    [string]$RunName = "",
    [string]$DataPath = "",
    [string]$OutputDir = "",
    [string]$Resume = "",
    [string]$ResumeLrPolicy = "",
    [int64]$TargetTokens = -1,
    [ValidateSet("", "total", "additional")]
    [string]$TargetTokensMode = "",
    [int]$SeqLen = -1,
    [int]$BatchSize = -1,
    [double]$LearningRate = -1.0,
    [int]$WarmupSteps = -1,
    [double]$MinLrRatio = -1.0,
    [double]$GradClip = -1.0,
    [ValidateSet("", "full", "dense")]
    [string]$CompileScope = "",
    [ValidateSet("", "inductor", "eager", "aot_eager")]
    [string]$CompileBackend = "",
    [double]$VramFraction = -1.0,
    [int]$AutoBatchMax = -1,
    [int]$EvalEvery = -1,
    [int]$EvalMaxBatches = -1,
    [int]$SaveEvery = -1,
    [int]$LatestEvery = -1,
    [int]$MetricsFlushEvery = -1,
    [int]$MetricsFsyncEvery = -1,
    [int]$NumWorkers = -1,
    [ValidateSet("", "auto", "torch", "triton_ce")]
    [string]$LmLossBackend = "",
    [double]$SemanticStateWriteScale = -1.0,
    [double]$SemanticMemoryHiddenScale = -1.0,
    [double]$SemanticGateMixerMaxStateWeight = -1.0,
    [double]$WorldRouterMaxRatio = -1.0,
    [double]$SelfStateWorldGateMin = -1.0,
    [double]$SelfStateWorldGateScale = -1.0,
    [string]$Device = "",
    [int]$MaxSteps = -1,
    [switch]$NoAutoBatch,
    [switch]$UseVoice,
    [switch]$SyncLatest,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $TemplateDir) {
    $TemplateDir = Join-Path $repoRoot "configs\training_templates"
}

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-NaimeWorkspaceConfig -AllowMissing

function Get-TemplatePath {
    param([string]$Name)
    if ([System.IO.Path]::IsPathRooted($Name) -or $Name.EndsWith(".json")) {
        return $Name
    }
    return Join-Path $TemplateDir "$Name.json"
}

function Get-NestedValue {
    param(
        [object]$Root,
        [string]$Path
    )
    if ($Path -eq "repo") { return $repoRoot }
    if ($Path.StartsWith("env:")) {
        return [Environment]::GetEnvironmentVariable($Path.Substring(4))
    }

    $current = $Root
    foreach ($part in $Path.Split(".")) {
        if ($null -eq $current) { return $null }
        $prop = $current.PSObject.Properties[$part]
        if ($null -eq $prop) { return $null }
        $current = $prop.Value
    }
    return $current
}

function Resolve-TemplateValue {
    param([object]$Value)
    if ($Value -is [string]) {
        $resolved = $Value
        $matches = [regex]::Matches($Value, "\$\{([^}]+)\}")
        foreach ($match in $matches) {
            $key = $match.Groups[1].Value
            $replacement = Get-NestedValue -Root $workspace -Path $key
            if ($null -eq $replacement) {
                throw "Template variable `${$key} could not be resolved. Check configs/workspace.local.json."
            }
            $resolved = $resolved.Replace($match.Value, [string]$replacement)
        }
        return $resolved
    }
    return $Value
}

function Add-Override {
    param(
        [hashtable]$Params,
        [string]$Name,
        [object]$Value,
        [scriptblock]$ShouldUse
    )
    if (& $ShouldUse) {
        $Params[$Name] = $Value
    }
}

if ($List) {
    Get-ChildItem -LiteralPath $TemplateDir -Filter "*.json" | Sort-Object Name | ForEach-Object {
        $json = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
        $description = if ($json.PSObject.Properties["description"]) { $json.description } else { "" }
        "{0,-34} {1}" -f $name, $description
    }
    exit 0
}

$templatePath = Get-TemplatePath -Name $Template
if (-not (Test-Path -LiteralPath $templatePath)) {
    throw "Training template not found: $templatePath. Use -List to see available templates."
}

$templateJson = Get-Content -LiteralPath $templatePath -Raw | ConvertFrom-Json
if (-not $templateJson.PSObject.Properties["params"]) {
    throw "Template has no params object: $templatePath"
}

$params = @{}
foreach ($prop in $templateJson.params.PSObject.Properties) {
    $value = Resolve-TemplateValue -Value $prop.Value
    if ($value -is [bool] -and -not $value) {
        continue
    }
    $params[$prop.Name] = $value
}

Add-Override $params "RunName" $RunName { -not [string]::IsNullOrWhiteSpace($RunName) }
Add-Override $params "DataPath" $DataPath { -not [string]::IsNullOrWhiteSpace($DataPath) }
Add-Override $params "OutputDir" $OutputDir { -not [string]::IsNullOrWhiteSpace($OutputDir) }
Add-Override $params "Resume" $Resume { -not [string]::IsNullOrWhiteSpace($Resume) }
Add-Override $params "ResumeLrPolicy" $ResumeLrPolicy { -not [string]::IsNullOrWhiteSpace($ResumeLrPolicy) }
Add-Override $params "TargetTokens" $TargetTokens { $TargetTokens -ge 0 }
Add-Override $params "TargetTokensMode" $TargetTokensMode { -not [string]::IsNullOrWhiteSpace($TargetTokensMode) }
Add-Override $params "SeqLen" $SeqLen { $SeqLen -gt 0 }
Add-Override $params "BatchSize" $BatchSize { $BatchSize -gt 0 }
Add-Override $params "LearningRate" $LearningRate { $LearningRate -gt 0.0 }
Add-Override $params "WarmupSteps" $WarmupSteps { $WarmupSteps -ge 0 }
Add-Override $params "MinLrRatio" $MinLrRatio { $MinLrRatio -ge 0.0 }
Add-Override $params "GradClip" $GradClip { $GradClip -ge 0.0 }
Add-Override $params "CompileScope" $CompileScope { -not [string]::IsNullOrWhiteSpace($CompileScope) }
Add-Override $params "CompileBackend" $CompileBackend { -not [string]::IsNullOrWhiteSpace($CompileBackend) }
Add-Override $params "VramFraction" $VramFraction { $VramFraction -gt 0.0 }
Add-Override $params "AutoBatchMax" $AutoBatchMax { $AutoBatchMax -gt 0 }
Add-Override $params "EvalEvery" $EvalEvery { $EvalEvery -ge 0 }
Add-Override $params "EvalMaxBatches" $EvalMaxBatches { $EvalMaxBatches -ge 0 }
Add-Override $params "SaveEvery" $SaveEvery { $SaveEvery -ge 0 }
Add-Override $params "LatestEvery" $LatestEvery { $LatestEvery -ge 0 }
Add-Override $params "MetricsFlushEvery" $MetricsFlushEvery { $MetricsFlushEvery -gt 0 }
Add-Override $params "MetricsFsyncEvery" $MetricsFsyncEvery { $MetricsFsyncEvery -ge 0 }
Add-Override $params "NumWorkers" $NumWorkers { $NumWorkers -ge 0 }
Add-Override $params "LmLossBackend" $LmLossBackend { -not [string]::IsNullOrWhiteSpace($LmLossBackend) }
Add-Override $params "SemanticStateWriteScale" $SemanticStateWriteScale { $SemanticStateWriteScale -gt 0.0 }
Add-Override $params "SemanticMemoryHiddenScale" $SemanticMemoryHiddenScale { $SemanticMemoryHiddenScale -gt 0.0 }
Add-Override $params "SemanticGateMixerMaxStateWeight" $SemanticGateMixerMaxStateWeight { $SemanticGateMixerMaxStateWeight -gt 0.0 }
Add-Override $params "WorldRouterMaxRatio" $WorldRouterMaxRatio { $WorldRouterMaxRatio -gt 0.0 }
Add-Override $params "SelfStateWorldGateMin" $SelfStateWorldGateMin { $SelfStateWorldGateMin -ge 0.0 }
Add-Override $params "SelfStateWorldGateScale" $SelfStateWorldGateScale { $SelfStateWorldGateScale -gt 0.0 }
Add-Override $params "Device" $Device { -not [string]::IsNullOrWhiteSpace($Device) }
Add-Override $params "MaxSteps" $MaxSteps { $MaxSteps -gt 0 }

if ($NoAutoBatch) {
    $params["NoAutoBatch"] = $true
}
if ($UseVoice) {
    $params["UseVoice"] = $true
}
if ($SyncLatest) {
    $params["SyncLatest"] = $true
}
if ($PrintArgs) {
    $params["PrintArgs"] = $true
}
$params["NoAdaptiveDefaults"] = $true

$switchNames = @(
    "UseVoice",
    "NoAutoBatch",
    "NoAdaptiveDefaults",
    "CompileModel",
    "DisableFlashSdp",
    "NoAmp",
    "AsyncLatest",
    "SyncLatest",
    "ResumeAllowFailed",
    "AllowLegacyResume",
    "SemanticRouterPriorGate",
    "NoSelfStateWorldGate",
    "StructuralStop",
    "ResearchUnsafe",
    "PrintArgs"
)
foreach ($name in $switchNames) {
    if ($params.ContainsKey($name) -and -not [bool]$params[$name]) {
        $params.Remove($name)
    }
}

if (-not $params.ContainsKey("RunName") -or [string]::IsNullOrWhiteSpace([string]$params["RunName"])) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $params["RunName"] = "$Template`_$timestamp"
}

Write-Host "Training template: $Template"
if ($templateJson.PSObject.Properties["description"]) {
    Write-Host $templateJson.description
}
Write-Host "RunName: $($params["RunName"])"

& "$PSScriptRoot\train_model.ps1" @params @ExtraArgs
exit $LASTEXITCODE
