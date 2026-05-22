param(
    [string]$RunPrefix = "v6_mechanism",
    [string]$Template = "v6_local_smoke",
    [string]$TemplateDir = "",
    [string]$WorkspaceConfig = "",
    [string]$DataPath = "",
    [string]$OutputDir = "",
    [int]$LatentThoughtSteps = 1,
    [ValidateSet("state_only", "final_hidden")]
    [string]$LatentThoughtWriteMode = "state_only",
    [double]$LatentThoughtHiddenScale = 0.0,
    [switch]$LatentFieldCoupling,
    [double]$LatentFieldTokenScale = 0.02,
    [double]$LatentFieldMaxRatio = 0.05,
    [switch]$UseVoice,
    [switch]$PrintArgs,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Invoke-MatrixRun {
    param(
        [string]$Suffix,
        [switch]$EnableStateCarry,
        [switch]$EnableThought
    )

    $argsForRun = @(
        "-Template", $Template,
        "-RunName", "$RunPrefix`_$Suffix"
    )
    if (-not [string]::IsNullOrWhiteSpace($TemplateDir)) {
        $argsForRun += @("-TemplateDir", $TemplateDir)
    }
    if (-not [string]::IsNullOrWhiteSpace($WorkspaceConfig)) {
        $argsForRun += @("-WorkspaceConfig", $WorkspaceConfig)
    }
    if (-not [string]::IsNullOrWhiteSpace($DataPath)) {
        $argsForRun += @("-DataPath", $DataPath)
    }
    if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
        $argsForRun += @("-OutputDir", $OutputDir)
    }
    if ($UseVoice) {
        $argsForRun += "-UseVoice"
    }
    if ($PrintArgs) {
        $argsForRun += "-PrintArgs"
    }
    if ($EnableStateCarry) {
        $argsForRun += "-EvalStateCarry"
    }
    if ($EnableThought) {
        $argsForRun += @(
            "-LatentThoughtSteps", "$LatentThoughtSteps",
            "-LatentThoughtWriteMode", $LatentThoughtWriteMode,
            "-LatentThoughtHiddenScale", "$LatentThoughtHiddenScale",
            "-EvalLatentThoughtGain"
        )
    } else {
        $argsForRun += @("-LatentThoughtSteps", "0")
    }
    if ($LatentFieldCoupling) {
        $argsForRun += @(
            "-LatentFieldCoupling",
            "-LatentFieldTokenScale", "$LatentFieldTokenScale",
            "-LatentFieldMaxRatio", "$LatentFieldMaxRatio"
        )
    }
    if ($ExtraArgs) {
        $argsForRun += @($ExtraArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }

    & "$PSScriptRoot\train_template.ps1" @argsForRun
    if ($LASTEXITCODE -ne 0) {
        throw "matrix run failed: $Suffix"
    }
}

Invoke-MatrixRun -Suffix "baseline"
Invoke-MatrixRun -Suffix "state_carry" -EnableStateCarry
Invoke-MatrixRun -Suffix "latent_thought" -EnableThought
Invoke-MatrixRun -Suffix "state_carry_latent_thought" -EnableStateCarry -EnableThought
