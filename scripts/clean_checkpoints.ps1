param(
    [string]$RunsRoot = "experiments\runs",
    [int]$KeepLastStepCheckpoints = 0,
    [switch]$RemoveTestRuns,
    [string[]]$SkipRuns = @(),
    [switch]$Apply
)

$resolvedRoot = (Resolve-Path -LiteralPath $RunsRoot).Path
$deleteItems = New-Object System.Collections.Generic.List[object]

function Test-InRunsRoot {
    param([string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    return $resolved.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)
}

function Add-DeleteFile {
    param([System.IO.FileInfo]$File)
    if (Test-InRunsRoot $File.FullName) {
        $deleteItems.Add([PSCustomObject]@{
            Type = "file"
            Path = $File.FullName
            Size = $File.Length
        })
    }
}

function Add-DeleteDirectory {
    param([System.IO.DirectoryInfo]$Directory)
    if (Test-InRunsRoot $Directory.FullName) {
        $size = (Get-ChildItem -LiteralPath $Directory.FullName -Recurse -File | Measure-Object Length -Sum).Sum
        $deleteItems.Add([PSCustomObject]@{
            Type = "dir"
            Path = $Directory.FullName
            Size = $size
        })
    }
}

$runDirs = Get-ChildItem -LiteralPath $resolvedRoot -Directory | Where-Object { $SkipRuns -notcontains $_.Name }

foreach ($run in $runDirs) {
    if ($RemoveTestRuns -and $run.Name -match "(smoke\d*|probe\d*)$") {
        Add-DeleteDirectory $run
        continue
    }

    $stepFiles = Get-ChildItem -LiteralPath $run.FullName -File -Filter "step_*.pt" |
        Sort-Object LastWriteTime -Descending
    $modelStepFiles = Join-Path $run.FullName "models" |
        ForEach-Object {
            if (Test-Path -LiteralPath $_) {
                Get-ChildItem -LiteralPath $_ -File -Filter "model_step_*.pt" |
                    Sort-Object LastWriteTime -Descending
            }
        }

    $stepFiles | Select-Object -Skip $KeepLastStepCheckpoints | ForEach-Object { Add-DeleteFile $_ }
    $modelStepFiles | Select-Object -Skip $KeepLastStepCheckpoints | ForEach-Object { Add-DeleteFile $_ }
}

$totalBytes = ($deleteItems | Measure-Object Size -Sum).Sum
$totalGb = [math]::Round(($totalBytes / 1GB), 2)

Write-Host "Runs root: $resolvedRoot"
Write-Host "Items selected: $($deleteItems.Count)"
Write-Host "Estimated reclaim: $totalGb GB"
Write-Host "Mode: $(if ($Apply) { 'APPLY' } else { 'DRY-RUN' })"

$deleteItems |
    Sort-Object Type, Path |
    Select-Object Type, @{Name="SizeGB"; Expression={[math]::Round($_.Size / 1GB, 3)}}, Path |
    Format-Table -AutoSize

if (-not $Apply) {
    Write-Host "No files deleted. Re-run with -Apply to delete selected items."
    exit 0
}

foreach ($item in $deleteItems | Sort-Object Type) {
    if ($item.Type -eq "file") {
        Remove-Item -LiteralPath $item.Path -Force -ErrorAction SilentlyContinue
    } elseif ($item.Type -eq "dir") {
        Remove-Item -LiteralPath $item.Path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Cleanup complete."
