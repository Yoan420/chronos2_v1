param(
    [string]$ProjectRoot = "C:\Users\BQ6757\chronos2_v1",
    [string]$PythonExe = "python",
    [string]$Config = "chronos2_inputs_extended_exogenous.yaml",
    [ValidateSet("family", "component")]
    [string]$Level = "family",
    [ValidateSet("full", "only", "loo", "both")]
    [string]$Mode = "only",
    [int]$BacktestWindows = 60,
    [string[]]$FoldAsOf = @(
        "2025-08-01T08:00:00+02:00",
        "2026-02-01T08:00:00+01:00",
        "2026-08-01T08:00:00+02:00"
    ),
    [string[]]$Subjects = @()
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$RunRoot = ".\runs\feature_selection\${Level}_${Mode}"
$SummaryRoot = Join-Path $RunRoot "summary"

$Arguments = @(
    ".\run_feature_selection_batch.py",
    "--config", ".\$Config",
    "--groups", ".\feature_groups.yaml",
    "--level", $Level,
    "--mode", $Mode,
    "--backtest-windows", "$BacktestWindows",
    "--output-dir", $RunRoot,
    "--local-files-only",
    "--fold-asof"
) + $FoldAsOf

if ($Subjects.Count -gt 0) {
    $Arguments += "--subjects"
    $Arguments += $Subjects
}

& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Le batch de sélection a échoué avec le code $LASTEXITCODE."
}

& $PythonExe ".\summarize_feature_selection.py" `
    --results (Join-Path $RunRoot "selection_results.csv") `
    --output-dir $SummaryRoot
if ($LASTEXITCODE -ne 0) {
    throw "La synthèse de sélection a échoué avec le code $LASTEXITCODE."
}
