param(
    [string]$StructuralConfig = "chronos2_selected_core_structural_covariates.yaml",
    [string]$ResidualConfig = "chronos2_selected_core_structural_residual.yaml",
    [string]$ScarcityConfig = "chronos2_selected_core_structural_scarcity_residual.yaml",
    [int]$HoldoutDays = 0,
    [double]$ExtremeThreshold = 150,
    [double]$ExtremeWeight = 2.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "=== Vérification des features MILP existantes ===" -ForegroundColor Cyan

$Features = ".\data\derived\structural_market_features.csv.gz"

if (-not (Test-Path $Features)) {
    throw "Features MILP absentes : $Features. Aucun rebuild automatique n'est lancé."
}

Write-Host "Features existantes : $Features" -ForegroundColor Green

Write-Host ""
Write-Host "=== Calibration du scarcity layer ===" -ForegroundColor Cyan

$FitArgs = @(
    ".\fit_structural_scarcity_layer.py",
    "--config", $StructuralConfig,
    "--extreme-threshold", "$ExtremeThreshold",
    "--extreme-weight", "$ExtremeWeight"
)

if ($HoldoutDays -gt 0) {
    $FitArgs += @(
        "--holdout-days",
        "$HoldoutDays"
    )
}

python @FitArgs
if ($LASTEXITCODE -ne 0) {
    throw "Échec de la calibration du scarcity layer."
}

Write-Host ""
Write-Host "=== Création du YAML C4 ===" -ForegroundColor Cyan

python .\install_structural_scarcity_layer.py `
    --config $ResidualConfig `
    --destination $ScarcityConfig

if ($LASTEXITCODE -ne 0) {
    throw "Échec de la création du YAML C4."
}

Write-Host ""
Write-Host "=== Run C4 : LP + scarcity adder + résidu Chronos ===" -ForegroundColor Cyan

python .\run_chronos2_structural_scarcity_residual.py `
    --config $ScarcityConfig `
    --zones FR `
    --local-files-only

if ($LASTEXITCODE -ne 0) {
    throw "Échec du run C4."
}

Write-Host ""
Write-Host "=== Comparaison C3 vs C4 ===" -ForegroundColor Cyan

python .\compare_structural_scarcity.py `
    --c3-config $ResidualConfig `
    --c4-config $ScarcityConfig `
    --zone FR

if ($LASTEXITCODE -ne 0) {
    Write-Host "Comparaison C3/C4 non disponible (C3 peut ne pas avoir été exécuté)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Scarcity layer terminé ===" -ForegroundColor Green
Write-Host "Calibration : data\derived\scarcity_layer_calibration.json"
Write-Host "Holdout     : data\derived\scarcity_layer_holdout_predictions.csv.gz"
Write-Host "Config C4   : $ScarcityConfig"
