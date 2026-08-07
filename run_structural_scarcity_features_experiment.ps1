param(
    [string]$StructuralConfig = "chronos2_selected_core_structural_covariates.yaml",
    [string]$ResidualConfig = "chronos2_selected_core_structural_residual.yaml",
    [string]$C4Config = "chronos2_selected_core_structural_scarcity_residual.yaml",
    [string]$C5Config = "chronos2_selected_core_structural_scarcity_features.yaml",
    [double]$EwmAlpha = 0.65
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$MilpFeatures = ".\data\derived\structural_market_features.csv.gz"
$Calibration = ".\data\derived\scarcity_layer_calibration.json"

Write-Host ""
Write-Host "=== C5 : vérification des entrées existantes ===" -ForegroundColor Cyan

if (-not (Test-Path $MilpFeatures)) {
    throw "Features MILP absentes : $MilpFeatures"
}
if (-not (Test-Path $Calibration)) {
    throw "Calibration scarcity C4 absente : $Calibration"
}

Write-Host "MILP existant       : $MilpFeatures" -ForegroundColor Green
Write-Host "Calibration existante : $Calibration" -ForegroundColor Green
Write-Host "Aucun refresh Saturn ou rebuild MILP ne sera lancé." -ForegroundColor Green

Write-Host ""
Write-Host "=== Construction des features scarcity C5 ===" -ForegroundColor Cyan

python .\build_structural_scarcity_features.py `
    --config $StructuralConfig `
    --calibration "data/derived/scarcity_layer_calibration.json" `
    --output "data/derived/structural_scarcity_features_c5.csv.gz" `
    --ewm-alpha $EwmAlpha

if ($LASTEXITCODE -ne 0) {
    throw "Échec de la construction des features C5."
}

Write-Host ""
Write-Host "=== Création du YAML C5 ===" -ForegroundColor Cyan

python .\install_structural_scarcity_features.py `
    --config $ResidualConfig `
    --destination $C5Config `
    --features-file "data/derived/structural_scarcity_features_c5.csv.gz" `
    --zone FR

if ($LASTEXITCODE -ne 0) {
    throw "Échec de la création du YAML C5."
}

Write-Host ""
Write-Host "=== Tests C5 ===" -ForegroundColor Cyan

python -m pytest `
    .\tests\test_structural_scarcity_features.py `
    -q

if ($LASTEXITCODE -ne 0) {
    throw "Échec des tests C5."
}

Write-Host ""
Write-Host "=== Run C5 : LP brut + résidu Chronos + scarcity features ===" -ForegroundColor Cyan

python .\run_chronos2_structural_scarcity_features.py `
    --config $C5Config `
    --zones FR `
    --local-files-only

if ($LASTEXITCODE -ne 0) {
    throw "Échec du run C5."
}

Write-Host ""
Write-Host "=== Comparaison propre C3 / C4 / C5 ===" -ForegroundColor Cyan

$CompareArgs = @(
    ".\compare_structural_c3_c4_c5.py",
    "--c3-config", $ResidualConfig,
    "--c5-config", $C5Config,
    "--zone", "FR"
)

if (Test-Path $C4Config) {
    $CompareArgs += @(
        "--c4-config",
        $C4Config
    )
}

python @CompareArgs

if ($LASTEXITCODE -ne 0) {
    throw "Échec de la comparaison C3/C4/C5."
}

Write-Host ""
Write-Host "=== Expérience C5 terminée ===" -ForegroundColor Green
Write-Host "Features C5 : data\derived\structural_scarcity_features_c5.csv.gz"
Write-Host "Config C5   : $C5Config"
Write-Host "Résultats   : runs\structural_c3_c4_c5_comparison.csv"
