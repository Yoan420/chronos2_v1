
param(
    [string]$BaseConfig = "chronos2_selected_core_structural_residual.yaml",
    [string]$C6Config = "chronos2_selected_core_zero_plateau.yaml"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "=== C6 : vérification des dépendances ===" -ForegroundColor Cyan

python -c "import catboost, sklearn; print('CatBoost', catboost.__version__, '| sklearn OK')"
if ($LASTEXITCODE -ne 0) {
    throw "CatBoost est requis. Exécute : python -m pip install catboost"
}

$MilpFeatures = ".\data\derived\structural_market_features.csv.gz"
if (-not (Test-Path $MilpFeatures)) {
    throw "Features MILP absentes : $MilpFeatures. Aucun rebuild automatique n'est lancé."
}

Write-Host "Features MILP existantes détectées." -ForegroundColor Green
Write-Host "Aucun refresh Saturn / backfill PIT / recalcul MILP." -ForegroundColor Green

Write-Host ""
Write-Host "=== Tests unitaires C6 ===" -ForegroundColor Cyan

python -m pytest `
    .\tests\test_zero_plateau_labels.py `
    .\tests\test_zero_plateau_decoder.py `
    .\tests\test_zero_plateau_gate.py `
    -q

if ($LASTEXITCODE -ne 0) {
    throw "Échec des tests C6."
}

Write-Host ""
Write-Host "=== 1-3. Features PIT + CatBoost + décodeur start/end ===" -ForegroundColor Cyan

python .\fit_zero_plateau_classifier.py `
    --config $BaseConfig `
    --zone FR `
    --output "data/derived/zero_plateau_predictions.csv.gz" `
    --model-output "data/models/zero_plateau_catboost.cbm" `
    --metadata-output "data/derived/zero_plateau_model_metadata.json"

if ($LASTEXITCODE -ne 0) {
    throw "Échec du classifieur zero plateau."
}

Write-Host ""
Write-Host "=== 4. Ajout des probabilités à Chronos ===" -ForegroundColor Cyan

python .\install_zero_plateau_experiment.py `
    --config $BaseConfig `
    --destination $C6Config `
    --predictions-file "data/derived/zero_plateau_predictions.csv.gz" `
    --zone FR

if ($LASTEXITCODE -ne 0) {
    throw "Échec de la création du YAML C6."
}

Write-Host ""
Write-Host "=== C6A. Chronos + probabilités, SANS soft gate ===" -ForegroundColor Cyan

python .\run_chronos2_zero_plateau.py `
    --config $C6Config `
    --zones FR `
    --local-files-only

if ($LASTEXITCODE -ne 0) {
    throw "Échec du run C6A."
}

Write-Host ""
Write-Host "=== 5-6. Soft gate + évaluation plateau dédiée ===" -ForegroundColor Cyan

python .\evaluate_zero_plateau_experiment.py `
    --config $C6Config `
    --zone FR `
    --predictions-file "data/derived/zero_plateau_predictions.csv.gz" `
    --metadata-file "data/derived/zero_plateau_model_metadata.json"

if ($LASTEXITCODE -ne 0) {
    throw "Échec de l'évaluation C6."
}

Write-Host ""
Write-Host "=== Expérience C6 terminée ===" -ForegroundColor Green
Write-Host "Classifier        : data\models\zero_plateau_catboost.cbm"
Write-Host "Probabilités      : data\derived\zero_plateau_predictions.csv.gz"
Write-Host "Importance        : data\derived\zero_plateau_feature_importance.csv"
Write-Host "Config C6         : $C6Config"
Write-Host "Évaluation dédiée : <output>\fr\zero_plateau_evaluation.json"
Write-Host "Gate backtest     : <output>\fr\backtest_zero_plateau_soft_gate.csv"
