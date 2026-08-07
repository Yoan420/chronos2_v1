param(
    [string]$BaseConfig = "chronos2_selected_core.yaml",
    [ValidateSet("covariates", "residual", "both")]
    [string]$Mode = "both",
    [int]$MaxDays = 0,
    [switch]$RefreshData,
    [switch]$FullDataRefresh,
    [string]$DataAsOf = "",
    [switch]$FailFast
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$CovConfig = "chronos2_selected_core_structural_covariates.yaml"
$ResidualConfig = "chronos2_selected_core_structural_residual.yaml"

Write-Host ""
Write-Host "=== Vérification de SciPy / HiGHS ===" -ForegroundColor Cyan
python -c "import scipy; from scipy.optimize import milp, linprog; print('SciPy', scipy.__version__, '| HiGHS MILP/LP disponible')"
if ($LASTEXITCODE -ne 0) {
    throw "SciPy avec scipy.optimize.milp est requis."
}

Write-Host ""
Write-Host "=== Création des configurations structurelles ===" -ForegroundColor Cyan
python .\install_structural_market.py `
    --config $BaseConfig `
    --covariates-output $CovConfig `
    --residual-output $ResidualConfig
if ($LASTEXITCODE -ne 0) {
    throw "Échec de la création des YAML structurels."
}

Write-Host ""
Write-Host "=== Tests unitaires ===" -ForegroundColor Cyan
python -m pytest `
    .\tests\test_structural_market_model.py `
    .\tests\test_structural_residual.py `
    -q
if ($LASTEXITCODE -ne 0) {
    throw "Échec des tests du modèle structurel."
}

$BuildArgs = @(
    ".\build_structural_market_features.py",
    "--config", $CovConfig,
    "--zone", "FR"
)
if ($MaxDays -gt 0) {
    $BuildArgs += @("--max-days", "$MaxDays")
}
if ($RefreshData) {
    $BuildArgs += "--refresh-data"
}
if ($FullDataRefresh) {
    $BuildArgs += "--full-data-refresh"
}
if ($DataAsOf -ne "") {
    $BuildArgs += @("--data-as-of", $DataAsOf)
}
if ($FailFast) {
    $BuildArgs += "--fail-fast"
}

Write-Host ""
Write-Host "=== MILP agrégé + Branch-and-Cut + LP de pricing ===" -ForegroundColor Cyan
python @BuildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Échec de la construction des covariables structurelles."
}

Write-Host ""
Write-Host "=== Validation du backfill MILP ===" -ForegroundColor Cyan
python .\validate_structural_feature_coverage.py `
    --config $CovConfig `
    --zone FR
if ($LASTEXITCODE -ne 0) {
    throw "Le backfill MILP n'est pas exploitable."
}

function Run-ChronosStructural {
    param(
        [string]$Runner,
        [string]$Config
    )
    $RunArgs = @(
        $Runner,
        "--config", $Config,
        "--zones", "FR",
        "--local-files-only"
    )
    python @RunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Échec du run : $Runner"
    }
}

if ($Mode -in @("covariates", "both")) {
    Write-Host ""
    Write-Host "=== Chronos-2 avec covariables MILP ===" -ForegroundColor Cyan
    Run-ChronosStructural `
        -Runner ".\run_chronos2_structural_covariates.py" `
        -Config $CovConfig
}

if ($Mode -in @("residual", "both")) {
    Write-Host ""
    Write-Host "=== Prix structurel + résidu Chronos-2 ===" -ForegroundColor Cyan
    Run-ChronosStructural `
        -Runner ".\run_chronos2_structural_residual.py" `
        -Config $ResidualConfig
}

Write-Host ""
Write-Host "=== Pipeline structurel terminé ===" -ForegroundColor Green
Write-Host "Features : data\derived\structural_market_features.csv.gz"
Write-Host "Diagnostic : data\derived\structural_market_daily_diagnostics.csv"
Write-Host "Config covariables : $CovConfig"
Write-Host "Config résiduelle  : $ResidualConfig"
