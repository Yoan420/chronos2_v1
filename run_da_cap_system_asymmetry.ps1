param(
    [string]$Config = "chronos2_selected_core.yaml",
    [string]$GeneratedConfig = "chronos2_selected_core_da_cap.yaml",
    [switch]$Full,
    [switch]$RefreshData
)

$ErrorActionPreference = "Stop"

$buildArgs = @(
    ".\build_da_cap_system_asymmetry.py",
    "--config", $Config,
    "--unit", "GW"
)
if ($Full) {
    $buildArgs += "--full"
}

Write-Host ""
Write-Host "=== Construction PIT de da_cap_system_asymmetry ==="
python @buildArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== Création de la configuration Chronos-2 ==="
python .\install_da_cap_system_asymmetry.py `
    --config $Config `
    --destination $GeneratedConfig
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$runArgs = @(
    ".\run_chronos2_extended_exogenous.py",
    "--config", $GeneratedConfig,
    "--zones", "FR",
    "--local-files-only"
)
if ($RefreshData) {
    $runArgs += "--refresh-data"
}

Write-Host ""
Write-Host "=== Lancement Chronos-2 avec asymétrie Day-Ahead ==="
python @runArgs
exit $LASTEXITCODE
