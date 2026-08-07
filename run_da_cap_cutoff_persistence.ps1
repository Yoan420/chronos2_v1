param(
    [string]$BaseConfig = "chronos2_selected_core.yaml",
    [string]$GeneratedConfig = "chronos2_selected_core_da_cap_cutoff_persistence.yaml",
    [string]$Cutoff = "08:00",
    [double]$MaxAgeHours = 72,
    [switch]$RefreshData
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Historique final de l'asymétrie ==="
python .\build_da_cap_system_asymmetry_history.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== Configuration cutoff_persistence ==="
python .\install_da_cap_cutoff_persistence.py `
    --config $BaseConfig `
    --destination $GeneratedConfig `
    --cutoff $Cutoff `
    --max-age-hours $MaxAgeHours
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$runArgs = @(
    ".\run_chronos2_cutoff_persistence.py",
    "--config", $GeneratedConfig,
    "--zones", "FR",
    "--local-files-only"
)
if ($RefreshData) {
    $runArgs += "--refresh-data"
}

Write-Host ""
Write-Host "=== Run Chronos-2 ==="
python @runArgs
exit $LASTEXITCODE
