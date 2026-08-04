param(
    [string]$Config = "$PSScriptRoot\chronos2_inputs_extended_exogenous.yaml",
    [string]$Zone = "FR",
    [switch]$RefreshSaturn,
    [switch]$RebuildOrderSignals,
    [string]$OrderSignalStartDay = "2024-01-01"
)

$ErrorActionPreference = "Stop"

$builder = @(
    "$PSScriptRoot\build_extended_exogenous_inputs.py",
    "--config", $Config,
    "--zone", $Zone
)
if (-not $RefreshSaturn) { $builder += "--local-only" }
python @builder
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RebuildOrderSignals) {
    python "$PSScriptRoot\build_order_signal_vintages.py" `
        --config $Config `
        --mode backfill `
        --zone $Zone `
        --start-day $OrderSignalStartDay
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$chronos = @(
    "$PSScriptRoot\run_chronos2_extended_exogenous.py",
    "--config", $Config,
    "--zones", $Zone,
    "--local-files-only"
)
if ($RefreshSaturn) { $chronos += "--refresh-data" }
python @chronos
exit $LASTEXITCODE
