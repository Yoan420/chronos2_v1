param(
    [string]$Config = "$PSScriptRoot\chronos2_inputs_asof_jplus1_regime_order_signals.yaml",
    [string]$Zone = "FR",
    [switch]$RefreshSaturn
)

$ErrorActionPreference = "Stop"

if ($RefreshSaturn) {
    python "$PSScriptRoot\update_saturn_data.py" --config $Config
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python "$PSScriptRoot\build_order_signal_vintages.py" `
    --config $Config `
    --mode live `
    --zone $Zone
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ChronosRunner = "$PSScriptRoot\run_chronos2_order_signals.py"

python $ChronosRunner `
    --config $Config `
    --zones $Zone `
    --local-files-only
exit $LASTEXITCODE
