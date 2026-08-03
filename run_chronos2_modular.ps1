param(
    [string]$Config = "$PSScriptRoot\chronos2_inputs_asof_jplus1.yaml",
    [string[]]$Zones = @("FR"),
    [switch]$RefreshData
)

$Arguments = @(
    "$PSScriptRoot\run_chronos2_modular.py",
    "--config", $Config,
    "--zones"
) + $Zones

if ($RefreshData) {
    $Arguments += "--refresh-data"
}

python @Arguments
exit $LASTEXITCODE
