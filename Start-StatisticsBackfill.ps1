<#
.SYNOPSIS
Reconstruit les forecasts PIT manquants pour completer les Statistics.

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Start-StatisticsBackfill.ps1' -Countries FR,DE,BE,NL,ES

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Start-StatisticsBackfill.ps1' -Countries FR,DE,BE,NL,ES -ThenRunForecast

.NOTES
Les jours manquants sont detectes automatiquement par pays. Chaque replay est
execute au cutoff civil causal D-1 08:00 et publie dans la racine _replays de
la configuration active. Les archives existantes sont validees puis ignorees;
elles ne sont jamais ecrasees.
#>
[CmdletBinding()]
param(
    [ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]
    [string[]]$Countries = @('FR', 'DE', 'BE', 'NL', 'ES'),

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [ValidateRange(1, 128)]
    [int]$Threads = 4,

    [ValidateRange(1, 128)]
    [int]$Workers = 4,

    [switch]$AllowModelDownload,
    [switch]$StopOnError,
    [switch]$DryRun,
    [switch]$ThenRunForecast,

    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$LiveDeliveryDay = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$Launcher = Join-Path $ProjectRoot 'run_statistics_backfill.py'

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python introuvable : $PythonExe"
}
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Launcher introuvable : $Launcher"
}
if ($Countries.Count -eq 0) {
    throw 'Selectionnez au moins un pays.'
}
if (-not [string]::IsNullOrWhiteSpace($LiveDeliveryDay) -and -not $ThenRunForecast) {
    throw '-LiveDeliveryDay exige aussi -ThenRunForecast.'
}

$Arguments = @(
    $Launcher,
    '--zones'
) + $Countries + @(
    '--device', $Device,
    '--threads', [string]$Threads,
    '--workers', [string]$Workers
)
if ($AllowModelDownload) {
    $Arguments += '--allow-model-download'
}
if ($StopOnError) {
    $Arguments += '--stop-on-error'
}
if ($DryRun) {
    $Arguments += '--dry-run'
}
if ($ThenRunForecast) {
    $Arguments += '--then-run-live'
}
if (-not [string]::IsNullOrWhiteSpace($LiveDeliveryDay)) {
    $Arguments += @('--live-delivery-day', $LiveDeliveryDay)
}

Push-Location $ProjectRoot
try {
    & $PythonExe @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
