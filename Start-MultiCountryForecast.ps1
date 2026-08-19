<#
.SYNOPSIS
Lance les forecasts day-ahead selectionnes et valide un rapport HTML detaille
par pays.

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Start-MultiCountryForecast.ps1' -Countries FR,DE,BE

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Start-MultiCountryForecast.ps1' -Countries NL -DeliveryDay '2026-08-20'

.NOTES
Sans -DeliveryDay, le jour de livraison est demain en Europe/Paris. Les pays
sont executes sequentiellement. Une archive existante n'est reutilisee qu'apres
validation complete de ses checksums et de sa grille horaire.
#>
[CmdletBinding()]
param(
    [ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]
    [string[]]$Countries = @('FR', 'DE', 'BE', 'NL', 'ES'),

    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [ValidateRange(1, 128)]
    [int]$Threads = 4,

    [ValidateRange(1, 128)]
    [int]$Workers = 4,

    [switch]$AllowModelDownload,
    [switch]$StopOnError,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$Launcher = Join-Path $ProjectRoot 'run_multicountry_forecast.py'

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python introuvable : $PythonExe"
}
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    throw "Launcher introuvable : $Launcher"
}
if ($Countries.Count -eq 0) {
    throw 'Selectionnez au moins un pays.'
}

$Arguments = @(
    $Launcher,
    '--zones'
) + $Countries + @(
    '--device', $Device,
    '--threads', [string]$Threads,
    '--workers', [string]$Workers
)
if (-not [string]::IsNullOrWhiteSpace($DeliveryDay)) {
    $Arguments += @('--delivery-day', $DeliveryDay)
}
if ($AllowModelDownload) {
    $Arguments += '--allow-model-download'
}
if ($StopOnError) {
    $Arguments += '--stop-on-error'
}
if ($DryRun) {
    $Arguments += '--dry-run'
}

Push-Location $ProjectRoot
try {
    & $PythonExe @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
