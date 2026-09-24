<#
.SYNOPSIS
Essai isole LoRA rang 16 + correcteur, avec et sans Kalman.
.DESCRIPTION
Prepare fige la recette et reutilise les anciens backtests comme calibration.
Bootstrap complete seulement les jours passes manquants (pas un test prospectif).
Run produit les deux chaines avant publication du prix puis genere le rapport.
Apres la limite horaire, Run produit une comparaison RETROSPECTIVE separee.
Compare demande explicitement cette comparaison, sans toucher au journal prospectif.
FullReport reconstruit un rejeu exploratoire de 365 jours et deux rapports HTML complets.
Resolve actualise les observations/rapports sans recalculer les forecasts.
Aucune modification du Forecast.ps1 operationnel, aucun entrainement NOAA.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Bootstrap', 'Run', 'Compare', 'FullReport', 'Resolve', 'Status')]
    [string]$Action = 'Status',
    [ValidateSet('FR', 'DE', 'BE', 'NL')]
    [string[]]$Zones = @('FR', 'DE', 'BE', 'NL'),
    [string]$DeliveryDay = '',
    [string]$Config = 'config\chronos2_exogenous_rank16_trial.yaml',
    [string]$Device = 'auto',
    [ValidateRange(1, 64)]
    [int]$Threads = 4,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$Python = if (Test-Path -LiteralPath $VenvPython -PathType Leaf) { $VenvPython } else { 'python' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $ProjectRoot $Config }
$Arguments = @((Join-Path $ProjectRoot 'run_lora_rank16_trial.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
$Arguments += @('--zones') + $Zones + @('--device', $Device, '--threads', [string]$Threads)
if ($DeliveryDay) { $Arguments += @('--delivery-day', $DeliveryDay) }
if ($Action -in @('Run', 'Compare', 'FullReport', 'Bootstrap') -and -not $DeliveryDay) { throw '-DeliveryDay est requis.' }
Write-Host ('Commande (argv): ' + (ConvertTo-Json -InputObject (@($Python) + $Arguments) -Compress))
if (-not $DryRun) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Le laboratoire LoRA rang 16 a retourne le code $LASTEXITCODE." }
}
