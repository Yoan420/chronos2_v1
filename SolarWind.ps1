<#
.SYNOPSIS
NYX + four hourly CWE solar + two DE/NL wind forecasts. Separate experiment, operational modes unchanged.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Sync','Audit','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\solar_wind.yaml',
    [string]$DeliveryDay,
    [ValidateSet('DE','NL')][string[]]$Zones,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [ValidateRange(1,32)][int]$Threads = 2,
    [ValidateRange(1,2)][int]$Workers = 2,
    [string]$PythonExecutable,
    [switch]$SkipAttribution,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$SolarProject = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PythonExecutable) { $PythonExecutable = Join-Path (Split-Path -Parent $SolarProject) 'venvs\pricefm311\Scripts\python.exe' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $SolarProject $Config }
$SolarScript = Join-Path $SolarProject 'run_solar_wind_forecast.py'
foreach ($SolarFile in @($PythonExecutable, $SolarScript, $Config)) {
    if (-not (Test-Path -LiteralPath $SolarFile -PathType Leaf)) { throw "Fichier introuvable : $SolarFile" }
}
$SolarArguments = @($SolarScript, '--config', $Config, '--action', $Action.ToLowerInvariant(),
    '--device', $Device, '--threads', "$Threads", '--workers', "$Workers")
if ($DeliveryDay) { $SolarArguments += @('--delivery-day', $DeliveryDay) }
if ($Zones) { $SolarArguments += @('--zones') + $Zones }
if ($SkipAttribution) { $SolarArguments += '--skip-attribution' }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $SolarArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @SolarArguments
if ($LASTEXITCODE -ne 0) { throw "SolarWind a retourne le code $LASTEXITCODE. La production reste inchangee." }
