<# Isolated NYX + clean gas/coal in Chronos, residual and Kalman. #>
[CmdletBinding()]
param(
    [ValidateSet('Audit','Sync','Prepare','Run','Report','Status')][string]$Action = 'Run',
    [string]$Config,
    [string]$DeliveryDay,
    [ValidateSet('FR','DE','BE','NL')][string[]]$Zones,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [ValidateRange(1,32)][int]$Threads = 4,
    [ValidateRange(1,2)][int]$Workers = 2,
    [string]$PythonExecutable,
    [switch]$SkipAttribution,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$FuelRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Config) { $Config = Join-Path $FuelRoot 'config\nyx_clean_fuel_full.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $FuelRoot $Config }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Configuration introuvable : $Config" }
if (-not $PythonExecutable) {
    $FuelPython = Join-Path (Split-Path -Parent $FuelRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $FuelPython -PathType Leaf) { $PythonExecutable = $FuelPython }
    else {
        $FuelCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $FuelCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $FuelCommand.Source
    }
}
$FuelArguments = @((Join-Path $FuelRoot 'run_clean_fuel_full.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config, '--threads', $Threads, '--workers', $Workers, '--device', $Device)
if ($DeliveryDay) { $FuelArguments += @('--delivery-day', $DeliveryDay) }
if ($Zones) { $FuelArguments += @('--zones') + $Zones }
if ($SkipAttribution) { $FuelArguments += @('--skip-attribution') }
Write-Host ('Commande (argv): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $FuelArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun fichier, calcul ou appel API.'; return }
& $PythonExecutable @FuelArguments
if ($LASTEXITCODE -ne 0) { throw "Clean Fuel a retourne le code $LASTEXITCODE. La production reste inchangee." }
