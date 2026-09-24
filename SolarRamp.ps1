<#
.SYNOPSIS
Laboratoire solaire/rampes NYX separe. Report ne reentraine aucun modele.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Backtest','Report','Status','Prospective')]
    [string]$Action = 'Run',
    [string]$Config = '',
    [string]$RunDirectory = '',
    [string]$PythonExecutable = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Config)) { $Config = Join-Path $ProjectRoot 'config\nyx_solar_ramp.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $ProjectRoot $Config }
$Arguments = @((Join-Path $ProjectRoot 'run_nyx_solar_ramp.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
if (-not [string]::IsNullOrWhiteSpace($RunDirectory)) {
    if (-not [System.IO.Path]::IsPathRooted($RunDirectory)) { $RunDirectory = Join-Path $ProjectRoot $RunDirectory }
    $Arguments += @('--run-directory', $RunDirectory)
}
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $Arguments) -Compress))
if ($DryRun) { exit 0 }
& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Le laboratoire solaire a retourne le code $LASTEXITCODE." }
