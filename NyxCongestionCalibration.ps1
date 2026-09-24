<# Isolated calibration research launcher; original runs and stage one stay read-only. #>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','Run','Backtest','Report','Status','DryRun')][string]$Action = 'Run',
    [string]$Config,
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$CalibrationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Config) { $Config = Join-Path $CalibrationRoot 'config\nyx_congestion_calibration.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $CalibrationRoot $Config }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Configuration introuvable : $Config" }
if ($Action -in @('Prepare','DryRun') -and $RunDirectory) { throw 'Prepare/DryRun ne reutilisent pas -RunDirectory.' }
if (-not $PythonExecutable) {
    $CalibrationPython = Join-Path (Split-Path -Parent $CalibrationRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $CalibrationPython -PathType Leaf) { $PythonExecutable = $CalibrationPython }
    else {
        $CalibrationCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $CalibrationCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $CalibrationCommand.Source
    }
}
if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) { $PythonExecutable = Join-Path $CalibrationRoot $PythonExecutable }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python introuvable : $PythonExecutable" }
$CalibrationArguments = @((Join-Path $CalibrationRoot 'run_nyx_congestion_calibration.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
if ($RunDirectory) { $CalibrationArguments += @('--run-directory', $RunDirectory) }
Write-Host ('Commande (argv): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $CalibrationArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun fichier, entrainement, forecast ou appel API.'; return }
& $PythonExecutable @CalibrationArguments
if ($LASTEXITCODE -ne 0) { throw "NYX Congestion Calibration a retourne le code $LASTEXITCODE. La production reste inchangee." }
