<# Isolated RMSE research: saved inputs only, no production invocation. #>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','Run','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config,
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$RmseRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Config) { $Config = Join-Path $RmseRoot 'config\nyx_rmse.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $RmseRoot $Config }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Configuration introuvable : $Config" }
if (-not $PythonExecutable) {
    $RmsePython = Join-Path (Split-Path -Parent $RmseRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $RmsePython -PathType Leaf) { $PythonExecutable = $RmsePython }
    else {
        $RmseCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $RmseCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $RmseCommand.Source
    }
}
if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) { $PythonExecutable = Join-Path $RmseRoot $PythonExecutable }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python introuvable : $PythonExecutable" }
$RmseArguments = @((Join-Path $RmseRoot 'run_nyx_rmse.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
if ($RunDirectory) { $RmseArguments += @('--run-directory', $RunDirectory) }
Write-Host ('Commande (argv): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $RmseArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun fit, fichier, forecast ou appel API.'; return }
& $PythonExecutable @RmseArguments
if ($LASTEXITCODE -ne 0) { throw "NYX RMSE a retourne le code $LASTEXITCODE. La production reste inchangee." }
