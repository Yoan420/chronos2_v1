<# Dedicated research launcher: no modification of Forecast.ps1 or production. #>
[CmdletBinding()]
param(
    [ValidateSet('Collect','Prepare','Run','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config,
    [string]$RunDirectory,
    [string]$StartDay,
    [string]$EndDay,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$CongestionRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Config) { $Config = Join-Path $CongestionRoot 'config\nyx_congestion.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $CongestionRoot $Config }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Configuration introuvable : $Config" }
if ($Action -eq 'Collect') {
    if (-not $StartDay -or -not $EndDay) { throw 'Collect exige -StartDay et -EndDay (1 a 366 jours).' }
    if ($RunDirectory) { throw 'Collect ne modifie pas un snapshot existant; ne pas fournir -RunDirectory.' }
} elseif ($StartDay -or $EndDay) { throw 'StartDay et EndDay sont reserves a Collect.' }
if ($Action -eq 'Prepare' -and $RunDirectory) { throw 'Prepare cree un nouveau snapshot; ne pas fournir -RunDirectory.' }
if (-not $PythonExecutable) {
    $CongestionPython = Join-Path (Split-Path -Parent $CongestionRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $CongestionPython -PathType Leaf) { $PythonExecutable = $CongestionPython }
    else {
        $CongestionCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $CongestionCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $CongestionCommand.Source
    }
}
if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) { $PythonExecutable = Join-Path $CongestionRoot $PythonExecutable }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python introuvable : $PythonExecutable" }
$CongestionArguments = @((Join-Path $CongestionRoot 'run_nyx_congestion.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
if ($RunDirectory) { $CongestionArguments += @('--run-directory', $RunDirectory) }
if ($StartDay) { $CongestionArguments += @('--start-day', $StartDay) }
if ($EndDay) { $CongestionArguments += @('--end-day', $EndDay) }
Write-Host ('Commande (argv): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $CongestionArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun fit, fichier, forecast ou appel API.'; return }
& $PythonExecutable @CongestionArguments
if ($LASTEXITCODE -ne 0) { throw "NYX Congestion a retourne le code $LASTEXITCODE. La production reste inchangee." }


