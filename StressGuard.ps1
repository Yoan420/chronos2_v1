<# Isolated physical-risk replay and prospective ledger; Forecast.ps1 unchanged. #>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Backtest','Report','Status','Freeze','Capture','Issue','Evaluate')][string]$Action = 'Run',
    [string]$Config = 'config\nyx_stress_guard.yaml',
    [string]$RunDirectory,
    [string]$LedgerDirectory,
    [string]$InputDirectory,
    [string]$DeliveryDay,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$StressGuardRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-StressGuardPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $StressGuardRoot $Value))
}
if ($RunDirectory -and $Action -in @('Run','Prepare')) { throw 'RunDirectory : utiliser Backtest pour reprendre.' }
if (-not $PythonExecutable) {
    $StressGuardPython = Join-Path (Split-Path -Parent $StressGuardRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $StressGuardPython -PathType Leaf) { $PythonExecutable = $StressGuardPython }
    else {
        $StressGuardCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $StressGuardCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $StressGuardCommand.Source
    }
}
$PythonExecutable = Resolve-StressGuardPath $PythonExecutable
$StressGuardScript = Join-Path $StressGuardRoot 'run_nyx_stress_guard.py'
$StressGuardConfig = Resolve-StressGuardPath $Config
foreach ($StressGuardPath in @($PythonExecutable, $StressGuardScript)) {
    if (-not (Test-Path -LiteralPath $StressGuardPath -PathType Leaf)) { throw "Fichier introuvable : $StressGuardPath" }
}
$StressGuardArguments = @($StressGuardScript, '--action', $Action.ToLowerInvariant(), '--config', $StressGuardConfig)
if ($RunDirectory) { $StressGuardArguments += @('--run-directory', (Resolve-StressGuardPath $RunDirectory)) }
if ($LedgerDirectory) { $StressGuardArguments += @('--ledger-directory', (Resolve-StressGuardPath $LedgerDirectory)) }
if ($InputDirectory) { $StressGuardArguments += @('--input-directory', (Resolve-StressGuardPath $InputDirectory)) }
if ($DeliveryDay) { $StressGuardArguments += @('--delivery-day', $DeliveryDay) }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $StressGuardArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul, appel API ou fichier cree.'; return }
& $PythonExecutable @StressGuardArguments
if ($LASTEXITCODE -ne 0) { throw "StressGuard a retourne le code $LASTEXITCODE. La production reste inchangee." }
