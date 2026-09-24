<# Isolated RMSE residual ablation. Forecast.ps1 and operational models are untouched. #>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','Smoke','Run','Status','Report')][string]$Action = 'Run',
    [string]$Config = 'config\catboost_rmse.yaml',
    [ValidateSet('FR','DE','BE','NL')][string[]]$Zones = @('FR','DE','BE','NL'),
    [ValidateRange(1,16)][int]$Threads = 2,
    [ValidateRange(0,365)][int]$MaxDays = 0,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$RmseRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PythonExecutable) { $PythonExecutable = Join-Path (Split-Path -Parent $RmseRoot) 'venvs\pricefm311\Scripts\python.exe' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $RmseRoot $Config }
$RmseScript = Join-Path $RmseRoot 'run_catboost_rmse.py'
foreach ($RmseFile in @($PythonExecutable, $RmseScript, $Config)) {
    if (-not (Test-Path -LiteralPath $RmseFile -PathType Leaf)) { throw "Fichier introuvable : $RmseFile" }
}
$RmseArgs = @('-u', $RmseScript, '--config', $Config, '--action', $Action.ToLowerInvariant(), '--zones') + $Zones + @('--threads', "$Threads")
if ($MaxDays -gt 0) { $RmseArgs += @('--max-days', "$MaxDays") }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $RmseArgs) -Compress))
if ($DryRun) { Write-Host 'Aucun calcul ni fichier cree.'; return }
& $PythonExecutable @RmseArgs
if ($LASTEXITCODE -ne 0) { throw "Le laboratoire CatBoost RMSE a retourne le code $LASTEXITCODE. Relancer la meme commande pour reprendre les checkpoints valides." }
