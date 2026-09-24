<#
.SYNOPSIS
Challenger NYX des pointes de prix. Ne modifie ni Forecast.ps1 ni ses sorties.
.DESCRIPTION
Run fige les sources, entraine uniquement l'expert et produit un rapport autonome.
Backtest reutilise un snapshot termine. Audit ne modifie aucun fichier.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Audit','Refresh','Prepare','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\nyx_scarcity.yaml',
    [string[]]$Countries,
    [string]$DeliveryDay,
    [string]$EndDay,
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$RefreshSources,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ScarcityRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-ScarcityPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ScarcityRoot $Value))
}
try {
    if ($RunDirectory -and $Action -in @('Run','Prepare','Audit','Refresh')) { throw 'RunDirectory est reserve a Backtest, Report et Status.' }
    if ($Action -in @('Backtest','Report','Status') -and ($Countries -or $DeliveryDay -or $EndDay)) { throw 'Le snapshot utilise ses pays et dates figes ; aucun override autorise.' }
    if ($RefreshSources -and $Action -notin @('Run','Prepare')) { throw 'RefreshSources est reserve a Run et Prepare. Audit reste en lecture seule.' }
    if (-not $PythonExecutable) {
        $ScarcityDefaultPython = Join-Path (Split-Path -Parent $ScarcityRoot) 'venvs\pricefm311\Scripts\python.exe'
        if (Test-Path -LiteralPath $ScarcityDefaultPython -PathType Leaf) { $PythonExecutable = $ScarcityDefaultPython }
        else {
            $ScarcityPythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -eq $ScarcityPythonCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
            $PythonExecutable = $ScarcityPythonCommand.Source
        }
    }
    $PythonExecutable = Resolve-ScarcityPath $PythonExecutable
    $ScarcityScript = Join-Path $ScarcityRoot 'run_nyx_scarcity.py'
    $ScarcityConfig = Resolve-ScarcityPath $Config
    $ScarcityRequired = @($PythonExecutable, $ScarcityScript)
    if (-not $RunDirectory) { $ScarcityRequired += $ScarcityConfig }
    foreach ($ScarcityPath in $ScarcityRequired) {
        if (-not (Test-Path -LiteralPath $ScarcityPath -PathType Leaf)) { throw "Fichier introuvable : $ScarcityPath" }
    }
    $ScarcityArguments = @($ScarcityScript, '--action', $Action.ToLowerInvariant(), '--config', $ScarcityConfig)
    if ($RunDirectory) { $ScarcityArguments += @('--run-directory', (Resolve-ScarcityPath $RunDirectory)) }
    if ($Countries) { $ScarcityArguments += '--zones'; $ScarcityArguments += $Countries }
    if ($DeliveryDay) { $ScarcityArguments += @('--delivery-day', $DeliveryDay) }
    if ($EndDay) { $ScarcityArguments += @('--end-day', $EndDay) }
    if ($RefreshSources) { $ScarcityArguments += '--refresh-sources' }
    Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $ScarcityArguments) -Compress))
    if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
    & $PythonExecutable @ScarcityArguments
    if ($LASTEXITCODE -ne 0) { throw "Le challenger Scarcity a retourne le code $LASTEXITCODE." }
}
catch { throw "[Scarcity] $($_.Exception.Message)" }
