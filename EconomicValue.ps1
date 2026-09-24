<#
.SYNOPSIS
Laboratoire Economic Value Added, independant de Forecast.ps1.
.DESCRIPTION
Lit les rapports locaux existants sans entrainement ni synchronisation.
Par defaut : dernier export commun, 365 jours, 100 MW TOTAL, proxy prix veille.
Audit est en lecture seule. Prepare fige les donnees. Backtest evalue un snapshot.
Report regenere uniquement le HTML. Run fait Prepare + Backtest + Report.
Les chemins relatifs sont resolus depuis ce script, quel que soit le dossier courant.
DryRun n'execute rien et ne cree aucun fichier. Aucun ordre n'est passe.
.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Run
.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Run -Countries FR -Models nuclear_kalman -PortfolioMW 100
.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\EconomicValue.ps1' -Action Report
#>
[CmdletBinding()]
param(
    [ValidateSet('Audit', 'Prepare', 'Backtest', 'Run', 'Report', 'Status')]
    [string]$Action = 'Run',
    [string]$Config = 'config\economic_value.yaml',
    [ValidateSet('FR', 'DE', 'BE', 'NL')][string[]]$Countries,
    [ValidateSet('autonomous', 'kalman', 'nuclear_autonomous', 'nuclear_kalman')][string[]]$Models,
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')][string]$DeliveryDay,
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')][string]$EndDay,
    [ValidateRange(0.001, 1000000)][double]$PortfolioMW,
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-EconomicPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}
try {
    foreach ($DayValue in @($DeliveryDay, $EndDay)) {
        if ($DayValue) { $null = [datetime]::ParseExact($DayValue, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture) }
    }
    if ($Countries) {
        $Countries = @($Countries | ForEach-Object { $_.ToUpperInvariant() })
        if (@($Countries | Select-Object -Unique).Count -ne $Countries.Count) { throw 'Pays en doublon.' }
    }
    if ($Models) {
        $Models = @($Models | ForEach-Object { $_.ToLowerInvariant() })
        if (@($Models | Select-Object -Unique).Count -ne $Models.Count) { throw 'Modeles en doublon.' }
    }
    if (-not $PythonExecutable) {
        $DefaultPython = Join-Path (Split-Path -Parent $ProjectRoot) 'venvs\pricefm311\Scripts\python.exe'
        if (Test-Path -LiteralPath $DefaultPython -PathType Leaf) { $PythonExecutable = $DefaultPython }
        else {
            $PythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -eq $PythonCommand) { throw 'Python introuvable. Indiquez -PythonExecutable.' }
            $PythonExecutable = $PythonCommand.Source
        }
    }
    $PythonExecutable = Resolve-EconomicPath $PythonExecutable
    $ConfigPath = Resolve-EconomicPath $Config
    $ScriptPath = Join-Path $ProjectRoot 'run_economic_value.py'
    $RequiredPaths = @($PythonExecutable, $ScriptPath)
    if (-not ($RunDirectory -and $Action -in @('Backtest', 'Report', 'Status'))) { $RequiredPaths += $ConfigPath }
    foreach ($RequiredPath in $RequiredPaths) {
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) { throw "Fichier introuvable : $RequiredPath" }
    }
    $Arguments = @($ScriptPath, '--action', $Action.ToLowerInvariant(), '--config', $ConfigPath)
    if ($Countries) { $Arguments += @('--zones') + $Countries }
    if ($Models) { $Arguments += @('--models') + $Models }
    if ($DeliveryDay) { $Arguments += @('--delivery-day', $DeliveryDay) }
    if ($EndDay) { $Arguments += @('--end-day', $EndDay) }
    if ($PSBoundParameters.ContainsKey('PortfolioMW')) {
        $Arguments += @('--portfolio-mw', $PortfolioMW.ToString([cultureinfo]::InvariantCulture))
    }
    if ($RunDirectory) { $Arguments += @('--run-directory', (Resolve-EconomicPath $RunDirectory)) }
    $DisplayCommand = @($PythonExecutable) + $Arguments
    Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject $DisplayCommand -Compress))
    if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Le laboratoire Economic Value a retourne le code $LASTEXITCODE." }
}
catch {
    throw "[Economic Value] $($_.Exception.Message)"
}
