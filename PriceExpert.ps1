<#
.SYNOPSIS
Teste nuclear_kalman_extreme, un correcteur de PRIX independant de Forecast.ps1.
.DESCRIPTION
Run fige les entrees et compare prix + EVA sur la meme annee que le test precedent.
Calibration progressive apres 90 jours, plafonnee a 365 jours de passe disponible.
Le prix de reference reste un proxy veille non executable. Aucune activation.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Audit','Prepare','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\nuclear_kalman_extreme.yaml',
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$PriceExpertRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-PriceExpertPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $PriceExpertRoot $Value))
}
try {
    if ($RunDirectory -and $Action -in @('Run','Prepare','Audit')) { throw 'RunDirectory est reserve a Backtest, Report et Status.' }
    if (-not $PythonExecutable) {
        $PriceDefaultPython = Join-Path (Split-Path -Parent $PriceExpertRoot) 'venvs\pricefm311\Scripts\python.exe'
        if (Test-Path -LiteralPath $PriceDefaultPython -PathType Leaf) { $PythonExecutable = $PriceDefaultPython }
        else {
            $PricePythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -eq $PricePythonCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
            $PythonExecutable = $PricePythonCommand.Source
        }
    }
    $PythonExecutable = Resolve-PriceExpertPath $PythonExecutable
    $PriceScript = Join-Path $PriceExpertRoot 'run_price_expert.py'
    $PriceConfig = Resolve-PriceExpertPath $Config
    $PriceRequired = @($PythonExecutable, $PriceScript)
    if (-not $RunDirectory) { $PriceRequired += $PriceConfig }
    foreach ($PricePath in $PriceRequired) {
        if (-not (Test-Path -LiteralPath $PricePath -PathType Leaf)) { throw "Fichier introuvable : $PricePath" }
    }
    $PriceArguments = @($PriceScript, '--action', $Action.ToLowerInvariant(), '--config', $PriceConfig)
    if ($RunDirectory) { $PriceArguments += @('--run-directory', (Resolve-PriceExpertPath $RunDirectory)) }
    Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $PriceArguments) -Compress))
    if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
    & $PythonExecutable @PriceArguments
    if ($LASTEXITCODE -ne 0) { throw "Le laboratoire Price Expert a retourne le code $LASTEXITCODE." }
}
catch { throw "[Price Expert] $($_.Exception.Message)" }
