<#
.SYNOPSIS
Expert des mouvements extremes + gouvernance economique, hors production.
.DESCRIPTION
Run fige les entrees, entraine en rolling365 et compare les decisions a Storm.
La reference reste le proxy non executable du prix day-ahead de la veille.
Forecast.ps1, Both et Complete ne sont jamais appeles ou modifies.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Audit','Prepare','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\economic_extreme_policy.yaml',
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ExpertProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-ExpertPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ExpertProjectRoot $Value))
}
try {
    if ($RunDirectory -and $Action -in @('Run','Prepare','Audit')) {
        throw 'RunDirectory est reserve a Backtest, Report et Status.'
    }
    if (-not $PythonExecutable) {
        $ExpertDefaultPython = Join-Path (Split-Path -Parent $ExpertProjectRoot) 'venvs\pricefm311\Scripts\python.exe'
        if (Test-Path -LiteralPath $ExpertDefaultPython -PathType Leaf) { $PythonExecutable = $ExpertDefaultPython }
        else {
            $ExpertPythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -eq $ExpertPythonCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
            $PythonExecutable = $ExpertPythonCommand.Source
        }
    }
    $PythonExecutable = Resolve-ExpertPath $PythonExecutable
    $ExpertScript = Join-Path $ExpertProjectRoot 'run_economic_extreme_policy.py'
    $ExpertConfig = Resolve-ExpertPath $Config
    $ExpertRequired = @($PythonExecutable, $ExpertScript)
    if (-not $RunDirectory) { $ExpertRequired += $ExpertConfig }
    foreach ($ExpertPath in $ExpertRequired) {
        if (-not (Test-Path -LiteralPath $ExpertPath -PathType Leaf)) { throw "Fichier introuvable : $ExpertPath" }
    }
    $ExpertArguments = @($ExpertScript, '--action', $Action.ToLowerInvariant(), '--config', $ExpertConfig)
    if ($RunDirectory) { $ExpertArguments += @('--run-directory', (Resolve-ExpertPath $RunDirectory)) }
    Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $ExpertArguments) -Compress))
    if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
    & $PythonExecutable @ExpertArguments
    if ($LASTEXITCODE -ne 0) { throw "Le laboratoire Economic Expert a retourne le code $LASTEXITCODE." }
}
catch { throw "[Economic Expert] $($_.Exception.Message)" }
