<#
.SYNOPSIS
Compare les variantes anti-spikes sur les memes entrees figees, hors production.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Backtest','Report','Status','Install')][string]$Action = 'Run',
    [string]$Config = 'config\nyx_scarcity_variants.yaml',
    [string]$SourceSnapshot,
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ScarcityVariantsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-ScarcityVariantPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ScarcityVariantsRoot $Value))
}
try {
    if ($SourceSnapshot -and $Action -notin @('Run','Prepare')) { throw 'SourceSnapshot est reserve a Run/Prepare.' }
    if ($RunDirectory -and $Action -notin @('Backtest','Report','Status')) { throw 'RunDirectory est reserve a Backtest/Report/Status.' }
    if (-not $PythonExecutable) {
        $ScarcityVariantsDefaultPython = Join-Path (Split-Path -Parent $ScarcityVariantsRoot) 'venvs\pricefm311\Scripts\python.exe'
        if (Test-Path -LiteralPath $ScarcityVariantsDefaultPython -PathType Leaf) { $PythonExecutable = $ScarcityVariantsDefaultPython }
        else {
            $ScarcityVariantsPythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -eq $ScarcityVariantsPythonCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
            $PythonExecutable = $ScarcityVariantsPythonCommand.Source
        }
    }
    $PythonExecutable = Resolve-ScarcityVariantPath $PythonExecutable
    $ScarcityVariantsScript = Join-Path $ScarcityVariantsRoot 'run_nyx_scarcity_variants.py'
    $ScarcityVariantsConfig = Resolve-ScarcityVariantPath $Config
    $ScarcityVariantsRequired = @($PythonExecutable, $ScarcityVariantsScript)
    if (-not $RunDirectory -and $Action -ne 'Install') { $ScarcityVariantsRequired += $ScarcityVariantsConfig }
    foreach ($ScarcityVariantsPath in $ScarcityVariantsRequired) {
        if (-not (Test-Path -LiteralPath $ScarcityVariantsPath -PathType Leaf)) { throw "Fichier introuvable : $ScarcityVariantsPath" }
    }
    $ScarcityVariantsArguments = @($ScarcityVariantsScript, '--action', $Action.ToLowerInvariant(), '--config', $ScarcityVariantsConfig)
    if ($RunDirectory) { $ScarcityVariantsArguments += @('--run-directory', (Resolve-ScarcityVariantPath $RunDirectory)) }
    if ($SourceSnapshot) { $ScarcityVariantsArguments += @('--source-snapshot', (Resolve-ScarcityVariantPath $SourceSnapshot)) }
    Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $ScarcityVariantsArguments) -Compress))
    if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
    & $PythonExecutable @ScarcityVariantsArguments
    if ($LASTEXITCODE -ne 0) { throw "Les variantes Scarcity ont retourne le code $LASTEXITCODE." }
}
catch { throw "[Scarcity variants] $($_.Exception.Message)" }
