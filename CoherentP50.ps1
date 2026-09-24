<# Isolated CoherentP50 research launcher. Forecast.ps1 and production stay unchanged. #>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\nyx_coherent_p50.yaml',
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$CoherentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-CoherentPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $CoherentRoot $Value))
}
if ($RunDirectory -and $Action -in @('Run','Prepare')) { throw 'RunDirectory : utiliser Backtest, Report ou Status.' }
if (-not $PythonExecutable) {
    $CoherentPython = Join-Path (Split-Path -Parent $CoherentRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $CoherentPython -PathType Leaf) { $PythonExecutable = $CoherentPython }
    else {
        $CoherentCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $CoherentCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $CoherentCommand.Source
    }
}
$PythonExecutable = Resolve-CoherentPath $PythonExecutable
$CoherentScript = Join-Path $CoherentRoot 'run_nyx_coherent_p50.py'
$CoherentConfig = Resolve-CoherentPath $Config
$CoherentRequired = @($PythonExecutable, $CoherentScript)
if (-not $RunDirectory) { $CoherentRequired += $CoherentConfig }
foreach ($CoherentPath in $CoherentRequired) {
    if (-not (Test-Path -LiteralPath $CoherentPath -PathType Leaf)) { throw "Fichier introuvable : $CoherentPath" }
}
$CoherentArguments = @($CoherentScript, '--action', $Action.ToLowerInvariant(), '--config', $CoherentConfig)
if ($RunDirectory) { $CoherentArguments += @('--run-directory', (Resolve-CoherentPath $RunDirectory)) }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $CoherentArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @CoherentArguments
if ($LASTEXITCODE -ne 0) { throw "Le laboratoire CoherentP50 a retourne le code $LASTEXITCODE. La production reste inchangee." }
