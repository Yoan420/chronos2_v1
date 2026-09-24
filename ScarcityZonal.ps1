<# Separate zonal research launcher. Forecast.ps1 and production remain unchanged. #>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\nyx_scarcity_zonal.yaml',
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ZonalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-ZonalPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ZonalRoot $Value))
}
if ($RunDirectory -and $Action -in @('Run','Prepare')) { throw 'RunDirectory : utiliser Backtest, Report ou Status.' }
if (-not $PythonExecutable) {
    $ZonalPython = Join-Path (Split-Path -Parent $ZonalRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $ZonalPython -PathType Leaf) { $PythonExecutable = $ZonalPython }
    else {
        $ZonalCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $ZonalCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $ZonalCommand.Source
    }
}
$PythonExecutable = Resolve-ZonalPath $PythonExecutable
$ZonalScript = Join-Path $ZonalRoot 'run_nyx_scarcity_zonal.py'
$ZonalConfig = Resolve-ZonalPath $Config
$ZonalRequired = @($PythonExecutable, $ZonalScript)
if (-not $RunDirectory) { $ZonalRequired += $ZonalConfig }
foreach ($ZonalPath in $ZonalRequired) {
    if (-not (Test-Path -LiteralPath $ZonalPath -PathType Leaf)) { throw "Fichier introuvable : $ZonalPath" }
}
$ZonalArguments = @($ZonalScript, '--action', $Action.ToLowerInvariant(), '--config', $ZonalConfig)
if ($RunDirectory) { $ZonalArguments += @('--run-directory', (Resolve-ZonalPath $RunDirectory)) }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $ZonalArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @ZonalArguments
if ($LASTEXITCODE -ne 0) { throw "Le laboratoire zonal a retourne le code $LASTEXITCODE. La production reste inchangee." }
