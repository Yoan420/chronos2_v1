<# Separate fundamental research launcher. Forecast.ps1 and production remain unchanged. #>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\nyx_fundamental_stress.yaml',
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$FundamentalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-FundamentalPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $FundamentalRoot $Value))
}
if ($RunDirectory -and $Action -in @('Run','Prepare')) { throw 'RunDirectory : utiliser Backtest, Report ou Status.' }
if (-not $PythonExecutable) {
    $FundamentalPython = Join-Path (Split-Path -Parent $FundamentalRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $FundamentalPython -PathType Leaf) { $PythonExecutable = $FundamentalPython }
    else {
        $FundamentalCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $FundamentalCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $FundamentalCommand.Source
    }
}
$PythonExecutable = Resolve-FundamentalPath $PythonExecutable
$FundamentalScript = Join-Path $FundamentalRoot 'run_nyx_fundamental_stress.py'
$FundamentalConfig = Resolve-FundamentalPath $Config
$FundamentalRequired = @($PythonExecutable, $FundamentalScript)
if (-not $RunDirectory) { $FundamentalRequired += $FundamentalConfig }
foreach ($FundamentalPath in $FundamentalRequired) {
    if (-not (Test-Path -LiteralPath $FundamentalPath -PathType Leaf)) { throw "Fichier introuvable : $FundamentalPath" }
}
$FundamentalArguments = @($FundamentalScript, '--action', $Action.ToLowerInvariant(), '--config', $FundamentalConfig)
if ($RunDirectory) { $FundamentalArguments += @('--run-directory', (Resolve-FundamentalPath $RunDirectory)) }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $FundamentalArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @FundamentalArguments
if ($LASTEXITCODE -ne 0) { throw "Le laboratoire fundamental a retourne le code $LASTEXITCODE. La production reste inchangee." }
