<# Dedicated research launcher: no modification of Forecast.ps1 or production. #>
[CmdletBinding()]
param(
    [ValidateSet('Collect','Prepare','Run','Backtest','Report','Status')][string]$Action = 'Run',
    [string]$Config,
    [string]$RunDirectory,
    [string]$StartDay,
    [string]$EndDay,
    [string]$CaBundle,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$PhysicalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Config) { $Config = Join-Path $PhysicalRoot 'config\nyx_physical_p50.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $PhysicalRoot $Config }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Configuration introuvable : $Config" }
if ($Action -eq 'Collect') {
    if (-not $StartDay -or -not $EndDay) { throw 'Collect exige -StartDay et -EndDay (1 a 30 jours).' }
    if ($RunDirectory) { throw 'Collect ne modifie pas un snapshot existant; ne pas fournir -RunDirectory.' }
} elseif ($StartDay -or $EndDay -or $CaBundle) { throw 'StartDay, EndDay et CaBundle sont reserves a Collect.' }
if ($Action -eq 'Prepare' -and $RunDirectory) { throw 'Prepare cree un nouveau snapshot; ne pas fournir -RunDirectory.' }
if ($CaBundle -and -not (Test-Path -LiteralPath $CaBundle -PathType Leaf)) { throw "Certificat introuvable : $CaBundle" }
if (-not $PythonExecutable) {
    $PhysicalPython = Join-Path (Split-Path -Parent $PhysicalRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $PhysicalPython -PathType Leaf) { $PythonExecutable = $PhysicalPython }
    else {
        $PhysicalCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $PhysicalCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $PhysicalCommand.Source
    }
}
if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) { $PythonExecutable = Join-Path $PhysicalRoot $PythonExecutable }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python introuvable : $PythonExecutable" }
$PhysicalArguments = @((Join-Path $PhysicalRoot 'run_nyx_physical_p50.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
if ($RunDirectory) { $PhysicalArguments += @('--run-directory', $RunDirectory) }
if ($StartDay) { $PhysicalArguments += @('--start-day', $StartDay) }
if ($EndDay) { $PhysicalArguments += @('--end-day', $EndDay) }
if ($CaBundle) { $PhysicalArguments += @('--ca-bundle', $CaBundle) }
Write-Host ('Commande (argv): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $PhysicalArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun fit, fichier, forecast ou appel API.'; return }
& $PythonExecutable @PhysicalArguments
if ($LASTEXITCODE -ne 0) { throw "NYX Physical P50 a retourne le code $LASTEXITCODE. La production reste inchangee." }
