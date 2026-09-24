<# Offline comparison of saved latest models; operational pipeline unchanged. #>
[CmdletBinding()]
param(
    [ValidateSet('Report','List','Status')][string]$Action = 'Report',
    [string]$EndDay,
    [string[]]$Countries,
    [string[]]$Models,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$KpiRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PythonExecutable) {
    $KpiPython = Join-Path (Split-Path -Parent $KpiRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $KpiPython -PathType Leaf) { $PythonExecutable = $KpiPython }
    else {
        $KpiCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $KpiCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $KpiCommand.Source
    }
}
if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) { $PythonExecutable = Join-Path $KpiRoot $PythonExecutable }
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python introuvable : $PythonExecutable" }
$KpiArguments = @((Join-Path $KpiRoot 'run_kpi_report.py'), '--action', $Action.ToLowerInvariant())
if ($EndDay) { $KpiArguments += @('--end-day', $EndDay) }
if ($Countries) { $KpiArguments += @('--zones') + $Countries }
if ($Models) { $KpiArguments += @('--models') + $Models }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $KpiArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun fichier, calcul, forecast ou appel API.'; return }
& $PythonExecutable @KpiArguments
if ($LASTEXITCODE -ne 0) { throw "KPI a retourne le code $LASTEXITCODE. La production reste inchangee." }
