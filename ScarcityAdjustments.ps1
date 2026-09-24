<# Diagnostic chiffre d'amplitude des pics, separe des forecasts operationnels. #>
[CmdletBinding()]
param(
    [ValidateSet('Run','Report','Status')][string]$Action = 'Run',
    [string]$SourceSuite,
    [string]$RunDirectory,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ScarcityAdjustmentsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-ScarcityAdjustmentPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ScarcityAdjustmentsRoot $Value))
}
if ($SourceSuite -and $Action -ne 'Run') { throw 'SourceSuite est reserve a Run.' }
if ($RunDirectory -and $Action -eq 'Run') { throw 'RunDirectory est reserve a Report/Status.' }
if (-not $PythonExecutable) {
    $ScarcityAdjustmentsDefaultPython = Join-Path (Split-Path -Parent $ScarcityAdjustmentsRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $ScarcityAdjustmentsDefaultPython -PathType Leaf) { $PythonExecutable = $ScarcityAdjustmentsDefaultPython }
    else {
        $ScarcityAdjustmentsPythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $ScarcityAdjustmentsPythonCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $ScarcityAdjustmentsPythonCommand.Source
    }
}
$PythonExecutable = Resolve-ScarcityAdjustmentPath $PythonExecutable
$ScarcityAdjustmentsScript = Join-Path $ScarcityAdjustmentsRoot 'run_nyx_scarcity_adjustments.py'
foreach ($ScarcityAdjustmentsPath in @($PythonExecutable, $ScarcityAdjustmentsScript)) {
    if (-not (Test-Path -LiteralPath $ScarcityAdjustmentsPath -PathType Leaf)) { throw "Fichier introuvable : $ScarcityAdjustmentsPath" }
}
$ScarcityAdjustmentsArguments = @($ScarcityAdjustmentsScript, '--action', $Action.ToLowerInvariant())
if ($SourceSuite) { $ScarcityAdjustmentsArguments += @('--source-suite', (Resolve-ScarcityAdjustmentPath $SourceSuite)) }
if ($RunDirectory) { $ScarcityAdjustmentsArguments += @('--run-directory', (Resolve-ScarcityAdjustmentPath $RunDirectory)) }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $ScarcityAdjustmentsArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @ScarcityAdjustmentsArguments
if ($LASTEXITCODE -ne 0) { throw "Le diagnostic d'ajustement a retourne le code $LASTEXITCODE." }
