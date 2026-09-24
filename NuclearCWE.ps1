<#
.SYNOPSIS
Independent CWE nuclear challenger; never calls Forecast.ps1 or publishes over its exports.
.DESCRIPTION
Run = isolated source sync + frozen snapshot + rolling replay + standard HTML reports.
Prepare downloads only missing BE/NL capacity vintages and freezes the incumbent comparison.
Report uses frozen results and does not fit any model. Status is read-only.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Sync','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\nuclear_cwe.yaml',
    [string]$DeliveryDay,
    [ValidateSet('FR','DE','BE','NL')][string[]]$Zones,
    [string]$Device = 'auto',
    [ValidateRange(1,64)][int]$Threads = 4,
    [ValidateRange(1,16)][int]$Workers = 4,
    [string]$PythonExecutable,
    [switch]$SkipAttribution,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$CweRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-CwePath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $CweRoot $Value))
}
if (-not $PythonExecutable) {
    $CweDefaultPython = Join-Path (Split-Path -Parent $CweRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $CweDefaultPython -PathType Leaf) { $PythonExecutable = $CweDefaultPython }
    else {
        $CwePython = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $CwePython) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $CwePython.Source
    }
}
$PythonExecutable = Resolve-CwePath $PythonExecutable
$CweScript = Join-Path $CweRoot 'run_nuclear_cwe_forecast.py'
$CweConfig = Resolve-CwePath $Config
foreach ($CweFile in @($PythonExecutable, $CweScript, $CweConfig)) {
    if (-not (Test-Path -LiteralPath $CweFile -PathType Leaf)) { throw "Fichier introuvable : $CweFile" }
}
$CweArguments = @($CweScript, '--action', $Action.ToLowerInvariant(), '--config', $CweConfig,
                  '--device', $Device, '--threads', "$Threads", '--workers', "$Workers")
if ($DeliveryDay) { $CweArguments += @('--delivery-day', $DeliveryDay) }
if ($Zones) { $CweArguments += @('--zones') + $Zones }
if ($SkipAttribution) { $CweArguments += '--skip-attribution' }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $CweArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @CweArguments
if ($LASTEXITCODE -ne 0) { throw "Le candidat Nuclear CWE a retourne le code $LASTEXITCODE. Les runs operationnels restent inchanges." }
