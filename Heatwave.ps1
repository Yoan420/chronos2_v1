<#
.SYNOPSIS
Independent five-country temperature/heat challenger; no operational writes.
.DESCRIPTION
Run: missing historical forecasts + pinned baseline + rolling replay + HTML.
Prepare: source sync and causal data preflight, without model inference.
Report: frozen results only. Status: read-only. Sync: source downloads only.
#>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Sync','Report','Status')][string]$Action = 'Run',
    [string]$Config = 'config\heatwave.yaml',
    [string]$DeliveryDay,
    [ValidateSet('FR','DE','BE','NL','ES')][string[]]$Zones,
    [string]$Device = 'auto',
    [ValidateRange(1,64)][int]$Threads = 4,
    [ValidateRange(1,16)][int]$Workers = 4,
    [string]$PythonExecutable,
    [switch]$SkipAttribution,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$HeatwaveRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
function Resolve-HeatwavePath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $HeatwaveRoot $Value))
}
if (-not $PythonExecutable) {
    $HeatwaveDefaultPython = Join-Path (Split-Path -Parent $HeatwaveRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $HeatwaveDefaultPython -PathType Leaf) { $PythonExecutable = $HeatwaveDefaultPython }
    else {
        $HeatwavePython = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $HeatwavePython) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $HeatwavePython.Source
    }
}
$PythonExecutable = Resolve-HeatwavePath $PythonExecutable
$HeatwaveScript = Join-Path $HeatwaveRoot 'run_heatwave_forecast.py'
$HeatwaveConfig = Resolve-HeatwavePath $Config
foreach ($HeatwaveFile in @($PythonExecutable, $HeatwaveScript, $HeatwaveConfig)) {
    if (-not (Test-Path -LiteralPath $HeatwaveFile -PathType Leaf)) { throw "Fichier introuvable : $HeatwaveFile" }
}
$HeatwaveArguments = @($HeatwaveScript, '--action', $Action.ToLowerInvariant(), '--config', $HeatwaveConfig,
                       '--device', $Device, '--threads', "$Threads", '--workers', "$Workers")
if ($DeliveryDay) { $HeatwaveArguments += @('--delivery-day', $DeliveryDay) }
if ($Zones) { $HeatwaveArguments += @('--zones') + $Zones }
if ($SkipAttribution) { $HeatwaveArguments += '--skip-attribution' }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $HeatwaveArguments) -Compress))
if ($DryRun) { Write-Host 'DryRun : aucun calcul ni fichier cree.'; return }
& $PythonExecutable @HeatwaveArguments
if ($LASTEXITCODE -ne 0) { throw "Le candidat Heatwave a retourne le code $LASTEXITCODE. Les runs operationnels restent inchanges." }
