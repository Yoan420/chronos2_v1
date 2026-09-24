<# Separate solar residual/Kalman ablation; no neural computation or production activation. #>
[CmdletBinding()]
param(
    [ValidateSet('Run','Prepare','Audit','Status','Report','Evaluate')][string]$Action = 'Run',
    [string]$Config = 'config\solar_correction.yaml',
    [string]$DeliveryDay,
    [ValidateSet('FR','DE','BE','NL')][string[]]$Zones,
    [ValidateRange(1,32)][int]$Threads = 4,
    [ValidateRange(1,2)][int]$Workers = 2,
    [ValidateRange(0,2147483647)][int]$AfterSolarPid = 0,
    [string]$PythonExecutable,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$SolarCorrectionRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PythonExecutable) { $PythonExecutable = Join-Path (Split-Path -Parent $SolarCorrectionRoot) 'venvs\pricefm311\Scripts\python.exe' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $SolarCorrectionRoot $Config }
$SolarCorrectionScript = Join-Path $SolarCorrectionRoot 'run_solar_correction.py'
foreach ($SolarCorrectionFile in @($PythonExecutable, $SolarCorrectionScript, $Config)) {
    if (-not (Test-Path -LiteralPath $SolarCorrectionFile -PathType Leaf)) { throw "Fichier introuvable : $SolarCorrectionFile" }
}
$SolarCorrectionArgs = @('-u', $SolarCorrectionScript, '--config', $Config, '--action', $Action.ToLowerInvariant(), '--threads', "$Threads", '--workers', "$Workers")
if ($DeliveryDay) { $SolarCorrectionArgs += @('--delivery-day', $DeliveryDay) }
if ($Zones) { $SolarCorrectionArgs += @('--zones') + $Zones }
if ($AfterSolarPid -gt 0) { $SolarCorrectionArgs += @('--after-solar-pid', "$AfterSolarPid") }
Write-Host ('Commande (argv, shell=False): ' + (ConvertTo-Json -InputObject (@($PythonExecutable) + $SolarCorrectionArgs) -Compress))
if ($DryRun) { Write-Host 'Aucun calcul ni fichier cree.'; return }
& $PythonExecutable @SolarCorrectionArgs
if ($LASTEXITCODE -ne 0) { throw "SolarCorrection a retourne le code $LASTEXITCODE. La production reste inchangee." }
