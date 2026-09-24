[CmdletBinding()]
param(
    [ValidateSet('Plan', 'Backfill', 'Audit', 'Preflight', 'Poc', 'All')]
    [string]$Action = 'Plan',
    [ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]
    [string]$Country = 'FR',
    [string]$DeliveryDay = (Get-Date).AddDays(1).ToString('yyyy-MM-dd'),
    [string]$SourceRun = '',
    [string]$BaseKalmanConfig = '',
    [string]$FlowStore = 'data\pit\jao_core_flowbased\flowbased_features.parquet',
    [string]$Config = 'config\kalman_flowbased_experimental.yaml',
    [ValidateRange(1, 8)]
    [int]$Workers = 4,
    [ValidateRange(1, 2)]
    [int]$JaoWorkers = 2,
    [switch]$AuditOnly,
    [switch]$Insecure,
    [string]$CaBundle = '',
    [switch]$AllowNonPit,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { 'python' }
$Materializer = Join-Path $ProjectRoot 'materialize_jao_core_flowbased.py'
$AuditRunner = Join-Path $ProjectRoot 'run_cnec_ram_audit.py'
$PocRunner = Join-Path $ProjectRoot 'run_kalman_flowbased_experiment.py'
$ResolvedFlowStore = if ([System.IO.Path]::IsPathRooted($FlowStore)) {
    [System.IO.Path]::GetFullPath($FlowStore)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $FlowStore))
}
if ([System.IO.Path]::GetFileName($ResolvedFlowStore) -ne 'flowbased_features.parquet') {
    throw '-FlowStore doit se terminer par flowbased_features.parquet.'
}
$FlowRoot = Split-Path -Parent $ResolvedFlowStore

if ($Insecure -and $CaBundle) {
    throw '-Insecure et -CaBundle sont mutuellement exclusifs.'
}
$ResolvedCaBundle = ''
if ($CaBundle) {
    $ResolvedCaBundle = if ([System.IO.Path]::IsPathRooted($CaBundle)) {
        [System.IO.Path]::GetFullPath($CaBundle)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $CaBundle))
    }
    if (-not (Test-Path -LiteralPath $ResolvedCaBundle -PathType Leaf)) {
        throw "-CaBundle introuvable: $ResolvedCaBundle. Fournissez un vrai fichier PEM ou omettez -CaBundle pour utiliser automatiquement les certificats configures sur ce poste."
    }
}
if ($Action -in @('Preflight', 'Poc', 'All') -and $Country -ne 'FR') {
    throw "Le POC kalman_flowbased v1 est limite a FR; Country=$Country est incompatible avec -Action $Action."
}

function Invoke-CheckedPython([string[]]$Arguments) {
    Write-Host ('Commande (argv): ' + ($Arguments | ConvertTo-Json -Compress))
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Le pipeline flow-based a retourne le code $LASTEXITCODE."
    }
}

function Get-MaterializerArguments([switch]$DryRun) {
    $TrainingDays = if ($AuditOnly) { '0' } else { '365' }
    $FutureDays = if ($AuditOnly) { '0' } else { '1' }
    $MaterializerEndDay = if ($AuditOnly) {
        ([datetime]::ParseExact(
            $DeliveryDay,
            'yyyy-MM-dd',
            [System.Globalization.CultureInfo]::InvariantCulture
        )).AddDays(-1).ToString('yyyy-MM-dd')
    } else {
        $DeliveryDay
    }
    $Arguments = @(
        $Materializer,
        '--end-day', $MaterializerEndDay,
        '--evaluation-days', '365',
        '--training-days', $TrainingDays,
        '--future-days', $FutureDays,
        '--workers', [string]$JaoWorkers,
        '--output-root', $FlowRoot
    )
    if ($DryRun) { $Arguments += '--dry-run' }
    if ($Insecure) { $Arguments += '--insecure' }
    if ($ResolvedCaBundle) { $Arguments += @('--ca-bundle', $ResolvedCaBundle) }
    if ($AllowNonPit) { $Arguments += '--allow-non-pit' }
    if ($Overwrite) { $Arguments += '--overwrite' }
    return $Arguments
}

function Get-AuditArguments {
    $Arguments = @(
        $AuditRunner,
        '--zone', $Country,
        '--delivery-day', $DeliveryDay,
        '--flowbased-features', $ResolvedFlowStore
    )
    if ($SourceRun) { $Arguments += @('--source-run', $SourceRun) }
    if ($Overwrite) { $Arguments += '--overwrite' }
    return $Arguments
}

function Get-PocArguments {
    $Arguments = @(
        $PocRunner,
        '--zone', $Country,
        '--delivery-day', $DeliveryDay,
        '--flowbased-features', $ResolvedFlowStore,
        '--config', $Config,
        '--rolling-refit-workers', [string]$Workers
    )
    if ($SourceRun) { $Arguments += @('--source-run', $SourceRun) }
    if ($BaseKalmanConfig) {
        $Arguments += @('--base-kalman-config', $BaseKalmanConfig)
    }
    if ($Overwrite) { $Arguments += '--overwrite' }
    return $Arguments
}

Push-Location $ProjectRoot
try {
    switch ($Action) {
        'Plan' {
            Invoke-CheckedPython (Get-MaterializerArguments -DryRun)
            Write-Host ''
            Write-Host 'Etapes suivantes:'
            $AuditOnlyToken = if ($AuditOnly) { ' -AuditOnly' } else { '' }
            Write-Host "  .\FlowBased.ps1 -Action Backfill -DeliveryDay $DeliveryDay$AuditOnlyToken"
            Write-Host "  .\FlowBased.ps1 -Action Audit -Country $Country -DeliveryDay $DeliveryDay"
            if (-not $AuditOnly -and $Country -eq 'FR') {
                Write-Host "  .\FlowBased.ps1 -Action Preflight -Country FR -DeliveryDay $DeliveryDay"
                Write-Host "  .\FlowBased.ps1 -Action Poc -Country FR -DeliveryDay $DeliveryDay"
            }
        }
        'Backfill' { Invoke-CheckedPython (Get-MaterializerArguments) }
        'Audit' { Invoke-CheckedPython (Get-AuditArguments) }
        'Preflight' {
            if ($AuditOnly) { throw '-AuditOnly est incompatible avec -Action Preflight.' }
            $PreflightArguments = Get-PocArguments
            $PreflightArguments += '--preflight-only'
            Invoke-CheckedPython $PreflightArguments
        }
        'Poc' {
            if ($AuditOnly) { throw '-AuditOnly est incompatible avec -Action Poc.' }
            Invoke-CheckedPython (Get-PocArguments)
        }
        'All' {
            if ($AuditOnly) { throw '-AuditOnly est incompatible avec -Action All.' }
            Invoke-CheckedPython (Get-MaterializerArguments)
            Invoke-CheckedPython (Get-AuditArguments)
            Invoke-CheckedPython (Get-PocArguments)
        }
    }
}
finally {
    Pop-Location
}
