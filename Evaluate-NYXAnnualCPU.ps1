<# Prepare, rejoue et evalue la chaine annuelle CPU sur une periode explicite. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^\d{4}-\d{2}-\d{2}$')][string]$FirstDay,
    [Parameter(Mandatory=$true)][ValidatePattern('^\d{4}-\d{2}-\d{2}$')][string]$StopDayExclusive,
    [string]$EvaluationRoot = 'runs/evaluations/annual_cpu_full_chain',
    [switch]$Activate
)
$ErrorActionPreference = 'Stop'
$NyxPython = Join-Path $PSScriptRoot '.venv-annual\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $NyxPython -PathType Leaf)) { throw 'Executer Setup-NYXAnnualCPU.ps1 auparavant.' }
$NyxFirst = [datetime]::ParseExact($FirstDay, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture)
$NyxStop = [datetime]::ParseExact($StopDayExclusive, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture)
$NyxDays = ($NyxStop - $NyxFirst).Days
if ($NyxDays -lt 1 -or $NyxDays -gt 365) { throw 'Choisir une periode de 1 a 365 jours.' }
if ($Activate -and $NyxDays -ne 365) { throw 'Une qualification exige exactement 365 jours consecutifs.' }
function Invoke-NyxAnnual([string[]]$Arguments) {
    & $NyxPython -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Evaluation arretee (code $LASTEXITCODE). Corriger la cause puis reprendre la meme commande." }
}
Push-Location -LiteralPath $PSScriptRoot
try {
    $NyxEvaluation = [IO.Path]::GetFullPath($EvaluationRoot)
    $NyxInputs = Join-Path $NyxEvaluation 'inputs'
    $NyxComparisons = Join-Path $NyxEvaluation 'comparisons'
    $NyxRun = Join-Path $NyxEvaluation 'evaluation'
    $NyxCache = Join-Path $NyxEvaluation 'source_cache'
    if (-not (Test-Path -LiteralPath (Join-Path $NyxRun 'plan.json'))) {
        for ($NyxDate = $NyxFirst; $NyxDate -lt $NyxStop; $NyxDate = $NyxDate.AddDays(1)) {
            $NyxDay = $NyxDate.ToString('yyyy-MM-dd')
            Invoke-NyxAnnual @('run_nyx_annual_pipeline.py','--action','prepare','--delivery-day',$NyxDay,
                '--bundle',(Join-Path $NyxInputs $NyxDay),'--output',(Join-Path $NyxEvaluation "preparation/$NyxDay"),
                '--source-cache-root',$NyxCache)
        }
        Invoke-NyxAnnual @('export_nyx_annual_comparisons.py','--first-day',$FirstDay,'--stop-day-exclusive',$StopDayExclusive,'--output',$NyxComparisons)
        Invoke-NyxAnnual @('evaluate_nyx_annual_cpu_full_chain.py','plan','--first-day',$FirstDay,'--stop-day-exclusive',$StopDayExclusive,
            '--bundles',$NyxInputs,'--comparisons',$NyxComparisons,'--output',$NyxRun)
    } else {
        $NyxPlan = Get-Content -LiteralPath (Join-Path $NyxRun 'plan.json') -Raw | ConvertFrom-Json
        if ($NyxPlan.first_delivery_day -ne $FirstDay -or $NyxPlan.stop_day_exclusive -ne $StopDayExclusive) {
            throw 'Ce dossier contient un plan pour une autre periode. Choisir un autre -EvaluationRoot.'
        }
    }
    Invoke-NyxAnnual @('evaluate_nyx_annual_cpu_full_chain.py','predict','--output',$NyxRun)
    Invoke-NyxAnnual @('evaluate_nyx_annual_cpu_full_chain.py','score','--output',$NyxRun)
    if ($Activate) { Invoke-NyxAnnual @('qualify_nyx_annual_cpu.py','--full-chain-evaluation',$NyxRun,'--activate') }
} finally { Pop-Location }
