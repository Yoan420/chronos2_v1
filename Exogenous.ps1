<#
.SYNOPSIS
Construit, entraîne, évalue et gouverne les candidats Chronos-2 + LoRA.

.DESCRIPTION
Les actions historiques conservent leur comportement strict. Le workflow
prospectif optionnel utilise Panel/CalibrationPanel avec -ProspectiveFreeze,
puis EpochPlan, EpochFreeze et Shadow avant l'origine. Après publication du
dernier prix holdout, ResolveHoldout, Backtest, FinalBacktest et
EpochFinalizeCheck lient la preuve tardive sans modifier le candidat gelé.
EpochFreeze refuse toute preuve PIT non opérationnelle et aucune action epoch
ne promeut ni n'active un modèle.

Les actions ResearchFitValidationCorrector et
ResearchEvaluateValidationCorrector constituent un diagnostic séparé : fit
sur les 30 jours de validation, puis lecture du holdout uniquement après
scellement. Cette preuve n'est pas OOF et ne peut jamais être promue.

.EXAMPLE
& '.\Exogenous.ps1' -Action EpochEarliest

.EXAMPLE
& '.\Exogenous.ps1' -Action EpochPlan -Zones FR -RunDirectory '<bundle>' `
  -ResidualCalibrationDirectory '<residual>' -FirstShadowDay 2026-09-06

.EXAMPLE
& '.\Exogenous.ps1' -Action RecoverTrainingSnapshot `
  -Config 'config\chronos2_exogenous_lora_rank8_reference.yaml' `
  -RunDirectory 'runs\experiments\chronos2_exogenous_lora_poc_v1\artifact' `
  -ZoneArtifactsRoot 'runs\experiments\chronos2_exogenous_lora_poc_v1_recovered' `
  -Zones FR,DE,BE,NL

.EXAMPLE
& '.\Exogenous.ps1' -Action ResearchFitValidationCorrector `
  -Config 'config\chronos2_exogenous_lora_poc.yaml' `
  -RunDirectory '<bundle-zone>' -Zones FR

.EXAMPLE
& '.\Exogenous.ps1' -Action ResearchEvaluateValidationCorrector `
  -Config 'config\chronos2_exogenous_lora_poc.yaml' `
  -RunDirectory '<bundle-zone>' -Zones FR
#>
[CmdletBinding()]
param(
    [ValidateSet('Panel', 'CalibrationPanel', 'ResolveHoldout', 'Validate', 'Train', 'PrepareZones', 'RecoverTrainingSnapshot', 'Backtest', 'CalibrateResidual', 'ResearchFitValidationCorrector', 'ResearchEvaluateValidationCorrector', 'FinalBacktest', 'CompareCandidates', 'EpochEarliest', 'EpochPlan', 'EpochFreeze', 'EpochVerify', 'EpochFinalizeCheck', 'Shadow', 'Govern', 'Verify', 'POC')]
    [string]$Action = 'Validate',
    [string]$Config = 'config\chronos2_exogenous_lora_poc.yaml',
    [ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]
    [string[]]$Zones = @('FR'),
    [string]$EndDay = '',
    [ValidateSet('residual_only', 'residual_weather', 'residual_fuel', 'residual_flowbased', 'full')]
    [string]$Pack = 'full',
    [string]$RunDirectory = '',
    [string]$ZoneArtifactsRoot = '',
    [string]$ResidualCalibrationDirectory = '',
    [string]$ResidualCalibrationPanel = '',
    [string]$ResearchCorrectorDirectory = '',
    [string]$OofCheckpointCacheDirectory = '',
    [string]$IncumbentStatistics = '',
    [string]$Rank8Candidate = '',
    [string]$Rank16Candidate = '',
    [string]$ComparisonOutput = 'runs\experiments\chronos2_exogenous\candidate_comparisons\rank8_vs_rank16',
    [ValidateRange(1, 365)]
    [int]$BlockDays = 30,
    [string]$Predictions = '',
    [string]$ShadowPredictions = '',
    [string]$ResolvedEvaluationPanel = '',
    [string]$LiveSourceManifest = '',
    [switch]$ProspectiveFreeze,
    [string]$ShadowEpochDirectory = '',
    [string]$FirstShadowDay = '',
    [string]$Policy = 'config\chronos2_exogenous_promotion_v1.yaml',
    [string]$OutputRoot = 'runs\experiments\chronos2_exogenous\promotion',
    [switch]$Overwrite,
    [string]$Device = 'auto',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ConfigWasExplicit = $PSBoundParameters.ContainsKey('Config')
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$Python = if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    $VenvPython
} else {
    'python'
}

$PanelRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_panel.py'
$FineTuneRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_finetune.py'
$PrepareZonesRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_prepare_zones.py'
$RecoverTrainingSnapshotRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_recover_training_snapshot.py'
$BacktestRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_evaluate.py'
$ResidualCalibrationRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_calibrate_residual.py'
$ResearchValidationCorrectorRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_research_validation_corrector.py'
$FinalBacktestRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_final_evaluate.py'
$CandidateComparisonRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_compare_candidates.py'
$FinalShadowRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_finalize_shadow.py'
$GovernanceRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_governance.py'
$ShadowEpochRunner = Join-Path $ProjectRoot 'run_chronos2_exogenous_shadow_epoch.py'

function Resolve-ProjectPath([string]$Value) {
    if (-not $Value) { return '' }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}

function Get-ConfigScalar([string]$Path, [string]$Section, [string]$Key) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return '' }
    $InSection = $false
    foreach ($Line in [System.IO.File]::ReadAllLines($Path)) {
        if ($Line -match ('^' + [regex]::Escape($Section) + ':\s*(?:#.*)?$')) {
            $InSection = $true
            continue
        }
        if ($InSection -and $Line -match '^\S') { break }
        if ($InSection -and $Line -match ('^\s+' + [regex]::Escape($Key) + ':\s*([^#]+?)\s*(?:#.*)?$')) {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ''
}

function Get-EffectiveRunDirectory {
    if ($RunDirectory) { return Resolve-ProjectPath $RunDirectory }
    $Configured = Get-ConfigScalar $ResolvedConfig 'output' 'directory'
    if (-not $Configured) {
        throw "Impossible d'inferer RunDirectory depuis output.directory; fournissez -RunDirectory."
    }
    return Resolve-ProjectPath $Configured
}

function Get-PanelOutput {
    $Configured = Get-ConfigScalar $ResolvedConfig 'data' 'panel_path'
    if (-not $Configured) {
        throw "Impossible d'inferer le panel depuis data.panel_path dans la configuration."
    }
    return Resolve-ProjectPath $Configured
}

function Get-ResidualCalibrationPanelOutput {
    if ($ResidualCalibrationPanel) {
        return Resolve-ProjectPath $ResidualCalibrationPanel
    }
    $TrainingPanel = Get-PanelOutput
    return Join-Path (Split-Path -Parent $TrainingPanel) 'residual_calibration_panel.parquet'
}

function Get-ResidualCalibrationOutput {
    if ($ResidualCalibrationDirectory) {
        return Resolve-ProjectPath $ResidualCalibrationDirectory
    }
    $EffectiveRun = Get-EffectiveRunDirectory
    return Join-Path (Join-Path $EffectiveRun 'residual_calibration') $Zones[0].ToLowerInvariant()
}

function Get-ShadowEpochOutput {
    if ($ShadowEpochDirectory) {
        return Resolve-ProjectPath $ShadowEpochDirectory
    }
    return Join-Path (Get-EffectiveRunDirectory) 'shadow_epoch'
}

function Get-IncumbentStatisticsOutput {
    if ($IncumbentStatistics) {
        $Explicit = Resolve-ProjectPath $IncumbentStatistics
        if (-not $DryRun -and -not (Test-Path -LiteralPath $Explicit -PathType Leaf)) {
            throw "Statistics incumbent introuvables: $Explicit"
        }
        return $Explicit
    }
    $LiveRoot = Join-Path $ProjectRoot 'runs\live'
    if (Test-Path -LiteralPath $LiveRoot -PathType Container) {
        $ArchivePattern = ($Zones[0].ToLowerInvariant() + '_day_ahead_')
        $Candidates = @(
            Get-ChildItem -LiteralPath $LiveRoot -Recurse -File -Filter 'statistics_history_hourly.csv.gz' -ErrorAction SilentlyContinue |
                Where-Object { $_.Directory.Name.StartsWith($ArchivePattern, [System.StringComparison]::OrdinalIgnoreCase) } |
                Sort-Object LastWriteTime -Descending
        )
        if ($Candidates.Count -gt 0) {
            return $Candidates[0].FullName
        }
    }
    if ($DryRun) {
        $Suffix = if ($EndDay) { $EndDay } else { 'YYYY-MM-DD' }
        return Join-Path $LiveRoot ($Zones[0].ToLowerInvariant() + '_day_ahead_' + $Suffix + '\statistics_history_hourly.csv.gz')
    }
    throw 'Aucun statistics_history_hourly.csv.gz incumbent trouve. Fournissez -IncumbentStatistics.'
}

function Assert-SingleZone([string]$ForAction) {
    if ($Zones.Count -ne 1) {
        throw "-$ForAction exige exactement une zone en v1. Lancez une evaluation et une gouvernance separees par zone."
    }
}

function Invoke-CheckedPython([string[]]$Arguments) {
    $FullArgv = @($Python) + $Arguments
    Write-Host ('Commande (argv, shell=False): ' + ($FullArgv | ConvertTo-Json -Compress))
    if ($DryRun) { return }
    if (-not (Test-Path -LiteralPath $Arguments[0] -PathType Leaf)) {
        throw "Runner introuvable: $($Arguments[0]). L'etape n'est pas encore installee."
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Le POC Chronos-2 exogene a retourne le code $LASTEXITCODE."
    }
}

function Invoke-Panel([switch]$ReuseExisting, [switch]$ForResidualCalibration) {
    $PanelOutput = if ($ForResidualCalibration) {
        Get-ResidualCalibrationPanelOutput
    } else {
        Get-PanelOutput
    }
    $ContextLength = Get-ConfigScalar $ResolvedConfig 'data' 'context_length'
    if (-not $ContextLength) {
        throw "Impossible d'inferer data.context_length pour le panel d'entrainement."
    }
    if (-not $DryRun -and (Test-Path -LiteralPath $PanelOutput) -and -not $Overwrite) {
        if ($ReuseExisting) {
            Write-Host "Panel existant reutilise: $PanelOutput"
            return
        }
        throw "Panel deja present: $PanelOutput. Utilisez -Overwrite pour le reconstruire."
    }
    $PanelMode = if ($ForResidualCalibration) { 'calibration' } else { 'training' }
    $TrainingDays = if ($ForResidualCalibration) { '730' } else { '365' }
    $Arguments = @(
        $PanelRunner,
        '--mode', $PanelMode,
        '--zones'
    )
    $Arguments += $Zones
    $Arguments += @(
        '--pack', $Pack,
        '--training-days', $TrainingDays,
        '--evaluation-days', '365',
        '--context-length', $ContextLength,
        '--output', $PanelOutput
    )
    if ($EndDay) { $Arguments += @('--end-day', $EndDay) }
    if ($ProspectiveFreeze) {
        $Arguments += @(
            '--allow-unresolved-final-evaluation-day',
            '--require-production-pit'
        )
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-ResolveHoldout {
    if (-not $ResolvedEvaluationPanel) {
        throw '-ResolvedEvaluationPanel est obligatoire avec ResolveHoldout.'
    }
    if (-not $EndDay) {
        throw '-EndDay est obligatoire avec ResolveHoldout et doit rester celui du holdout gele.'
    }
    $FrozenPanel = [System.IO.Path]::GetFullPath((Get-PanelOutput))
    $ResolvedPanel = [System.IO.Path]::GetFullPath((Resolve-ProjectPath $ResolvedEvaluationPanel))
    if ([System.StringComparer]::OrdinalIgnoreCase.Equals($FrozenPanel, $ResolvedPanel)) {
        throw '-ResolvedEvaluationPanel doit etre un nouveau chemin: le panel gele ne sera jamais ecrase.'
    }
    $ContextLength = Get-ConfigScalar $ResolvedConfig 'data' 'context_length'
    if (-not $ContextLength) {
        throw "Impossible d'inferer data.context_length pour le panel resolu."
    }
    $Arguments = @(
        $PanelRunner,
        '--mode', 'training',
        '--zones'
    )
    $Arguments += $Zones
    $Arguments += @(
        '--pack', $Pack,
        '--training-days', '365',
        '--evaluation-days', '365',
        '--context-length', $ContextLength,
        '--end-day', $EndDay,
        '--output', $ResolvedPanel,
        '--allow-unresolved-final-evaluation-day',
        '--require-production-pit'
    )
    Invoke-CheckedPython $Arguments
}

function Invoke-CalibrateResidual {
    Assert-SingleZone 'Action CalibrateResidual'
    if ($Overwrite) {
        throw '-Overwrite est interdit avec CalibrateResidual: les folds sont immuables et repris automatiquement. Utilisez un nouveau -ResidualCalibrationDirectory.'
    }
    $EffectiveRun = Get-EffectiveRunDirectory
    $CalibrationPanel = Get-ResidualCalibrationPanelOutput
    $Arguments = @(
        $ResidualCalibrationRunner,
        '--config', $ResolvedConfig,
        '--run-directory', $EffectiveRun,
        '--panel', $CalibrationPanel,
        '--panel-audit', ($CalibrationPanel + '.audit.json'),
        '--item-id', $Zones[0],
        '--block-days', $BlockDays.ToString(),
        '--device-map', $Device
    )
    if ($ResidualCalibrationDirectory) {
        $Arguments += @(
            '--output-directory', (Resolve-ProjectPath $ResidualCalibrationDirectory)
        )
    }
    if ($OofCheckpointCacheDirectory) {
        $Arguments += @(
            '--shared-checkpoint-cache-directory',
            (Resolve-ProjectPath $OofCheckpointCacheDirectory)
        )
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-ResearchFitValidationCorrector {
    Assert-SingleZone 'Action ResearchFitValidationCorrector'
    $Arguments = @(
        $ResearchValidationCorrectorRunner,
        'fit',
        '--config', $ResolvedConfig,
        '--run-directory', (Get-EffectiveRunDirectory),
        '--item-id', $Zones[0],
        '--device-map', $Device
    )
    if ($ResearchCorrectorDirectory) {
        $Arguments += @(
            '--research-directory',
            (Resolve-ProjectPath $ResearchCorrectorDirectory)
        )
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-ResearchEvaluateValidationCorrector {
    Assert-SingleZone 'Action ResearchEvaluateValidationCorrector'
    $Arguments = @(
        $ResearchValidationCorrectorRunner,
        'evaluate',
        '--config', $ResolvedConfig,
        '--run-directory', (Get-EffectiveRunDirectory),
        '--item-id', $Zones[0]
    )
    if ($ResearchCorrectorDirectory) {
        $Arguments += @(
            '--research-directory',
            (Resolve-ProjectPath $ResearchCorrectorDirectory)
        )
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-FinalBacktest {
    Assert-SingleZone 'Action FinalBacktest'
    $EffectiveRun = Get-EffectiveRunDirectory
    $CalibrationOutput = Get-ResidualCalibrationOutput
    $Corrector = Join-Path $CalibrationOutput 'residual_corrector.json'
    $OofAudit = Join-Path $CalibrationOutput 'oof_predictions_365.csv.gz.audit.json'
    if (-not $DryRun) {
        foreach ($Required in @($Corrector, $OofAudit)) {
            if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
                throw "Artefact de calibration residuelle absent: $Required"
            }
        }
    }
    $Arguments = @(
        $FinalBacktestRunner,
        '--run-directory', $EffectiveRun,
        '--residual-corrector', $Corrector,
        '--oof-audit', $OofAudit,
        '--incumbent-statistics', (Get-IncumbentStatisticsOutput),
        '--zone', $Zones[0]
    )
    if ($Overwrite) { $Arguments += '--overwrite' }
    Invoke-CheckedPython $Arguments
}

function Invoke-CompareCandidates {
    Assert-SingleZone 'Action CompareCandidates'
    if (-not $Rank8Candidate -or -not $Rank16Candidate) {
        throw '-Rank8Candidate et -Rank16Candidate sont obligatoires avec CompareCandidates.'
    }
    $Arguments = @(
        $CandidateComparisonRunner,
        '--rank8', (Resolve-ProjectPath $Rank8Candidate),
        '--rank16', (Resolve-ProjectPath $Rank16Candidate),
        '--policy', $ResolvedPolicy,
        '--output-directory', (Resolve-ProjectPath $ComparisonOutput),
        '--zone', $Zones[0]
    )
    if ($Overwrite) { $Arguments += '--overwrite' }
    Invoke-CheckedPython $Arguments
}

function Invoke-Validate {
    Invoke-CheckedPython @($FineTuneRunner, 'validate', '--config', $ResolvedConfig)
}

function Invoke-Train {
    $Arguments = @($FineTuneRunner, 'train', '--config', $ResolvedConfig)
    if ($Overwrite) { $Arguments += '--overwrite' }
    Invoke-CheckedPython $Arguments
}

function Invoke-PrepareZones {
    $EffectiveRun = Get-EffectiveRunDirectory
    $Arguments = @(
        $PrepareZonesRunner,
        '--source-run-directory', $EffectiveRun,
        '--zones'
    )
    $Arguments += $Zones
    if ($ZoneArtifactsRoot) {
        $Arguments += @('--output-root', (Resolve-ProjectPath $ZoneArtifactsRoot))
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-RecoverTrainingSnapshot {
    if (-not $ConfigWasExplicit) {
        throw '-Config est obligatoire avec RecoverTrainingSnapshot: fournissez la reference YAML exacte du candidat evalue.'
    }
    $EffectiveRun = Get-EffectiveRunDirectory
    $Arguments = @(
        $RecoverTrainingSnapshotRunner,
        '--source-run-directory', $EffectiveRun,
        '--config-reference', $ResolvedConfig,
        '--zones'
    )
    $Arguments += $Zones
    if ($ZoneArtifactsRoot) {
        $Arguments += @('--output-root', (Resolve-ProjectPath $ZoneArtifactsRoot))
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-Backtest {
    Assert-SingleZone 'Action Backtest/POC'
    $EffectiveRun = Get-EffectiveRunDirectory
    $Arguments = @(
        $BacktestRunner,
        '--config', $ResolvedConfig,
        '--run-directory', $EffectiveRun,
        '--item-id', $Zones[0],
        '--device-map', $Device,
        '--mode', 'backtest'
    )
    if ($ResolvedEvaluationPanel) {
        $ResolvedPanel = Resolve-ProjectPath $ResolvedEvaluationPanel
        $Arguments += @(
            '--panel', $ResolvedPanel,
            '--panel-audit', ($ResolvedPanel + '.audit.json')
        )
    }
    if ($Overwrite) { $Arguments += '--overwrite' }
    Invoke-CheckedPython $Arguments
}

function Get-EpochCalibrationInputs {
    $CalibrationOutput = Get-ResidualCalibrationOutput
    return @(
        (Join-Path $CalibrationOutput 'residual_corrector.json'),
        (Join-Path $CalibrationOutput 'oof_predictions_365.csv.gz.audit.json')
    )
}

function Invoke-EpochEarliest {
    $Arguments = @(
        $ShadowEpochRunner,
        'earliest',
        '--timezone', 'Europe/Paris',
        '--cutoff', '08:00',
        '--shadow-days', '30'
    )
    Invoke-CheckedPython $Arguments
}

function Invoke-EpochPlan([switch]$Freeze) {
    Assert-SingleZone 'Action EpochPlan/EpochFreeze'
    $Inputs = Get-EpochCalibrationInputs
    $Arguments = @(
        $ShadowEpochRunner,
        $(if ($Freeze) { 'freeze' } else { 'plan' }),
        '--run-directory', (Get-EffectiveRunDirectory),
        '--residual-corrector', $Inputs[0],
        '--oof-audit', $Inputs[1],
        '--zone', $Zones[0],
        '--shadow-days', '30'
    )
    if ($FirstShadowDay) { $Arguments += @('--first-shadow-day', $FirstShadowDay) }
    if ($Freeze) { $Arguments += @('--output-directory', (Get-ShadowEpochOutput)) }
    Invoke-CheckedPython $Arguments
}

function Invoke-EpochVerify([switch]$Finalize) {
    Assert-SingleZone 'Action EpochVerify/EpochFinalizeCheck'
    $Arguments = @(
        $ShadowEpochRunner,
        $(if ($Finalize) { 'finalize-check' } else { 'verify' }),
        (Get-ShadowEpochOutput),
        '--run-directory', (Get-EffectiveRunDirectory)
    )
    Invoke-CheckedPython $Arguments
}

function Invoke-Shadow {
    Assert-SingleZone 'Action Shadow'
    if ($Overwrite) {
        throw '-Overwrite est interdit avec Shadow: le journal est append-only et idempotent.'
    }
    if (-not $EndDay) {
        throw '-EndDay (jour de livraison D+1) est obligatoire avec Shadow.'
    }
    $EffectiveRun = Get-EffectiveRunDirectory
    $ShadowJournal = if ($ShadowPredictions) {
        Resolve-ProjectPath $ShadowPredictions
    } else {
        Join-Path $EffectiveRun 'shadow_predictions.csv.gz'
    }
    $ProspectiveConfig = (Get-ConfigScalar $ResolvedConfig 'data' 'allow_unresolved_final_evaluation_day')
    if ($ProspectiveConfig -and $ProspectiveConfig.ToLowerInvariant() -eq 'true') {
        if (-not $ShadowEpochDirectory) {
            throw '-ShadowEpochDirectory est obligatoire pour un candidat prospectif two-phase.'
        }
        Invoke-CheckedPython @(
            $ShadowEpochRunner,
            'delivery-check',
            (Get-ShadowEpochOutput),
            '--run-directory', $EffectiveRun,
            '--journal', $ShadowJournal,
            '--delivery-day', $EndDay
        )
    }
    $CalibrationOutput = Get-ResidualCalibrationOutput
    $Corrector = Join-Path $CalibrationOutput 'residual_corrector.json'
    $OofAudit = Join-Path $CalibrationOutput 'oof_predictions_365.csv.gz.audit.json'
    if (-not $DryRun) {
        foreach ($Required in @($Corrector, $OofAudit)) {
            if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
                throw "Artefact de calibration residuelle absent: $Required. Lancez d'abord CalibrateResidual puis FinalBacktest."
            }
        }
    }
    $Incumbent = Get-IncumbentStatisticsOutput
    $ContextLength = Get-ConfigScalar $ResolvedConfig 'data' 'context_length'
    if (-not $ContextLength) {
        throw "Impossible d'inferer data.context_length pour le panel shadow."
    }
    $ShadowInputDirectory = Join-Path $EffectiveRun 'shadow_inputs'
    $ShadowPanel = Join-Path $ShadowInputDirectory ("shadow_panel_" + $EndDay + '.parquet')
    $ShadowPanelAudit = $ShadowPanel + '.audit.json'
    $PanelArguments = @(
        $PanelRunner,
        '--mode', 'shadow',
        '--zones', $Zones[0],
        '--pack', $Pack,
        '--end-day', $EndDay,
        '--context-length', $ContextLength,
        '--output', $ShadowPanel
    )
    if ($LiveSourceManifest) {
        $PanelArguments += @(
            '--live-source-manifest', (Resolve-ProjectPath $LiveSourceManifest)
        )
    }
    Invoke-CheckedPython $PanelArguments
    $Arguments = @(
        $BacktestRunner,
        '--config', $ResolvedConfig,
        '--run-directory', $EffectiveRun,
        '--item-id', $Zones[0],
        '--device-map', $Device,
        '--mode', 'shadow',
        '--panel', $ShadowPanel,
        '--panel-audit', $ShadowPanelAudit
    )
    $Arguments += @('--shadow-predictions', $ShadowJournal)
    Invoke-CheckedPython $Arguments

    $ShadowDirectory = Split-Path -Parent $ShadowJournal
    $RawObserved = Join-Path $ShadowDirectory 'shadow_observed_evidence.csv.gz'
    $RawManifest = Join-Path $ShadowDirectory 'shadow_manifest.json'
    $FinalShadowDirectory = Join-Path $ShadowDirectory 'shadow_final'
    $CanFinalizeShadow = $DryRun
    if (-not $DryRun) {
        $ExperimentManifest = Join-Path $EffectiveRun 'experiment_manifest.json'
        if (Test-Path -LiteralPath $ExperimentManifest -PathType Leaf) {
            $Experiment = Get-Content -LiteralPath $ExperimentManifest -Raw | ConvertFrom-Json
            $CanFinalizeShadow = ($Experiment.production_pipeline_evidence -eq $true)
        }
    }
    if ($CanFinalizeShadow) {
        Invoke-CheckedPython @(
            $FinalShadowRunner,
            '--run-directory', $EffectiveRun,
            '--raw-observed-evidence', $RawObserved,
            '--raw-shadow-manifest', $RawManifest,
            '--raw-shadow-journal', $ShadowJournal,
            '--residual-corrector', $Corrector,
            '--oof-audit', $OofAudit,
            '--incumbent-statistics', $Incumbent,
            '--zone', $Zones[0],
            '--output-directory', $FinalShadowDirectory,
            '--allow-no-observed'
        )
    } else {
        Write-Host 'Shadow brut scelle. Finalisation differee: executez Backtest avec le panel resolu, puis FinalBacktest et EpochFinalizeCheck.'
    }
}

function Get-ShadowManifest([string]$ShadowEvidence) {
    if ([System.IO.Path]::GetFileName($ShadowEvidence) -ne 'shadow_final_evidence.csv.gz') {
        throw "La gouvernance accepte uniquement shadow_final_evidence.csv.gz (LoRA corrige contre incumbent residual_corrected), jamais le journal LoRA brut."
    }
    $Parent = Split-Path -Parent $ShadowEvidence
    $Manifest = Join-Path $Parent 'shadow_final_manifest.json'
    if (-not $DryRun -and -not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "Manifeste shadow final introuvable: $Manifest. -ShadowPredictions doit pointer vers une preuve v4 produite par -Action Shadow."
    }
    return $Manifest
}

function Invoke-Govern([string]$PredictionsOverride = '') {
    Assert-SingleZone 'Action Govern/POC'
    $Evidence = if ($PredictionsOverride) { $PredictionsOverride } elseif ($Predictions) { Resolve-ProjectPath $Predictions } else { '' }
    if (-not $Evidence) {
        throw '-Predictions est obligatoire pour Govern.'
    }
    $EffectiveRun = Get-EffectiveRunDirectory
    $ProspectiveConfig = (Get-ConfigScalar $ResolvedConfig 'data' 'allow_unresolved_final_evaluation_day')
    if ($ProspectiveConfig -and $ProspectiveConfig.ToLowerInvariant() -eq 'true') {
        if (-not $ShadowEpochDirectory) {
            throw '-ShadowEpochDirectory est obligatoire pour gouverner un candidat prospectif two-phase.'
        }
        Invoke-EpochVerify -Finalize
    }
    $Manifest = Join-Path $EffectiveRun 'experiment_manifest.json'
    $Checkpoint = Join-Path $EffectiveRun 'checkpoint'
    $Schema = Join-Path $EffectiveRun 'schema.json'
    $Arguments = @(
        $GovernanceRunner,
        'evaluate',
        '--zone', $Zones[0],
        '--rolling-predictions', $Evidence,
        '--experiment-manifest', $Manifest,
        '--policy', $ResolvedPolicy,
        '--output-root', $ResolvedOutputRoot,
        '--artifact', ('checkpoint=' + $Checkpoint),
        '--artifact', ('schema=' + $Schema)
    )
    if ($ProspectiveConfig -and $ProspectiveConfig.ToLowerInvariant() -eq 'true') {
        $Arguments += @('--shadow-epoch-directory', (Get-ShadowEpochOutput))
    }
    $CalibrationOutput = Get-ResidualCalibrationOutput
    $Corrector = Join-Path $CalibrationOutput 'residual_corrector.json'
    $OofAudit = Join-Path $CalibrationOutput 'oof_predictions_365.csv.gz.audit.json'
    if ((Test-Path -LiteralPath $Corrector -PathType Leaf) -and (Test-Path -LiteralPath $OofAudit -PathType Leaf)) {
        $Arguments += @(
            '--artifact', ('residual_corrector=' + $Corrector),
            '--artifact', ('oof_audit=' + $OofAudit)
        )
    }
    if ($EndDay) { $Arguments += @('--rolling-end-day', $EndDay) }
    if ($ShadowPredictions) {
        $ShadowEvidence = Resolve-ProjectPath $ShadowPredictions
        $Arguments += @(
            '--shadow-predictions', $ShadowEvidence,
            '--shadow-manifest', (Get-ShadowManifest $ShadowEvidence)
        )
    }
    Invoke-CheckedPython $Arguments
}

function Invoke-Verify {
    $EffectiveRun = Get-EffectiveRunDirectory
    if (Test-Path -LiteralPath (Join-Path $EffectiveRun 'bundle_manifest.json') -PathType Leaf) {
        Invoke-CheckedPython @($GovernanceRunner, 'verify', $EffectiveRun)
    } else {
        Invoke-CheckedPython @($FineTuneRunner, 'inspect', '--run-dir', $EffectiveRun)
    }
}

if (-not $PSBoundParameters.ContainsKey('Config')) {
    $DefaultConfigs = @{
        full = 'config\chronos2_exogenous_lora_poc.yaml'
        residual_only = 'config\chronos2_exogenous_lora_residual_only_poc.yaml'
        residual_weather = 'config\chronos2_exogenous_lora_residual_weather_poc.yaml'
        residual_fuel = 'config\chronos2_exogenous_lora_residual_fuel_poc.yaml'
        residual_flowbased = 'config\chronos2_exogenous_lora_residual_flowbased_poc.yaml'
    }
    $Config = $DefaultConfigs[$Pack]
}
$ResolvedConfig = Resolve-ProjectPath $Config
$ResolvedPolicy = Resolve-ProjectPath $Policy
$ResolvedOutputRoot = Resolve-ProjectPath $OutputRoot
if (-not $DryRun -and -not (Test-Path -LiteralPath $ResolvedConfig -PathType Leaf)) {
    throw "Configuration introuvable: $ResolvedConfig"
}
if ($LiveSourceManifest -and $Action -ne 'Shadow') {
    throw '-LiveSourceManifest est reserve a -Action Shadow.'
}
if ($ResolvedEvaluationPanel -and $Action -notin @('Backtest', 'ResolveHoldout')) {
    throw '-ResolvedEvaluationPanel est reserve aux actions Backtest et ResolveHoldout.'
}
if ($ProspectiveFreeze -and $Action -notin @('Panel', 'CalibrationPanel')) {
    throw '-ProspectiveFreeze est reserve aux actions Panel et CalibrationPanel.'
}
if ($FirstShadowDay -and $Action -notin @('EpochPlan', 'EpochFreeze')) {
    throw '-FirstShadowDay est reserve aux actions EpochPlan et EpochFreeze.'
}
if ($ResearchCorrectorDirectory -and $Action -notin @('ResearchFitValidationCorrector', 'ResearchEvaluateValidationCorrector')) {
    throw '-ResearchCorrectorDirectory est reserve aux actions ResearchFitValidationCorrector et ResearchEvaluateValidationCorrector.'
}
if ($Overwrite -and $Action -in @('ResearchFitValidationCorrector', 'ResearchEvaluateValidationCorrector')) {
    throw '-Overwrite est interdit pour les preuves research scellees; utilisez un nouveau -ResearchCorrectorDirectory.'
}
if ($Action -in @('CompareCandidates', 'Govern', 'POC') -and -not $DryRun -and -not (Test-Path -LiteralPath $ResolvedPolicy -PathType Leaf)) {
    throw "Politique de promotion introuvable: $ResolvedPolicy"
}

Push-Location $ProjectRoot
try {
    switch ($Action) {
        'Panel' { Invoke-Panel }
        'CalibrationPanel' { Invoke-Panel -ForResidualCalibration }
        'ResolveHoldout' { Invoke-ResolveHoldout }
        'Validate' { Invoke-Validate }
        'Train' { Invoke-Train }
        'PrepareZones' { Invoke-PrepareZones }
        'RecoverTrainingSnapshot' { Invoke-RecoverTrainingSnapshot }
        'Backtest' { Invoke-Backtest }
        'CalibrateResidual' { Invoke-CalibrateResidual }
        'ResearchFitValidationCorrector' { Invoke-ResearchFitValidationCorrector }
        'ResearchEvaluateValidationCorrector' { Invoke-ResearchEvaluateValidationCorrector }
        'FinalBacktest' { Invoke-FinalBacktest }
        'CompareCandidates' { Invoke-CompareCandidates }
        'EpochEarliest' { Invoke-EpochEarliest }
        'EpochPlan' { Invoke-EpochPlan }
        'EpochFreeze' { Invoke-EpochPlan -Freeze }
        'EpochVerify' { Invoke-EpochVerify }
        'EpochFinalizeCheck' { Invoke-EpochVerify -Finalize }
        'Shadow' { Invoke-Shadow }
        'Govern' { Invoke-Govern }
        'Verify' { Invoke-Verify }
        'POC' {
            Assert-SingleZone 'Action POC'
            Invoke-Panel -ReuseExisting
            Invoke-Validate
            Invoke-Train
            Invoke-Backtest
            $PocEvidence = if ($Predictions) {
                Resolve-ProjectPath $Predictions
            } else {
                Join-Path (Get-EffectiveRunDirectory) 'evaluation_predictions.csv.gz'
            }
            Invoke-Govern $PocEvidence
        }
    }
}
finally {
    Pop-Location
}
