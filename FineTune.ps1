[CmdletBinding()]
param(
    [ValidateSet('List', 'Validate', 'Train', 'Retrain', 'Evaluate', 'Predict', 'Compare', 'Weather')]
    [string]$Action = 'List',
    [string]$Config = 'config\auxiliary_lab.yaml',
    [string]$Models = '',
    [string]$RunDirectory = '',
    [string[]]$Runs = @(),
    [string]$Artifact = '',
    [string]$SourceRun = '',
    [string]$Output = '',
    [string]$StartDay = '',
    [string]$EndDay = '',
    [string[]]$Countries = @('FR', 'DE', 'BE', 'NL', 'ES'),
    [ValidateRange(1, 32)]
    [int]$SeriesWorkers = 2,
    [ValidateRange(1, 32)]
    [int]$DayWorkers = 4,
    [switch]$DryRun,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { 'python' }
$Runner = Join-Path $ProjectRoot 'run_auxiliary_lab.py'
$SaturnWeatherRunner = Join-Path $ProjectRoot 'materialize_saturn_kalman_weather.py'

$Arguments = @($Runner)
switch ($Action) {
    'List' { $Arguments += 'list-models' }
    'Validate' { $Arguments += @('validate', '--config', $Config) }
    'Train' {
        $Arguments += @('train', '--config', $Config)
        if ($Models) { $Arguments += @('--models', $Models) }
        if ($Overwrite) { $Arguments += '--overwrite' }
    }
    'Retrain' {
        $Arguments += @('retrain', '--config', $Config, '--overwrite')
        if ($Models) { $Arguments += @('--models', $Models) }
    }
    'Evaluate' {
        if (-not $RunDirectory) { throw '-RunDirectory est obligatoire pour Evaluate.' }
        $Arguments += @('evaluate', '--run-dir', $RunDirectory)
        if ($Output) { $Arguments += @('--output-dir', $Output) }
    }
    'Predict' {
        if (-not $Artifact -or -not $SourceRun -or -not $Output) {
            throw '-Artifact, -SourceRun et -Output sont obligatoires pour Predict.'
        }
        $Arguments += @('predict', '--artifact', $Artifact, '--source-run', $SourceRun, '--output', $Output)
    }
    'Compare' {
        if ($Runs.Count -lt 2 -or -not $Output) {
            throw '-Runs (au moins deux) et -Output sont obligatoires pour Compare.'
        }
        $Arguments += @('compare', '--runs')
        $Arguments += $Runs
        $Arguments += @('--output-dir', $Output)
    }
    'Weather' {
        if (-not $StartDay -or -not $EndDay) {
            throw '-StartDay et -EndDay sont obligatoires pour Weather.'
        }
        if ($Countries.Count -eq 0) {
            throw '-Countries doit contenir au moins une zone pour Weather.'
        }
        if (($SeriesWorkers * $DayWorkers) -gt 32) {
            throw 'SeriesWorkers * DayWorkers doit etre inferieur ou egal a 32.'
        }
        $Arguments = @(
            $SaturnWeatherRunner,
            '--start-day', $StartDay,
            '--end-day', $EndDay,
            '--zones'
        )
        $Arguments += $Countries
        $Arguments += @(
            '--series-workers', [string]$SeriesWorkers,
            '--day-workers', [string]$DayWorkers
        )
        if ($Output) { $Arguments += @('--output-dir', $Output) }
        if ($DryRun) { $Arguments += '--dry-run' }
    }
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Le laboratoire a retourne le code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
