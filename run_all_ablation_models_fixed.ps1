param(
    [string]$ProjectRoot = 'C:\Users\BQ6757\chronos2_v1',
    [string]$PythonExe = 'python',
    [string]$StartDay = '2024-01-01'
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

$StartedAt = Get-Date
$RunStamp = $StartedAt.ToString('yyyyMMdd_HHmmss')
$LogDirectory = Join-Path $ProjectRoot 'runs\ablation_batch_logs'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$MainLog = Join-Path $LogDirectory "ablation_batch_$RunStamp.log"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class PowerState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
[uint32]$ES_CONTINUOUS = 2147483648
[uint32]$ES_SYSTEM_REQUIRED = 1
[PowerState]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

function Write-Step {
    param([string]$Message)
    $Line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host ''
    Write-Host ('=' * 90)
    Write-Host $Line
    Write-Host ('=' * 90)
    Add-Content -Path $MainLog -Value $Line
}

function Invoke-Python {
    param([string]$StepName, [string[]]$Arguments)
    Write-Step $StepName
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $MainLog -Append
    $Code = $LASTEXITCODE
    if ($Code -ne 0) { throw "$StepName a échoué avec le code $Code." }
}

function Reset-Signals {
    param([string]$Directory)
    Remove-Item $Directory -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $ProjectRoot 'data\pit\vintages\fr_order_*_fcst.parquet') -Force -ErrorAction SilentlyContinue
}

function Backup-Signals {
    param([string]$Name)
    $Destination = Join-Path $ProjectRoot "runs\order_signals_$Name\pit_vintages"
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Get-ChildItem (Join-Path $ProjectRoot 'data\pit\vintages') -Filter 'fr_order_*_fcst.parquet' -ErrorAction SilentlyContinue |
        Copy-Item -Destination $Destination -Force
}

function Assert-File {
    param([string]$Path)
    if (-not (Test-Path $Path)) { throw "Fichier requis introuvable : $Path" }
}

try {
    Write-Step 'Vérification des fichiers et de Python'
    @(
        'chronos2_inputs_extended_exogenous.yaml',
        'build_ablation_configs.py',
        'build_order_signal_vintages_extended.py',
        'build_extended_exogenous_inputs.py',
        'run_chronos2_extended_exogenous.py',
        'chronos2_modular\exogenous_extensions.py'
    ) | ForEach-Object { Assert-File (Join-Path $ProjectRoot $_) }

    & $PythonExe --version 2>&1 | Tee-Object -FilePath $MainLog -Append
    if ($LASTEXITCODE -ne 0) { throw "Python inaccessible : $PythonExe" }

    Invoke-Python 'Création des configurations M1 à M4' @('.\build_ablation_configs.py')

    Reset-Signals (Join-Path $ProjectRoot 'runs\order_signals_m1_calendar')
    Invoke-Python 'M1 — reconstruction des scores OOF' @(
        '.\build_order_signal_vintages_extended.py', '--config', '.\chronos2_m1_calendar.yaml',
        '--mode', 'backfill', '--zone', 'FR', '--start-day', $StartDay
    )
    Backup-Signals 'm1_calendar'
    Invoke-Python 'M1 — run Chronos-2' @(
        '.\run_chronos2_extended_exogenous.py', '--config', '.\chronos2_m1_calendar.yaml',
        '--zones', 'FR', '--local-files-only'
    )

    Invoke-Python 'M2 — téléchargement des prix voisins' @(
        '.\build_extended_exogenous_inputs.py', '--config', '.\chronos2_m2_neighbour_prices.yaml', '--zone', 'FR'
    )
    Reset-Signals (Join-Path $ProjectRoot 'runs\order_signals_m2_neighbour_prices')
    Invoke-Python 'M2 — reconstruction des scores OOF' @(
        '.\build_order_signal_vintages_extended.py', '--config', '.\chronos2_m2_neighbour_prices.yaml',
        '--mode', 'backfill', '--zone', 'FR', '--start-day', $StartDay
    )
    Backup-Signals 'm2_neighbour_prices'
    Invoke-Python 'M2 — run Chronos-2' @(
        '.\run_chronos2_extended_exogenous.py', '--config', '.\chronos2_m2_neighbour_prices.yaml',
        '--zones', 'FR', '--local-files-only'
    )

    Invoke-Python 'M3 — préparation locale des prix voisins' @(
        '.\build_extended_exogenous_inputs.py', '--config', '.\chronos2_m3_neighbour_spreads.yaml', '--zone', 'FR', '--local-only'
    )
    Reset-Signals (Join-Path $ProjectRoot 'runs\order_signals_m3_neighbour_spreads')
    Invoke-Python 'M3 — reconstruction des scores OOF' @(
        '.\build_order_signal_vintages_extended.py', '--config', '.\chronos2_m3_neighbour_spreads.yaml',
        '--mode', 'backfill', '--zone', 'FR', '--start-day', $StartDay
    )
    Backup-Signals 'm3_neighbour_spreads'
    Invoke-Python 'M3 — run Chronos-2' @(
        '.\run_chronos2_extended_exogenous.py', '--config', '.\chronos2_m3_neighbour_spreads.yaml',
        '--zones', 'FR', '--local-files-only'
    )

    Invoke-Python 'M4 — construction des incertitudes PIT' @(
        '.\build_extended_exogenous_inputs.py', '--config', '.\chronos2_m4_forecast_uncertainty.yaml', '--zone', 'FR', '--local-only'
    )
    Reset-Signals (Join-Path $ProjectRoot 'runs\order_signals_m4_forecast_uncertainty')
    Invoke-Python 'M4 — reconstruction des scores OOF' @(
        '.\build_order_signal_vintages_extended.py', '--config', '.\chronos2_m4_forecast_uncertainty.yaml',
        '--mode', 'backfill', '--zone', 'FR', '--start-day', $StartDay
    )
    Backup-Signals 'm4_forecast_uncertainty'
    Invoke-Python 'M4 — run Chronos-2' @(
        '.\run_chronos2_extended_exogenous.py', '--config', '.\chronos2_m4_forecast_uncertainty.yaml',
        '--zones', 'FR', '--local-files-only'
    )

    $Duration = New-TimeSpan -Start $StartedAt -End (Get-Date)
    Write-Step "TOUS LES RUNS SONT TERMINÉS — durée : $Duration"
    @(
        '',
        'Résultats :',
        '  M1 : runs\ablation_m1_calendar',
        '  M2 : runs\ablation_m2_neighbour_prices',
        '  M3 : runs\ablation_m3_neighbour_spreads',
        '  M4 : runs\ablation_m4_forecast_uncertainty',
        '',
        "Log principal : $MainLog"
    ) | Tee-Object -FilePath $MainLog -Append
}
catch {
    $Message = @"

================================================================================
ÉCHEC DU PIPELINE
Date   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Erreur : $($_.Exception.Message)
Log    : $MainLog
================================================================================
"@
    $Message | Tee-Object -FilePath $MainLog -Append
    exit 1
}
finally {
    [PowerState]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
