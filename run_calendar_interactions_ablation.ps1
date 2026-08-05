param(
    [string]$ProjectRoot = "C:\Users\BQ6757\chronos2_v1",
    [string]$PythonExe = "python",
    [string]$StartDay = "2024-01-01",
    [string]$BaseConfig = "chronos2_m1_calendar.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDirectory = Join-Path $ProjectRoot "runs\interaction_logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path `
    $LogDirectory `
    "m1_calendar_interactions_$RunStamp.log"

$OutputDirectory = Join-Path `
    $ProjectRoot `
    "runs\ablation_m1_calendar_interactions"

$SignalDirectory = Join-Path `
    $ProjectRoot `
    "runs\order_signals_m1_calendar_interactions"

$PitDirectory = Join-Path $ProjectRoot "data\pit\vintages"

$PowerStateCode = 'using System; using System.Runtime.InteropServices; public static class InteractionPowerState { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
Add-Type -TypeDefinition $PowerStateCode
[uint32]$ES_CONTINUOUS = 2147483648
[uint32]$ES_SYSTEM_REQUIRED = 1
[InteractionPowerState]::SetThreadExecutionState(
    $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
) | Out-Null

function Invoke-Python {
    param(
        [string]$StepName,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ("=" * 90)
    Write-Host $StepName -ForegroundColor Cyan
    Write-Host ("=" * 90)

    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        & $PythonExe @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }

    if ($ExitCode -ne 0) {
        throw "$StepName a échoué avec le code $ExitCode."
    }
}

try {
    Invoke-Python `
        -StepName "Création de la configuration M1 + interactions" `
        -Arguments @(
            ".\build_calendar_interactions_config.py",
            "--base-config",
            $BaseConfig,
            "--output",
            ".\chronos2_m1_calendar_interactions.yaml",
            "--output-dir",
            "runs/ablation_m1_calendar_interactions"
        )

    Remove-Item `
        $OutputDirectory `
        -Recurse -Force -ErrorAction SilentlyContinue

    Remove-Item `
        $SignalDirectory `
        -Recurse -Force -ErrorAction SilentlyContinue

    New-Item `
        -ItemType Directory `
        -Path $PitDirectory `
        -Force |
        Out-Null

    Remove-Item `
        (Join-Path $PitDirectory "fr_order_*_fcst.parquet") `
        -Force `
        -ErrorAction SilentlyContinue

    Invoke-Python `
        -StepName "Reconstruction des scores OOF avec interactions" `
        -Arguments @(
            ".\build_order_signal_vintages_calendar_interactions.py",
            "--config",
            ".\chronos2_m1_calendar_interactions.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay
        )

    $SignalBackup = Join-Path $SignalDirectory "pit_vintages"
    New-Item `
        -ItemType Directory `
        -Path $SignalBackup `
        -Force |
        Out-Null

    Get-ChildItem `
        -Path $PitDirectory `
        -Filter "fr_order_*_fcst.parquet" `
        -ErrorAction SilentlyContinue |
        Copy-Item `
            -Destination $SignalBackup `
            -Force

    Invoke-Python `
        -StepName "Run Chronos-2 M1 + interactions calendaires" `
        -Arguments @(
            ".\run_chronos2_calendar_interactions.py",
            "--config",
            ".\chronos2_m1_calendar_interactions.yaml",
            "--zones",
            "FR",
            "--local-files-only"
        )

    Write-Host ""
    Write-Host "RUN TERMINÉ" -ForegroundColor Green
    Write-Host (
        "Rapport : " +
        (Join-Path `
            $OutputDirectory `
            "chronos2_m1_calendar_interactions.html")
    )
    Write-Host "Log : $LogPath"
}
catch {
    Write-Host ""
    Write-Host "ÉCHEC DU RUN" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Log : $LogPath"
    exit 1
}
finally {
    [InteractionPowerState]::SetThreadExecutionState(
        $ES_CONTINUOUS
    ) | Out-Null
}
