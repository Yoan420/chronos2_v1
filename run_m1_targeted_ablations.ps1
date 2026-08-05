param(
    [string]$ProjectRoot = "C:\Users\BQ6757\chronos2_v1",
    [string]$PythonExe = "python",
    [string]$StartDay = "2024-01-01",
    [string]$InterconnectionMap = "interconnection_series.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$StartedAt = Get-Date
$RunStamp = $StartedAt.ToString("yyyyMMdd_HHmmss")
$LogDirectory = Join-Path $ProjectRoot "runs\targeted_ablation_logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$MainLog = Join-Path $LogDirectory "m1_targeted_$RunStamp.log"

$PowerStateCode = 'using System; using System.Runtime.InteropServices; public static class PowerStateTargeted { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
Add-Type -TypeDefinition $PowerStateCode
[uint32]$ES_CONTINUOUS = 2147483648
[uint32]$ES_SYSTEM_REQUIRED = 1
[PowerStateTargeted]::SetThreadExecutionState(
    $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
) | Out-Null

function Write-Step {
    param([string]$Message)
    $Line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host ""
    Write-Host ("=" * 90)
    Write-Host $Line
    Write-Host ("=" * 90)
    Add-Content -Path $MainLog -Value $Line
}

function Invoke-Python {
    param(
        [string]$StepName,
        [string[]]$Arguments
    )

    Write-Step $StepName
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        & $PythonExe @Arguments 2>&1 |
            Tee-Object -FilePath $MainLog -Append
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }

    if ($ExitCode -ne 0) {
        throw "$StepName a échoué avec le code $ExitCode."
    }
}

function Reset-Signals {
    param([string]$OutputDirectory)

    Remove-Item $OutputDirectory `
        -Recurse -Force -ErrorAction SilentlyContinue

    Remove-Item (
        Join-Path $ProjectRoot "data\pit\vintages\fr_order_*_fcst.parquet"
    ) -Force -ErrorAction SilentlyContinue
}

function Backup-Signals {
    param([string]$Name)

    $Source = Join-Path $ProjectRoot "data\pit\vintages"
    $Destination = Join-Path (
        $ProjectRoot
    ) "runs\order_signals_$Name\pit_vintages"

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Get-ChildItem `
        -Path $Source `
        -Filter "fr_order_*_fcst.parquet" `
        -ErrorAction SilentlyContinue |
        Copy-Item -Destination $Destination -Force
}

try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Write-Step "Création des deux configurations ciblées"

    Invoke-Python `
        -StepName "Génération des YAML" `
        -Arguments @(
            ".\build_m1_targeted_configs.py",
            "--base-config",
            ".\chronos2_m1_calendar.yaml",
            "--interconnection-map",
            $InterconnectionMap
        )

    # --------------------------------------------------------------
    # M1 + incertitude uniquement
    # --------------------------------------------------------------
    Invoke-Python `
        -StepName "Construction des métriques d'incertitude PIT" `
        -Arguments @(
            ".\build_extended_exogenous_inputs.py",
            "--config",
            ".\chronos2_m1_uncertainty_only.yaml",
            "--zone",
            "FR",
            "--local-only"
        )

    Reset-Signals `
        -OutputDirectory (
            Join-Path $ProjectRoot "runs\order_signals_m1_uncertainty_only"
        )

    Invoke-Python `
        -StepName "M1 + incertitude — scores OOF" `
        -Arguments @(
            ".\build_order_signal_vintages_targeted.py",
            "--config",
            ".\chronos2_m1_uncertainty_only.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay
        )

    Backup-Signals -Name "m1_uncertainty_only"

    Invoke-Python `
        -StepName "M1 + incertitude — Chronos-2" `
        -Arguments @(
            ".\run_chronos2_targeted_ablation.py",
            "--config",
            ".\chronos2_m1_uncertainty_only.yaml",
            "--zones",
            "FR",
            "--local-files-only"
        )

    # --------------------------------------------------------------
    # M1 + capacités d'interconnexion uniquement
    # --------------------------------------------------------------
    Reset-Signals `
        -OutputDirectory (
            Join-Path $ProjectRoot (
                "runs\order_signals_m1_interconnection_only"
            )
        )

    Invoke-Python `
        -StepName "M1 + interconnexions — chargement Saturn et scores OOF" `
        -Arguments @(
            ".\build_order_signal_vintages_targeted.py",
            "--config",
            ".\chronos2_m1_interconnection_only.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay,
            "--refresh-data"
        )

    Backup-Signals -Name "m1_interconnection_only"

    Invoke-Python `
        -StepName "M1 + interconnexions — Chronos-2" `
        -Arguments @(
            ".\run_chronos2_targeted_ablation.py",
            "--config",
            ".\chronos2_m1_interconnection_only.yaml",
            "--zones",
            "FR",
            "--local-files-only",
            "--refresh-data"
        )

    $Duration = New-TimeSpan -Start $StartedAt -End (Get-Date)
    Write-Step (
        "RUNS TERMINÉS — durée : " + $Duration.ToString()
    )

    @(
        "",
        "Résultats :",
        "  Incertitude : runs\ablation_m1_uncertainty_only",
        "  Interconnexions : runs\ablation_m1_interconnection_only",
        "",
        "Log : $MainLog"
    ) | Tee-Object -FilePath $MainLog -Append
}
catch {
    $Message = (
        "`n" +
        ("=" * 80) + "`n" +
        "ÉCHEC DU PIPELINE`n" +
        "Date   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" +
        "Erreur : $($_.Exception.Message)`n" +
        "Log    : $MainLog`n" +
        ("=" * 80)
    )
    $Message | Tee-Object -FilePath $MainLog -Append
    exit 1
}
finally {
    [PowerStateTargeted]::SetThreadExecutionState(
        $ES_CONTINUOUS
    ) | Out-Null
}
