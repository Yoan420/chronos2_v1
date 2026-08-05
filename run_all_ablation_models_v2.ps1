param(
    [string]$ProjectRoot = "C:\Users\BQ6757\chronos2_v1",
    [string]$PythonExe = "python",
    [string]$StartDay = "2024-01-01"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$StartedAt = Get-Date
$RunStamp = $StartedAt.ToString("yyyyMMdd_HHmmss")
$LogDirectory = Join-Path $ProjectRoot "runs\ablation_batch_logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$MainLog = Join-Path $LogDirectory "ablation_batch_$RunStamp.log"

$PowerStateCode = 'using System; using System.Runtime.InteropServices; public static class PowerState { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
Add-Type -TypeDefinition $PowerStateCode

[uint32]$ES_CONTINUOUS = 2147483648
[uint32]$ES_SYSTEM_REQUIRED = 1
[PowerState]::SetThreadExecutionState(
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

    # Les logs Python INFO/WARNING sont souvent ?crits sur stderr.
    # Ils ne doivent pas ?tre consid?r?s comme des erreurs PowerShell.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        & $PythonExe @Arguments 2>&1 |
            Tee-Object -FilePath $MainLog -Append

        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        throw "$StepName a ?chou? avec le code $ExitCode."
    }
}

function Reset-OrderSignalOutputs {
    param([string]$RunDirectory)

    Remove-Item $RunDirectory `
        -Recurse -Force -ErrorAction SilentlyContinue

    Remove-Item (
        Join-Path $ProjectRoot "data\pit\vintages\fr_order_*_fcst.parquet"
    ) -Force -ErrorAction SilentlyContinue
}

function Backup-OrderSignalVintages {
    param([string]$ModelName)

    $SourceDirectory = Join-Path $ProjectRoot "data\pit\vintages"
    $Destination = Join-Path (
        $ProjectRoot
    ) "runs\order_signals_$ModelName\pit_vintages"

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null

    Get-ChildItem `
        -Path $SourceDirectory `
        -Filter "fr_order_*_fcst.parquet" `
        -ErrorAction SilentlyContinue |
        Copy-Item -Destination $Destination -Force
}

function Assert-RequiredFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Fichier requis introuvable : $Path"
    }
}

try {
    Write-Step "Vérification de l'environnement"

    $RequiredFiles = @(
        "chronos2_inputs_extended_exogenous.yaml",
        "build_order_signal_vintages.py",
        "build_extended_exogenous_inputs.py",
        "run_chronos2_extended_exogenous.py",
        "chronos2_modular\exogenous_extensions.py",
        "build_ablation_configs.py",
        "build_order_signal_vintages_extended.py"
    )

    foreach ($RelativePath in $RequiredFiles) {
        Assert-RequiredFile (Join-Path $ProjectRoot $RelativePath)
    }

    & $PythonExe --version 2>&1 |
        Tee-Object -FilePath $MainLog -Append

    if ($LASTEXITCODE -ne 0) {
        throw "L'exécutable Python est inaccessible : $PythonExe"
    }

    Invoke-Python `
        -StepName "Création des configurations M1, M2, M3 et M4" `
        -Arguments @(".\build_ablation_configs.py")

    Reset-OrderSignalOutputs `
        -RunDirectory (
            Join-Path $ProjectRoot "runs\order_signals_m1_calendar"
        )

    Invoke-Python `
        -StepName "M1 — reconstruction des scores OOF" `
        -Arguments @(
            ".\build_order_signal_vintages_extended.py",
            "--config",
            ".\chronos2_m1_calendar.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay
        )

    Backup-OrderSignalVintages -ModelName "m1_calendar"

    Invoke-Python `
        -StepName "M1 — run Chronos-2" `
        -Arguments @(
            ".\run_chronos2_extended_exogenous.py",
            "--config",
            ".\chronos2_m1_calendar.yaml",
            "--zones",
            "FR",
            "--local-files-only"
        )

    Invoke-Python `
        -StepName "M2 — téléchargement des prix voisins" `
        -Arguments @(
            ".\build_extended_exogenous_inputs.py",
            "--config",
            ".\chronos2_m2_neighbour_prices.yaml",
            "--zone",
            "FR"
        )

    Reset-OrderSignalOutputs `
        -RunDirectory (
            Join-Path $ProjectRoot "runs\order_signals_m2_neighbour_prices"
        )

    Invoke-Python `
        -StepName "M2 — reconstruction des scores OOF" `
        -Arguments @(
            ".\build_order_signal_vintages_extended.py",
            "--config",
            ".\chronos2_m2_neighbour_prices.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay
        )

    Backup-OrderSignalVintages -ModelName "m2_neighbour_prices"

    Invoke-Python `
        -StepName "M2 — run Chronos-2" `
        -Arguments @(
            ".\run_chronos2_extended_exogenous.py",
            "--config",
            ".\chronos2_m2_neighbour_prices.yaml",
            "--zones",
            "FR",
            "--local-files-only"
        )

    Invoke-Python `
        -StepName "M3 — préparation locale des prix voisins" `
        -Arguments @(
            ".\build_extended_exogenous_inputs.py",
            "--config",
            ".\chronos2_m3_neighbour_spreads.yaml",
            "--zone",
            "FR",
            "--local-only"
        )

    Reset-OrderSignalOutputs `
        -RunDirectory (
            Join-Path $ProjectRoot "runs\order_signals_m3_neighbour_spreads"
        )

    Invoke-Python `
        -StepName "M3 — reconstruction des scores OOF" `
        -Arguments @(
            ".\build_order_signal_vintages_extended.py",
            "--config",
            ".\chronos2_m3_neighbour_spreads.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay
        )

    Backup-OrderSignalVintages -ModelName "m3_neighbour_spreads"

    Invoke-Python `
        -StepName "M3 — run Chronos-2" `
        -Arguments @(
            ".\run_chronos2_extended_exogenous.py",
            "--config",
            ".\chronos2_m3_neighbour_spreads.yaml",
            "--zones",
            "FR",
            "--local-files-only"
        )

    Invoke-Python `
        -StepName "M4 — construction des incertitudes PIT" `
        -Arguments @(
            ".\build_extended_exogenous_inputs.py",
            "--config",
            ".\chronos2_m4_forecast_uncertainty.yaml",
            "--zone",
            "FR",
            "--local-only"
        )

    Reset-OrderSignalOutputs `
        -RunDirectory (
            Join-Path $ProjectRoot "runs\order_signals_m4_forecast_uncertainty"
        )

    Invoke-Python `
        -StepName "M4 — reconstruction des scores OOF" `
        -Arguments @(
            ".\build_order_signal_vintages_extended.py",
            "--config",
            ".\chronos2_m4_forecast_uncertainty.yaml",
            "--mode",
            "backfill",
            "--zone",
            "FR",
            "--start-day",
            $StartDay
        )

    Backup-OrderSignalVintages -ModelName "m4_forecast_uncertainty"

    Invoke-Python `
        -StepName "M4 — run Chronos-2" `
        -Arguments @(
            ".\run_chronos2_extended_exogenous.py",
            "--config",
            ".\chronos2_m4_forecast_uncertainty.yaml",
            "--zones",
            "FR",
            "--local-files-only"
        )

    $FinishedAt = Get-Date
    $Duration = New-TimeSpan -Start $StartedAt -End $FinishedAt

    Write-Step (
        "TOUS LES RUNS SONT TERMINÉS — durée : " +
        $Duration.ToString()
    )

    @(
        "",
        "Résultats :",
        "  M1 : runs\ablation_m1_calendar",
        "  M2 : runs\ablation_m2_neighbour_prices",
        "  M3 : runs\ablation_m3_neighbour_spreads",
        "  M4 : runs\ablation_m4_forecast_uncertainty",
        "",
        "Log principal : $MainLog"
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
    [PowerState]::SetThreadExecutionState(
        $ES_CONTINUOUS
    ) | Out-Null
}
