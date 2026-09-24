<#
.SYNOPSIS
Point d'entree unique pour l'application et les forecasts multi-pays.

.DESCRIPTION
Actions disponibles :
  App        ouvre l'application Streamlit.
  Run        produit les forecasts selectionnes.
  Backfill   reconstruit les jours manquants des Statistics.
  Experiment audite ou lance un test isole de nouvelles series PIT.

DryRun affiche la commande exacte (argv) sans lancer Python.

.EXAMPLE
& '.\Forecast.ps1' -Action App

.EXAMPLE
& '.\Forecast.ps1' -Action Run -Countries FR,DE,BE -Mode Autonomous

.EXAMPLE
& '.\Forecast.ps1' -Action Run -Countries FR,NL -Mode Both -DeliveryDay 2026-08-21

.EXAMPLE
& '.\Forecast.ps1' -Action Backfill -Countries FR,DE,BE,NL,ES

.EXAMPLE
& '.\Forecast.ps1' -Action Experiment -Countries FR -ExperimentConfig config\experiment.yaml -RunExperiment
#>
[CmdletBinding()]
param(
    [ValidateSet('App', 'Run', 'Backfill', 'Experiment')]
    [string]$Action = 'App',

    [ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]
    [string[]]$Countries = @('FR', 'DE', 'BE', 'NL', 'ES'),

    [ValidateSet('Production', 'Autonomous', 'Blend', 'Both')]
    [string]$Mode = 'Production',

    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [ValidateRange(1, 128)]
    [int]$Threads = 4,

    [ValidateRange(1, 128)]
    [int]$Workers = 4,

    [string]$ExperimentConfig = 'config\experiment.yaml',
    [switch]$RunExperiment,
    [switch]$ThenRunForecast,
    [switch]$AllowModelDownload,
    [switch]$StopOnError,
    [switch]$DryRun,

    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$Headless,

    [string]$PythonExecutable = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
}

$BundledHuggingFaceCache = Join-Path $ProjectRoot 'huggingface_cache'
if (
    [string]::IsNullOrWhiteSpace($env:HF_HUB_CACHE) -and
    (Test-Path -LiteralPath $BundledHuggingFaceCache -PathType Container)
) {
    $env:HF_HUB_CACHE = $BundledHuggingFaceCache
}

if ([string]::IsNullOrWhiteSpace($env:SATURN_AUTHOR)) {
    $env:SATURN_AUTHOR = $env:USERNAME
}

function Test-IsLoopbackBlackholeProxy {
    param(
        [AllowNull()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    # Port 9 on loopback is the deliberate no-network proxy used by isolated
    # Codex sessions.  The narrow match never removes a real corporate proxy.
    return $Value.Trim() -match '^(?:(?:https?|socks5?)://)?(?:127\.0\.0\.1|localhost|\[::1\]):9/?$'
}

function Enter-SafeNetworkEnvironment {
    $BlackholeNames = @(
        foreach ($Name in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')) {
            $Value = [System.Environment]::GetEnvironmentVariable(
                $Name,
                [System.EnvironmentVariableTarget]::Process
            )
            if (Test-IsLoopbackBlackholeProxy -Value $Value) {
                $Name
            }
        }
    )

    if ($BlackholeNames.Count -eq 0) {
        return
    }

    $Names = ($BlackholeNames | Sort-Object -Unique) -join ', '
    $IsCodexSandbox = (
        $env:CODEX_SANDBOX_NETWORK_DISABLED -eq '1' -or
        -not [string]::IsNullOrWhiteSpace($env:CODEX_THREAD_ID)
    )
    if ($IsCodexSandbox) {
        throw (
            "Cette console appartient encore au sandbox Codex " +
            "(proxy 127.0.0.1:9 dans $Names). Lancez Forecast.ps1 depuis " +
            "une fenetre PowerShell Windows normale."
        )
    }

    $SavedValues = [System.Collections.Generic.List[object]]::new()
    foreach ($Name in $BlackholeNames) {
        $UserValue = [System.Environment]::GetEnvironmentVariable($Name, 'User')
        $MachineValue = [System.Environment]::GetEnvironmentVariable($Name, 'Machine')
        if (
            (Test-IsLoopbackBlackholeProxy -Value $UserValue) -or
            (Test-IsLoopbackBlackholeProxy -Value $MachineValue)
        ) {
            throw (
                "Le proxy 127.0.0.1:9 est configure durablement dans Windows " +
                "pour $Name. Supprimez cette variable utilisateur/machine avant " +
                "de relancer le programme."
            )
        }

        $ProcessValue = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
        $SavedValues.Add([pscustomobject]@{ Name = $Name; Value = $ProcessValue })
        [System.Environment]::SetEnvironmentVariable($Name, $null, 'Process')
    }

    Write-Warning (
        "Ancien proxy local 127.0.0.1:9 retire de ce processus Windows " +
        "($Names). Les configurations utilisateur/machine restent intactes."
    )
    return $SavedValues
}

function Exit-SafeNetworkEnvironment {
    param([object[]]$SavedValues)

    foreach ($Saved in $SavedValues) {
        [System.Environment]::SetEnvironmentVariable(
            $Saved.Name,
            $Saved.Value,
            [System.EnvironmentVariableTarget]::Process
        )
    }
}

function Resolve-ProjectFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python introuvable : $PythonExecutable"
}
if ($Action -ne 'App' -and $Countries.Count -eq 0) {
    throw 'Selectionnez au moins un pays.'
}

$Arguments = @()
switch ($Action) {
    'App' {
        $AppPath = Join-Path $ProjectRoot 'app_multizone.py'
        if (-not (Test-Path -LiteralPath $AppPath -PathType Leaf)) {
            throw "Application introuvable : $AppPath"
        }
        $Arguments = @(
            '-m', 'streamlit', 'run', $AppPath,
            '--server.port', [string]$Port,
            '--browser.gatherUsageStats', 'false'
        )
        if ($Headless) {
            $Arguments += @('--server.headless', 'true')
        }
    }

    'Run' {
        $Launcher = Join-Path $ProjectRoot 'run_multicountry_forecast.py'
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            throw "Launcher introuvable : $Launcher"
        }
        $Arguments = @($Launcher, '--zones') + $Countries + @(
            '--mode', $Mode,
            '--device', $Device,
            '--threads', [string]$Threads,
            '--workers', [string]$Workers
        )
        if (-not [string]::IsNullOrWhiteSpace($DeliveryDay)) {
            $Arguments += @('--delivery-day', $DeliveryDay)
        }
        if ($AllowModelDownload) {
            $Arguments += '--allow-model-download'
        }
        if ($StopOnError) {
            $Arguments += '--stop-on-error'
        }
    }

    'Backfill' {
        if ($Mode -ne 'Production') {
            throw 'Backfill utilise les configurations Production; retirez -Mode ou choisissez Production.'
        }
        if (-not [string]::IsNullOrWhiteSpace($DeliveryDay) -and -not $ThenRunForecast) {
            throw '-DeliveryDay exige aussi -ThenRunForecast pour Backfill.'
        }
        $Launcher = Join-Path $ProjectRoot 'run_statistics_backfill.py'
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            throw "Launcher introuvable : $Launcher"
        }
        $Arguments = @($Launcher, '--zones') + $Countries + @(
            '--device', $Device,
            '--threads', [string]$Threads,
            '--workers', [string]$Workers
        )
        if ($AllowModelDownload) {
            $Arguments += '--allow-model-download'
        }
        if ($StopOnError) {
            $Arguments += '--stop-on-error'
        }
        if ($ThenRunForecast) {
            $Arguments += '--then-run-live'
        }
        if (-not [string]::IsNullOrWhiteSpace($DeliveryDay)) {
            $Arguments += @('--live-delivery-day', $DeliveryDay)
        }
    }

    'Experiment' {
        if ($Mode -in @('Blend', 'Both')) {
            throw 'Les experiences de series sont isolees et autonomes; Blend et Both ne sont pas disponibles.'
        }
        $Launcher = Join-Path $ProjectRoot 'run_input_experiment.py'
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            throw "Launcher introuvable : $Launcher"
        }
        $ResolvedExperiment = Resolve-ProjectFile -Path $ExperimentConfig
        if (-not (Test-Path -LiteralPath $ResolvedExperiment -PathType Leaf)) {
            throw "Configuration d'experience introuvable : $ResolvedExperiment"
        }
        $Arguments = @(
            $Launcher,
            '--experiment', $ResolvedExperiment,
            '--zones'
        ) + $Countries + @('--device', $Device)
        if ($RunExperiment) {
            $Arguments += '--run'
        }
        if ($AllowModelDownload) {
            $Arguments += '--allow-model-download'
        }
    }
}

$DisplayCommand = @($PythonExecutable) + $Arguments
$DisplayJson = ConvertTo-Json -InputObject $DisplayCommand -Compress
Write-Host "Commande (argv, shell=False): $DisplayJson"
if ($DryRun) {
    Write-Host 'DryRun : aucune commande executee.'
    exit 0
}

$SavedProxyValues = @()
Push-Location $ProjectRoot
try {
    $SavedProxyValues = @(Enter-SafeNetworkEnvironment)
    & $PythonExecutable @Arguments
    $ProcessExitCode = $LASTEXITCODE
    if ($null -eq $ProcessExitCode) {
        $ProcessExitCode = 1
    }
    exit ([int]$ProcessExitCode)
}
finally {
    Exit-SafeNetworkEnvironment -SavedValues $SavedProxyValues
    Pop-Location
}
