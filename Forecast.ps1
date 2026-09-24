<#
.SYNOPSIS
Point d'entree unique pour l'application et les forecasts multi-pays.

.DESCRIPTION
Actions disponibles :
  App        ouvre l'application Streamlit.
  Run        produit les forecasts selectionnes.
  RegimeChallenger lance le challenger de regime sans remplacer le live.
  Backfill   reconstruit les jours manquants des Statistics.
  Experiment audite ou lance un test isole de nouvelles series PIT.
  Topology   audite, prepare ou applique la couche PriceFM/topologie isolee.
  ResidualCompare reconstruit le benchmark historique Saturn vs Chronos-2.

Avec -Action Run, Both exporte autonomous et kalman pour chaque pays : la
chaine autonome incumbent (ou LoRA promu et active), puis le meme upstream
suivi du Kalman standard. Both ne produit aucun blend MKOnline.
Both et All exigent -ResidualLoadSource Saturn pour cette action.

Avec -Action Run -Mode Both -WithNuclear, publier nuclear_autonomous et
nuclear_kalman dans runs/exports, au meme format comparatif Storm.
-NuclearStage Sync collecte les sources, Audit les controle, Prepare valide
les entrees, Run calcule les modeles, Report actualise seulement les rapports.
Run synchronise les sources nucleaires manquantes. Run et Report actualisent
les observations et Storm; -SkipObservedSync utilise explicitement le cache local.
Sans -WithNuclear le parcours habituel est inchange.
Voir NUCLEAR_FORECAST.md pour les limites de comparaison et les checkpoints.

Avec -Action Run -Mode Complete, lancer les quatre variantes pour chaque pays :
autonomous, kalman, nuclear_autonomous, nuclear_kalman. Une seule date de
livraison est fixee pour tout le batch; les pipelines se suivent sans concurrence.
Complete utilise le Both habituel, pas le mode All qui exige LoRA actif.
Ne pas ajouter -WithNuclear. -StopOnError interrompt le batch au premier echec.

Le mode All est reserve a l'action Run. Pour chaque pays, il exige un bundle
LoRA final promu et exporte exactement deux chaines : Chronos-2 + LoRA +
correcteur residuel sous autonomous, puis le meme upstream + Kalman standard
sous kalman. Il n'utilise jamais l'incumbent comme fallback silencieux. Le
blend MKOnline reste disponible avec Blend, uniquement pour FR et NL.
Pour RegimeChallenger et Topology, Both conserve sa semantique propre
autonome/blend, sans ajout du Kalman. Les anciens challengers
kalman_weather et kalman_hybrid restent disponibles dans le laboratoire.

DryRun affiche la commande exacte (argv) sans lancer Python.

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action App

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE -Mode Autonomous

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,NL -Mode Both -DeliveryDay 2026-08-21

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode All

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL -Mode Complete

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode All -KalmanConfig config\kalman_operational.yaml

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR -Mode All -LoraActivationConfig config\chronos2_exogenous_activation_v1.yaml

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action RegimeChallenger -Countries FR,DE,BE,NL,ES -Mode Both

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Run -Countries FR,DE,BE,NL,ES -Mode Autonomous -ResidualLoadSource Chronos2

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Backfill -Countries FR,DE,BE,NL,ES

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Experiment -Countries FR -ExperimentConfig config\experiment.yaml -RunExperiment

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -Countries FR,DE,BE,NL,ES

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Prepare

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Apply -Countries FR,DE,BE,NL,ES -Mode Production

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Apply -Countries FR,DE,BE,NL,ES -Mode Both -DeliveryDay 2026-08-22

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Report365 -Countries FR,DE,BE,NL,ES

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Rolling365 -Countries FR,DE,BE,NL,ES -Workers 5

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage Plan -Countries FR,DE,BE,NL,ES

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action ResidualCompare -ResidualComparisonStage All -Countries FR,DE,BE,NL,ES -Device cuda
#>
[CmdletBinding()]
param(
    [ValidateSet('App', 'Run', 'RegimeChallenger', 'Backfill', 'Experiment', 'Topology', 'ResidualCompare')]
    [string]$Action = 'App',

    [ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]
    [string[]]$Countries = @('FR', 'DE', 'BE', 'NL', 'ES'),

    [ValidateSet('Production', 'Autonomous', 'Blend', 'Both', 'All', 'Complete')]
    [string]$Mode = 'Production',

    [ValidateSet('Saturn', 'Chronos2')]
    [string]$ResidualLoadSource = 'Saturn',

    [ValidateSet('Audit', 'Calibrate', 'Prepare', 'Apply', 'Report365', 'Rolling365')]
    [string]$TopologyStage = 'Audit',

    [ValidateSet('Plan', 'SyncObserved', 'ResidualReplay', 'PriceReplay', 'Downstream', 'Blend', 'Report', 'All')]
    [string]$ResidualComparisonStage = 'Plan',

    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [ValidateRange(1, 128)]
    [int]$Threads = 4,

    [ValidateRange(1, 128)]
    [int]$Workers = 4,

    [string]$ExperimentConfig = 'config\pricefm_topology_experiment.yaml',
    [string]$RegimeChallengerConfig = 'config\price_regime_challenger.yaml',
    [string]$KalmanConfig = 'config\kalman_operational.yaml',
    [string]$LoraActivationConfig = 'config\chronos2_exogenous_activation_v1.yaml',
    [switch]$WithNuclear,
    [ValidateSet('Audit', 'Sync', 'Prepare', 'Run', 'Report')]
    [string]$NuclearStage = 'Run',
    [string]$NuclearConfig = 'config\nuclear_forecast.yaml',
    [switch]$RunExperiment,
    [switch]$ThenRunForecast,
    [switch]$AllowModelDownload,
    [switch]$StopOnError,
    [switch]$ReuseForecasts,
    [switch]$RunForecastsFirst,
    [switch]$OverwriteChallenger,
    [switch]$OverwriteComparison,
    [switch]$NoResume,
    [switch]$SkipObservedSync,
    [switch]$DryRun,

    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$Headless,

    [string]$PythonExecutable = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

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
if ($Mode -in @('All', 'Complete') -and $Action -ne 'Run') {
    throw "Le mode $Mode est disponible uniquement avec -Action Run."
}
if ($Action -eq 'Run' -and $Mode -in @('Both', 'All', 'Complete') -and $ResidualLoadSource -ne 'Saturn') {
    throw "Le mode $Mode exige -ResidualLoadSource Saturn pour conserver un historique Kalman causal homogene."
}
if ($Mode -eq 'Complete') {
    if ($WithNuclear -or $PSBoundParameters.ContainsKey('NuclearStage')) {
        throw 'Complete inclut deja les deux variantes nucleaires en Run; retirer -WithNuclear et -NuclearStage.'
    }
    if ($PSBoundParameters.ContainsKey('KalmanConfig') -or $PSBoundParameters.ContainsKey('LoraActivationConfig') -or $AllowModelDownload -or $SkipObservedSync) {
        throw 'Complete utilise les configurations actuelles, les poids locaux et actualise les observations; les overrides Kalman/LoRA/download/SkipObservedSync ne sont pas acceptes.'
    }
}
if ($WithNuclear) {
    if ($Action -ne 'Run' -or $Mode -ne 'Both' -or $ResidualLoadSource -ne 'Saturn') {
        throw '-WithNuclear exige -Action Run -Mode Both -ResidualLoadSource Saturn.'
    }
    if ($PSBoundParameters.ContainsKey('KalmanConfig') -or $PSBoundParameters.ContainsKey('LoraActivationConfig') -or $AllowModelDownload) {
        throw '-WithNuclear utilise ses configurations isolees et les poids locaux; les overrides Kalman/LoRA/download ne sont pas acceptes.'
    }
}
elseif ($Mode -ne 'Complete' -and ($PSBoundParameters.ContainsKey('NuclearStage') -or $PSBoundParameters.ContainsKey('NuclearConfig'))) {
    throw '-NuclearStage et -NuclearConfig exigent -WithNuclear.'
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
        if ($Mode -eq 'Complete') {
            $CompleteLauncher = Join-Path $ProjectRoot 'run_complete_forecast.py'
            if (-not (Test-Path -LiteralPath $CompleteLauncher -PathType Leaf)) {
                throw "Launcher introuvable : $CompleteLauncher"
            }
            $Arguments = @($CompleteLauncher, '--zones') + $Countries + @(
                '--device', $Device, '--threads', [string]$Threads, '--workers', [string]$Workers,
                '--nuclear-config', (Resolve-ProjectFile -Path $NuclearConfig))
            if (-not [string]::IsNullOrWhiteSpace($DeliveryDay)) {
                $Arguments += @('--delivery-day', $DeliveryDay)
            }
            if ($StopOnError) {
                $Arguments += '--stop-on-error'
            }
            break
        }
        if ($WithNuclear) {
            $NuclearLauncher = Join-Path $ProjectRoot 'run_nuclear_forecast.py'
            $Arguments = @($NuclearLauncher, '--config', (Resolve-ProjectFile -Path $NuclearConfig),
                '--stage', $NuclearStage, '--zones') + $Countries + @(
                '--device', $Device, '--threads', [string]$Threads, '--workers', [string]$Workers)
            if (-not [string]::IsNullOrWhiteSpace($DeliveryDay)) {
                $Arguments += @('--delivery-day', $DeliveryDay)
            }
            if ($SkipObservedSync) {
                $Arguments += '--skip-observed-sync'
            }
            break
        }
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
        if ($ResidualLoadSource -eq 'Chronos2') {
            $Arguments += @('--residual-load-source', 'chronos2')
        }
        if ($PSBoundParameters.ContainsKey('KalmanConfig')) {
            $ResolvedKalmanConfig = Resolve-ProjectFile -Path $KalmanConfig
            if (-not (Test-Path -LiteralPath $ResolvedKalmanConfig -PathType Leaf)) {
                throw "Configuration Kalman introuvable : $ResolvedKalmanConfig"
            }
            $Arguments += @('--kalman-config', $ResolvedKalmanConfig)
        }
        if ($PSBoundParameters.ContainsKey('LoraActivationConfig')) {
            $ResolvedLoraActivationConfig = Resolve-ProjectFile -Path $LoraActivationConfig
            if (-not (Test-Path -LiteralPath $ResolvedLoraActivationConfig -PathType Leaf)) {
                throw "Configuration d'activation LoRA introuvable : $ResolvedLoraActivationConfig"
            }
            $Arguments += @('--lora-activation-config', $ResolvedLoraActivationConfig)
        }
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

    'RegimeChallenger' {
        $Launcher = Join-Path $ProjectRoot 'run_price_regime_challenger.py'
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            throw "Challenger introuvable : $Launcher"
        }
        if ($ResidualLoadSource -ne 'Saturn') {
            throw 'RegimeChallenger exige les controles Saturn officiels; ResidualLoadSource Chronos2 est refuse.'
        }
        if ($Mode -eq 'Production') {
            throw 'RegimeChallenger exige -Mode Autonomous, Blend ou Both; Production ne fournit aucune vue comparable.'
        }
        if ($ReuseForecasts -and $RunForecastsFirst) {
            throw '-ReuseForecasts et -RunForecastsFirst sont mutuellement exclusifs.'
        }
        $ResolvedConfig = Resolve-ProjectFile -Path $RegimeChallengerConfig
        $Arguments = @(
            $Launcher,
            '--config', $ResolvedConfig,
            '--zones'
        ) + $Countries + @(
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
        if ($RunForecastsFirst) {
            $Arguments += '--run-forecasts-first'
        }
        else {
            $Arguments += '--reuse-forecasts'
        }
        if ($OverwriteChallenger) {
            $Arguments += '--overwrite'
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
        # Conserve le comportement historique de l'action Experiment lorsque
        # son fichier n'est pas precise, tout en donnant a Topology son fichier
        # dedie comme valeur par defaut du launcher unique.
        $InputExperimentConfig = $ExperimentConfig
        if (-not $PSBoundParameters.ContainsKey('ExperimentConfig')) {
            $InputExperimentConfig = 'config\experiment.yaml'
        }
        $ResolvedExperiment = Resolve-ProjectFile -Path $InputExperimentConfig
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

    'Topology' {
        $ResolvedTopologyStage = $TopologyStage
        if ($RunExperiment) {
            if (
                $PSBoundParameters.ContainsKey('TopologyStage') -and
                $TopologyStage -ne 'Calibrate'
            ) {
                throw '-RunExperiment ne peut pas etre combine avec -TopologyStage; utilisez -TopologyStage Calibrate.'
            }
            $ResolvedTopologyStage = 'Calibrate'
        }

        if (
            $ResolvedTopologyStage -in @('Audit', 'Calibrate', 'Prepare', 'Report365', 'Rolling365') -and
            $Mode -in @('Blend', 'Both')
        ) {
            throw (
                'Avant l''application quotidienne, Topology accepte uniquement ' +
                'Production ou Autonomous. Utilisez -TopologyStage Apply pour ' +
                'demander Blend ou Both.'
            )
        }
        if (
            $ResolvedTopologyStage -eq 'Apply' -and
            $Mode -eq 'Blend' -and
            @($Countries | Where-Object { $_ -notin @('FR', 'NL') }).Count -gt 0
        ) {
            throw 'Le mode Blend topologique est disponible uniquement pour FR et NL.'
        }

        $Launcher = Join-Path $ProjectRoot 'run_topology_experiment.py'
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            throw "Launcher introuvable : $Launcher"
        }
        $ResolvedExperiment = Resolve-ProjectFile -Path $ExperimentConfig
        if (-not (Test-Path -LiteralPath $ResolvedExperiment -PathType Leaf)) {
            throw "Configuration d'experience introuvable : $ResolvedExperiment"
        }

        $CalibrationDirectory = Join-Path $ProjectRoot 'runs\experiments\pricefm_topology_v1'
        $OperationalDirectory = Join-Path $ProjectRoot 'runs\experiments\pricefm_topology_operational_v1'
        $DailyOutputRoot = Join-Path $ProjectRoot 'runs\experiments\pricefm_topology_daily'
        $AnnualOutputDirectory = Join-Path $ProjectRoot 'runs\experiments\pricefm_topology_annual_365_v1'
        $Rolling365OutputDirectory = Join-Path $ProjectRoot 'runs\experiments\pricefm_topology_rolling365_v1'
        $CalibrationManifestSha256 = 'b11548b6a1d520e9163aacd3204b0bc97e8ca99fbb9d48856ea92dda82cbf405'
        $OperationalManifestSha256 = '231988e94ded3798f3edefc1cdbe7629971c9e3b56ccda42dc4b8831106c2bac'

        switch ($ResolvedTopologyStage) {
            'Audit' {
                $Arguments = @(
                    $Launcher,
                    '--config', $ResolvedExperiment,
                    '--zones'
                ) + $Countries + @('--audit-only')
            }
            'Calibrate' {
                $Arguments = @(
                    $Launcher,
                    '--config', $ResolvedExperiment,
                    '--zones'
                ) + $Countries
            }
            'Prepare' {
                $Arguments = @(
                    $Launcher,
                    '--config', $ResolvedExperiment,
                    '--prepare-operational',
                    '--calibration-dir', $CalibrationDirectory,
                    '--calibration-manifest-sha256', $CalibrationManifestSha256,
                    '--operational-dir', $OperationalDirectory,
                    '--operational-manifest-sha256', $OperationalManifestSha256
                )
            }
            'Apply' {
                $ResolvedDeliveryDay = $DeliveryDay
                if ([string]::IsNullOrWhiteSpace($ResolvedDeliveryDay)) {
                    $ResolvedDeliveryDay = (Get-Date).Date.AddDays(1).ToString('yyyy-MM-dd')
                }
                $Arguments = @(
                    $Launcher,
                    '--config', $ResolvedExperiment,
                    '--apply-daily',
                    '--operational-dir', $OperationalDirectory,
                    '--operational-manifest-sha256', $OperationalManifestSha256,
                    '--delivery-day', $ResolvedDeliveryDay,
                    '--zones'
                ) + $Countries + @(
                    '--mode', $Mode.ToLowerInvariant(),
                    '--daily-output-root', $DailyOutputRoot
                )
            }
            'Report365' {
                $Arguments = @(
                    $Launcher,
                    '--config', $ResolvedExperiment,
                    '--report-365',
                    '--calibration-dir', $CalibrationDirectory,
                    '--calibration-manifest-sha256', $CalibrationManifestSha256,
                    '--annual-output-dir', $AnnualOutputDirectory,
                    '--zones'
                ) + $Countries
            }
            'Rolling365' {
                $Arguments = @(
                    $Launcher,
                    '--config', $ResolvedExperiment,
                    '--rolling365-backtest',
                    '--calibration-dir', $CalibrationDirectory,
                    '--calibration-manifest-sha256', $CalibrationManifestSha256,
                    '--rolling365-output-dir', $Rolling365OutputDirectory,
                    '--rolling365-workers', [string]$Workers,
                    '--zones'
                ) + $Countries
            }
        }
    }

    'ResidualCompare' {
        $Launcher = Join-Path $ProjectRoot 'run_residual_load_historical_comparison.py'
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            throw "Launcher introuvable : $Launcher"
        }
        $StageMap = @{
            Plan = 'plan'
            SyncObserved = 'sync-observed'
            ResidualReplay = 'residual-replay'
            PriceReplay = 'price-replay'
            Downstream = 'downstream'
            Blend = 'blend'
            Report = 'report'
            All = 'all'
        }
        $Arguments = @(
            $Launcher,
            '--config', (Join-Path $ProjectRoot 'config\residual_load_historical_comparison.yaml'),
            '--stage', $StageMap[$ResidualComparisonStage],
            '--zones'
        ) + $Countries + @(
            '--device', $Device,
            '--threads', [string]$Threads
        )
        if ($AllowModelDownload) {
            $Arguments += '--allow-model-download'
        }
        if ($OverwriteComparison) {
            $Arguments += '--overwrite'
        }
        if ($NoResume) {
            $Arguments += '--no-resume'
        }
        if ($SkipObservedSync) {
            $Arguments += '--skip-observed-sync'
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
