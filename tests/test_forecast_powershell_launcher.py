from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "Forecast.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _invoke(arguments: str) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available")
    command = f"& {_quote(LAUNCHER)} {arguments}"
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _dry_run_argv(arguments: str) -> list[str]:
    completed = _invoke(arguments + " -DryRun")
    assert completed.returncode == 0, completed.stderr
    prefix = "Commande (argv, shell=False): "
    line = next(
        item for item in completed.stdout.splitlines() if item.startswith(prefix)
    )
    return json.loads(line[len(prefix) :])


def test_launcher_has_valid_powershell_syntax() -> None:
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available")
    script = (
        "$tokens=$null; $errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile({_quote(LAUNCHER)},"
        "[ref]$tokens,[ref]$errors) > $null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    completed = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr


def test_launcher_uses_argv_without_dynamic_shell_evaluation() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Invoke-Expression" not in source
    assert "cmd /c" not in source.lower()
    assert "& $PythonExecutable @Arguments" in source


def test_app_action_builds_streamlit_argv() -> None:
    argv = _dry_run_argv("-Action App -Port 9001 -Headless")
    assert argv[1:5] == [
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app_multizone.py"),
    ]
    assert argv[-4:] == [
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
    ]
    assert argv[argv.index("--server.port") + 1] == "9001"


def test_run_action_preserves_country_order_and_mode() -> None:
    argv = _dry_run_argv(
        "-Action Run -Countries FR,NL -Mode Both -DeliveryDay 2026-08-21 "
        "-Device cpu -Threads 3 -Workers 2 -AllowModelDownload -StopOnError"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_multicountry_forecast.py")
    assert argv[2:5] == ["--zones", "FR", "NL"]
    assert argv[argv.index("--mode") + 1] == "Both"
    assert argv[argv.index("--delivery-day") + 1] == "2026-08-21"
    assert argv[-2:] == ["--allow-model-download", "--stop-on-error"]


def test_run_action_maps_all_mode_without_changing_other_arguments() -> None:
    argv = _dry_run_argv(
        "-Action Run -Countries FR,DE,BE,NL,ES -Mode All "
        "-DeliveryDay 2026-08-28 -Device cpu -Threads 3 -Workers 2"
    )

    assert argv[1:] == [
        str(PROJECT_ROOT / "run_multicountry_forecast.py"),
        "--zones",
        "FR",
        "DE",
        "BE",
        "NL",
        "ES",
        "--mode",
        "All",
        "--device",
        "cpu",
        "--threads",
        "3",
        "--workers",
        "2",
        "--delivery-day",
        "2026-08-28",
    ]


@pytest.mark.parametrize("mode", ["Both", "All"])
def test_run_action_forwards_an_explicit_kalman_sidecar(mode: str) -> None:
    config = PROJECT_ROOT / "config" / "kalman_operational.yaml"
    argv = _dry_run_argv(
        f"-Action Run -Countries FR -Mode {mode} "
        f"-KalmanConfig {_quote(config)}"
    )

    assert argv[argv.index("--kalman-config") + 1] == str(config)


def test_run_action_keeps_the_default_kalman_sidecar_implicit() -> None:
    argv = _dry_run_argv("-Action Run -Countries FR -Mode All")

    assert "--kalman-config" not in argv


@pytest.mark.parametrize("mode", ["Both", "All"])
def test_run_action_forwards_an_explicit_lora_activation_contract(mode: str) -> None:
    config = PROJECT_ROOT / "config" / "chronos2_exogenous_activation_v1.yaml"
    argv = _dry_run_argv(
        f"-Action Run -Countries FR -Mode {mode} "
        f"-LoraActivationConfig {_quote(config)}"
    )

    assert argv[argv.index("--lora-activation-config") + 1] == str(config)


def test_run_action_keeps_default_lora_activation_contract_implicit() -> None:
    argv = _dry_run_argv("-Action Run -Countries FR -Mode Autonomous")

    assert "--lora-activation-config" not in argv


@pytest.mark.parametrize(
    "action",
    [
        "App",
        "RegimeChallenger",
        "Backfill",
        "Experiment",
        "Topology",
        "ResidualCompare",
    ],
)
def test_all_mode_is_restricted_to_run(action: str) -> None:
    completed = _invoke(f"-Action {action} -Mode All -DryRun")

    assert completed.returncode != 0
    assert "uniquement avec -Action Run" in completed.stdout + completed.stderr


def test_all_mode_is_documented_in_the_unified_launcher() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "-Action Run -Countries FR,DE,BE,NL,ES -Mode All" in source
    assert "Le mode All est reserve a l'action Run" in source
    assert "exporte exactement deux chaines" in source
    assert "blend MKOnline reste disponible avec Blend, uniquement pour FR et NL" in source
    assert "fallback silencieux" in source


def test_both_mode_is_documented_as_autonomous_and_standard_kalman() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Both exporte autonomous et kalman pour chaque pays" in source
    assert "chaine autonome incumbent (ou LoRA promu et active)" in source
    assert "Both ne produit aucun blend MKOnline" in source
    assert "Pour RegimeChallenger et Topology, Both conserve sa semantique propre" in source
    assert "-Mode Both -ResidualLoadSource Chronos2" not in source


@pytest.mark.parametrize("mode", ["Both", "All"])
def test_run_kalman_modes_reject_chronos2_residual_load(mode: str) -> None:
    completed = _invoke(
        f"-Action Run -Countries FR -Mode {mode} "
        "-ResidualLoadSource Chronos2 -DryRun"
    )

    assert completed.returncode != 0
    assert f"Le mode {mode} exige -ResidualLoadSource Saturn" in completed.stdout + completed.stderr


def test_run_action_keeps_default_saturn_argv_exactly_unchanged() -> None:
    arguments = (
        "-Action Run -Countries FR,DE,BE,NL,ES -Mode Both "
        "-Device cpu -Threads 3 -Workers 2"
    )

    default_argv = _dry_run_argv(arguments)
    explicit_saturn_argv = _dry_run_argv(
        arguments + " -ResidualLoadSource Saturn"
    )

    assert explicit_saturn_argv == default_argv
    assert "--residual-load-source" not in default_argv
    assert "--residual-load-bundle-manifest" not in default_argv


def test_run_action_adds_only_chronos2_residual_load_opt_in() -> None:
    default_argv = _dry_run_argv(
        "-Action Run -Countries FR,DE -Mode Autonomous -Device cpu "
        "-Threads 3 -Workers 2"
    )
    chronos2_argv = _dry_run_argv(
        "-Action Run -Countries FR,DE -Mode Autonomous -Device cpu "
        "-Threads 3 -Workers 2 -ResidualLoadSource Chronos2"
    )

    assert chronos2_argv == default_argv + [
        "--residual-load-source",
        "chronos2",
    ]


def test_regime_challenger_builds_both_multizone_argv() -> None:
    argv = _dry_run_argv(
        "-Action RegimeChallenger -Countries NL,FR,DE -Mode Both "
        "-DeliveryDay 2026-08-28 -Device cpu -Threads 3 -Workers 2"
    )

    assert argv[1:] == [
        str(PROJECT_ROOT / "run_price_regime_challenger.py"),
        "--config",
        str(PROJECT_ROOT / "config" / "price_regime_challenger.yaml"),
        "--zones",
        "NL",
        "FR",
        "DE",
        "--mode",
        "Both",
        "--device",
        "cpu",
        "--threads",
        "3",
        "--workers",
        "2",
        "--delivery-day",
        "2026-08-28",
        "--reuse-forecasts",
    ]


def test_regime_challenger_maps_all_switches() -> None:
    argv = _dry_run_argv(
        "-Action RegimeChallenger -Countries FR -Mode Autonomous "
        "-RegimeChallengerConfig config\\challenger-test.yaml "
        "-AllowModelDownload -StopOnError -ReuseForecasts "
        "-OverwriteChallenger"
    )

    assert argv[argv.index("--config") + 1] == str(
        PROJECT_ROOT / "config" / "challenger-test.yaml"
    )
    assert argv[-4:] == [
        "--allow-model-download",
        "--stop-on-error",
        "--reuse-forecasts",
        "--overwrite",
    ]


def test_regime_challenger_can_explicitly_run_official_forecasts_first() -> None:
    argv = _dry_run_argv(
        "-Action RegimeChallenger -Countries FR,NL -Mode Both "
        "-RunForecastsFirst"
    )
    assert "--run-forecasts-first" in argv
    assert "--reuse-forecasts" not in argv


def test_regime_challenger_rejects_conflicting_control_switches() -> None:
    completed = _invoke(
        "-Action RegimeChallenger -Countries FR -Mode Both "
        "-ReuseForecasts -RunForecastsFirst -DryRun"
    )
    assert completed.returncode != 0
    assert "mutuellement exclusifs" in completed.stdout + completed.stderr


def test_regime_challenger_does_not_change_run_action_argv() -> None:
    argv = _dry_run_argv(
        "-Action Run -Countries FR,NL -Mode Both -DeliveryDay 2026-08-21 "
        "-Device cpu -Threads 3 -Workers 2 -AllowModelDownload -StopOnError"
    )

    assert argv[1:] == [
        str(PROJECT_ROOT / "run_multicountry_forecast.py"),
        "--zones",
        "FR",
        "NL",
        "--mode",
        "Both",
        "--device",
        "cpu",
        "--threads",
        "3",
        "--workers",
        "2",
        "--delivery-day",
        "2026-08-21",
        "--allow-model-download",
        "--stop-on-error",
    ]


def test_regime_challenger_rejects_challenger_on_challenger_input() -> None:
    completed = _invoke(
        "-Action RegimeChallenger -Countries FR -Mode Both "
        "-ResidualLoadSource Chronos2 -DryRun"
    )
    assert completed.returncode != 0
    assert "controles Saturn officiels" in (
        completed.stdout + completed.stderr
    )


def test_regime_challenger_rejects_production_mode_during_dry_run() -> None:
    completed = _invoke(
        "-Action RegimeChallenger -Countries FR -Mode Production -DryRun"
    )
    assert completed.returncode != 0
    assert "Mode Autonomous, Blend ou Both" in (
        completed.stdout + completed.stderr
    )


def test_backfill_action_maps_delivery_day_to_optional_live_run() -> None:
    argv = _dry_run_argv(
        "-Action Backfill -Countries DE,BE -ThenRunForecast "
        "-DeliveryDay 2026-08-21"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_statistics_backfill.py")
    assert argv[2:5] == ["--zones", "DE", "BE"]
    assert "--then-run-live" in argv
    assert argv[argv.index("--live-delivery-day") + 1] == "2026-08-21"


def test_experiment_action_passes_isolated_config_and_run_switch() -> None:
    argv = _dry_run_argv(
        "-Action Experiment -Countries FR -Mode Autonomous "
        "-ExperimentConfig config\\experiment.yaml -RunExperiment"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_input_experiment.py")
    assert argv[argv.index("--experiment") + 1] == str(
        PROJECT_ROOT / "config" / "experiment.yaml"
    )
    assert argv[argv.index("--zones") + 1] == "FR"
    assert argv[-1] == "--run"


def test_experiment_rejects_blend_mode_before_running() -> None:
    completed = _invoke(
        "-Action Experiment -Countries FR -Mode Blend "
        "-ExperimentConfig config\\experiment.yaml -DryRun"
    )
    assert completed.returncode != 0
    assert "Blend et Both ne sont pas disponibles" in (
        completed.stdout + completed.stderr
    )


def test_topology_action_audits_by_default_with_dedicated_config() -> None:
    argv = _dry_run_argv(
        "-Action Topology -Countries FR,DE,BE,NL,ES -Mode Production"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_topology_experiment.py")
    assert argv[2:4] == [
        "--config",
        str(PROJECT_ROOT / "config" / "pricefm_topology_experiment.yaml"),
    ]
    assert argv[4:10] == ["--zones", "FR", "DE", "BE", "NL", "ES"]
    assert argv[-1] == "--audit-only"


def test_topology_run_preserves_country_order_and_omits_audit_only() -> None:
    argv = _dry_run_argv(
        "-Action Topology -Countries NL,FR -Mode Autonomous -RunExperiment"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_topology_experiment.py")
    assert argv[argv.index("--zones") + 1 :] == ["NL", "FR"]
    assert "--audit-only" not in argv


def test_topology_prepare_uses_pinned_calibration_and_bundle_anchors() -> None:
    argv = _dry_run_argv(
        "-Action Topology -TopologyStage Prepare -Countries FR,DE,BE,NL,ES"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_topology_experiment.py")
    assert "--prepare-operational" in argv
    assert argv[argv.index("--calibration-dir") + 1] == str(
        PROJECT_ROOT / "runs" / "experiments" / "pricefm_topology_v1"
    )
    assert argv[argv.index("--calibration-manifest-sha256") + 1] == (
        "b11548b6a1d520e9163aacd3204b0bc97e8ca99fbb9d48856ea92dda82cbf405"
    )
    assert argv[argv.index("--operational-dir") + 1] == str(
        PROJECT_ROOT
        / "runs"
        / "experiments"
        / "pricefm_topology_operational_v1"
    )
    assert argv[argv.index("--operational-manifest-sha256") + 1] == (
        "231988e94ded3798f3edefc1cdbe7629971c9e3b56ccda42dc4b8831106c2bac"
    )
    assert "--zones" not in argv


def test_topology_apply_builds_daily_both_argv() -> None:
    argv = _dry_run_argv(
        "-Action Topology -TopologyStage Apply "
        "-Countries FR,DE,BE,NL,ES -Mode Both -DeliveryDay 2026-08-22"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_topology_experiment.py")
    assert "--apply-daily" in argv
    assert argv[argv.index("--delivery-day") + 1] == "2026-08-22"
    assert argv[argv.index("--zones") + 1 : argv.index("--mode")] == [
        "FR",
        "DE",
        "BE",
        "NL",
        "ES",
    ]
    assert argv[argv.index("--mode") + 1] == "both"
    assert argv[argv.index("--daily-output-root") + 1] == str(
        PROJECT_ROOT / "runs" / "experiments" / "pricefm_topology_daily"
    )


def test_both_saturn_restriction_does_not_change_topology_apply() -> None:
    arguments = (
        "-Action Topology -TopologyStage Apply -Countries FR,DE,BE,NL,ES "
        "-Mode Both -DeliveryDay 2026-08-22"
    )
    assert _dry_run_argv(arguments + " -ResidualLoadSource Chronos2") == _dry_run_argv(arguments)


def test_topology_apply_accepts_blend_for_fr_and_nl() -> None:
    argv = _dry_run_argv(
        "-Action Topology -TopologyStage Apply "
        "-Countries NL,FR -Mode Blend -DeliveryDay 2026-08-22"
    )
    assert argv[argv.index("--zones") + 1 : argv.index("--mode")] == ["NL", "FR"]
    assert argv[argv.index("--mode") + 1] == "blend"


def test_topology_report365_uses_sealed_calibration_and_selected_zones() -> None:
    argv = _dry_run_argv(
        "-Action Topology -TopologyStage Report365 -Countries BE,FR"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_topology_experiment.py")
    assert "--report-365" in argv
    assert argv[argv.index("--calibration-dir") + 1] == str(
        PROJECT_ROOT / "runs" / "experiments" / "pricefm_topology_v1"
    )
    assert argv[argv.index("--calibration-manifest-sha256") + 1] == (
        "b11548b6a1d520e9163aacd3204b0bc97e8ca99fbb9d48856ea92dda82cbf405"
    )
    assert argv[argv.index("--annual-output-dir") + 1] == str(
        PROJECT_ROOT
        / "runs"
        / "experiments"
        / "pricefm_topology_annual_365_v1"
    )
    assert argv[argv.index("--zones") + 1 :] == ["BE", "FR"]


def test_topology_rolling365_uses_daily_refits_and_selected_workers() -> None:
    argv = _dry_run_argv(
        "-Action Topology -TopologyStage Rolling365 "
        "-Countries FR,DE,BE,NL,ES -Workers 5"
    )
    assert argv[1] == str(PROJECT_ROOT / "run_topology_experiment.py")
    assert "--rolling365-backtest" in argv
    assert argv[argv.index("--rolling365-workers") + 1] == "5"
    assert argv[argv.index("--rolling365-output-dir") + 1] == str(
        PROJECT_ROOT
        / "runs"
        / "experiments"
        / "pricefm_topology_rolling365_v1"
    )
    assert argv[argv.index("--zones") + 1 :] == ["FR", "DE", "BE", "NL", "ES"]


def test_topology_apply_rejects_blend_for_unavailable_country() -> None:
    completed = _invoke(
        "-Action Topology -TopologyStage Apply "
        "-Countries FR,DE -Mode Blend -DeliveryDay 2026-08-22 -DryRun"
    )
    assert completed.returncode != 0
    assert "disponible uniquement pour FR et NL" in (
        completed.stdout + completed.stderr
    )


@pytest.mark.parametrize("mode", ["Blend", "Both"])
def test_topology_rejects_explicit_blend_modes(mode: str) -> None:
    completed = _invoke(
        f"-Action Topology -Countries FR -Mode {mode} -DryRun"
    )
    assert completed.returncode != 0
    assert "Topology accepte uniquement" in (
        completed.stdout + completed.stderr
    )


def test_residual_compare_plan_preserves_country_order() -> None:
    argv = _dry_run_argv(
        "-Action ResidualCompare -ResidualComparisonStage Plan "
        "-Countries NL,FR,DE -Device cpu -Threads 3"
    )
    assert argv[1] == str(
        PROJECT_ROOT / "run_residual_load_historical_comparison.py"
    )
    assert argv[argv.index("--stage") + 1] == "plan"
    assert argv[argv.index("--zones") + 1 : argv.index("--device")] == [
        "NL",
        "FR",
        "DE",
    ]
    assert argv[argv.index("--threads") + 1] == "3"


def test_residual_compare_all_maps_resume_and_download_switches() -> None:
    argv = _dry_run_argv(
        "-Action ResidualCompare -ResidualComparisonStage All "
        "-Countries FR,DE -AllowModelDownload -OverwriteComparison "
        "-NoResume -SkipObservedSync"
    )
    assert argv[argv.index("--stage") + 1] == "all"
    assert argv[-4:] == [
        "--allow-model-download",
        "--overwrite",
        "--no-resume",
        "--skip-observed-sync",
    ]


def test_double_click_launcher_uses_unified_entry_point() -> None:
    source = (PROJECT_ROOT / "launch_forecast_app.cmd").read_text(
        encoding="utf-8"
    )
    assert 'Forecast.ps1" -Action App' in source
    assert "Start-ForecastApp.ps1" not in source
