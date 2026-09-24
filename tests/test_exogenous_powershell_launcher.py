from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "Exogenous.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _invoke(arguments: str) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available")
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"& {_quote(LAUNCHER)} {arguments}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _dry_run_commands(arguments: str) -> list[list[str]]:
    completed = _invoke(arguments + " -DryRun")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prefix = "Commande (argv, shell=False): "
    return [
        json.loads(line[len(prefix) :])
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]


def test_launcher_has_valid_powershell_syntax() -> None:
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available")
    command = (
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
            command,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr


def test_panel_dry_run_preserves_zone_order_pack_and_365_contract() -> None:
    commands = _dry_run_commands(
        "-Action Panel -Zones FR,DE,BE,NL -Pack residual_weather "
        "-EndDay 2026-09-02 -Overwrite"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert argv[1] == str(PROJECT_ROOT / "run_chronos2_exogenous_panel.py")
    assert argv[argv.index("--zones") + 1 : argv.index("--pack")] == [
        "FR",
        "DE",
        "BE",
        "NL",
    ]
    assert argv[argv.index("--pack") + 1] == "residual_weather"
    assert argv[argv.index("--output") + 1] == str(
        PROJECT_ROOT
        / "runs"
        / "experiments"
        / "chronos2_exogenous_lora_residual_weather_poc_v1"
        / "inputs"
        / "training_panel.parquet"
    )
    assert argv[argv.index("--training-days") + 1] == "365"
    assert argv[argv.index("--evaluation-days") + 1] == "365"
    assert argv[argv.index("--end-day") + 1] == "2026-09-02"


def test_prospective_panel_opt_in_is_explicit_and_default_remains_strict() -> None:
    default = _dry_run_commands(
        "-Action Panel -Zones FR -EndDay 2026-09-05 -Overwrite"
    )[0]
    assert "--allow-unresolved-final-evaluation-day" not in default

    prospective = _dry_run_commands(
        "-Action Panel -Zones FR -EndDay 2026-09-05 -Overwrite -ProspectiveFreeze"
    )[0]
    assert "--allow-unresolved-final-evaluation-day" in prospective
    assert "--require-production-pit" in prospective

    invalid = _invoke(
        "-Action POC -Zones FR -EndDay 2026-09-05 "
        "-ProspectiveFreeze -DryRun"
    )
    assert invalid.returncode != 0
    assert "reserve aux actions Panel et CalibrationPanel" in (
        invalid.stdout + invalid.stderr
    )


def test_calibration_panel_dry_run_requests_the_required_1095_origins() -> None:
    commands = _dry_run_commands(
        "-Action CalibrationPanel -Zones FR,DE,BE,NL -Pack full "
        "-EndDay 2026-09-02 -Overwrite"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == "run_chronos2_exogenous_panel.py"
    assert argv[argv.index("--mode") + 1] == "calibration"
    assert argv[argv.index("--training-days") + 1] == "730"
    assert argv[argv.index("--evaluation-days") + 1] == "365"


def test_calibrate_residual_dry_run_uses_fold_specific_resume_runner() -> None:
    output = PROJECT_ROOT / "runs" / "tmp" / "lora residual calibration"
    shared_cache = PROJECT_ROOT / "runs" / "tmp" / "lora shared oof checkpoints"
    commands = _dry_run_commands(
        "-Action CalibrateResidual -Zones FR -Pack full -BlockDays 21 "
        f"-Device cpu -ResidualCalibrationDirectory {_quote(output)} "
        f"-OofCheckpointCacheDirectory {_quote(shared_cache)}"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == "run_chronos2_exogenous_calibrate_residual.py"
    assert argv[argv.index("--item-id") + 1] == "FR"
    assert argv[argv.index("--block-days") + 1] == "21"
    assert argv[argv.index("--device-map") + 1] == "cpu"
    assert argv[argv.index("--output-directory") + 1] == str(output)
    assert argv[argv.index("--shared-checkpoint-cache-directory") + 1] == str(
        shared_cache
    )
    assert "--overwrite" not in argv


def test_research_validation_fit_dry_run_is_single_target_and_non_promotable() -> None:
    run_directory = PROJECT_ROOT / "runs" / "tmp" / "rank16 zone" / "artifact"
    research_directory = PROJECT_ROOT / "runs" / "tmp" / "research validation FR"
    commands = _dry_run_commands(
        "-Action ResearchFitValidationCorrector -Zones FR -Device cpu "
        f"-RunDirectory {_quote(run_directory)} "
        f"-ResearchCorrectorDirectory {_quote(research_directory)}"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == (
        "run_chronos2_exogenous_research_validation_corrector.py"
    )
    assert argv[2] == "fit"
    assert argv[argv.index("--run-directory") + 1] == str(run_directory)
    assert argv[argv.index("--item-id") + 1] == "FR"
    assert argv[argv.index("--device-map") + 1] == "cpu"
    assert argv[argv.index("--research-directory") + 1] == str(
        research_directory
    )
    assert "--target-column" not in argv
    assert "--output-directory" not in argv
    assert "--overwrite" not in argv


def test_research_validation_evaluate_uses_only_reserved_child_output() -> None:
    run_directory = PROJECT_ROOT / "runs" / "tmp" / "rank16 zone" / "artifact"
    research_directory = PROJECT_ROOT / "runs" / "tmp" / "research validation FR"
    commands = _dry_run_commands(
        "-Action ResearchEvaluateValidationCorrector -Zones FR "
        f"-RunDirectory {_quote(run_directory)} "
        f"-ResearchCorrectorDirectory {_quote(research_directory)}"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == (
        "run_chronos2_exogenous_research_validation_corrector.py"
    )
    assert argv[2] == "evaluate"
    assert argv[argv.index("--run-directory") + 1] == str(run_directory)
    assert argv[argv.index("--item-id") + 1] == "FR"
    assert argv[argv.index("--research-directory") + 1] == str(
        research_directory
    )
    assert "--target-column" not in argv
    assert "--output-directory" not in argv
    assert "--device-map" not in argv
    assert "--overwrite" not in argv


@pytest.mark.parametrize(
    "action",
    ["ResearchFitValidationCorrector", "ResearchEvaluateValidationCorrector"],
)
def test_research_validation_actions_reject_overwrite(action: str) -> None:
    completed = _invoke(f"-Action {action} -Zones FR -Overwrite -DryRun")

    assert completed.returncode != 0
    assert "preuves research scellees" in completed.stdout + completed.stderr


def test_final_backtest_dry_run_pairs_corrected_lora_with_incumbent() -> None:
    run_directory = PROJECT_ROOT / "runs" / "tmp" / "lora final"
    calibration = PROJECT_ROOT / "runs" / "tmp" / "lora calibration"
    incumbent = PROJECT_ROOT / "runs" / "tmp" / "incumbent statistics.csv.gz"
    commands = _dry_run_commands(
        "-Action FinalBacktest -Zones FR "
        f"-RunDirectory {_quote(run_directory)} "
        f"-ResidualCalibrationDirectory {_quote(calibration)} "
        f"-IncumbentStatistics {_quote(incumbent)} -Overwrite"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == "run_chronos2_exogenous_final_evaluate.py"
    assert argv[argv.index("--run-directory") + 1] == str(run_directory)
    assert argv[argv.index("--residual-corrector") + 1] == str(
        calibration / "residual_corrector.json"
    )
    assert argv[argv.index("--oof-audit") + 1] == str(
        calibration / "oof_predictions_365.csv.gz.audit.json"
    )
    assert argv[argv.index("--incumbent-statistics") + 1] == str(incumbent)
    assert argv[argv.index("--zone") + 1] == "FR"
    assert "--overwrite" in argv


def test_backtest_can_bind_a_separate_resolved_holdout_panel() -> None:
    panel = PROJECT_ROOT / "runs" / "tmp" / "resolved holdout.parquet"
    commands = _dry_run_commands(
        "-Action Backtest -Zones FR "
        f"-ResolvedEvaluationPanel {_quote(panel)}"
    )
    argv = commands[0]
    assert argv[argv.index("--panel") + 1] == str(panel)
    assert argv[argv.index("--panel-audit") + 1] == str(
        panel.with_suffix(panel.suffix + ".audit.json")
    )

    materialize = _dry_run_commands(
        "-Action ResolveHoldout -Zones FR -EndDay 2026-09-05 "
        f"-ResolvedEvaluationPanel {_quote(panel)}"
    )[0]
    assert materialize[materialize.index("--output") + 1] == str(panel)
    assert "--allow-unresolved-final-evaluation-day" in materialize
    assert "--require-production-pit" in materialize


def test_resolve_holdout_never_overwrites_the_frozen_training_panel() -> None:
    frozen = (
        PROJECT_ROOT
        / "runs"
        / "experiments"
        / "chronos2_exogenous_lora_poc_v1"
        / "inputs"
        / "training_panel.parquet"
    )
    completed = _invoke(
        "-Action ResolveHoldout -Zones FR -EndDay 2026-09-05 "
        f"-ResolvedEvaluationPanel {_quote(frozen)} -DryRun"
    )

    assert completed.returncode != 0
    assert "panel gele ne sera jamais ecrase" in (
        completed.stdout + completed.stderr
    )


def test_shadow_epoch_commands_are_linked_without_promotion() -> None:
    run = PROJECT_ROOT / "runs" / "tmp" / "future candidate"
    calibration = PROJECT_ROOT / "runs" / "tmp" / "future calibration"
    epoch = PROJECT_ROOT / "runs" / "tmp" / "future epoch"
    commands = _dry_run_commands(
        "-Action EpochFreeze -Zones FR "
        f"-RunDirectory {_quote(run)} "
        f"-ResidualCalibrationDirectory {_quote(calibration)} "
        f"-ShadowEpochDirectory {_quote(epoch)} "
        "-FirstShadowDay 2026-09-06"
    )
    argv = commands[0]
    assert Path(argv[1]).name == "run_chronos2_exogenous_shadow_epoch.py"
    assert argv[2] == "freeze"
    assert argv[argv.index("--first-shadow-day") + 1] == "2026-09-06"
    assert argv[argv.index("--output-directory") + 1] == str(epoch)
    assert not any("promot" in value.casefold() for value in argv)

    verify = _dry_run_commands(
        "-Action EpochVerify -Zones FR "
        f"-RunDirectory {_quote(run)} -ShadowEpochDirectory {_quote(epoch)}"
    )[0]
    assert verify[2] == "verify"
    assert verify[3] == str(epoch)


def test_compare_candidates_dry_run_is_explicit_and_read_only() -> None:
    rank8 = PROJECT_ROOT / "runs" / "tmp" / "rank 8" / "fr"
    rank16 = PROJECT_ROOT / "runs" / "tmp" / "rank 16" / "fr"
    output = PROJECT_ROOT / "runs" / "tmp" / "rank comparison" / "fr"
    commands = _dry_run_commands(
        "-Action CompareCandidates -Zones FR "
        f"-Rank8Candidate {_quote(rank8)} "
        f"-Rank16Candidate {_quote(rank16)} "
        f"-ComparisonOutput {_quote(output)} -Overwrite"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == "run_chronos2_exogenous_compare_candidates.py"
    assert argv[argv.index("--rank8") + 1] == str(rank8)
    assert argv[argv.index("--rank16") + 1] == str(rank16)
    assert argv[argv.index("--output-directory") + 1] == str(output)
    assert argv[argv.index("--zone") + 1] == "FR"
    assert argv[argv.index("--policy") + 1] == str(
        PROJECT_ROOT / "config" / "chronos2_exogenous_promotion_v1.yaml"
    )
    assert "--overwrite" in argv


def test_pack_selects_its_matching_training_configuration() -> None:
    commands = _dry_run_commands("-Action Validate -Pack residual_fuel")

    assert len(commands) == 1
    argv = commands[0]
    assert argv[argv.index("--config") + 1] == str(
        PROJECT_ROOT / "config" / "chronos2_exogenous_lora_residual_fuel_poc.yaml"
    )


def test_prepare_zones_dry_run_forks_one_finished_training_artifact() -> None:
    source = PROJECT_ROOT / "runs" / "tmp" / "rank16" / "artifact"
    output_root = PROJECT_ROOT / "runs" / "tmp" / "rank16 prepared"
    commands = _dry_run_commands(
        "-Action PrepareZones -Zones FR,DE,BE,NL "
        f"-RunDirectory {_quote(source)} "
        f"-ZoneArtifactsRoot {_quote(output_root)}"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == "run_chronos2_exogenous_prepare_zones.py"
    assert argv[argv.index("--source-run-directory") + 1] == str(source)
    assert argv[argv.index("--zones") + 1 : argv.index("--output-root")] == [
        "FR",
        "DE",
        "BE",
        "NL",
    ]
    assert argv[argv.index("--output-root") + 1] == str(output_root)


def test_recover_training_snapshot_is_a_separate_explicit_fail_closed_action() -> None:
    source = PROJECT_ROOT / "runs" / "tmp" / "evaluated rank8" / "artifact"
    output_root = PROJECT_ROOT / "runs" / "tmp" / "recovered rank8"
    config = PROJECT_ROOT / "config" / "chronos2_exogenous_lora_rank8_reference.yaml"
    commands = _dry_run_commands(
        "-Action RecoverTrainingSnapshot -Zones FR,DE,BE,NL "
        f"-Config {_quote(config)} -RunDirectory {_quote(source)} "
        f"-ZoneArtifactsRoot {_quote(output_root)}"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert Path(argv[1]).name == (
        "run_chronos2_exogenous_recover_training_snapshot.py"
    )
    assert argv[argv.index("--source-run-directory") + 1] == str(source)
    assert argv[argv.index("--config-reference") + 1] == str(config)
    assert argv[argv.index("--zones") + 1 : argv.index("--output-root")] == [
        "FR",
        "DE",
        "BE",
        "NL",
    ]
    assert argv[argv.index("--output-root") + 1] == str(output_root)
    assert "--overwrite" not in argv

    missing_config = _invoke(
        "-Action RecoverTrainingSnapshot -Zones FR "
        f"-RunDirectory {_quote(source)} -DryRun"
    )
    assert missing_config.returncode != 0
    assert "-Config est obligatoire" in missing_config.stdout + missing_config.stderr


def test_govern_dry_run_infers_only_artifacts_below_run_directory() -> None:
    run_directory = PROJECT_ROOT / "runs" / "tmp" / "candidate with spaces"
    predictions = PROJECT_ROOT / "runs" / "tmp" / "paired predictions.csv.gz"
    commands = _dry_run_commands(
        "-Action Govern -Zones FR "
        f"-RunDirectory {_quote(run_directory)} "
        f"-Predictions {_quote(predictions)}"
    )

    assert len(commands) == 1
    argv = commands[0]
    assert argv[1:4] == [
        str(PROJECT_ROOT / "run_chronos2_exogenous_governance.py"),
        "evaluate",
        "--zone",
    ]
    assert argv[argv.index("--rolling-predictions") + 1] == str(predictions)
    assert argv[argv.index("--experiment-manifest") + 1] == str(
        run_directory / "experiment_manifest.json"
    )
    assert argv[argv.index("--artifact") + 1] == (
        "checkpoint=" + str(run_directory / "checkpoint")
    )
    second_artifact = argv.index("--artifact", argv.index("--artifact") + 1)
    assert argv[second_artifact + 1] == "schema=" + str(run_directory / "schema.json")


def test_govern_requires_explicit_rolling_predictions() -> None:
    completed = _invoke("-Action Govern -Zones FR -DryRun")

    assert completed.returncode != 0
    assert "-Predictions est obligatoire" in completed.stdout + completed.stderr


def test_prospective_govern_requires_and_rechecks_the_shadow_epoch() -> None:
    config = PROJECT_ROOT / "config" / "chronos2_exogenous_lora_prospective.example.yaml"
    rolling = PROJECT_ROOT / "runs" / "tmp" / "rolling.csv.gz"
    run = PROJECT_ROOT / "runs" / "tmp" / "prospective candidate"
    missing = _invoke(
        "-Action Govern -Zones FR "
        f"-Config {_quote(config)} -RunDirectory {_quote(run)} "
        f"-Predictions {_quote(rolling)} -DryRun"
    )
    assert missing.returncode != 0
    assert "ShadowEpochDirectory est obligatoire" in (
        missing.stdout + missing.stderr
    )

    epoch = PROJECT_ROOT / "runs" / "tmp" / "prospective epoch"
    commands = _dry_run_commands(
        "-Action Govern -Zones FR "
        f"-Config {_quote(config)} -RunDirectory {_quote(run)} "
        f"-Predictions {_quote(rolling)} "
        f"-ShadowEpochDirectory {_quote(epoch)}"
    )
    assert len(commands) == 2
    assert Path(commands[0][1]).name == "run_chronos2_exogenous_shadow_epoch.py"
    assert commands[0][2] == "finalize-check"
    assert commands[0][3] == str(epoch)
    assert Path(commands[1][1]).name == "run_chronos2_exogenous_governance.py"
    assert commands[1][commands[1].index("--shadow-epoch-directory") + 1] == str(
        epoch
    )


@pytest.mark.parametrize(
    "action",
    [
        "Backtest",
        "CalibrateResidual",
        "ResearchFitValidationCorrector",
        "ResearchEvaluateValidationCorrector",
        "FinalBacktest",
        "CompareCandidates",
        "Govern",
        "POC",
    ],
)
def test_performance_actions_fail_closed_on_multiple_zones(action: str) -> None:
    if action == "Govern":
        extra = " -Predictions placeholder.csv.gz"
    elif action == "CompareCandidates":
        extra = " -Rank8Candidate rank8 -Rank16Candidate rank16"
    else:
        extra = ""
    overwrite = "" if action.startswith("Research") else " -Overwrite"
    completed = _invoke(f"-Action {action} -Zones FR,DE{extra}{overwrite} -DryRun")

    assert completed.returncode != 0
    assert "exige exactement une zone" in completed.stdout + completed.stderr


def test_poc_dry_run_is_isolated_and_orders_the_five_stages() -> None:
    commands = _dry_run_commands(
        "-Action POC -Zones FR -EndDay 2026-09-02 -Overwrite -Device cpu"
    )

    assert [Path(command[1]).name for command in commands] == [
        "run_chronos2_exogenous_panel.py",
        "run_chronos2_exogenous_finetune.py",
        "run_chronos2_exogenous_finetune.py",
        "run_chronos2_exogenous_evaluate.py",
        "run_chronos2_exogenous_governance.py",
    ]
    assert commands[1][2] == "validate"
    assert commands[2][2] == "train"
    assert commands[3][commands[3].index("--device-map") + 1] == "cpu"
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Forecast.ps1" not in source
    assert "Invoke-Expression" not in source
    assert "cmd /c" not in source.casefold()


def test_shadow_dry_run_uses_append_only_mode_and_explicit_journal() -> None:
    journal = PROJECT_ROOT / "runs" / "tmp" / "shadow predictions.csv.gz"
    live_sources = PROJECT_ROOT / "runs" / "tmp" / "live sources.json"
    commands = _dry_run_commands(
        f"-Action Shadow -Zones FR -EndDay 2026-09-04 -Device cpu "
        f"-ShadowPredictions {_quote(journal)} "
        f"-LiveSourceManifest {_quote(live_sources)}"
    )

    assert len(commands) == 3
    panel_argv, argv, final_argv = commands
    assert Path(panel_argv[1]).name == "run_chronos2_exogenous_panel.py"
    assert panel_argv[panel_argv.index("--mode") + 1] == "shadow"
    assert panel_argv[panel_argv.index("--end-day") + 1] == "2026-09-04"
    assert panel_argv[panel_argv.index("--live-source-manifest") + 1] == str(
        live_sources
    )
    assert Path(argv[1]).name == "run_chronos2_exogenous_evaluate.py"
    assert argv[argv.index("--mode") + 1] == "shadow"
    assert argv[argv.index("--item-id") + 1] == "FR"
    assert argv[argv.index("--device-map") + 1] == "cpu"
    panel_path = Path(argv[argv.index("--panel") + 1])
    assert argv[argv.index("--panel-audit") + 1] == str(
        panel_path.with_suffix(panel_path.suffix + ".audit.json")
    )
    assert argv[argv.index("--shadow-predictions") + 1] == str(journal)
    assert "--overwrite" not in argv
    assert Path(final_argv[1]).name == (
        "run_chronos2_exogenous_finalize_shadow.py"
    )
    assert final_argv[final_argv.index("--raw-observed-evidence") + 1] == str(
        journal.with_name("shadow_observed_evidence.csv.gz")
    )
    assert final_argv[final_argv.index("--raw-shadow-manifest") + 1] == str(
        journal.with_name("shadow_manifest.json")
    )
    assert final_argv[final_argv.index("--raw-shadow-journal") + 1] == str(
        journal
    )
    assert final_argv[final_argv.index("--output-directory") + 1] == str(
        journal.parent / "shadow_final"
    )
    assert final_argv[final_argv.index("--zone") + 1] == "FR"
    assert "--allow-no-observed" in final_argv


def test_prospective_shadow_preflights_the_frozen_day_and_journal() -> None:
    config = PROJECT_ROOT / "config" / "chronos2_exogenous_lora_prospective.example.yaml"
    run = PROJECT_ROOT / "runs" / "tmp" / "prospective candidate"
    epoch = PROJECT_ROOT / "runs" / "tmp" / "prospective epoch"
    journal = PROJECT_ROOT / "runs" / "tmp" / "prospective shadow.csv.gz"
    commands = _dry_run_commands(
        "-Action Shadow -Zones FR -EndDay 2026-09-06 "
        f"-Config {_quote(config)} -RunDirectory {_quote(run)} "
        f"-ShadowEpochDirectory {_quote(epoch)} "
        f"-ShadowPredictions {_quote(journal)}"
    )

    assert len(commands) == 4
    preflight = commands[0]
    assert Path(preflight[1]).name == "run_chronos2_exogenous_shadow_epoch.py"
    assert preflight[2] == "delivery-check"
    assert preflight[3] == str(epoch)
    assert preflight[preflight.index("--journal") + 1] == str(journal)
    assert preflight[preflight.index("--delivery-day") + 1] == "2026-09-06"

    missing = _invoke(
        "-Action Shadow -Zones FR -EndDay 2026-09-06 "
        f"-Config {_quote(config)} -RunDirectory {_quote(run)} -DryRun"
    )
    assert missing.returncode != 0
    assert "ShadowEpochDirectory est obligatoire" in (
        missing.stdout + missing.stderr
    )


def test_shadow_rejects_overwrite_even_in_dry_run() -> None:
    completed = _invoke(
        "-Action Shadow -Zones FR -EndDay 2026-09-04 -Overwrite -DryRun"
    )

    assert completed.returncode != 0
    assert "append-only" in completed.stdout + completed.stderr


def test_govern_accepts_only_the_final_shadow_snapshot() -> None:
    rolling = PROJECT_ROOT / "runs" / "tmp" / "rolling.csv.gz"
    raw = PROJECT_ROOT / "runs" / "tmp" / "shadow_observed_evidence.csv.gz"
    refused = _invoke(
        "-Action Govern -Zones FR -DryRun "
        f"-Predictions {_quote(rolling)} -ShadowPredictions {_quote(raw)}"
    )
    assert refused.returncode != 0
    assert "shadow_final_evidence.csv.gz" in refused.stdout + refused.stderr

    final = raw.parent / "shadow_final" / "shadow_final_evidence.csv.gz"
    commands = _dry_run_commands(
        "-Action Govern -Zones FR "
        f"-Predictions {_quote(rolling)} -ShadowPredictions {_quote(final)}"
    )
    argv = commands[0]
    assert argv[argv.index("--shadow-predictions") + 1] == str(final)
    assert argv[argv.index("--shadow-manifest") + 1] == str(
        final.with_name("shadow_final_manifest.json")
    )


def test_live_source_manifest_is_reserved_for_shadow() -> None:
    completed = _invoke(
        "-Action Validate -LiveSourceManifest placeholder.json -DryRun"
    )

    assert completed.returncode != 0
    assert "reserve a -Action Shadow" in completed.stdout + completed.stderr
