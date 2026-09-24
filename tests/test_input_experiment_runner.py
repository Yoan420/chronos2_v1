from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run_input_experiment as runner
from chronos2_hourly.experiment_contract import (
    ExperimentContract,
    ExperimentInput,
    ResolvedExperimentConfig,
)


def _plan(tmp_path: Path, zones: tuple[str, ...] = ("FR",)) -> runner.ExperimentPlan:
    source = tmp_path / "experiment.yaml"
    source.write_text("schema_version: 1\n", encoding="utf-8")
    production = tmp_path / "production.yaml"
    production.write_text("zones: {}\n", encoding="utf-8")
    pit = tmp_path / "data" / "pit" / "vintages" / "wind.parquet"
    pit.parent.mkdir(parents=True)
    pit.write_bytes(b"sealed pit bytes")
    input_item = ExperimentInput(
        alias="wind_test",
        enabled=True,
        zones=zones,
        kind="hourly_numeric_pit",
        timezone="UTC",
        series="power.wind.fcst",
        pit_file=pit,
        delivery_column="value_time_utc",
        value_column="value",
        availability_column="snapshot_time_utc",
        revision_column="revision_time_utc",
        minimum_coverage=0.9,
    )
    output_root = tmp_path / "runs" / "experiments" / "wind_test"
    contract = ExperimentContract(
        schema_version=1,
        experiment_id="wind_test",
        source_path=source,
        project_root=tmp_path,
        production_configs={zone: production for zone in zones},
        output_directory=output_root,
        inputs=(input_item,),
    )
    resolved = tuple(
        ResolvedExperimentConfig(
            experiment_id="wind_test",
            zone=zone,
            production_config=production,
            production_config_directory=tmp_path,
            output_directory=output_root / zone.lower(),
            enabled_aliases=("wind_test",),
            config={"output": {"directory": str(output_root / zone.lower())}},
        )
        for zone in zones
    )
    return runner.ExperimentPlan(contract=contract, resolved=resolved)


def test_normalize_zones_is_strict() -> None:
    assert runner.normalize_zones(("fr", "NL")) == ("FR", "NL")
    with pytest.raises(ValueError, match="duplique"):
        runner.normalize_zones(("FR", "fr"))
    with pytest.raises(ValueError, match="non supporte"):
        runner.normalize_zones(("GB",))


def test_runner_command_is_argv_only_and_keeps_experiment_output(tmp_path: Path) -> None:
    item = _plan(tmp_path).resolved[0]
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"python")
    config = tmp_path / "resolved.yaml"
    command = runner.build_runner_command(
        item,
        resolved_config_path=config,
        python_executable=executable,
        local_files_only=True,
        device="cpu",
    )
    assert isinstance(command, tuple)
    assert command[0] == str(executable.resolve())
    assert command[command.index("--output-dir") + 1] == str(item.output_directory)
    assert "--local-files-only" in command


def test_execute_writes_audited_request_then_runs_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, ("FR", "NL"))
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"python")
    calls: list[tuple[str, ...]] = []

    def fake_run(command, *, check, cwd):
        assert check is True
        assert cwd == runner.PROJECT_ROOT
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    outputs = runner.execute_experiment_plan(
        plan,
        python_executable=executable,
        local_files_only=True,
        device="auto",
    )

    assert outputs == tuple(item.output_directory for item in plan.resolved)
    assert len(calls) == 2
    for item in plan.resolved:
        assert (item.output_directory / "resolved_experiment.yaml").is_file()
        manifest = item.output_directory / "experiment_request.json"
        assert manifest.is_file()
        text = manifest.read_text(encoding="utf-8")
        assert '"production_changed": false' in text
        assert '"storm_used_as_input": false' in text
        assert '"mkonline_used_as_input": false' in text


def test_invalid_device_fails_before_output(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"python")
    with pytest.raises(ValueError, match="device"):
        runner.execute_experiment_plan(
            plan,
            python_executable=executable,
            device="quantum",
        )
    assert not plan.resolved[0].output_directory.exists()
