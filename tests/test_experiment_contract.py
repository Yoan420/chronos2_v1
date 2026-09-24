from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from chronos2_hourly.experiment_contract import (
    ExperimentContractError,
    load_experiment_contract,
    overlay_experiment_inputs,
    resolve_experiment_config,
    validate_experiment_inputs,
)


ZONE_TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}


def _write_project(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    pit_dir = root / "data" / "pit" / "vintages"
    pit_dir.mkdir(parents=True)

    delivery = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "value_time_utc": delivery,
            "snapshot_time_utc": delivery - pd.Timedelta(days=2),
            "revision_time_utc": delivery - pd.Timedelta(days=2),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame.to_parquet(pit_dir / "wind.parquet", index=False)

    production_configs: dict[str, str] = {}
    for zone, timezone in ZONE_TIMEZONES.items():
        path = root / f"production_{zone.lower()}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "data": {
                        "pit_vintage_dir": "data/pit/vintages",
                        "pit_files": {"existing": "existing.parquet"},
                    },
                    "output": {"directory": f"runs/live/{zone.lower()}"},
                    "zones": {
                        zone: {
                            "enabled": True,
                            "timezone": timezone,
                            "target": {"series": f"target.{zone.lower()}"},
                            "covariates": {
                                "existing": {
                                    "enabled": True,
                                    "source": "pit_parquet",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        production_configs[zone] = path.name

    payload = {
        "schema_version": 1,
        "experiment_id": "wind_test",
        "production_configs": production_configs,
        "output_directory": "runs/experiments/wind_test",
        "series": {
            "wind_fcst_test": {
                "enabled": True,
                "zones": ["FR", "DE", "BE", "NL", "ES"],
                "kind": "hourly_numeric_pit",
                "timezone": "UTC",
                "series": "power.fr.generation.wind.hourly.gw.fcst",
                "pit_file": "data/pit/vintages/wind.parquet",
                "columns": {
                    "delivery": "value_time_utc",
                    "value": "value",
                    "availability": "snapshot_time_utc",
                    "revision": "revision_time_utc",
                },
                "minimum_coverage": 0.5,
            }
        },
    }
    contract_path = root / "config" / "experiment.yaml"
    contract_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return contract_path, payload


def _rewrite(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_resolves_all_five_zones_in_memory_and_never_targets_live(tmp_path: Path) -> None:
    contract_path, _ = _write_project(tmp_path)
    contract = load_experiment_contract(contract_path)
    validate_experiment_inputs(contract)

    for zone in ZONE_TIMEZONES:
        resolved = resolve_experiment_config(contract, zone=zone)
        assert resolved.zone == zone
        assert resolved.enabled_aliases == ("wind_fcst_test",)
        assert resolved.output_directory == (
            contract.output_directory / zone.lower()
        ).resolve()
        assert resolved.output_directory.is_relative_to(
            contract.project_root / "runs" / "experiments"
        )
        assert not resolved.output_directory.is_relative_to(
            contract.project_root / "runs" / "live"
        )
        covariate = resolved.config["zones"][zone]["covariates"][
            "wind_fcst_test"
        ]
        assert covariate == {
            "enabled": True,
            "source": "pit_parquet",
            "series": "power.fr.generation.wind.hourly.gw.fcst",
            "pit_file": str(
                contract.project_root / "data" / "pit" / "vintages" / "wind.parquet"
            ),
            "timestamp_col": "value_time_utc",
            "value_col": "value",
            "availability_col": "snapshot_time_utc",
            "revision_col": "revision_time_utc",
            "fill_method": "none",
            "fill_limit": 0,
            "minimum_coverage": 0.5,
            "include_base_context": True,
            "future": {"known_future": True, "strategies": ["oracle"]},
        }


def test_overlay_does_not_mutate_the_production_mapping(tmp_path: Path) -> None:
    contract_path, _ = _write_project(tmp_path)
    contract = load_experiment_contract(contract_path)
    production = yaml.safe_load(
        contract.production_configs["FR"].read_text(encoding="utf-8")
    )
    before = copy.deepcopy(production)

    resolved = overlay_experiment_inputs(
        contract,
        zone="FR",
        production_config=production,
    )

    assert production == before
    assert production["output"]["directory"] == "runs/live/fr"
    assert resolved.config["output"]["directory"] == str(
        contract.output_directory / "fr"
    )


def test_enabled_false_excludes_input_and_allows_a_not_yet_materialized_file(
    tmp_path: Path,
) -> None:
    contract_path, payload = _write_project(tmp_path)
    item = payload["series"]["wind_fcst_test"]
    item["enabled"] = False
    item["pit_file"] = "data/pit/vintages/not_materialized.parquet"
    _rewrite(contract_path, payload)

    contract = load_experiment_contract(contract_path)
    resolved = resolve_experiment_config(contract, zone="FR")

    assert contract.enabled_aliases == ()
    assert resolved.enabled_aliases == ()
    assert "wind_fcst_test" not in resolved.config["zones"]["FR"]["covariates"]


@pytest.mark.parametrize("token", ["storm", "MKOnline"])
def test_storm_and_mkonline_are_forbidden_as_inputs(
    tmp_path: Path,
    token: str,
) -> None:
    contract_path, payload = _write_project(tmp_path)
    payload["series"]["wind_fcst_test"]["series"] = f"power.hidden.{token}.fcst"
    _rewrite(contract_path, payload)

    with pytest.raises(ExperimentContractError, match="forbidden"):
        load_experiment_contract(contract_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "python_callable", "kind"),
        ("timezone", "Europe/Paris", "UTC"),
        ("timezone", "Mars/Olympus", "timezone"),
    ],
)
def test_kind_and_timezone_are_strict(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    contract_path, payload = _write_project(tmp_path)
    payload["series"]["wind_fcst_test"][field] = value
    _rewrite(contract_path, payload)

    with pytest.raises(ExperimentContractError, match=message):
        load_experiment_contract(contract_path)


def test_alias_must_be_lower_snake_case(tmp_path: Path) -> None:
    contract_path, payload = _write_project(tmp_path)
    payload["series"]["Wind-Feature"] = payload["series"].pop("wind_fcst_test")
    _rewrite(contract_path, payload)

    with pytest.raises(ExperimentContractError, match="snake_case"):
        load_experiment_contract(contract_path)


def test_unknown_transform_or_code_key_is_rejected(tmp_path: Path) -> None:
    contract_path, payload = _write_project(tmp_path)
    payload["series"]["wind_fcst_test"]["transform"] = "__import__('os')"
    _rewrite(contract_path, payload)

    with pytest.raises(ExperimentContractError, match="unknown=.*transform"):
        load_experiment_contract(contract_path)


@pytest.mark.parametrize(
    "output_directory",
    ["runs/live/wind_test", "runs/experiments", "../outside/wind_test"],
)
def test_output_is_confined_below_runs_experiments(
    tmp_path: Path,
    output_directory: str,
) -> None:
    contract_path, payload = _write_project(tmp_path)
    payload["output_directory"] = output_directory
    _rewrite(contract_path, payload)

    with pytest.raises(ExperimentContractError, match="output_directory"):
        load_experiment_contract(contract_path)


def test_input_file_must_stay_in_pit_vintage_directory(tmp_path: Path) -> None:
    contract_path, payload = _write_project(tmp_path)
    payload["series"]["wind_fcst_test"]["pit_file"] = "secrets.parquet"
    _rewrite(contract_path, payload)

    with pytest.raises(ExperimentContractError, match="pit_file must stay below"):
        load_experiment_contract(contract_path)


def test_numeric_hourly_parquet_is_verified(tmp_path: Path) -> None:
    contract_path, _ = _write_project(tmp_path)
    contract = load_experiment_contract(contract_path)
    pit_file = contract.inputs[0].pit_file
    bad = pd.DataFrame(
        {
            "value_time_utc": [pd.Timestamp("2026-01-01 00:30", tz="UTC")],
            "snapshot_time_utc": [pd.Timestamp("2025-12-30", tz="UTC")],
            "revision_time_utc": [pd.Timestamp("2025-12-30", tz="UTC")],
            "value": ["not-numeric"],
        }
    )
    bad.to_parquet(pit_file, index=False)

    with pytest.raises(ExperimentContractError, match="numeric dtype"):
        validate_experiment_inputs(contract)


def test_checked_in_example_resolves_without_writing() -> None:
    project_root = Path(__file__).resolve().parents[1]
    contract = load_experiment_contract(
        project_root / "config" / "experiment.yaml"
    )

    assert contract.enabled_aliases == ("fr_wind_generation_fcst_test",)
    assert contract.output_directory == (
        project_root / "runs" / "experiments" / "fr_wind_pit_test"
    ).resolve()
    validate_experiment_inputs(contract)
    for zone in ZONE_TIMEZONES:
        resolved = resolve_experiment_config(contract, zone=zone)
        assert resolved.output_directory.parent == contract.output_directory
