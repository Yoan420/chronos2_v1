"""Small, offline integration tests for the isolated NOAA LoRA candidate."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from auxiliary_lab.noaa_gfs import GfsError
from chronos2_exogenous.evaluation import build_inference_input
from chronos2_exogenous.feature_bank import (
    ExogenousBankError,
    cutoff_by_delivery_hour,
    delivery_utc_index,
)
from chronos2_exogenous.lora_finetune import (
    ExogenousFineTuneError,
    load_config,
    validate_panel,
)
from chronos2_exogenous.panel import OriginPanelError
import run_chronos2_noaa_panel as noaa


CONFIG = Path(__file__).resolve().parents[1] / "config/chronos2_exogenous_noaa_lora_v1.yaml"
CALENDAR = (
    "known_hour_sin", "known_hour_cos", "known_dow_sin", "known_dow_cos",
    "known_doy_sin", "known_doy_cos", "known_is_weekend",
)
WEATHER_COLUMNS = tuple(f"local_gfs_{name}" for name in noaa.VARIABLES)
QUALITY_COLUMNS = tuple(f"noaa_gfs_weather__{name}" for name in ("coverage", "available", "age_hours"))


@pytest.mark.parametrize(
    ("mode", "days", "start", "weather_start"),
    [
        ("training", 730, "2024-09-03", "2024-06-07"),
        ("calibration", 1095, "2023-09-04", "2023-06-08"),
    ],
)
def test_exact_full_periods_and_context_support(mode, days, start, weather_start):
    period = noaa.panel_period(mode=mode, end_day="2026-09-02")
    assert period == {
        "mode": mode, "start_day": start, "end_day": "2026-09-02",
        "origin_days": days, "weather_start_day": weather_start,
        "context_length": 2048, "holdout_days": 365,
        "holdout_start_day": "2025-09-03",
    }
    deliveries = pd.date_range(start, period["end_day"], freq="D")
    assert len(deliveries) == days
    required_context = delivery_utc_index(start, start)[0] - pd.Timedelta(hours=2048)
    assert delivery_utc_index(weather_start, weather_start)[0] <= required_context


def test_full_training_and_calibration_reserve_identical_holdout_origins():
    origins = []
    for mode in ("training", "calibration"):
        period = noaa.panel_period(mode=mode, end_day="2026-09-02")
        days = pd.date_range(period["start_day"], period["end_day"], freq="D")[-365:]
        origins.append(tuple(
            cutoff_by_delivery_hour(delivery_utc_index(str(day.date()), str(day.date())))[0]
            for day in days
        ))
    assert origins[0] == origins[1]
    assert len(origins[0]) == 365
    # Origins remain 08:00 local across both clock changes, not fixed 24h UTC steps.
    assert {ts.tz_convert("Europe/Paris").hour for ts in origins[0]} == {8}
    assert {ts.hour for ts in origins[0]} == {6, 7}


@pytest.mark.parametrize("zone", noaa.ZONES)
def test_country_mapping_keeps_one_schema_without_proxy_aliases(tmp_path, zone):
    source = noaa.noaa_source(tmp_path / "weather.parquet", tmp_path / "audit.json", zone)
    assert source.name == "noaa_gfs_weather"
    assert source.family == "weather"
    assert source.value_columns == {
        f"local_gfs_{name}": f"{zone.lower()}_gfs_{name}" for name in noaa.VARIABLES
    }
    assert source.route.consumers == ("chronos",)
    assert source.timestamp_column == "delivery_start_utc"
    assert source.cutoff_column == "cutoff_utc"
    assert source.information_time_columns == ("run_init_utc", "publication_max_utc")
    assert source.age_column == "publication_max_utc"
    assert source.production_evidence_kind is None
    assert source.historical_evidence_manifest_path is None


@pytest.fixture
def small_project(tmp_path, monkeypatch):
    """Real canonical contracts and audited Parquet, without mocking the data path."""
    end_day = "2026-03-30"  # Includes the 23-hour spring delivery in the holdout.
    original_period = noaa.panel_period

    def small_period(*, mode, end_day, context_length=24):
        result = original_period(mode=mode, end_day=end_day, context_length=context_length)
        days = 10 if mode == "training" else 13
        start = pd.Timestamp(end_day) - pd.Timedelta(days=days - 1)
        margin = (context_length + 23) // 24 + 2
        return {
            **result, "start_day": str(start.date()), "origin_days": days,
            "weather_start_day": str((start - pd.Timedelta(days=margin)).date()),
            "holdout_days": 2,
            "holdout_start_day": str((pd.Timestamp(end_day) - pd.Timedelta(days=1)).date()),
        }

    monkeypatch.setattr(noaa, "panel_period", small_period)
    period = small_period(mode="calibration", end_day=end_day, context_length=24)
    index = delivery_utc_index(period["weather_start_day"], end_day)
    zone_config = {}
    for zone in noaa.ZONES:
        domain = "de_lu" if zone == "DE" else zone.lower()
        series = f"power.price.da.{domain}.bzn.hourly.entsoe.utc.cdh.eurmwh"
        zone_config[zone] = {"target": {"series": series, "naive_timezone": "UTC"}}
        live = {"live": {"base_config": "base.yaml", "target_series": series}}
        (tmp_path / f"chronos2_hourly_{zone.lower()}_mkonline_live_v1.yaml").write_text(
            yaml.safe_dump(live), encoding="utf-8",
        )
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"data": {"cache_dir": "data/cache"}, "zones": zone_config}),
        encoding="utf-8",
    )
    target_paths = {}
    for number, zone in enumerate(noaa.ZONES):
        target, _ = noaa._canonical_target_path(tmp_path, zone)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "timestamp": index.astype(str),
            "value": 40.0 + number * 10 + np.arange(len(index)) / 100,
        }).to_csv(target, index=False, compression="gzip")
        target_paths[zone] = target

    cutoffs = cutoff_by_delivery_hour(index)
    weather_frame = pd.DataFrame({
        "delivery_start_utc": index,
        "cutoff_utc": cutoffs,
        "run_init_utc": cutoffs.normalize(),
        "publication_max_utc": cutoffs - pd.Timedelta(minutes=30),
    })
    expected = {}
    for number, zone in enumerate(noaa.ZONES):
        expected[zone] = dict(zip(WEATHER_COLUMNS, (10.0 + number, 4.0 + number, 100.0 + number)))
        for alias, value in expected[zone].items():
            weather_frame[alias.replace("local_", f"{zone.lower()}_", 1)] = value
    weather = tmp_path / "synthetic_noaa.parquet"
    weather_audit = weather.with_suffix(".manifest.json")

    def persist_weather(frame):
        frame.to_parquet(weather, index=False)
        weather_audit.write_text(json.dumps({
            "dataset_sha256": noaa.file_sha256(weather),
            "output_sha256": noaa.file_sha256(weather),
            "evidence_kind": "synthetic_test_fixture_not_archive_evidence",
            "local_prospective_capture": False,
            "production_pit_evidence": False,
            "production_pipeline_evidence": False,
            "promotion_eligible": False,
        }), encoding="utf-8")

    persist_weather(weather_frame)
    return SimpleNamespace(
        root=tmp_path, end_day=end_day, weather=weather, weather_audit=weather_audit,
        weather_frame=weather_frame, persist_weather=persist_weather,
        expected=expected, target_paths=target_paths,
    )


def _build(project, mode="training", output=None):
    return noaa.build_noaa_panel(
        root=project.root, weather=project.weather, weather_audit=project.weather_audit,
        output=output or project.root / "inputs" / f"{mode}.parquet",
        mode=mode, end_day=project.end_day, context_length=24,
    )


def _config(project, panel):
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["project_root"] = str(project.root)
    payload["data"].update({
        "panel_path": str(panel), "context_length": 24,
        "training_window_days": 8, "validation_days": 2, "evaluation_days": 2,
    })
    payload["output"]["directory"] = str(project.root / "unused_artifact")
    config_path = project.root / f"{Path(panel).stem}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_config(config_path)


def test_real_bank_panel_and_lora_validation_preserve_country_values_and_research_gates(small_project):
    before = {path: noaa.file_sha256(path) for path in small_project.target_paths.values()}
    result = _build(small_project)
    panel = pd.read_parquet(result["panel"])
    audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
    config = _config(small_project, result["panel"])
    selected, split, validation = validate_panel(panel, config)

    assert result["rows"] == 4 * (10 * (24 + 24) - 1)
    assert len(split.train) == 6 and len(split.validation) == 2 and len(split.evaluation) == 2
    assert audit["horizon_day_counts"] == {"24": 9, "23": 1}
    assert set(panel["item_id"]) == set(noaa.ZONES)
    assert set(audit["feature_columns"]) == set((*WEATHER_COLUMNS, *CALENDAR, *QUALITY_COLUMNS))
    assert config.known_future_covariates == (*WEATHER_COLUMNS, *CALENDAR)
    assert not set(QUALITY_COLUMNS).intersection(selected.columns)
    for zone, expected in small_project.expected.items():
        country = selected.loc[selected["item_id"].eq(zone)]
        for column, value in expected.items():
            assert country[column].eq(value).all()
        assert result["target_coverage"][zone]["missing_hours"] == 0
        assert Path(audit["target_contracts"][zone]["cache_path"]) == small_project.target_paths[zone]
        assert set(audit["exogenous_banks"][zone]["source_hashes"]) == {"noaa_gfs_weather", "deterministic_calendar"}

    assert before == {path: noaa.file_sha256(path) for path in before}
    assert audit["panel_sha256"] == noaa.file_sha256(Path(result["panel"]))
    assert audit["canonical_target_contracts_verified"] is True
    assert audit["target_publication_evidence"] == "not_attested_latest_canonical_cache"
    assert audit["pack"] == "noaa_weather"
    assert audit["purpose"] == "noaa_lora_research"
    assert audit["promotion_eligible"] is False
    assert audit["production_pipeline_evidence"] is False
    assert audit["production_ready"] is False
    assert audit["production_pit_evidence"] == dict.fromkeys(noaa.ZONES, False)
    assert config.evaluation_role == "diagnostic_only"
    assert config.production_pit_evidence is False
    assert validation["production_pit_evidence"] is False
    with pytest.raises(ExogenousFineTuneError, match="production_pit_evidence=true"):
        validate_panel(panel, replace(config, production_pit_evidence=True))

    # Published historical D+1 labels are evaluation-only, never model inputs.
    group = selected.loc[
        selected["item_id"].eq("FR") & selected["origin_timestamp"].eq(split.evaluation[0])
    ]
    inputs, delivery, actuals = build_inference_input(group, config)
    assert inputs["target"].shape == (1, 24)
    assert len(delivery) == 23 and actuals.shape == (1, 23)
    changed = group.copy()
    changed.loc[changed["timestamp"].isin(delivery), "target"] += 1000
    other_inputs, _, other_actuals = build_inference_input(changed, config)
    np.testing.assert_array_equal(inputs["target"], other_inputs["target"])
    np.testing.assert_array_equal(other_actuals, actuals + 1000)


def test_training_and_calibration_integration_keep_the_same_holdout(small_project):
    splits = []
    for mode, total in (("training", 10), ("calibration", 13)):
        result = _build(small_project, mode)
        frame = pd.read_parquet(result["panel"])
        selected, split, audit = validate_panel(frame, _config(small_project, result["panel"]))
        assert audit["origins_total"] == total
        assert audit["origins_used"] == 10
        assert selected["origin_timestamp"].nunique() == 10
        splits.append(split)
    assert splits[0] == splits[1]


@pytest.mark.parametrize("column", ("publication_max_utc", "run_init_utc"))
def test_late_weather_information_is_rejected_without_publishing(small_project, column):
    frame = small_project.weather_frame.copy()
    row = frame.index[-1]
    frame.loc[row, column] = frame.loc[row, "cutoff_utc"] + pd.Timedelta(seconds=1)
    small_project.persist_weather(frame)
    output = small_project.root / "inputs" / "rejected.parquet"
    with pytest.raises(ExogenousBankError, match=f"{column}.*posterieure au cutoff"):
        _build(small_project, output=output)
    assert not output.exists()
    assert not output.with_suffix(".parquet.audit.json").exists()
    assert list(output.parent.iterdir()) == []


@pytest.mark.parametrize("existing", ("panel", "audit", "both"))
def test_existing_outputs_are_preserved_byte_for_byte(small_project, existing):
    output = small_project.root / "inputs" / "existing.parquet"
    sidecar = output.with_suffix(".parquet.audit.json")
    output.parent.mkdir()
    protected = {}
    for name, path in (("panel", output), ("audit", sidecar)):
        if existing in (name, "both"):
            protected[path] = f"user-owned-{name}".encode()
            path.write_bytes(protected[path])
    with pytest.raises(GfsError, match="Final outputs already exist"):
        _build(small_project, output=output)
    assert {path: path.read_bytes() for path in protected} == protected
    assert set(output.parent.iterdir()) == set(protected)


def test_missing_canonical_target_hour_is_not_filled_from_another_cache(small_project):
    canonical = small_project.target_paths["NL"]
    target = pd.read_csv(canonical)
    target.iloc[:-1].to_csv(canonical, index=False, compression="gzip")
    # A complete unrelated cache must never replace the contracted series.
    target.to_csv(canonical.with_name("target__unrelated.csv.gz"), index=False, compression="gzip")
    with pytest.raises(OriginPanelError, match="NL: 1 heures cibles canoniques manquantes"):
        _build(small_project)
    assert not (small_project.root / "inputs").exists()


def test_committed_candidate_configuration_is_isolated_and_diagnostic():
    config = load_config(CONFIG)
    assert config.evaluation_role == "diagnostic_only"
    assert config.production_pit_evidence is False
    assert config.context_length == 2048
    assert config.training_window_days == config.evaluation_days == 365
    assert config.validation_days == 30
    assert config.known_future_covariates == (*WEATHER_COLUMNS, *CALENDAR)
    assert config.past_only_covariates == ()
    assert config.lora_config["r"] == 16
    assert "chronos2_exogenous_noaa_lora_v1" in config.panel_path.parts
    assert "chronos2_exogenous_noaa_lora_v1" in config.output_directory.parts
