"""Real file/PIT preparation smoke tests, without network or neural inference."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import run_nuclear_forecast as launcher
from chronos2_hourly import nuclear_forecast as engine
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_modular.common import build_zone_configs
from chronos2_modular.data import prepare_zone_data
from run_chronos2_hourly import _feature_inputs


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL", "ES"])
def test_snapshot_real_zone_config_prepares_full_nuclear_pit_inputs(
    zone: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = launcher.load_settings(launcher.ROOT / "config/nuclear_forecast.yaml")
    source_path = launcher.ROOT / settings["zone_configs"][zone]
    original = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    spec = build_zone_configs(original, [zone], None, None)[0]
    day = pd.Timestamp("2026-10-25")  # delivery includes both autumn folds
    physical = pd.date_range(
        (day - pd.Timedelta(days=820)).tz_localize(spec.timezone),
        (day + pd.Timedelta(days=1)).tz_localize(spec.timezone),
        freq="h", inclusive="left",
    ).tz_convert("UTC")
    # Match the real production target cache: gzip CSV with timestamp/value
    # and explicit local offsets, including both offsets at the autumn fold.
    target_path = tmp_path / "target.csv.gz"
    pd.DataFrame({
        "timestamp": physical.tz_convert(spec.timezone),
        "value": 50.0 + np.sin(np.arange(len(physical)) / 24.0),
    }).to_csv(target_path, index=False, compression="gzip")
    local = physical.tz_convert("Europe/Paris")
    cutoff = (
        local.tz_localize(None).normalize() - pd.Timedelta(days=1)
        + pd.Timedelta(hours=8)
    ).tz_localize("Europe/Paris").tz_convert("UTC")
    pit = pd.DataFrame({
        "value_time_utc": physical,
        "snapshot_time_utc": cutoff,
        "revision_time_utc": cutoff,
        "value": 30.0,
    })
    residual_path = tmp_path / "residual.parquet"
    pit.to_parquet(residual_path, index=False)
    nuclear_path = tmp_path / "nuclear.parquet"
    nuclear_start = (day - pd.Timedelta(days=730)).tz_localize("Europe/Paris").tz_convert("UTC")
    pit.loc[pit.value_time_utc >= nuclear_start].assign(value=45.0).to_parquet(
        nuclear_path, index=False
    )
    nuclear_path.with_name(nuclear_path.name + ".audit.json").write_text("{}", encoding="utf-8")
    inputs = {"target": target_path}
    inputs.update({alias: residual_path for alias, item in spec.covariates.items() if item.enabled})
    inputs[engine.NUCLEAR_ALIAS] = nuclear_path
    settings["project_root"] = tmp_path
    settings["output_root"] = tmp_path / "runs/experiments/nuclear_preparation"
    settings["nuclear_store"] = nuclear_path
    monkeypatch.setattr(launcher, "zone_inputs", lambda settings, code: (deepcopy(original), source_path, inputs))
    monkeypatch.setattr(launcher, "audit_nuclear_store", lambda *args: {"complete": True})
    monkeypatch.setattr(launcher, "resolve_local_model_revision", lambda config: "a" * 40)
    workdir = tmp_path / "runs/experiments/nuclear_preparation" / zone.lower()
    workdir.mkdir(parents=True)
    resolved = launcher.snapshot_config(settings, zone, day, workdir)
    resolved_spec = build_zone_configs(resolved, [zone], None, None)[0]
    data = prepare_zone_data(resolved_spec, resolved, workdir, False, workdir / "prepared")

    assert data.zone == zone
    assert data.timezone == engine.ZONE_TIMEZONES[zone]
    assert resolved["model"]["context_length"] >= 2048
    assert resolved["model"]["revision"] == "a" * 40
    assert engine.NUCLEAR_ALIAS in data.model_context_covariates
    assert engine.NUCLEAR_KNOWN_COLUMN in data.known_future_columns
    target, _, future_covariates, features = _feature_inputs(data, resolved)
    expected_future = local_delivery_day_index(day, timezone=spec.timezone)
    assert len(expected_future) == 25
    assert future_covariates.index.equals(expected_future)
    assert target.index[-1] == expected_future[0] - pd.Timedelta(hours=1)
    assert target.index.get_indexer([nuclear_start])[0] >= 2048
    assert engine.NUCLEAR_KNOWN_COLUMN in features
    selected_index = physical[physical >= nuclear_start]
    covariates = engine._covariates(data, selected_index)
    assert len(covariates) == len(selected_index)
    assert covariates[engine.NUCLEAR_ALIAS].eq(45.0).all()
    assert features.loc[selected_index, engine.NUCLEAR_KNOWN_COLUMN].eq(45.0).all()
    assert data.target.index.tz_convert("UTC")[0] == physical[0]

    # Reach the first runtime request only after the engine validates the real
    # prepared configuration, context support, future grid and feature schema.
    class RuntimeBoundaryReached(Exception):
        pass

    def no_neural_runtime(*args, **kwargs):
        raise RuntimeBoundaryReached

    with pytest.raises(RuntimeBoundaryReached):
        engine.run_nuclear_forecast(
            config=resolved, data=data, zone=zone, delivery_day=day.date(),
            workdir=workdir, runtime_factory=no_neural_runtime,
        )


@pytest.mark.parametrize("invalid", ["naive", "duplicate", "nonfinite"])
def test_target_cache_guard_rejects_ambiguous_or_invalid_physical_rows(
    invalid: str, tmp_path: Path
) -> None:
    frame = pd.DataFrame({
        "timestamp": ["2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00"],
        "value": [40.0, 41.0],
    })
    if invalid == "naive":
        frame.loc[0, "timestamp"] = "2026-10-25 02:00:00"
    elif invalid == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    else:
        frame.loc[0, "value"] = np.nan
    path = tmp_path / "target.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError):
        launcher.validate_target_cache(path)


def test_target_cache_guard_preserves_distinct_autumn_physical_hours(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "timestamp": ["2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00"],
        "value": [40.0, 41.0],
    })
    path = tmp_path / "target.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    before = path.read_bytes()
    launcher.validate_target_cache(path)
    assert path.read_bytes() == before
