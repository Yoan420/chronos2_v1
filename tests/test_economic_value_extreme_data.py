from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from economic_value.data import EconomicDataError, TARGET_PATHS, _origin, load_reference_proxy
from economic_value.extreme_data import CORE_PATHS, CORE_ZONES, load_extreme_inputs
from marginal_cost_expert.evaluation import physical_index


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pit(root, key, frame):
    path = root / CORE_PATHS[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    series = ({f"{z.lower()}_residual_load_fcst": f"power.{z.lower()}.residual.load.hourly.gw.fcst" for z in CORE_ZONES}
              if key == "residual_load" else "power.fr.generation.nuclear.gw.fcst")
    audit = {"sha256": _sha(path), "series": series, "cutoff_time": "08:00", "cutoff_timezone": "Europe/Paris",
             "provider_revision_timestamp_available": False, "revision_time_semantics": "query_asof_cutoff"}
    path.with_name(path.name + ".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return path


def _reseal(snapshot):
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["snapshot_files"] = {name: _sha(snapshot / name) for name in manifest["snapshot_files"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    index = physical_index("2024-09-11", "2026-09-10")
    origin = _origin(index, "Europe/Paris")
    basic = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": origin, "revision_time_utc": origin})
    residual = basic.copy()
    for i, zone in enumerate(CORE_ZONES):
        residual[f"{zone.lower()}_residual_load_fcst"] = 40.0 + i * 5.0
    _write_pit(tmp_path, "residual_load", residual)
    _write_pit(tmp_path, "nuclear", basic.assign(value=35.0))
    target_path = tmp_path / TARGET_PATHS["FR"]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_index = physical_index("2024-09-10", "2026-09-10")
    pd.DataFrame({"timestamp": target_index, "value": 100.0}).to_csv(target_path, index=False, compression="gzip")
    eval_index = physical_index("2025-09-11", "2026-09-10")
    panel = pd.DataFrame({"timestamp_utc": eval_index, "zone": "FR", "model": "nuclear_kalman", "sample": "evaluation",
                          "forecast": 120.0, "q10": 110.0, "q90": 130.0, "actual": 125.0, "benchmark_forecast": 118.0,
                          "forecast_origin_utc": _origin(eval_index, "Europe/Paris"), "duration_hours": 1.0,
                          "forecast_eligible": True, "forecast_pit_certified": False, "benchmark_pit_certified": False})
    ref, _ = load_reference_proxy(tmp_path, panel)
    panel = panel.merge(ref, on=["timestamp_utc", "zone"], validate="one_to_one")
    snapshot = tmp_path / "runs/experiments/economic_value_v1/snapshots/frozen"
    snapshot.mkdir(parents=True)
    panel.to_parquet(snapshot / "panel.parquet", index=False)
    for name in ["config.json", "data_audit.json", "reference_audit.json"]:
        (snapshot / name).write_text("{}", encoding="utf-8")
    manifest = {"evaluation_start": "2025-09-11", "evaluation_end": "2026-09-10", "evaluation_days": 365,
                "cutoff_time": "08:00", "zones": ["FR"], "models": ["nuclear_kalman"],
                "snapshot_files": {name: "" for name in ["panel.parquet", "config.json", "data_audit.json", "reference_audit.json"]}}
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _reseal(snapshot)
    config = {"source_snapshot": str(snapshot), "zones": ["FR"], "baseline_model": "nuclear_kalman", "training_days": 365}
    return tmp_path, config, snapshot


def test_730_support_and_exact_frozen_365_evaluation(inputs):
    root, config, snapshot = inputs
    h, p, a = load_extreme_inputs(root, config)
    assert len(h) == 17520 and len(p) == 8760
    assert a["history_start_day"] == "2024-09-11"
    assert a["evaluation_end_day"] == "2026-09-10"
    original = pd.read_parquet(snapshot / "panel.parquet").sort_values("timestamp_utc")
    for col in original:
        assert p[col].reset_index(drop=True).equals(original[col].reset_index(drop=True)), col
    assert a["evaluation_prices_forecasts_references_unchanged"] is True
    assert a["forecast_and_label_vintages_certified"] is False
    assert not h.label_eligible.eq(False).any()


def test_frozen_evaluation_labels_used_for_later_training_not_refreshed_cache(inputs):
    root, config, _ = inputs
    h, p, _ = load_extreme_inputs(root, config)
    days = h.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    assert h.loc[days.lt("2025-09-11"), "actual"].eq(100).all()
    assert h.loc[days.ge("2025-09-11"), "actual"].eq(125).all()
    assert p.actual.eq(125).all()
    assert h.feature_previous_da_price_eur_mwh.dropna().eq(100).all()


def test_actual_and_incumbent_changes_do_not_enter_expert_features(inputs):
    root, config, snapshot = inputs
    before, _, a = load_extreme_inputs(root, config)
    panel = pd.read_parquet(snapshot / "panel.parquet")
    panel[["actual", "forecast", "benchmark_forecast", "q10", "q90"]] += 999999
    panel.to_parquet(snapshot / "panel.parquet", index=False)
    _reseal(snapshot)
    after, _, _ = load_extreme_inputs(root, config)
    assert before[a["feature_columns"]].equals(after[a["feature_columns"]])
    assert not any("storm" in c or "benchmark" in c or "actual" in c for c in a["feature_columns"])


def test_missing_core_hour_is_abstention_preserving_full_year(inputs):
    root, config, _ = inputs
    path = root / CORE_PATHS["residual_load"]
    frame = pd.read_parquet(path)
    target = frame.value_time_utc.eq(pd.Timestamp("2026-06-24T10:00Z"))
    frame.loc[target, "be_residual_load_fcst"] = np.nan
    _write_pit(root, "residual_load", frame)
    h, p, a = load_extreme_inputs(root, config)
    assert len(p) == 8760 and p.forecast.notna().all()
    row = p.loc[p.timestamp_utc.eq(pd.Timestamp("2026-06-24T10:00Z"))].iloc[0]
    assert not row.feature_eligible
    assert row.feature_be_residual_load_gw_missing == 1
    assert a["features"]["residual_load"]["missing_hours_by_feature"]["feature_be_residual_load_gw"] == 1


def test_post08_vintage_excluded_not_reinterpreted(inputs):
    root, config, _ = inputs
    path = root / CORE_PATHS["nuclear"]
    frame = pd.read_parquet(path)
    mask = frame.value_time_utc.eq(pd.Timestamp("2026-06-24T10:00Z"))
    frame.loc[mask, "revision_time_utc"] += pd.Timedelta(hours=3)
    _write_pit(root, "nuclear", frame)
    _, p, a = load_extreme_inputs(root, config)
    row = p.loc[p.timestamp_utc.eq(pd.Timestamp("2026-06-24T10:00Z"))].iloc[0]
    assert not row.feature_eligible and pd.isna(row.feature_fr_nuclear_generation_gw)
    assert a["features"]["nuclear"]["late_vintage_rows_excluded"] == 1
    assert a["selected_cutoff_violations"] == 0


def test_future_revision_does_not_replace_valid_prior_vintage(inputs):
    root, config, _ = inputs
    frame = pd.read_parquet(root / CORE_PATHS["nuclear"])
    late = frame.iloc[[100]].copy()
    late["revision_time_utc"] += pd.Timedelta(hours=3)
    late["value"] = 999999
    _write_pit(root, "nuclear", pd.concat([frame, late], ignore_index=True))
    h, _, _ = load_extreme_inputs(root, config)
    assert h.feature_fr_nuclear_generation_gw.eq(35).all()


def test_optional_missing_sources_do_not_invalidate_core(inputs):
    root, config, _ = inputs
    h, _, _ = load_extreme_inputs(root, config)
    extended, _, a = load_extreme_inputs(root, {**config, "include_optional_fuels": True, "include_optional_weather": True})
    assert extended.feature_eligible.equals(h.feature_eligible)
    assert extended.feature_ttf_m1_eur_mwh_th.isna().all()
    assert extended.feature_ttf_m1_eur_mwh_th_missing.eq(1).all()
    assert extended.feature_local_temperature_fcst.isna().all()
    assert a["optional_missing_policy"].startswith("NaN")


def test_target_and_cutoff_times_are_explicit_civil_instants(inputs):
    root, config, _ = inputs
    h, _, _ = load_extreme_inputs(root, config)
    row = h.loc[h.timestamp_utc.eq(pd.Timestamp("2026-03-30T10:00Z"))].iloc[0]
    assert row.forecast_origin_utc == pd.Timestamp("2026-03-29T06:00Z")
    assert row.label_available_at_utc == pd.Timestamp("2026-03-29T16:00Z")
    assert row.label_publication_time_assumed
    assert row.label_available_at_utc > row.forecast_origin_utc


def test_snapshot_tampering_refused(inputs):
    root, config, snapshot = inputs
    (snapshot / "config.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(EconomicDataError, match="snapshot checksum"):
        load_extreme_inputs(root, config)


def test_feature_source_checksum_tampering_refused(inputs):
    root, config, _ = inputs
    path = root / CORE_PATHS["nuclear"]
    f = pd.read_parquet(path).assign(value=99)
    f.to_parquet(path, index=False)
    with pytest.raises(EconomicDataError, match="PIT artifact checksum"):
        load_extreme_inputs(root, config)


def test_price_series_cannot_be_disguised_as_nuclear_forecast(inputs):
    root, config, _ = inputs
    path = root / (CORE_PATHS["nuclear"] + ".audit.json")
    audit = json.loads(path.read_text())
    audit["series"] = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(EconomicDataError, match="series identity"):
        load_extreme_inputs(root, config)


def test_source_naive_delivery_refused(inputs):
    root, config, _ = inputs
    f = pd.read_parquet(root / CORE_PATHS["nuclear"])
    f["value_time_utc"] = f.value_time_utc.dt.tz_localize(None)
    _write_pit(root, "nuclear", f)
    with pytest.raises(EconomicDataError, match="timezone-aware"):
        load_extreme_inputs(root, config)


def test_shorter_training_period_refused(inputs):
    root, config, _ = inputs
    with pytest.raises(EconomicDataError, match="365-day"):
        load_extreme_inputs(root, {**config, "training_days": 180})


def test_frozen_nan_is_not_filled_from_current_canonical_cache(inputs):
    root, config, snapshot = inputs
    panel = pd.read_parquet(snapshot / "panel.parquet")
    stamp = pd.Timestamp("2026-06-24T10:00Z")
    mask = panel.timestamp_utc.eq(stamp)
    panel.loc[mask, ["actual", "reference_price"]] = np.nan
    panel.loc[mask, "reference_eligible"] = False
    panel.to_parquet(snapshot / "panel.parquet", index=False)
    _reseal(snapshot)
    h, p, _ = load_extreme_inputs(root, config)
    for frame in (h, p):
        row = frame.loc[frame.timestamp_utc.eq(stamp)].iloc[0]
        assert pd.isna(row.actual) and pd.isna(row.reference_price)
        assert not row.feature_eligible


def test_conflicting_same_vintage_refused(inputs):
    root, config, _ = inputs
    frame = pd.read_parquet(root / CORE_PATHS["nuclear"])
    conflicting = frame.iloc[[100]].copy().assign(value=1000.0)
    _write_pit(root, "nuclear", pd.concat([frame, conflicting], ignore_index=True))
    with pytest.raises(EconomicDataError, match="conflicting PIT"):
        load_extreme_inputs(root, config)


def test_fuel_observation_after_cutoff_is_not_used(inputs):
    from economic_value.extreme_data import _pit
    root, _, _ = inputs
    index = physical_index("2026-06-24", "2026-06-24")
    origin = _origin(index, "Europe/Paris")
    frame = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": origin, "revision_time_utc": origin,
                          "ttf_source_value_time_utc": origin + pd.Timedelta(hours=2),
                          "eua_source_value_time_utc": origin - pd.Timedelta(hours=20), "ttf_m1_eur_mwh_th": 50.0})
    path = root / "fuel.parquet"
    frame.to_parquet(path, index=False)
    audit = {"sha256": _sha(path), "cutoff_time": "08:00", "cutoff_timezone": "Europe/Paris",
             "series": {"ttf_m1_eur_mwh_th": "gas.ttf.price.everyday.month.1.ice.eurmwh",
                        "eua_first_dec_eur_tco2": "carbon.eu.price.everyday.eua.ice.1st.dec"}}
    path.with_name(path.name + ".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    values, evidence = _pit(root, "fuels", str(path), {"ttf_m1_eur_mwh_th": "feature_ttf"}, index)
    assert values.feature_ttf.isna().all()
    assert evidence["late_fuel_observation_rows_excluded"] == 24
