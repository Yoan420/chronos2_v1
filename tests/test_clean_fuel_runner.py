"""Boundaries of the isolated fuel ablation: no live writes or changed labels."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from nyx_clean_fuel import features as f
from nyx_clean_fuel import runner as r


def bank_for(index):
    rows = []
    for day in index.tz_convert("Europe/Paris").strftime("%Y-%m-%d").unique():
        cutoff = (pd.Timestamp(day) - pd.DateOffset(days=1)).replace(hour=8).tz_localize("Europe/Paris").tz_convert("UTC")
        stamp = (cutoff.tz_convert("Europe/Paris").normalize() - pd.DateOffset(days=1)).tz_convert("UTC")
        row = {"delivery_day": day, "cutoff_time_utc": cutoff}
        for n, alias in enumerate(f.ALIASES):
            row.update({alias: 100.0 + n, alias + "__value_time_utc": stamp,
                        alias + "__age_hours": (cutoff - stamp).total_seconds() / 3600})
        rows.append(row)
    return pd.DataFrame(rows)


def day_index(day):
    start = pd.Timestamp(day, tz="Europe/Paris")
    return pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25), ("2026-09-17", 24)])
@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_features_keep_physical_dst_hours_and_native_costs(day, hours, zone):
    index = day_index(day)
    base = pd.DataFrame({"q50": 200.0}, index=index)
    bank = bank_for(index)
    actual = f.build_features(bank, index, zone=zone, timezone="Europe/Paris", base=base)
    assert len(actual) == hours
    assert actual.index.equals(index)
    assert len(actual.columns) == 15
    for alias in f.ALIASES:
        # Native values already include EUA and efficiency, never rescale them.
        np.testing.assert_array_equal(actual[f.PREFIX + alias], np.repeat(bank[alias].iloc[0], hours))
    local = bank["cgc_" + zone.lower()].iloc[0]
    assert (actual[f.PREFIX + "chronos_minus_local_cgc"] == 200 - local).all()
    assert (actual[f.PREFIX + "local_gas_minus_coal"] == local - 104).all()
    assert (actual[f.PREFIX + "local_minus_ttf_cgc"] == local - 103).all()
    assert (actual[f.PREFIX + "regional_cgc_spread"] == 3).all()
    assert not any("eua" in col or "actual" in col or "storm" in col for col in actual)


@pytest.mark.parametrize("issue", ["missing_day", "duplicate_day", "nonfinite", "wrong_cutoff", "same_day_close", "stale", "age_mismatch"])
def test_features_fail_closed_on_incomplete_or_noncausal_bank(issue):
    index = day_index("2026-09-17")
    bank = bank_for(index)
    if issue == "missing_day":
        bank.loc[0, "delivery_day"] = "2026-09-16"
    elif issue == "duplicate_day":
        bank = pd.concat([bank, bank], ignore_index=True)
    elif issue == "nonfinite":
        bank.loc[0, "ccc"] = np.nan
    elif issue == "wrong_cutoff":
        bank.loc[0, "cutoff_time_utc"] += pd.Timedelta(hours=1)
    elif issue == "same_day_close":
        bank.loc[0, "ccc__value_time_utc"] = bank.loc[0, "cutoff_time_utc"].normalize()
        bank.loc[0, "ccc__age_hours"] = 6
    elif issue == "stale":
        bank.loc[0, "ccc__value_time_utc"] -= pd.Timedelta(days=10)
        bank.loc[0, "ccc__age_hours"] += 240
    else:
        bank.loc[0, "ccc__age_hours"] += 1
    with pytest.raises(ValueError):
        f.build_features(bank, index, zone="FR", timezone="Europe/Paris", base=pd.DataFrame({"q50": 200}, index=index))


@pytest.mark.parametrize("issue", ["naive", "unordered", "duplicate", "quarterhour", "wrong_base", "nonfinite_base"])
def test_features_require_exact_hourly_base_alignment(issue):
    index = day_index("2026-09-17")
    bank = bank_for(index)
    if issue == "naive":
        index = index.tz_localize(None)
    elif issue == "unordered":
        index = index[::-1]
    elif issue == "duplicate":
        index = index.insert(1, index[0])
    elif issue == "quarterhour":
        index += pd.Timedelta(minutes=15)
    base = pd.DataFrame({"q50": 200.0}, index=index)
    if issue == "wrong_base":
        base.index = index + pd.Timedelta(hours=1)
    elif issue == "nonfinite_base":
        base.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        f.build_features(bank, index, zone="FR", timezone="Europe/Paris", base=base)


def test_existing_residual_builder_really_consumes_all_fuel_columns():
    from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder
    index = day_index("2026-10-25")
    base = pd.DataFrame({"q10": 180., "q50": 200., "q90": 220.}, index=index)
    inputs = f.build_features(bank_for(index), index, zone="DE", timezone="Europe/Berlin", base=base)
    X = inputs.assign(known_fr_nuclear_generation_fcst_gw=40., price_lag_24=999999.)
    experts = base.rename(columns=lambda col: "chronos2__" + col)
    builder = ResidualMetaFeatureBuilder(timezone="Europe/Berlin")
    fitted = builder.fit_transform(X, experts)
    assert set(inputs).issubset(fitted.columns)
    assert "price_lag_24" not in fitted
    for name in inputs:
        np.testing.assert_array_equal(fitted[name], inputs[name])
    transformed = builder.transform(X, experts)
    pd.testing.assert_frame_equal(fitted, transformed)


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "NAMESPACE", tmp_path / "runs/experiments/nyx_clean_fuel_v1")
    return r.NAMESPACE


def test_safe_paths_and_artifact_checksums(scoped):
    artifact = scoped / "2026-09-18/fr/test.json"
    r.write_json(artifact, {"production_modified": False})
    r.verify_files(artifact.parent, {artifact.name: r.sha(artifact)})
    with pytest.raises(ValueError, match="Missing or modified"):
        r.verify_files(artifact.parent, {artifact.name: "bad"})
    for path in (r.ROOT / "runs/exports/result.json", scoped / ".." / "escape.json"):
        with pytest.raises(ValueError, match="Writes must stay"):
            r.write_json(path, {})
    assert not (r.ROOT / "runs/exports/result.json").exists()
    with pytest.raises(ValueError, match="basenames|Writes must stay"):
        r.verify_files(artifact.parent, {"../../../../escape.json": "irrelevant"})


def config():
    return {"schema_version": 1, "output_root": "runs/experiments/nyx_clean_fuel_v1",
            "source_root": "runs/experiments/nuclear_forecast_v1", "zones": ["FR", "DE"],
            "delivery_day": "2026-09-18", "threads": 4, "workers": 2,
            "include_kalman": True, "fuel_sources": {"maximum_age_hours": 176},
            "diagnostic_only": True, "production_modified": False}


def test_configuration_is_opt_in_and_source_locked(scoped, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config()))
    actual = r.load_config(path)
    assert actual == config()
    assert r.source_dir(actual, "FR") == r.ROOT / "runs/experiments/nuclear_forecast_v1/2026-09-18/fr/civil_pit_v2"
    actual["source_root"] = "runs/exports"
    with pytest.raises(ValueError, match="frozen nuclear baseline"):
        r.source_dir(actual, "FR")


@pytest.mark.parametrize("key,value", [("zones", []), ("zones", ["FR", "FR"]), ("zones", ["ES"]),
    ("threads", True), ("threads", 0), ("threads", 33), ("workers", True), ("workers", 0),
    ("workers", 3), ("delivery_day", "2026-09-18T00:00:00"), ("delivery_day", "2026-09-18T08:00:00"),
    ("diagnostic_only", False), ("production_modified", True), ("include_kalman", "false"),
    ("output_root", "runs/exports")])
def test_invalid_configuration_rejected_before_writes(scoped, tmp_path, key, value):
    cfg = config()
    cfg[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        r.load_config(path)
    assert not scoped.exists()


@pytest.fixture
def raw_checkpoint(tmp_path):
    work = tmp_path / "source"
    folder = work / "checkpoints"
    folder.mkdir(parents=True)
    index = pd.date_range("2026-09-12T00:00:00Z", periods=5, freq="h", name="delivery_start_utc")
    full = pd.DataFrame({"forecast_origin_utc": pd.Timestamp("2026-09-11T06:00:00Z"),
                         "q10": np.asarray([50.1234567 + i for i in range(5)], dtype="float32"),
                         "q50": np.asarray([70.1234567 + i for i in range(5)], dtype="float32"),
                         "q90": np.asarray([90.1234567 + i for i in range(5)], dtype="float32"),
                         "actual": np.asarray([80.1234567 + i for i in range(5)], dtype="float32")}, index=index)
    full["forecast_origin_utc"] = full.forecast_origin_utc.astype("datetime64[ns, UTC]")
    path = folder / "nuclear_chronos_oof.csv.gz"
    full.to_csv(path)
    manifest = {"status": "complete", "output_sha256": r.sha(path), "source_hashes": {"features": "frozen"}}
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    source = SimpleNamespace(raw_history=full.iloc[2:].reset_index(), audit={"source_hashes": deepcopy(manifest["source_hashes"])})
    return work, full, source, path, manifest_path


def test_raw_checkpoint_restores_float32_and_exact_prefix(raw_checkpoint):
    work, full, source, *_ = raw_checkpoint
    actual = r.load_raw_history(work, source)
    pd.testing.assert_frame_equal(actual, full, check_freq=False)
    assert actual.index[0] < pd.Timestamp(source.raw_history.delivery_start_utc.iloc[0])


@pytest.mark.parametrize("mutation", ["status", "checksum", "source_hashes", "immutable_overlap"])
def test_raw_checkpoint_rejects_mismatched_frozen_source(raw_checkpoint, mutation):
    work, full, source, path, manifest_path = raw_checkpoint
    manifest = json.loads(manifest_path.read_text())
    if mutation == "status":
        manifest["status"] = "running"
    elif mutation == "checksum":
        manifest["output_sha256"] = "bad"
    elif mutation == "source_hashes":
        manifest["source_hashes"] = {"features": "other"}
    else:
        source.raw_history.loc[0, "q50"] += 0.1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint"):
        r.load_raw_history(work, source)


@pytest.mark.parametrize("include_kalman", [True, False])
@pytest.mark.parametrize("mutation", [None, "manifest", "status", "missing_file", "extra_file", "checksum", "empty"])
def test_results_inventory_and_identity_bound_to_input_snapshot(scoped, include_kalman, mutation):
    directory = scoped / "2026-09-18/fr/candidate"
    directory.mkdir(parents=True)
    cfg = config()
    cfg["include_kalman"] = include_kalman
    manifest = {"identity": {"config": cfg}}
    r.write_json(directory / "manifest.json", manifest)
    names = {"candidate_residual.parquet", "candidate_forecast.parquet", "residual_fit_audit.parquet"}
    if include_kalman:
        names.update({"candidate_kalman.parquet", "candidate_kalman_forecast.parquet", "kalman_audit.json"})
    for name in names:
        r.write_json(directory / name, {"fixture": "content checksum only, not deserialized"})
    files = {name: r.sha(directory / name) for name in names}
    result = {"files": files, "manifest_sha256": r.sha(directory / "manifest.json"), "status": "complete"}
    if mutation == "manifest":
        result["manifest_sha256"] = "different"
    elif mutation == "status":
        result["status"] = "running"
    elif mutation == "missing_file":
        files.pop("candidate_residual.parquet")
    elif mutation == "extra_file":
        files["unexpected.parquet"] = "bad"
    elif mutation == "checksum":
        files["candidate_residual.parquet"] = "bad"
    elif mutation == "empty":
        result["files"] = {}
    r.write_json(directory / "results.json", result)
    if mutation is None:
        assert r.verify_results(directory, manifest) == result
    else:
        with pytest.raises(ValueError, match="manifest mismatch|Missing or modified"):
            r.verify_results(directory, manifest)


def test_report_panel_uses_latest_exact_365_days_and_preserves_future_missing_actual(scoped):
    directory = scoped / "2026-09-18/fr/candidate"
    directory.mkdir(parents=True)
    cfg = config()
    cfg["include_kalman"] = True
    index = pd.date_range(pd.Timestamp("2024-09-18", tz="Europe/Paris"),
                          pd.Timestamp("2026-09-18", tz="Europe/Paris"), freq="h", inclusive="left").tz_convert("UTC")
    future_index = day_index("2026-09-18")
    index.name = future_index.name = "delivery_start_utc"
    labels = pd.DataFrame({"actual": 100.0, "storm_q50": 110.0}, index=index.append(future_index))
    labels.loc[future_index, "actual"] = np.nan
    frames = {"labels": labels}
    for prefix in ("reference", "candidate"):
        for kind, qprefix in (("residual", "residual_corrected__"), ("kalman", "residual_kalman__")):
            future_name = prefix + ("_forecast" if kind == "residual" else "_kalman_forecast")
            for name, idx in ((prefix + "_" + kind, index), (future_name, future_index)):
                frames[name] = pd.DataFrame({qprefix + "q10": 80., qprefix + "q50": 100., qprefix + "q90": 120.}, index=idx)
    frames["residual_fit_audit"] = pd.DataFrame({"generation_source": ["daily_prequential_refit"]})
    for name, frame in frames.items():
        frame.to_parquet(directory / (name + ".parquet"))
    inputs = {name + ".parquet": r.sha(directory / (name + ".parquet")) for name in frames
              if name.startswith("reference") or name == "labels"}
    manifest = {"identity": {"config": cfg, "zone": "FR"}, "files": inputs}
    r.write_json(directory / "manifest.json", manifest)
    outputs = {name + ".parquet": r.sha(directory / (name + ".parquet")) for name in frames
               if name.startswith("candidate") or name == "residual_fit_audit"}
    r.write_json(directory / "kalman_audit.json", {"fixture": True})
    outputs["kalman_audit.json"] = r.sha(directory / "kalman_audit.json")
    r.write_json(directory / "results.json", {"files": outputs, "status": "complete", "manifest_sha256": r.sha(directory / "manifest.json")})
    panel = r.panel_for_report(directory)
    history, future = panel.loc[panel.phase.eq("history")], panel.loc[panel.phase.eq("future")]
    assert len(history) == 4 * 8760
    assert history.delivery_start_utc.min() == pd.Timestamp("2025-09-18", tz="Europe/Paris").tz_convert("UTC")
    assert history.delivery_start_utc.max() == pd.Timestamp("2026-09-17T23:00:00", tz="Europe/Paris").tz_convert("UTC")
    assert len(future) == 4 * 24
    assert future.actual.isna().all()
    assert (history.actual == 100).all()
