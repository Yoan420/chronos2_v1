from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

import chronos2_exogenous.prospective_trial as trial
from chronos2_exogenous.prospective_auxiliary import MARKET_COLUMNS, RAW_MODEL, QUANTILES, ResidualRecipe


TZ = "Europe/Paris"


def _config_file(tmp_path, **overrides):
    config = {
        "version": 1, "project_root": ".", "output_root": "runs/experiments/r16_trial",
        "candidate_config": "candidate.yaml", "zone_artifacts_root": "bundles",
        "calibration_panel": "panel.parquet", "calibration_panel_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64, "zones": ["FR"], "residual_recipe": asdict(ResidualRecipe()),
    }
    config.update(overrides)
    path = tmp_path / "trial.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _raw(start="2025-09-03", days=365):
    first = pd.Timestamp(start, tz=TZ)
    index = pd.date_range(first, first + pd.DateOffset(days=days), freq="h", inclusive="left").tz_convert("UTC")
    local_days = index.tz_convert(TZ).date
    origins = pd.DatetimeIndex([pd.Timestamp(f"{pd.Timestamp(day) - pd.Timedelta(days=1):%Y-%m-%d} 08:00", tz=TZ) for day in local_days]).tz_convert("UTC")
    frame = pd.DataFrame({"delivery_start_utc": index, "forecast_origin_utc": origins, "actual": 52.0})
    for q, value in zip(QUANTILES, (40, 50, 60)):
        frame[f"{RAW_MODEL}__{q}"] = value
    for i, column in enumerate(MARKET_COLUMNS):
        frame[column] = 15 + i
    return frame


@pytest.mark.parametrize(("day", "origin", "deadline"), [
    ("2026-09-09", "2026-09-08T06:00:00+00:00", "2026-09-08T09:45:00+00:00"),
    ("2026-03-29", "2026-03-28T07:00:00+00:00", "2026-03-28T10:45:00+00:00"),
    ("2026-03-30", "2026-03-29T06:00:00+00:00", "2026-03-29T09:45:00+00:00"),
    ("2025-10-26", "2025-10-25T06:00:00+00:00", "2025-10-25T09:45:00+00:00"),
    ("2025-10-27", "2025-10-26T07:00:00+00:00", "2025-10-26T10:45:00+00:00"),
])
def test_emission_window_physical_dst_and_exclusive_deadline(day, origin, deadline):
    result = trial.emission_window(day, now=pd.Timestamp(origin))
    assert result == {"forecast_origin_utc": origin, "deadline_utc": deadline}
    trial.emission_window(day, now=pd.Timestamp(deadline) - pd.Timedelta(nanoseconds=1))
    with pytest.raises(trial.ProspectiveTrialError):
        trial.emission_window(day, now=pd.Timestamp(origin) - pd.Timedelta(nanoseconds=1))
    with pytest.raises(trial.ProspectiveTrialError):
        trial.emission_window(day, now=pd.Timestamp(deadline))


def test_emission_uses_real_clock_by_default(monkeypatch):
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-08T06:10:00Z"))
    assert trial.emission_window("2026-09-09")["forecast_origin_utc"] == "2026-09-08T06:00:00+00:00"
    with pytest.raises(trial.ProspectiveTrialError):
        trial.emission_window("2026-09-08")
    with pytest.raises(trial.ProspectiveTrialError):
        trial.emission_window("2026-09-09", now=pd.Timestamp("2026-09-08T08:30:00"))


@pytest.mark.parametrize("day", ["2026-09-09T01:00", "2026-09-09T00:00Z", "NaT"])
def test_delivery_day_must_be_civil_date(day):
    with pytest.raises(trial.ProspectiveTrialError):
        trial.emission_window(day, now=pd.Timestamp("2026-09-08T06:10:00Z"))


def test_config_paths_resolve_without_creating_output(tmp_path):
    path = _config_file(tmp_path)
    result = trial.load_trial_config(path)
    assert result["output_root"] == str(tmp_path / "runs/experiments/r16_trial")
    assert result["project_root"] == str(tmp_path)
    assert not Path(result["output_root"]).exists()


@pytest.mark.parametrize("overrides", [
    {"version": 2}, {"version": True}, {"zones": []}, {"zones": ["FR", "FR"]},
    {"zones": ["ES"]}, {"zones": {"FR": "unexpected"}},
    {"output_root": "runs/experiments"}, {"output_root": "runs/live"},
    {"output_root": "../outside"},
])
def test_invalid_config_is_rejected(tmp_path, overrides):
    with pytest.raises(trial.ProspectiveTrialError):
        trial.load_trial_config(_config_file(tmp_path, **overrides))


@pytest.mark.parametrize("content", ["[]", "null", "a string"])
def test_non_mapping_configuration_has_clear_error(tmp_path, content):
    path = tmp_path / "trial.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(trial.ProspectiveTrialError):
        trial.load_trial_config(path)


def test_json_and_csv_publication_refuse_overwrite(tmp_path):
    json_path = tmp_path / "manifest.json"
    trial._write_json(json_path, {"immutable": True})
    before = json_path.read_bytes()
    with pytest.raises((trial.ProspectiveTrialError, FileExistsError)):
        trial._write_json(json_path, {"immutable": False})
    assert json_path.read_bytes() == before
    csv_path = tmp_path / "predictions.csv.gz"
    trial._write_frame(csv_path, _raw(days=1))
    before = csv_path.read_bytes()
    with pytest.raises(trial.ProspectiveTrialError, match="Ecrasement"):
        trial._write_frame(csv_path, _raw(days=2))
    assert csv_path.read_bytes() == before


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from chronos2_exogenous import lora_finetune
    path = _config_file(tmp_path)
    config = trial.load_trial_config(path)
    candidate_path = Path(config["candidate_config"])
    candidate_path.write_text("candidate: fixture", encoding="utf-8")
    panel = Path(config["calibration_panel"])
    panel.write_bytes(b"immutable-panel-fixture")
    config["calibration_panel_sha256"] = trial.sha256(panel)
    # Manifest config and config bytes must describe the same SHA.
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["calibration_panel_sha256"] = config["calibration_panel_sha256"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = trial.load_trial_config(path)
    bundle = Path(config["zone_artifacts_root"]) / "FR/artifact"
    bundle.mkdir(parents=True)
    (bundle / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    schema = {"context_length": 4, "known_future_covariates": list(MARKET_COLUMNS)}
    trial._write_json(bundle / "schema.json", schema)
    raw = _raw()
    old = raw.rename(columns={f"{RAW_MODEL}__{q}": f"candidate_{q}" for q in QUANTILES})
    evidence = trial._write_frame(bundle / "evaluation_predictions.csv.gz", old)
    trial._write_json(bundle / "evaluation_manifest.json", {
        "bundle_manifest_sha256": trial.sha256(bundle / "experiment_manifest.json"),
        "item_id": "FR", "comparison": {"residual_corrector_applied": False},
        "artifacts": {"evidence": {"relative_path": "evaluation_predictions.csv.gz", "sha256": evidence["sha256"]}},
    })
    source = tmp_path / "scientific.py"
    source.write_text("# frozen numerical implementation\n", encoding="utf-8")
    monkeypatch.setattr(trial, "CODE_FILES", ("scientific.py",))
    candidate = SimpleNamespace(context_length=4, timezone=TZ, timestamp_column="timestamp",
        origin_column="origin_timestamp", item_column="item_id", feature_available_at_column="feature_available_at_utc",
        target_columns=("target",), covariate_columns=MARKET_COLUMNS, known_future_covariates=MARKET_COLUMNS,
        past_only_covariates=())
    monkeypatch.setattr(lora_finetune, "load_config", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(lora_finetune, "_schema_payload", lambda *_args: schema)
    monkeypatch.setattr(lora_finetune, "verify_bundle", lambda *_args: {
        "checkpoint_sha256": config["checkpoint_sha256"], "training": {"lora_config": {"r": 16}}})
    covs = raw[["delivery_start_utc", "forecast_origin_utc", *MARKET_COLUMNS]].rename(
        columns={"delivery_start_utc": "timestamp", "forecast_origin_utc": "origin_timestamp"})
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: covs.copy())
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-02T05:00:00Z"))
    manifest = trial.prepare_trial(config)
    return config, manifest, raw, candidate


def test_prepare_is_immutable_and_idempotent(prepared):
    config, manifest, _, _ = prepared
    path = Path(config["output_root"]) / "trial_manifest.json"
    before = path.read_bytes()
    reused = trial.prepare_trial(config)
    assert path.read_bytes() == before
    assert reused["historical_role"] == "previously_inspected_calibration_not_independent_test"
    assert reused["models"] == ["lora16_residual", "lora16_residual_kalman"]
    for name, expected in trial.FLAGS.items():
        assert reused[name] is expected
    assert manifest["zones"]["FR"]["calibration"]["sha256"] == reused["zones"]["FR"]["calibration"]["sha256"]


def test_verify_refuses_calibration_or_code_changes(prepared):
    config, manifest, _, _ = prepared
    calibration = Path(manifest["zones"]["FR"]["calibration"]["path"])
    payload = calibration.read_bytes()
    calibration.write_bytes(payload + b"changed")
    with pytest.raises(trial.ProspectiveTrialError, match="Empreinte"):
        trial.verify_trial(config)


def _daily(prepared, *, day="2026-09-03", zone="FR", manifest_zone=None, manifest_day=None,
           prospective=False, foreign_protocol=False):
    config, manifest, _, _ = prepared
    output = Path(config["output_root"])
    directory = output / "days" / day / zone
    directory.mkdir(parents=True)
    raw = _raw(day, 1)
    if prospective:
        raw["actual"] = np.nan
    capture = {"fresh_source_refresh": True, "capture_completed_at_utc": "2026-09-02T06:10:00+00:00",
               "horizon_labels_available_at_capture": {day: {zone: 0 if prospective else 24}}}
    trial._write_json(directory / "input_manifest.json", capture)
    daily = {**trial.FLAGS, "zone": manifest_zone or zone, "delivery_day": manifest_day or day,
             "prospective_eligible": prospective, "completed_at_utc": "2026-09-02T06:30:00+00:00",
             "role": "prospective_two_chain_forecast" if prospective else "retrospective_calibration_only",
             "trial_manifest_sha256": "f" * 64 if foreign_protocol else trial.sha256(output / "trial_manifest.json"),
             "input_manifest": trial._file(directory / "input_manifest.json"),
             "raw_predictions": trial._write_frame(directory / "raw.csv.gz", raw)}
    if prospective:
        predictions = raw.copy()
        predictions["zone"], predictions["prospective_eligible"] = zone, True
        for model in ("lora16_residual", "lora16_residual_kalman"):
            for q in QUANTILES:
                predictions[f"{model}__{q}"] = predictions[f"{RAW_MODEL}__{q}"] + 1
        daily["predictions"] = trial._write_frame(directory / "predictions.csv.gz", predictions)
        trial._write_json(directory / "auxiliary_audit.json", {**trial.FLAGS, "fixture": True})
        daily["auxiliary_audit"] = trial._file(directory / "auxiliary_audit.json")
        daily["prepublication_target_check"] = {"fresh_api_read": True, "fallback_used": False,
            "capture_completed_at_utc": "2026-09-02T06:20:00+00:00"}
        daily["prepublication_observed_hours"] = 0
        daily["emission_window"] = trial.emission_window(day, now=pd.Timestamp(daily["completed_at_utc"]))
    trial._write_json(directory / "manifest.json", daily)
    return daily, raw


def test_load_history_keeps_unresolved_actuals_and_raw_bytes_unchanged(prepared):
    config, manifest, raw, _ = prepared
    daily, added = _daily(prepared, prospective=True)
    raw_path = Path(daily["raw_predictions"]["path"])
    before = raw_path.read_bytes()
    combined = trial.load_history(manifest, "FR")
    assert len(combined) == len(raw) + 24
    assert combined.actual.iloc[-24:].isna().all()
    assert raw_path.read_bytes() == before
    assert combined.delivery_start_utc.is_unique


def test_load_history_rejects_foreign_protocol(prepared):
    _, manifest, _, _ = prepared
    _daily(prepared, foreign_protocol=True)
    with pytest.raises(trial.ProspectiveTrialError):
        trial.load_history(manifest, "FR")


@pytest.mark.parametrize("values", [{"manifest_zone": "DE"}, {"manifest_day": "2026-09-04"}])
def test_load_history_rejects_cross_zone_or_day_manifest(prepared, values):
    _, manifest, _, _ = prepared
    _daily(prepared, **values)
    with pytest.raises(trial.ProspectiveTrialError):
        trial.load_history(manifest, "FR")


def test_actual_attachment_updates_only_labels_and_preserves_inputs():
    raw = _raw(days=1)
    raw["actual"] = np.nan
    before = raw.copy()
    target = pd.Series([100.0, np.nan], index=pd.DatetimeIndex(raw.delivery_start_utc.iloc[:2]))
    result = trial._attach_actuals(raw, target)
    assert result.actual.iloc[0] == 100
    assert result.actual.iloc[1:].isna().all()
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_frame_equal(result.drop(columns="actual"), before.drop(columns="actual"))


def test_execute_prospective_keeps_panel_phase_metadata_and_emits_two_chains(prepared, monkeypatch):
    from chronos2_exogenous import lora_finetune, prospective_inputs, prospective_auxiliary
    config, manifest, _, candidate = prepared
    output = Path(config["output_root"])
    day = "2026-09-03"
    raw = _raw(day, 1)
    horizon = pd.DatetimeIndex(raw.delivery_start_utc)
    context = pd.date_range(end=horizon[0] - pd.Timedelta(hours=1), periods=4, freq="h")
    index = context.append(horizon)
    panel = pd.DataFrame({"timestamp": index, "origin_timestamp": pd.Timestamp("2026-09-02T06:00Z"),
        "item_id": "FR", "feature_available_at_utc": pd.Timestamp("2026-09-02T05:00Z"),
        "target": [50.0] * 4 + [np.nan] * 24,
        "phase": ["context"] * 4 + ["horizon"] * 24, "delivery_day": day})
    for name in MARKET_COLUMNS:
        panel[name] = 20.0
    capture_dir = output / "test_capture"
    capture_dir.mkdir()
    capture_manifest = capture_dir / "manifest.json"
    capture_record = {"fresh_source_refresh": True, "capture_completed_at_utc": "2026-09-02T06:10:00+00:00",
                      "horizon_labels_available_at_capture": {day: {"FR": 0}}}
    trial._write_json(capture_manifest, capture_record)
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-02T06:30Z"))
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: panel.copy())
    monkeypatch.setattr(prospective_inputs, "prepare_trial_inputs", lambda **_kwargs: {
        **capture_record,
        "panel_path": str(capture_dir / "panel.parquet"), "manifest_path": str(capture_manifest),
    })
    monkeypatch.setattr(lora_finetune, "load_checkpoint", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trial, "_targets", lambda *_args: pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")))
    monkeypatch.setattr(prospective_inputs, "_capture_target", lambda **_kwargs: (
        pd.Series([np.nan] * len(horizon), index=horizon), {"fresh_api_read": True, "fallback_used": False,
            "capture_completed_at_utc": "2026-09-02T06:20:00+00:00"}))
    predictions_seen = []
    def predict(group, _candidate, _pipeline):
        assert set(group.phase) == {"context", "horizon"}
        assert set(group.delivery_day) == {day}
        predictions_seen.append(len(group))
        result = raw.copy()
        result["actual"] = np.nan
        return result
    monkeypatch.setattr(trial, "_predict_group", predict)
    def chains(history, future, **_kwargs):
        assert history.actual.notna().all()
        assert future.actual.isna().all()
        result = future.drop(columns="actual").copy()
        for prefix in ("lora16_residual", "lora16_residual_kalman"):
            for q in QUANTILES:
                result[f"{prefix}__{q}"] = future[f"{RAW_MODEL}__{q}"] + 1
        return SimpleNamespace(kalman_future=result, audit={"promotion_eligible": False})
    monkeypatch.setattr(prospective_auxiliary, "forecast_trial_chains", chains)
    result = trial.execute_trial(config, delivery_day=day, zones=["FR"], prospective=True)
    assert result["status"] == "completed"
    assert predictions_seen == [28]
    daily = trial._json(Path(result["published"][0]))
    assert daily["prospective_eligible"] is True
    emitted = trial._read_frame(Path(daily["predictions"]["path"]))
    assert emitted.actual.isna().all()
    assert "lora16_residual__q50" in emitted
    assert "lora16_residual_kalman__q50" in emitted
    trial._verify_daily(Path(result["published"][0]), output=output, expected_zone="FR")
    with pytest.raises(trial.ProspectiveTrialError, match="reecriture"):
        trial.execute_trial(config, delivery_day=day, zones=["FR"], prospective=True)
    assert predictions_seen == [28]


@pytest.mark.parametrize("alteration", ["after_deadline", "known_actual", "labels_already_available",
                                         "wrong_window", "non_fresh_target", "wrong_time_order", "outside_raw_path"])
def test_daily_prospective_reclassification_or_invalid_clock_is_refused(prepared, alteration):
    config, manifest, _, _ = prepared
    daily, _ = _daily(prepared, prospective=True)
    directory = Path(daily["raw_predictions"]["path"]).parent
    if alteration == "after_deadline":
        daily["completed_at_utc"] = "2026-09-02T12:00:00+00:00"
    elif alteration == "known_actual":
        raw = trial._read_frame(Path(daily["raw_predictions"]["path"]))
        raw["actual"] = 999
        raw.to_csv(directory / "raw.csv.gz", index=False, compression="gzip")
        daily["raw_predictions"] = trial._file(directory / "raw.csv.gz")
    elif alteration == "labels_already_available":
        daily["prepublication_observed_hours"] = 1
    elif alteration == "wrong_window":
        daily["emission_window"]["deadline_utc"] = "2026-09-02T16:00Z"
    elif alteration == "non_fresh_target":
        daily["prepublication_target_check"]["fresh_api_read"] = False
    elif alteration == "wrong_time_order":
        daily["prepublication_target_check"]["capture_completed_at_utc"] = "2026-09-02T07:00Z"
    else:
        foreign = directory / "foreign.csv.gz"
        foreign.write_bytes((directory / "raw.csv.gz").read_bytes())
        daily["raw_predictions"] = trial._file(foreign)
    (directory / "manifest.json").write_text(json.dumps(daily), encoding="utf-8")
    with pytest.raises(trial.ProspectiveTrialError):
        trial.load_history(manifest, "FR")


def test_resolve_creates_new_observation_snapshots_without_reforecast_or_mutating_issuance(prepared, monkeypatch):
    from chronos2_exogenous import lora_finetune, prospective_inputs, prospective_reporting
    config, _, _, _ = prepared
    daily, raw = _daily(prepared, prospective=True)
    issuance = Path(daily["predictions"]["path"])
    before = issuance.read_bytes()
    observation = {"price": 51.0}
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Actual resolution must never reload the neural model")
    monkeypatch.setattr(lora_finetune, "load_checkpoint", forbidden)
    monkeypatch.setattr(trial, "_predict_group", forbidden)
    monkeypatch.setattr(prospective_inputs, "_capture_target", lambda **_kwargs: (
        pd.Series(observation["price"], index=pd.DatetimeIndex(raw.delivery_start_utc)), {"fresh_api_read": True}))
    def report(frame, path, **_kwargs):
        path.write_text("<html>research report</html>", encoding="utf-8")
        return path
    monkeypatch.setattr(prospective_reporting, "render_trial_report", report)
    first = trial.resolve_and_report(config)
    first_manifest = trial._json(Path(first["report"]).parent / "report_manifest.json")
    first_evidence = Path(first_manifest["evidence"]["path"])
    before_evidence = first_evidence.read_bytes()
    assert trial._read_frame(first_evidence).actual.eq(51).all()
    observation["price"] = 53.0
    second = trial.resolve_and_report(config)
    assert second["report"] != first["report"]
    second_manifest = trial._json(Path(second["report"]).parent / "report_manifest.json")
    assert trial._read_frame(Path(second_manifest["evidence"]["path"])).actual.eq(53).all()
    assert issuance.read_bytes() == before
    assert first_evidence.read_bytes() == before_evidence
    assert first_manifest["promotion_eligible"] is False
