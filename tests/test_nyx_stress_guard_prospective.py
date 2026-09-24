"""Prospective interfaces use fixed models and strictly earlier sealed evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_stress_guard import ledger, prospective as lab
from nyx_scarcity import policy as base

FEATURE = "feature_fundamental_x"


def panel_for(day, zones=("DE", "FR"), *, actual=np.nan):
    origin = ledger._origin(day)
    result = pd.concat([pd.DataFrame({"zone": z, "timestamp_utc": ledger._index(day),
        "forecast_origin_utc": origin, "forecast": 100., "q10": 80., "q90": 120.,
        FEATURE: 2., "feature_eligible": True, "forecast_eligible": True,
        "feature_available_at_utc": origin, "actual": actual,
        "label_available_at_utc": origin+pd.Timedelta(hours=10), "label_eligible": not np.isnan(actual)})
        for z in zones], ignore_index=True)
    return result


def historical(day, zones=("DE", "FR")):
    result = panel_for(day, zones, actual=110.)
    return result.assign(candidate_forecast=105., candidate_q10=80., candidate_q90=120.,
        precalibration_q10=80., precalibration_q90=120., bounded_correction=10.,
        raw_correction=10., expert_ready=True, intervention_active=True, threshold_eur_mwh=50.)


@pytest.fixture
def history_case(tmp_path, monkeypatch):
    history = pd.concat([historical("2026-09-14"), historical("2026-09-15")], ignore_index=True)
    # A source revision received after the future decision must not reach any helper.
    history.loc[0, "label_available_at_utc"] = pd.Timestamp("2026-09-17T00:00:00Z")
    path = tmp_path/"oos.parquet"; history.to_parquet(path, index=False)
    manifest = {"history_files": {"oos": {"path": str(path), "sha256": ledger._digest(path)}}}
    events = []
    monkeypatch.setattr(ledger, "_read", lambda directory, root: (tmp_path, manifest, events))
    monkeypatch.setattr(ledger, "_verify_files", lambda root, entries: None)
    return history, manifest, events


def add_events(tmp_path, events, *, day="2026-09-16", received="2026-09-16T10:00:00Z"):
    forecast = historical(day)
    # Even if someone supplied actuals in this fixture, resolved_history must erase them.
    predicted = tmp_path/(day+"_predictions.parquet"); forecast.to_parquet(predicted, index=False)
    issued = ledger._origin(day)+pd.Timedelta(hours=2)
    event = {"kind": "forecast", "delivery_day": day, "forecast_origin_utc": ledger._origin(day).isoformat(),
        "issued_at_utc": issued.isoformat(), "event_sha256": "matching_issue",
        "artifacts": {"predictions": {"path": str(predicted)}}}
    observed = tmp_path/(day+"_observations.parquet")
    forecast.loc[forecast.zone.eq("DE"), ["zone", "timestamp_utc", "actual"]].assign(actual=150.).to_parquet(observed, index=False)
    resolution = {"kind": "observation_resolution", "delivery_day": day, "zone": "DE",
        "forecast_event_sha256": "matching_issue", "observations_received_at_utc": received,
        "artifacts": {"observations": {"path": str(observed)}}}
    events.extend([event, resolution])
    return event, resolution


def test_resolved_history_masks_source_future_labels_and_excludes_current_origin(history_case, tmp_path):
    result = lab.resolved_history(tmp_path, root=tmp_path, cutoff=ledger._origin("2026-09-15"))
    assert result.forecast_origin_utc.lt(ledger._origin("2026-09-15")).all()
    assert pd.isna(result.iloc[0].actual) and pd.isna(result.iloc[0].label_available_at_utc)
    assert not result.iloc[0].label_eligible
    assert result.actual.notna().any()


def test_only_resolved_country_and_pre_cutoff_receipt_are_visible(history_case, tmp_path):
    _, _, events = history_case
    add_events(tmp_path, events)
    cutoff = ledger._origin("2026-09-17")
    before = lab.resolved_history(tmp_path, root=tmp_path, cutoff=cutoff)
    block = before.loc[before.timestamp_utc.isin(ledger._index("2026-09-16"))]
    assert block.actual.isna().all()  # 10 UTC receipt is later than 06 UTC decision.
    after = lab.resolved_history(tmp_path, root=tmp_path, cutoff=ledger._origin("2026-09-18"))
    block = after.loc[after.timestamp_utc.isin(ledger._index("2026-09-16"))]
    assert block.loc[block.zone.eq("DE"), "actual"].eq(150.).all()
    assert block.loc[block.zone.eq("FR"), "actual"].isna().all()


def test_late_issued_forecast_is_not_evidence_at_earlier_cutoff(history_case, tmp_path):
    _, _, events = history_case
    event, _ = add_events(tmp_path, events)
    event["issued_at_utc"] = "2026-09-17T12:00:00Z"
    result = lab.resolved_history(tmp_path, root=tmp_path, cutoff=ledger._origin("2026-09-17"))
    assert not result.timestamp_utc.isin(ledger._index("2026-09-16")).any()


@pytest.mark.parametrize("failure", ["foreign_issue", "early_receipt", "wrong_zone", "duplicate", "missing_hour", "extra_hour", "overlap"])
def test_inconsistent_resolution_cannot_enter_governance(history_case, tmp_path, failure):
    _, _, events = history_case
    event, receipt = add_events(tmp_path, events)
    if failure == "foreign_issue":
        receipt["forecast_event_sha256"] = "other"
    elif failure == "early_receipt":
        receipt["observations_received_at_utc"] = event["issued_at_utc"]
    elif failure == "overlap":
        events.append(deepcopy(event))
    else:
        path = Path(receipt["artifacts"]["observations"]["path"])
        frame = pd.read_parquet(path)
        if failure == "wrong_zone":
            frame["zone"] = "FR"
        elif failure == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]])
        elif failure == "extra_hour":
            extra = frame.iloc[:1].copy(); extra["timestamp_utc"] += pd.Timedelta(days=1)
            frame = pd.concat([frame, extra])
        else:
            frame = frame.iloc[1:]
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        lab.resolved_history(tmp_path, root=tmp_path, cutoff=ledger._origin("2026-09-18"))


@pytest.fixture
def prediction_case(tmp_path, monkeypatch):
    p = base._parameters({"feature_columns": [FEATURE], "required_feature_columns": [FEATURE], "threads": 1})
    p["correction_clip_eur_mwh"] = 400.
    state = {"fit_day": "2026-09-15", "zones": ("DE", "FR")}
    manifest = {"candidate_files": {"model": {"path": str(tmp_path/"latest_model.joblib")},
                                   "suite_manifest": {"path": str(tmp_path/"manifest.json")}}}
    calls = {"verification": 0, "model_load": 0, "predict_rows": [], "govern": [], "calibration": []}
    def verify(root, entries):
        calls["verification"] += 1
    def load(path):
        assert calls["verification"] > 0
        calls["model_load"] += 1
        return {"state": state, "settings": p}
    monkeypatch.setattr(ledger, "_verify_files", verify)
    monkeypatch.setattr(lab, "read_suite", lambda *a, **kw: (tmp_path, {}, {}))
    monkeypatch.setattr(lab, "verify_result", lambda *a: None)
    monkeypatch.setattr(lab.joblib, "load", load)
    monkeypatch.setattr(lab, "make_stress_features", lambda frame: (frame.copy(), [FEATURE], [FEATURE], {}))
    past = pd.concat([historical("2026-09-14"), historical("2026-09-15")], ignore_index=True)
    # Known eligible labels are all before cutoff. Missing labels stay missing.
    past.loc[0, "actual"] = np.nan; past.loc[0, "label_eligible"] = False
    monkeypatch.setattr(lab, "resolved_history", lambda *a, **kw: past.copy())
    def predict(state, frame, parameters):
        calls["predict_rows"].append(frame.copy())
        detail = frame[["zone", "timestamp_utc", "forecast_origin_utc"]].copy()
        detail["mixture_error_q10"] = -30.
        detail["mixture_error_q50"] = 200.
        detail["mixture_error_q90"] = 500.
        detail["strong_risk_gate"] = True; detail["physical_gate_passed"] = True
        detail["proposal_reason"] = "probability_above_half_coherent_median"
        return np.full(len(frame), .75), np.full(len(frame), 200.), np.full(len(frame), 100.), detail
    monkeypatch.setattr(lab, "predict_model", predict)
    def govern(history, zone, day, cutoff, parameters):
        calls["govern"].append((history.copy(), zone, day, cutoff))
        assert history.actual.notna().all()
        assert history.label_available_at_utc.le(cutoff).all()
        assert history.zone.eq(zone).all()
        return (.5 if zone == "DE" else 0.), "test_governance", []
    monkeypatch.setattr(base, "_govern", govern)
    real_calibration = lab.fit_interval_state
    def calibrate(history, cutoff, **kwargs):
        calls["calibration"].append((history.copy(), cutoff, kwargs))
        return real_calibration(history, cutoff, **kwargs)
    monkeypatch.setattr(lab, "fit_interval_state", calibrate)
    return manifest, calls, p


def test_fixed_inference_preserves_input_values_types_order_and_causal_daily_decisions(prediction_case, tmp_path):
    manifest, calls, _ = prediction_case
    panel = panel_for("2026-09-16").drop(columns=["actual", "label_available_at_utc"])
    panel["forecast"] = panel.forecast.astype("int32")
    panel["input_note"] = "untouched"
    panel = panel.iloc[::-1].copy()
    panel.loc[panel.index[0], "forecast_eligible"] = False
    original = panel.copy(deep=True)
    result = lab.predict_fixed(panel, manifest, ledger_directory=tmp_path, root=tmp_path)
    pd.testing.assert_frame_equal(panel, original, check_exact=True)
    pd.testing.assert_frame_equal(result[panel.columns], panel.reset_index(drop=True), check_exact=True)
    assert "actual" not in result and "label_available_at_utc" not in result
    assert not result.loc[~result.forecast_eligible, "expert_ready"].any()
    assert all(frame.forecast_eligible.all() for frame in calls["predict_rows"])
    active = result.zone.eq("DE") & result.forecast_eligible
    assert result.loc[active, "candidate_forecast"].eq(200.).all()
    assert result.loc[~active, "candidate_forecast"].eq(100.).all()
    assert result.loc[active, "mixture_raw_p50_eur_mwh"].eq(300.).all()
    assert (result.candidate_q10 <= result.q10).all() and (result.candidate_q90 >= result.q90).all()
    assert (result.candidate_q10 <= result.candidate_forecast).all() and (result.candidate_q90 >= result.candidate_forecast).all()
    assert calls["model_load"] == 1 and len(calls["calibration"]) == 1
    assert {entry[1] for entry in calls["govern"]} == {"DE", "FR"}
    assert all(entry[2] == "2026-09-16" and entry[3] == ledger._origin("2026-09-16") for entry in calls["govern"])


def test_current_target_refused_before_loading_any_model(prediction_case, tmp_path):
    manifest, calls, _ = prediction_case
    with pytest.raises(ValueError, match="target prices"):
        lab.predict_fixed(panel_for("2026-09-16", actual=123.), manifest, ledger_directory=tmp_path, root=tmp_path)
    assert calls["model_load"] == 0


def test_changed_candidate_refused_before_deserialisation(prediction_case, tmp_path, monkeypatch):
    manifest, calls, _ = prediction_case
    monkeypatch.setattr(ledger, "_verify_files", lambda *a: (_ for _ in ()).throw(ValueError("changed candidate")))
    with pytest.raises(ValueError, match="changed candidate"):
        lab.predict_fixed(panel_for("2026-09-16"), manifest, ledger_directory=tmp_path, root=tmp_path)
    assert calls["model_load"] == 0


def test_runtime_and_result_are_verified_before_fresh_process_model_load(prediction_case, tmp_path, monkeypatch):
    manifest, _, _ = prediction_case
    order = []
    original_load = lab.joblib.load
    def read(path, *, root):
        order.append("runtime_initialised_and_verified")
        assert path == tmp_path
        return tmp_path, {}, {}
    def verify(*args):
        order.append("result_verified")
    def load(path):
        assert order == ["runtime_initialised_and_verified", "result_verified"]
        order.append("model_loaded")
        return original_load(path)
    monkeypatch.setattr(lab, "read_suite", read)
    monkeypatch.setattr(lab, "verify_result", verify)
    monkeypatch.setattr(lab.joblib, "load", load)
    lab.predict_fixed(panel_for("2026-09-16"), manifest, ledger_directory=tmp_path, root=tmp_path)
    assert order[-1] == "model_loaded"


@pytest.mark.parametrize("stage", ["runtime", "result", "model_path"])
def test_stale_runtime_result_or_model_identity_refused_before_loading(prediction_case, tmp_path, monkeypatch, stage):
    manifest, calls, _ = prediction_case
    if stage == "runtime":
        monkeypatch.setattr(lab, "read_suite", lambda *a, **kw: (_ for _ in ()).throw(ValueError("runtime changed")))
    elif stage == "result":
        monkeypatch.setattr(lab, "verify_result", lambda *a: (_ for _ in ()).throw(ValueError("result changed")))
    else:
        manifest["candidate_files"]["model"]["path"] = str(tmp_path/"other.joblib")
    with pytest.raises(ValueError):
        lab.predict_fixed(panel_for("2026-09-16"), manifest, ledger_directory=tmp_path, root=tmp_path)
    assert calls["model_load"] == 0


def test_future_feature_contract_and_multiday_inputs_are_rejected(prediction_case, tmp_path, monkeypatch):
    manifest, _, _ = prediction_case
    with pytest.raises(ValueError, match="one common delivery"):
        lab.predict_fixed(pd.concat([panel_for("2026-09-16"), panel_for("2026-09-17")]), manifest, ledger_directory=tmp_path, root=tmp_path)
    monkeypatch.setattr(lab, "make_stress_features", lambda frame: (frame, [FEATURE, "extra"], [FEATURE], {}))
    with pytest.raises(ValueError, match="feature contract"):
        lab.predict_fixed(panel_for("2026-09-16"), manifest, ledger_directory=tmp_path, root=tmp_path)


@pytest.mark.parametrize("status", ["fallback", "empty", "missing_columns"])
def test_freeze_refuses_stale_latest_state_before_deserialisation(tmp_path, monkeypatch, status):
    monkeypatch.setattr(lab, "read_suite", lambda *a, **kw: (tmp_path, {}, {}))
    monkeypatch.setattr(lab, "verify_result", lambda *a: None)
    folds = pd.DataFrame({"fit_day": ["2026-09-14", "2026-09-15"], "status": ["trained", "fallback"]})
    if status == "empty": folds = folds.iloc[:0]
    if status == "missing_columns": folds = folds[["fit_day"]]
    folds.to_parquet(tmp_path/"folds.parquet", index=False)
    monkeypatch.setattr(lab.joblib, "load", lambda *a: pytest.fail("Stale model must not load"))
    with pytest.raises(ValueError):
        lab.freeze(tmp_path, root=tmp_path)


def test_freeze_refuses_artifact_from_earlier_trained_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "read_suite", lambda *a, **kw: (tmp_path, {}, {}))
    monkeypatch.setattr(lab, "verify_result", lambda *a: None)
    pd.DataFrame({"fit_day": ["2026-09-15"], "status": ["trained"]}).to_parquet(tmp_path/"folds.parquet", index=False)
    monkeypatch.setattr(lab.joblib, "load", lambda *a: {"state": {"fit_day": "2026-09-14"}})
    with pytest.raises(ValueError, match="resurrect"):
        lab.freeze(tmp_path, root=tmp_path)


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    import run_nyx_stress_guard as cli
    from nyx_stress_guard import inputs
    manifest = {"start_day": "2026-09-16", "created_at_utc": "2026-09-14T16:00:00Z",
        "zones": ["DE", "FR"], "issue_policy": "pre_observation_asof08", "candidate_files": {}, "history_files": {}}
    events = []
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "resolve_ledger", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(ledger, "_read", lambda *a: (tmp_path, manifest, events))
    monkeypatch.setattr(ledger, "now_utc", lambda: pd.Timestamp("2026-09-15T08:00:00Z"))
    monkeypatch.setattr(ledger, "_verify_files", lambda *a: None)
    monkeypatch.setattr(ledger, "ledger_status", lambda *a, **kw: {"status": "test_status"})
    monkeypatch.setattr(inputs, "capture_inputs", lambda **kw: pytest.fail("No capture/API expected"))
    monkeypatch.setattr(inputs, "collect_observations", lambda **kw: pytest.fail("No target/API expected"))
    monkeypatch.setattr(lab.joblib, "load", lambda *a, **kw: pytest.fail("CLI must not deserialize for these actions"))
    return cli, inputs, manifest, events


@pytest.mark.parametrize("action", ["capture", "issue"])
@pytest.mark.parametrize("day", ["2026-09-15", "2027-09-16", "2026-9-16"])
def test_cli_refuses_seen_or_outside_window_day_before_any_collection(cli_case, action, day):
    cli, _, _, _ = cli_case
    assert cli.main(["--action", action, "--delivery-day", day]) == 2


def test_cli_refuses_late_issue_before_any_collection(cli_case, monkeypatch):
    cli, _, _, _ = cli_case
    monkeypatch.setattr(ledger, "now_utc", lambda: pd.Timestamp("2026-09-15T09:45:00Z"))
    assert cli.main(["--action", "issue", "--delivery-day", "2026-09-16"]) == 2


def test_cli_evaluate_and_status_without_pending_issues_do_not_load_or_fetch(cli_case, capsys):
    cli, _, _, _ = cli_case
    assert cli.main(["--action", "evaluate"]) == 0
    assert cli.main(["--action", "status", "--ledger-directory", "anything"]) == 0
    assert '"production_modified": false' in capsys.readouterr().out


def test_cli_evaluate_only_previously_issued_days_and_bounds_batch(cli_case, monkeypatch):
    cli, inputs, manifest, events = cli_case
    days = pd.date_range("2026-09-16", periods=40).strftime("%Y-%m-%d").tolist()
    events.extend({"kind": "forecast", "delivery_day": day} for day in days)
    calls = []
    def collect(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(), {"receipt": True}
    monkeypatch.setattr(inputs, "collect_observations", collect)
    monkeypatch.setattr(ledger, "resolve_observations", lambda *a, **kw: {"status": "resolved", "forecast_recalculated": False})
    assert cli.main(["--action", "evaluate"]) == 0
    assert calls[0]["days"] == days[:31] and calls[0]["zones"] == manifest["zones"]
    calls.clear()
    assert cli.main(["--action", "evaluate", "--delivery-day", "2026-09-15"]) == 2
    assert not calls


def test_cli_rejects_mutated_candidate_before_capture(cli_case, monkeypatch):
    cli, _, _, _ = cli_case
    monkeypatch.setattr(ledger, "_verify_files", lambda *a: (_ for _ in ()).throw(ValueError("checksum")))
    assert cli.main(["--action", "capture", "--delivery-day", "2026-09-16"]) == 2


@pytest.mark.parametrize("mutated", [False, True])
def test_cli_saved_capture_is_bound_to_evidence_before_issue(tmp_path, monkeypatch, mutated):
    from test_nyx_stress_guard_ledger import make_case
    from nyx_stress_guard import inputs
    import run_nyx_stress_guard as cli
    case = make_case(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", case["root"])
    monkeypatch.setattr(inputs, "capture_inputs", lambda **kw: pytest.fail("Saved capture must not be recollected"))
    capture = case["root"]/ledger.NAMESPACE/"captures/saved"
    capture.mkdir(parents=True)
    frame = case["frame"].copy()
    if mutated:
        frame.loc[0, "feature_stress"] += 100.
    frame.to_parquet(capture/"panel.parquet", index=False)
    (capture/"input_evidence.json").write_text(json.dumps(case["evidence"]), encoding="utf8")
    def issue(directory, *, root, panel, input_evidence):
        return ledger.issue_forecast(directory, root=root, panel=panel, input_evidence=input_evidence,
            predictor=case["predictor"], fresh_label_check=case["check"])
    monkeypatch.setattr(lab, "issue", issue)
    result = cli.main(["--action", "issue", "--ledger-directory", str(case["directory"]),
        "--input-directory", str(capture), "--delivery-day", case["day"]])
    assert result == (2 if mutated else 0)
    assert case["counts"]["predict"] == (0 if mutated else 1)
    assert ledger.ledger_status(case["directory"], root=case["root"])["forecast_days"] == (0 if mutated else 1)
