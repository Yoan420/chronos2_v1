from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_stress_guard import ledger as lab


def make_case(tmp_path, monkeypatch, grade="pre_observation_asof08", day="2026-09-16"):
    root = tmp_path/"project"; root.mkdir()
    files = {}
    for name in ("model", "config", "code", "baseline", "features"):
        path = root/(name+".txt"); path.write_text(name, encoding="utf8"); files[name] = path
    monkeypatch.setattr(lab, "__file__", str(files["code"]))
    clock = {"now": pd.Timestamp("2026-09-14T16:30:00Z")}
    monkeypatch.setattr(lab, "now_utc", lambda: clock["now"])
    history = root/"history.parquet"
    pd.DataFrame({"timestamp_utc": [pd.Timestamp("2026-09-15T21:00:00Z")], "actual": [90.]}).to_parquet(history, index=False)
    directory = lab.freeze_ledger(root/lab.NAMESPACE/"prospective/test", root=root,
        candidate_files={k: files[k] for k in ("model", "config", "code")}, history_files={"history": history},
        explored_through_day="2026-09-15", zones=["FR", "DE", "BE", "NL"], issue_policy=grade,
        requested_start_day=day)
    origin = lab._origin(day)
    clock["now"] = origin if grade == "strict_08_issue" else origin+pd.Timedelta(hours=2)
    zones = ["BE", "DE", "FR", "NL"]
    frame = pd.concat([pd.DataFrame({"zone": z, "timestamp_utc": lab._index(day), "forecast_origin_utc": origin,
        "forecast": 100., "q10": 80., "q90": 120., "feature_stress": 2., "actual": np.nan}) for z in zones], ignore_index=True)
    files["panel"] = root/"panel.parquet"
    frame.to_parquet(files["panel"], index=False)
    evidence = {"schema_version": 1, "information_cutoff_utc": origin.isoformat(), "captured_at_utc": clock["now"].isoformat(),
        "artifacts": {k: {"path": str(files[k]), "sha256": lab._digest(files[k])} for k in ("baseline", "features", "panel")},
        "source_information_times_utc": {"physical": (origin-pd.Timedelta(hours=1)).isoformat()}, "publication_evidence": "historical_asof"}
    sources = {z: {"series": "canonical_"+z, "canonical_cache_sha256": "a"*64} for z in zones}
    counts = {"predict": 0, "check": 0}
    def predictor(panel, manifest):
        counts["predict"] += 1
        assert manifest["start_day"] == day
        result = panel.copy()
        result["candidate_forecast"] = 105.; result["candidate_q10"] = 81.; result["candidate_q90"] = 125.
        return result
    def check(target_day, countries):
        counts["check"] += 1
        return {"schema_version": 1, "kind": "fresh_canonical_target_check", "delivery_day": target_day, "zones": countries,
            "started_at_utc": clock["now"].isoformat(), "received_at_utc": clock["now"].isoformat(),
            "fresh_api_read": True, "nocache": True, "observed_hours_by_zone": {z: 0 for z in countries}, "target_sources": deepcopy(sources)}
    return {"root": root, "directory": directory, "files": files, "history": history, "clock": clock,
        "frame": frame, "evidence": evidence, "predictor": predictor, "check": check, "sources": sources, "counts": counts, "day": day}


def issue(case, **changes):
    args = {"root": case["root"], "panel": case["frame"], "input_evidence": case["evidence"],
        "predictor": case["predictor"], "fresh_label_check": case["check"], **changes}
    return lab.issue_forecast(case["directory"], **args)


def observations(case):
    frame = case["frame"][["zone", "timestamp_utc"]].copy(); frame["actual"] = 108.
    receipt = {"schema_version": 1, "kind": "canonical_target_observations", "received_at_utc": case["clock"]["now"].isoformat(),
        "fresh_api_read": True, "target_sources": deepcopy(case["sources"])}
    return frame, receipt


def test_freeze_starts_after_all_seen_live_days_and_real_clock(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    status = lab.ledger_status(case["directory"], root=case["root"])
    assert status["start_day"] == "2026-09-16" and status["explored_through_day"] == "2026-09-15"
    assert status["status"] == "armed_no_forecasts" and status["forecast_days"] == 0
    assert status["evaluation_by_grade_and_zone"] == {grade: {} for grade in lab.GRADES}
    manifest = json.loads((case["directory"]/"manifest.json").read_text())
    assert set(manifest["candidate_files"]) == {"model", "config", "code", "ledger_code"}
    assert not manifest["deadline_is_market_publication_time"]


def test_cannot_freeze_past_day_or_rewrite_ledger(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    kwargs = dict(root=case["root"], candidate_files={k: case["files"][k] for k in ("model", "config", "code")},
        history_files={"history": case["history"]}, explored_through_day="2026-09-14", zones=["FR"])
    with pytest.raises(lab.LedgerError, match="seen day|origin"):
        lab.freeze_ledger(case["directory"].parent/"new", requested_start_day="2026-09-15", **kwargs)
    with pytest.raises(FileExistsError):
        lab.freeze_ledger(case["directory"], **kwargs)


@pytest.mark.parametrize("grade", lab.GRADES)
def test_genuine_issue_and_two_fresh_receipts_are_immutable(tmp_path, monkeypatch, grade):
    case = make_case(tmp_path, monkeypatch, grade)
    original = case["frame"].copy(deep=True)
    result = issue(case)
    assert result["status"] == "issued" and result["issue_grade"] == grade
    assert case["counts"] == {"predict": 1, "check": 2}
    pd.testing.assert_frame_equal(original, case["frame"])
    _, _, events = lab._read(case["directory"], case["root"])
    saved = events[0]
    assert saved["issue_grade"] == grade and saved["source_publication_evidence"] == "historical_asof"
    assert saved["source_publication_certified_by_ledger"] is False
    before = {str(p): lab._digest(p) for p in case["directory"].rglob("*") if p.is_file() and not p.name.endswith(".guard")}
    case["clock"]["now"] += pd.Timedelta(days=2)
    assert issue(case)["status"] == "already_issued"
    assert case["counts"] == {"predict": 1, "check": 2}
    assert before == {str(p): lab._digest(p) for p in case["directory"].rglob("*") if p.is_file() and not p.name.endswith(".guard")}
    altered = case["frame"].copy(); altered.loc[0, "feature_stress"] += 1
    with pytest.raises(lab.LedgerError, match="different immutable"):
        issue(case, panel=altered)


@pytest.mark.parametrize("grade,offset", [("strict_08_issue", 1), ("pre_observation_asof08", 225), ("pre_observation_asof08", -1)])
def test_real_issue_window_rejects_late_or_wrong_grade(tmp_path, monkeypatch, grade, offset):
    case = make_case(tmp_path, monkeypatch, grade)
    case["clock"]["now"] = lab._origin(case["day"])+pd.Timedelta(minutes=offset)
    with pytest.raises(lab.LedgerError, match="wall-clock"):
        issue(case)
    assert case["counts"] == {"predict": 0, "check": 0}


@pytest.mark.parametrize("day,hours", [("2027-03-28", 23), ("2026-10-25", 25)])
def test_native_dst_complete_frames_issue_without_fabricated_hours(tmp_path, monkeypatch, day, hours):
    case = make_case(tmp_path, monkeypatch, day=day)
    assert len(case["frame"]) == 4*hours
    assert issue(case)["status"] == "issued"
    case["clock"]["now"] += pd.Timedelta(days=1)
    frame, receipt = observations(case)
    lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=receipt)
    status = lab.ledger_status(case["directory"], root=case["root"])
    assert status["evaluation_by_grade_and_zone"]["pre_observation_asof08"]["FR"]["hours"] == hours


@pytest.mark.parametrize("change", ["duplicate", "missing_hour", "missing_country", "origin", "actual", "target_alias", "baseline_quantile"])
def test_input_leak_or_incomplete_identity_fails_before_inference(tmp_path, monkeypatch, change):
    case = make_case(tmp_path, monkeypatch); panel = case["frame"].copy()
    if change == "duplicate": panel = pd.concat([panel, panel.iloc[:1]])
    elif change == "missing_hour": panel = panel.iloc[1:]
    elif change == "missing_country": panel = panel[panel.zone.ne("BE")]
    elif change == "origin": panel.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    elif change == "actual": panel.loc[0, "actual"] = 140.
    elif change == "target_alias": panel["target_fr"] = 140.
    elif change == "baseline_quantile": panel.loc[0, "q10"] = 110.
    with pytest.raises(lab.LedgerError): issue(case, panel=panel)
    assert case["counts"]["predict"] == 0


@pytest.mark.parametrize("change", ["future_source", "future_capture", "source_mutated", "wrong_cutoff", "uncertified_upgrade"])
def test_input_capture_and_source_hashes_fail_closed(tmp_path, monkeypatch, change):
    case = make_case(tmp_path, monkeypatch); evidence = deepcopy(case["evidence"])
    if change == "future_source": evidence["source_information_times_utc"]["physical"] = case["clock"]["now"].isoformat()
    elif change == "future_capture": evidence["captured_at_utc"] = (case["clock"]["now"]+pd.Timedelta(minutes=1)).isoformat()
    elif change == "source_mutated": case["files"]["features"].write_text("changed")
    elif change == "wrong_cutoff": evidence["information_cutoff_utc"] = case["clock"]["now"].isoformat()
    elif change == "uncertified_upgrade": evidence["publication_evidence"] = "certified_by_metadata"
    with pytest.raises(lab.LedgerError): issue(case, input_evidence=evidence)
    assert case["counts"]["predict"] == 0


@pytest.mark.parametrize("stage", ["before", "after"])
def test_published_price_before_or_during_inference_prevents_registration(tmp_path, monkeypatch, stage):
    case = make_case(tmp_path, monkeypatch)
    calls = []
    def check(day, zones):
        value = case["check"](day, zones); calls.append(1)
        if stage == "before" or len(calls) == 2: value["observed_hours_by_zone"]["BE"] = 1
        return value
    with pytest.raises(lab.LedgerError, match="already known"): issue(case, fresh_label_check=check)
    assert not list((case["directory"]/"events").iterdir())


@pytest.mark.parametrize("change", ["stale_time", "cached", "wrong_zone", "bad_hash"])
def test_fresh_receipt_cannot_be_faked_from_old_or_cached_check(tmp_path, monkeypatch, change):
    case = make_case(tmp_path, monkeypatch)
    def check(day, zones):
        value = case["check"](day, zones)
        if change == "stale_time": value["started_at_utc"] = (case["clock"]["now"]-pd.Timedelta(seconds=1)).isoformat()
        elif change == "cached": value["fresh_api_read"] = False
        elif change == "wrong_zone": value["zones"] = ["FR"]
        elif change == "bad_hash": value["target_sources"]["FR"]["canonical_cache_sha256"] = "bad"
        return value
    with pytest.raises(lab.LedgerError): issue(case, fresh_label_check=check)
    assert case["counts"]["predict"] == 0


@pytest.mark.parametrize("mutation", ["model", "source", "deadline", "receipt_expiry", "baseline", "quantiles"])
def test_inference_mutation_or_time_expiry_cannot_publish(tmp_path, monkeypatch, mutation):
    case = make_case(tmp_path, monkeypatch)
    def predict(panel, manifest):
        result = case["predictor"](panel, manifest)
        if mutation == "model": case["files"]["model"].write_text("altered")
        elif mutation == "source": case["files"]["features"].write_text("altered")
        elif mutation == "deadline": case["clock"]["now"] += pd.Timedelta(hours=3)
        elif mutation == "receipt_expiry": case["clock"]["now"] += pd.Timedelta(seconds=121)
        elif mutation == "baseline": result.loc[0, "forecast"] += 1.
        elif mutation == "quantiles": result.loc[0, "candidate_q10"] = 200.
        return result
    with pytest.raises(lab.LedgerError): issue(case, predictor=predict)
    assert not list((case["directory"]/"events").iterdir())


def test_concurrent_lock_refuses_before_predictor(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    with exclusive_process_lock(case["directory"]/"ledger.lock"):
        with pytest.raises(ValueError, match="verrou|Verrou"): issue(case)
    assert case["counts"] == {"predict": 0, "check": 0}


def test_resolve_only_after_issue_without_model_read_and_idempotent_first_complete_labels(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch); issue(case)
    frame, evidence = observations(case)
    with pytest.raises(lab.LedgerError, match="postdate"):
        lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=evidence)
    case["clock"]["now"] += pd.Timedelta(hours=3)
    frame, evidence = observations(case)
    # Historical issued predictions remain evaluable even if an external model changes later.
    case["files"]["model"].write_text("do not read/load this modified model during evaluation")
    result = lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=evidence)
    assert result["new_complete_country_days"] == 4 and not result["forecast_recalculated"]
    status = lab.ledger_status(case["directory"], root=case["root"])
    assert status["status"] == "evaluated" and status["journal_events"] == 5
    assert status["evaluation_by_grade_and_zone"]["strict_08_issue"] == {}
    for values in status["evaluation_by_grade_and_zone"]["pre_observation_asof08"].values():
        assert values["candidate_mae_eur_mwh"] == 3. and values["baseline_mae_eur_mwh"] == 8.
        assert values["coverage_p10_p90"] == 1. and not values["independent_annual_validation_complete"]
    assert lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=evidence)["status"] == "no_new_observations"
    frame.loc[0, "actual"] += 1.
    with pytest.raises(lab.LedgerError, match="changed after first"):
        lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=evidence)
    assert case["counts"] == {"predict": 1, "check": 2}


def test_no_retroactive_resolution_and_partial_country_days_remain_pending(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch); frame, evidence = observations(case)
    with pytest.raises(lab.LedgerError, match="No pre-existing"):
        lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=evidence)
    issue(case); case["clock"]["now"] += pd.Timedelta(hours=3); frame, evidence = observations(case)
    frame.loc[frame.zone.eq("BE"), "actual"] = np.nan
    result = lab.resolve_observations(case["directory"], root=case["root"], observations=frame, observation_evidence=evidence)
    assert result["new_complete_country_days"] == 3 and result["pending"][0]["zone"] == "BE"
    assert lab.ledger_status(case["directory"], root=case["root"])["pending_observations"] == [{"delivery_day": case["day"], "zone": "BE"}]


@pytest.mark.parametrize("mutation", ["manifest", "event", "artifact", "truncated_tail", "head"])
def test_hash_chain_manifest_artifacts_and_tail_removal_are_detected(tmp_path, monkeypatch, mutation):
    case = make_case(tmp_path, monkeypatch); issue(case)
    if mutation == "manifest":
        path = case["directory"]/"manifest.json"; value = json.loads(path.read_text()); value["issue_policy"] = "strict_08_issue"; path.write_text(json.dumps(value))
    elif mutation == "event":
        path = case["directory"]/"events/00000001.json"; value = json.loads(path.read_text()); value["issued_at_utc"] = "2026-09-15T06:00:00Z"; path.write_text(json.dumps(value))
    elif mutation == "artifact": next((case["directory"]/"artifacts").glob("*/predictions.parquet")).write_bytes(b"changed")
    elif mutation == "truncated_tail": (case["directory"]/"events/00000001.json").unlink()
    elif mutation == "head": (case["directory"]/"heads/00000001.json").write_text("{}")
    with pytest.raises(lab.LedgerError): lab.ledger_status(case["directory"], root=case["root"])


def test_missed_days_are_visible_not_cherry_picked(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    case["clock"]["now"] += pd.Timedelta(days=2)
    status = lab.ledger_status(case["directory"], root=case["root"])
    assert status["missed_emission_days"] == ["2026-09-16", "2026-09-17"]
    assert status["complete_country_days"] == 0


def test_path_escape_and_symlink_rejected(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    with pytest.raises(lab.LedgerError, match="namespace"):
        lab.ledger_status(case["root"]/"outside", root=case["root"])
    with pytest.raises(lab.LedgerError, match="traversal"):
        lab.ledger_status(case["directory"]/".."/"test", root=case["root"])


def test_memory_inputs_must_match_the_hash_pinned_capture(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    panel = case["frame"].copy()
    panel.loc[0, "feature_stress"] += 1.
    with pytest.raises(lab.LedgerError, match="In-memory inputs differ"):
        issue(case, panel=panel)
    assert case["counts"] == {"predict": 0, "check": 0}
