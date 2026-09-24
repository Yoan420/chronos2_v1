"""Read-only verification of exact hourly Statistics comparator snapshots."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.rolling_research_comparators import (
    RollingReportComparatorError, load_rolling_report_comparators,
)


def _fixture(tmp_path, *, day="2026-09-08", zone="FR", missing_actual=False,
             binary=True, aware=False, variant="kalman"):
    first = pd.Timestamp(day, tz="Europe/Paris")
    end = first + pd.DateOffset(days=1)
    hours = pd.date_range(first, end, freq="h", inclusive="left").tz_convert("UTC")
    native = hours.tz_convert("Europe/Paris").tz_localize(None)
    standard = native.tz_localize("Europe/Paris", ambiguous=False).tz_convert("UTC")
    keep = np.ones(len(hours), dtype=bool) if aware else hours == standard
    actual = np.arange(len(hours), dtype=float) + 50.0
    if missing_actual:
        actual[:] = np.nan
    storm = np.arange(len(hours), dtype=float) + 45.0
    def values(array):
        if binary:
            return {"dtype": "f8", "bdata": base64.b64encode(np.asarray(array, dtype="<f8").tobytes()).decode("ascii")}
        return [None if not np.isfinite(x) else float(x) for x in array]
    x = [stamp.isoformat() for stamp in (hours if aware else native)]
    official = [
        {"name": "Observé", "x": list(np.asarray(x)[keep]), "y": values(actual[keep])},
        {"name": "Storm officiel dashboard P50", "x": list(np.asarray(x)[keep]), "y": values(storm[keep])},
        {"name": "My model P50", "x": list(np.asarray(x)[keep]), "y": values(storm[keep] + 999)},
    ]
    observed = [{"name": "Observé", "x": x, "y": values(actual)}]
    record = {"zone": zone, "sample": "daily", "period_start": day,
        "observed_mean_price": None if missing_actual else float(actual[keep].mean()),
        "benchmark_mean_price": float(storm[keep].mean())}
    path = tmp_path / "runs/exports" / day / zone.lower() / variant / f"forecast_{zone.lower()}_{day}_{variant}.html"
    path.parent.mkdir(parents=True)
    batch = path.parents[2]
    manifest = {"schema_version": 1, "delivery_day": day, "exports": [{"zone": zone, "variant": variant,
        "source_model": "residual_kalman" if variant == "kalman" else "residual_corrected",
        "html": {"path": str(path.relative_to(batch)).replace("\\", "/"), "sha256": ""}}]}
    case = {"root": tmp_path, "path": path, "batch": batch, "manifest": manifest,
        "day": day, "zone": zone, "hours": hours, "keep": keep, "actual": actual, "storm": storm,
        "official": official, "observed": observed, "record": record, "extra": ""}
    _publish(case)
    return case


def _publish(case):
    rendered = "<html><p>Généré le 2026-09-07 14:41 CEST · Script 2.3.5</p>"
    rendered += "<p>Observations (extraction 2026-09-07 12:27:27.307945+00:00)</p>"
    for name, traces, title in (("official", case["official"], "Comparaison à Storm"),
                                ("physical", case["observed"], f"{case['zone']} — backtest glissant")):
        rendered += f'<script>Plotly.newPlot("{name}", ' + json.dumps(traces) + ", " + json.dumps({"title": {"text": title}}) + ", {});</script>"
    rendered += "<script>const payload = " + json.dumps({"records": [case["record"]], "metrics": [], "zones": [case["zone"]]}) + ";</script>"
    rendered += case["extra"] + "</html>"
    case["path"].write_text(rendered, encoding="utf-8")
    case["manifest"]["exports"][0]["html"]["sha256"] = hashlib.sha256(case["path"].read_bytes()).hexdigest()
    (case["batch"] / "current_batch_manifest.json").write_text(json.dumps(case["manifest"]), encoding="utf-8")


def _load(case):
    return load_rolling_report_comparators(case["root"], zone=case["zone"], delivery_day=case["day"], start_day=case["day"])


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_exact_reported_values_not_model_predictions_with_snapshot_audit(tmp_path, binary, zone):
    case = _fixture(tmp_path, binary=binary, zone=zone)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    frame, audit = _load(case)
    np.testing.assert_array_equal(frame.actual, case["actual"])
    np.testing.assert_array_equal(frame.q50, case["storm"])
    assert pd.DatetimeIndex(frame.timestamp).equals(case["hours"])
    assert frame.timestamp.equals(frame.delivery_start_utc)
    assert audit["physical_hours"] == audit["observed_hours"] == audit["storm_hours"] == 24
    assert audit["snapshot_generated_text"] == "2026-09-07 14:41 CEST"
    assert audit["observation_extraction_timestamps"] == ["2026-09-07 12:27:27.307945+00:00"]
    assert audit["network_refreshed"] is audit["latest_known_verified"] is False
    assert audit["used_for_prediction"] is audit["used_for_calibration"] is False
    assert audit["statistics_daily_means_verified"] is True
    assert audit["storm_contract"]["report_label"] == "Storm officiel dashboard"
    assert audit["storm_contract"]["series"] == f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"
    assert audit["storm_contract"]["official_dashboard_metric"] is True
    assert audit["storm_contract"]["used_for_prediction"] is False
    assert audit["storm_contract"]["used_for_calibration"] is False
    assert audit["storm_contract"]["materialization_audit"]["source"]["sha256"] == audit["html"]["sha256"]
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_autumn_dst_keeps_all_observed_hours_and_only_one_official_storm_fold(tmp_path):
    case = _fixture(tmp_path, day="2025-10-26")
    frame, audit = _load(case)
    assert len(frame) == 25
    assert frame.actual.notna().sum() == 25 and frame.q50.notna().sum() == 24
    assert audit["missing_storm_utc"] == ["2025-10-26T00:00:00+00:00"]
    assert audit["interpolation"] is False
    first = frame.loc[frame.timestamp.eq(pd.Timestamp("2025-10-26T00:00Z"))].iloc[0]
    second = frame.loc[frame.timestamp.eq(pd.Timestamp("2025-10-26T01:00Z"))].iloc[0]
    assert first.actual == 52 and np.isnan(first.q50)
    assert second.actual == 53 and second.q50 == 48


def test_spring_dst_retains_twenty_three_physical_hours(tmp_path):
    case = _fixture(tmp_path, day="2026-03-29")
    frame, audit = _load(case)
    assert len(frame) == audit["paired_hours"] == 23
    assert audit["missing_storm_utc"] == []


def test_utc_aware_trace_preserves_both_autumn_storm_folds(tmp_path):
    case = _fixture(tmp_path, day="2025-10-26", aware=True)
    frame, audit = _load(case)
    assert len(frame) == audit["storm_hours"] == 25
    assert audit["missing_storm_utc"] == []


def test_missing_current_day_actuals_remain_empty_while_storm_stays_available(tmp_path):
    case = _fixture(tmp_path, missing_actual=True)
    frame, audit = _load(case)
    assert frame.actual.isna().all() and frame.q50.notna().all()
    assert audit["observed_hours"] == audit["paired_hours"] == 0


def test_missing_kalman_allows_existing_autonomous_html(tmp_path):
    case = _fixture(tmp_path, variant="autonomous")
    _, audit = _load(case)
    assert audit["variant"] == "autonomous"


def test_html_checksum_tamper_refused(tmp_path):
    case = _fixture(tmp_path)
    case["path"].write_text(case["path"].read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(RollingReportComparatorError, match="SHA256"):
        _load(case)


@pytest.mark.parametrize("field", ["observed_mean_price", "benchmark_mean_price"])
def test_inconsistent_daily_statistics_mean_refused(tmp_path, field):
    case = _fixture(tmp_path)
    case["record"][field] += 1
    _publish(case)
    with pytest.raises(RollingReportComparatorError, match="divergent"):
        _load(case)


def test_non_dst_missing_storm_hour_is_not_filled_or_ignored(tmp_path):
    case = _fixture(tmp_path, binary=False)
    for trace in case["official"]:
        trace["x"].pop(3)
        trace["y"].pop(3)
    _publish(case)
    with pytest.raises(RollingReportComparatorError, match="hors omission DST"):
        _load(case)


def test_disagreeing_observed_traces_refused_not_silently_patched(tmp_path):
    case = _fixture(tmp_path, binary=False)
    case["observed"][0]["y"][0] += 1
    _publish(case)
    with pytest.raises(RollingReportComparatorError, match="contradictoires"):
        _load(case)


def test_official_trace_must_be_unique(tmp_path):
    case = _fixture(tmp_path)
    case["official"].append(case["official"][1])
    _publish(case)
    with pytest.raises(RollingReportComparatorError, match="uniques"):
        _load(case)


@pytest.mark.parametrize("alteration", ["path_escape", "source_model", "manifest_day", "duplicate_entry"])
def test_manifest_identity_or_target_path_mismatch_refused(tmp_path, alteration):
    case = _fixture(tmp_path)
    manifest = case["manifest"]
    entry = manifest["exports"][0]
    if alteration == "path_escape":
        entry["html"]["path"] = "../../../external.html"
    elif alteration == "source_model":
        entry["source_model"] = "some_other_model"
    elif alteration == "manifest_day":
        manifest["delivery_day"] = "2026-09-07"
    else:
        manifest["exports"].append(entry)
    _publish(case)
    with pytest.raises(RollingReportComparatorError):
        _load(case)


def test_other_javascript_is_never_executed(tmp_path):
    case = _fixture(tmp_path)
    case["extra"] = '<script>throw new Error("This JavaScript must never execute");</script>'
    _publish(case)
    frame, _ = _load(case)
    assert len(frame) == 24


def test_unknown_binary_dtype_is_rejected(tmp_path):
    case = _fixture(tmp_path)
    case["official"][1]["y"]["dtype"] = "O"
    _publish(case)
    with pytest.raises(RollingReportComparatorError, match="Encodage"):
        _load(case)


@pytest.mark.parametrize(("start", "end"), [("2026-09-09", "2026-09-08"), ("2025-09-01", "2026-09-08")])
def test_invalid_window_refused_before_source_lookup(tmp_path, start, end):
    with pytest.raises(RollingReportComparatorError, match="Fenêtre"):
        load_rolling_report_comparators(tmp_path, zone="FR", delivery_day=end, start_day=start)
