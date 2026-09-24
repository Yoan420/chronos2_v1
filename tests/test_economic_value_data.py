from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from economic_value.data import (
    EconomicDataError, MODEL_LABELS, load_reference_proxy, load_report_panel, read_report,
)
from marginal_cost_expert.evaluation import physical_index


def _report(root, *, model="kalman", zone="FR", day="2026-09-10", start="2025-09-10",
            end="2026-09-10", quantiles=True, actual=50.0, storm=True, naive=True):
    path = root / "runs" / "exports" / day / zone.lower() / model / f"forecast_{zone.lower()}_{day}_{model}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = physical_index(start, end)
    stamps = index.tz_convert("Europe/Paris").tz_localize(None) if naive else index
    x = [v.isoformat() for v in stamps]
    y = [60.0] * len(index)
    traces = [{"name": MODEL_LABELS[model], "x": x, "y": y},
              {"name": "Observé", "x": x, "y": [actual] * len(index)}]
    if quantiles:
        traces.extend([{"name": "P90", "x": x, "y": [70.] * len(index)},
                       {"name": "P10–P90", "x": x, "y": [40.] * len(index)}])
    paired = [{"name": MODEL_LABELS[model], "x": x, "y": y},
              {"name": "Observé", "x": x, "y": [actual] * len(index)},
              {"name": "Storm officiel dashboard P50", "x": x, "y": [55.] * len(index)}]
    document = "<script>throw new Error('must not execute');</script>"
    document += "<script>Plotly.newPlot(" + json.dumps("history") + "," + json.dumps(traces) + ",{});</script>"
    if storm:
        document += "<script>Plotly.newPlot(\"paired\"," + json.dumps(paired) + ",{});</script>"
    path.write_text(document, encoding="utf-8")
    return path


def _target(root: Path, index: pd.DatetimeIndex, values=None):
    path = root / "target.csv"
    pd.DataFrame({"timestamp": index, "value": np.arange(len(index)) if values is None else values}).to_csv(path, index=False)
    return path


def test_real_quantiles_and_no_javascript_execution(tmp_path):
    path = _report(tmp_path, start="2026-08-31")
    frame, audit = read_report(path, zone="FR", model="kalman")
    assert len(frame) == 264
    assert frame.q10.eq(40).all() and frame.q90.eq(70).all()
    assert frame.benchmark_forecast.eq(55).all()
    assert audit["forecast_pit_certified"] is False


def test_missing_quantiles_not_invented(tmp_path):
    path = _report(tmp_path, start="2026-08-31", quantiles=False)
    frame, _ = read_report(path, zone="FR", model="kalman")
    assert frame[["q10", "q90"]].isna().all().all()


def test_invalid_quantile_order_is_explicitly_discarded(tmp_path):
    path = _report(tmp_path, start="2026-08-31")
    text = path.read_text(encoding="utf-8").replace("40.0", "80.0")
    path.write_text(text, encoding="utf-8")
    frame, audit = read_report(path, zone="FR", model="kalman")
    assert frame[["q10", "q90"]].isna().all().all()
    assert audit["invalid_quantile_rows_discarded"] == len(frame)


def test_last365_calendar_days_preserve_dst(tmp_path):
    _report(tmp_path)
    panel, audit = load_report_panel(tmp_path, ["FR"], ["kalman"])
    assert audit["evaluation_start_day"] == "2025-09-11"
    assert audit["evaluation_end_day"] == "2026-09-10"
    assert len(panel) == 8760
    days = panel.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    assert (days == "2025-10-26").sum() == 25
    assert (days == "2026-03-29").sum() == 23
    assert not panel.duplicated(["zone", "model", "timestamp_utc"]).any()
    assert set(panel["sample"]) == {"evaluation"}
    assert len(audit["live_rows"]) == 24


def test_latest_common_export_not_individual_latest(tmp_path):
    _report(tmp_path, model="autonomous")
    _report(tmp_path, model="kalman")
    _report(tmp_path, model="kalman", day="2026-09-11", end="2026-09-11")
    panel, audit = load_report_panel(tmp_path, ["FR"], ["autonomous", "kalman"])
    assert audit["delivery_day"] == "2026-09-10"
    assert audit["latest_export_by_model"]["FR/kalman"] == "2026-09-11"
    assert len(panel) == 17520
    with pytest.raises(EconomicDataError, match="Incomplete export"):
        load_report_panel(tmp_path, ["FR"], ["autonomous", "kalman"], delivery_day="2026-09-11")


def test_shared_labels_and_storm_despite_source_revisions(tmp_path):
    _report(tmp_path, model="autonomous", actual=49)
    _report(tmp_path, model="kalman", actual=51)
    panel, audit = load_report_panel(tmp_path, ["FR"], ["autonomous", "kalman"])
    assert panel.groupby(["timestamp_utc", "zone"]).actual.nunique().eq(1).all()
    assert panel.groupby(["timestamp_utc", "zone"]).benchmark_forecast.nunique().eq(1).all()
    discrepancies = audit["shared_observations"]["FR"]["actual_revision_differences_by_model"]
    assert max(discrepancies.values()) > 0


def test_future_live_rows_do_not_enter_evaluation(tmp_path):
    path = _report(tmp_path, end="2026-09-09")
    current_index = physical_index("2026-09-10", "2026-09-10")
    pd.DataFrame({"delivery_start_utc": current_index, "zone": "FR", "q50": 100, "q10": 90, "q90": 110}).to_csv(path.with_suffix(".csv"), index=False)
    panel, audit = load_report_panel(tmp_path, ["FR"], ["kalman"])
    assert audit["evaluation_end_day"] == "2026-09-09"
    assert panel["sample"].eq("evaluation").sum() == 8760
    assert panel["sample"].eq("live").sum() == 24
    assert panel.loc[panel["sample"].eq("live"), "actual"].isna().all()


def test_end_day_cannot_use_incomplete_future(tmp_path):
    _report(tmp_path, end="2026-09-09")
    with pytest.raises(EconomicDataError, match="exceeds common"):
        load_report_panel(tmp_path, ["FR"], ["kalman"], end_day="2026-09-10")


def test_csv_html_conflict_rejected(tmp_path):
    path = _report(tmp_path)
    index = physical_index("2026-09-10", "2026-09-10")
    pd.DataFrame({"delivery_start_utc": index, "zone": "FR", "q50": 100}).to_csv(path.with_suffix(".csv"), index=False)
    with pytest.raises(EconomicDataError, match="CSV forecasts differ"):
        read_report(path, zone="FR", model="kalman")


def test_previous_civil_day_not_utc_minus24(tmp_path):
    index = physical_index("2026-03-28", "2026-03-30")
    path = _target(tmp_path, index)
    evaluation = pd.DatetimeIndex([pd.Timestamp("2026-03-30 12:00", tz="Europe/Paris")]).tz_convert("UTC")
    panel = pd.DataFrame({"timestamp_utc": evaluation, "zone": "FR"})
    ref, audit = load_reference_proxy(tmp_path, panel, target_paths={"FR": path})
    assert ref.reference_source_timestamp_utc.iloc[0] == pd.Timestamp("2026-03-29 12:00", tz="Europe/Paris").tz_convert("UTC")
    assert ref.reference_available_at_utc.iloc[0] == pd.Timestamp("2026-03-28 18:00", tz="Europe/Paris").tz_convert("UTC")
    assert audit["executable_reference"] is False and audit["reference_pit_certified"] is False


@pytest.mark.parametrize("day,reason", [("2025-10-27", "ambiguous_previous_civil_hour"),
                                         ("2026-03-30", "missing_previous_civil_hour")])
def test_dst_reference_is_abstention_not_average_or_fill(tmp_path, day, reason):
    start = str((pd.Timestamp(day) - pd.Timedelta(days=2)).date())
    path = _target(tmp_path, physical_index(start, day))
    panel = pd.DataFrame({"timestamp_utc": [pd.Timestamp(day + " 02:00", tz="Europe/Paris").tz_convert("UTC")], "zone": "FR"})
    ref, _ = load_reference_proxy(tmp_path, panel, target_paths={"FR": path})
    assert ref.reference_price.isna().all()
    assert ref.reference_missing_reason.eq(reason).all()
    assert not ref.reference_eligible.any()


def test_reference_rejects_naive_and_quarter_hour_timestamps(tmp_path):
    path = _target(tmp_path, physical_index("2026-09-08", "2026-09-10"))
    for stamp in ["2026-09-10 08:00", "2026-09-10 08:15+02:00"]:
        with pytest.raises(EconomicDataError):
            load_reference_proxy(tmp_path, pd.DataFrame({"timestamp_utc": [stamp], "zone": "FR"}), target_paths={"FR": path})


def test_reference_does_not_substitute_same_day_actual(tmp_path):
    path = _target(tmp_path, physical_index("2026-09-10", "2026-09-10"))
    panel = pd.DataFrame({"timestamp_utc": physical_index("2026-09-10", "2026-09-10"), "zone": "FR", "actual": 999})
    ref, _ = load_reference_proxy(tmp_path, panel, target_paths={"FR": path})
    assert ref.reference_price.isna().all()


def test_proxy_duplicate_target_physical_hour_rejected(tmp_path):
    index = pd.DatetimeIndex([pd.Timestamp("2026-09-09T00:00Z")] * 2)
    path = _target(tmp_path, index)
    panel = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2026-09-10T00:00Z")], "zone": "FR"})
    with pytest.raises(EconomicDataError, match="physical grid"):
        load_reference_proxy(tmp_path, panel, target_paths={"FR": path})


def test_unaudited_naive_autumn_missing_hour_refused(tmp_path):
    path = _report(tmp_path, start="2025-10-20", end="2025-11-01", storm=False)
    text = path.read_text(encoding="utf-8")
    # Drop one occurrence of the autumn fold from just the forecast axis.
    text = text.replace('"2025-10-26T02:00:00", ', '', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="exact physical grid"):
        read_report(path, zone="FR", model="kalman")


def test_export_csv_other_delivery_day_refused(tmp_path):
    path = _report(tmp_path, end="2026-09-09")
    index = physical_index("2026-09-11", "2026-09-11")
    pd.DataFrame({"delivery_start_utc": index, "zone": "FR", "q50": 100}).to_csv(path.with_suffix(".csv"), index=False)
    with pytest.raises(EconomicDataError, match="declared complete delivery day"):
        read_report(path, zone="FR", model="kalman")


def test_source_rewritten_during_read_refused(tmp_path, monkeypatch):
    from economic_value import data
    path = _report(tmp_path, start="2026-08-31")
    real_read = Path.read_bytes
    calls = 0

    def unstable_read(self):
        nonlocal calls
        raw = real_read(self)
        if self == path:
            calls += 1
            if calls > 1:
                return raw + b"changed"
        return raw

    monkeypatch.setattr(Path, "read_bytes", unstable_read)
    with pytest.raises(EconomicDataError, match="changed during capture"):
        data.read_report(path, zone="FR", model="kalman")
