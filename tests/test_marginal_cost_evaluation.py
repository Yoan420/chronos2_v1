from __future__ import annotations

import base64
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from marginal_cost_expert.evaluation import (
    _array, physical_index, read_report_comparator, rolling_select, score_results,
)
from marginal_cost_expert.governance import GuardConfig, walkforward_guard


LABEL = "Chronos-2 + nucl. + correcteur + Kalman P50"


def _encoded(values, dtype="f8"):
    array = np.asarray(values, dtype=dtype)
    return {"dtype": dtype, "bdata": base64.b64encode(array.tobytes()).decode("ascii")}


def _x(index, naive=False):
    if naive:
        return index.tz_convert("Europe/Paris").tz_localize(None).astype(str).tolist()
    return index.astype(str).tolist()


def _traces(index, base=100.0, actual=101.0, storm=None, encoded=False, naive=False):
    def values(value):
        array = np.full(len(index), value) if np.isscalar(value) else np.asarray(value)
        return _encoded(array) if encoded else array.tolist()
    traces = [{"name": LABEL, "x": _x(index, naive), "y": values(base)},
              {"name": "Observé", "x": _x(index, naive), "y": values(actual)}]
    if storm is not None:
        traces.append({"name": "Storm officiel dashboard P50", "x": _x(index, naive), "y": values(storm)})
    return traces


def _html(tmp_path, plots):
    path = tmp_path / "report.html"
    # Reader must only decode these JSON arguments, never execute surrounding JS.
    parts = ["<script>throw new Error('never execute report script');</script>"]
    parts += [f"<script>Plotly.newPlot({json.dumps(str(i))}, {json.dumps(traces)}, {{}});</script>"
              for i, traces in enumerate(plots)]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


@pytest.mark.parametrize("encoded", [False, True])
def test_explicit_large_trace_wins_over_delivery_day_and_wrong_model(tmp_path, encoded):
    index = physical_index("2025-09-10", "2026-09-09")
    wrong = _traces(index, base=9000.0)
    wrong[0]["name"] = "Some other model P50"
    path = _html(tmp_path, [_traces(index[-24:], base=-123.0), wrong,
                            _traces(index, encoded=encoded)])
    result, audit = read_report_comparator(path, zone="BE", label=LABEL)
    assert len(result) == 8760
    assert result.base.eq(100).all()
    assert result.actual.eq(101).all()
    assert result.storm.isna().all()
    assert pd.DatetimeIndex(result.timestamp).equals(index)
    assert audit["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["selection"] == "explicit_fixed_reference_not_daily_oracle"
    assert not audit["reference_training_pit_certified"]


def test_paired_actual_refresh_storm_join_and_forecast_day_extension(tmp_path):
    history = physical_index("2025-09-09", "2026-09-08")
    paired_full = physical_index("2025-09-10", "2026-09-09")
    missing = pd.Timestamp("2025-10-26T00:00:00Z")
    paired = paired_full[paired_full != missing]
    path = _html(tmp_path, [_traces(history), _traces(paired, actual=102, storm=103, encoded=True)])
    result, _ = read_report_comparator(path, zone="BE", label=LABEL)
    by_time = result.set_index("timestamp")
    assert len(result) == 8784
    assert by_time.loc[paired[-1], "actual"] == 102
    assert by_time.loc[paired[-1], "base"] == 100
    assert by_time.loc[paired[-1], "storm"] == 103
    assert by_time.loc[missing, "actual"] == 101
    assert np.isnan(by_time.loc[missing, "storm"])
    assert by_time.loc[paired[0], "actual"] == 102


def test_naive_axes_use_exact_local_dst_grid_not_utc_or_duplicated_fold(tmp_path):
    index = physical_index("2025-09-10", "2026-09-09")
    missing = pd.Timestamp("2025-10-26T00:00:00Z")
    paired = index[index != missing]
    path = _html(tmp_path, [_traces(index, naive=True),
                            _traces(paired, actual=102, storm=103, naive=True)])
    with pytest.raises(ValueError, match="physical grid"):
        read_report_comparator(path, zone="BE", label=LABEL)
    result, audit = read_report_comparator(path, zone="BE", label=LABEL,
                                         allowed_missing_utc=[missing.isoformat()])
    assert pd.DatetimeIndex(result.timestamp).equals(index)
    assert result.timestamp.iloc[0] == pd.Timestamp("2025-09-09T22:00:00Z")
    row = result.set_index("timestamp")
    assert np.isnan(row.loc[missing, "storm"])
    assert row.loc[pd.Timestamp("2025-10-26T01:00:00Z"), "storm"] == 103
    assert len(audit["allowed_missing_pair_utc"]) == 1


def test_naive_dst_gap_can_be_loaded_from_matching_audit(tmp_path):
    index = physical_index("2025-09-10", "2026-09-09")
    missing = pd.Timestamp("2025-10-26T00:00:00Z")
    paired = index[index != missing]
    path = _html(tmp_path, [_traces(index, naive=True), _traces(paired, storm=103, naive=True)])
    audit_path = tmp_path / "nuclear_report_audit.json"
    audit_path.write_text(json.dumps({"zone": "BE", "storm": {
        "series": "power.price.be.euromwh.h.fcst.3mv.storm.da.cache",
        "source": {"zone": "BE"}, "missing_timestamps": [{"utc": missing.isoformat()}],
        "missing_timestamps_truncated": False}}), encoding="utf-8")
    result, audit = read_report_comparator(path, zone="BE", label=LABEL)
    assert len(result) == 8760
    assert result.storm.isna().sum() == 1
    assert audit["time_axis_audits"][0]["sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()


def test_forecast_conflict_between_full_and_paired_trace_is_rejected(tmp_path):
    index = physical_index("2025-09-10", "2026-09-09")
    path = _html(tmp_path, [_traces(index), _traces(index, base=101.0, storm=103)])
    with pytest.raises(ValueError, match="disagree"):
        read_report_comparator(path, zone="BE", label=LABEL)


def test_duplicate_or_mismatched_explicit_axes_are_rejected(tmp_path):
    index = physical_index("2025-09-10", "2026-09-09")
    traces = _traces(index)
    traces[0]["x"][10] = traces[0]["x"][9]
    with pytest.raises(ValueError, match="Duplicate"):
        read_report_comparator(_html(tmp_path, [traces]), zone="BE", label=LABEL)
    traces = _traces(index)
    traces[1]["x"] = _x(index + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="hours disagree"):
        read_report_comparator(_html(tmp_path, [traces]), zone="BE", label=LABEL)


def test_ambiguous_trace_and_missing_label_raise(tmp_path):
    index = physical_index("2025-09-10", "2026-09-09")
    traces = _traces(index)
    traces.append(traces[0])
    with pytest.raises(ValueError, match="Ambiguous"):
        read_report_comparator(_html(tmp_path, [traces]), zone="BE", label=LABEL)
    with pytest.raises(ValueError, match="No explicitly named"):
        read_report_comparator(_html(tmp_path, [_traces(index)]), zone="BE", label="Missing P50")


def test_encoded_array_dtype_shape_and_payload_guards():
    np.testing.assert_allclose(_array(_encoded([1, 2, 3], "f4")), [1, 2, 3])
    encoded = _encoded([1, 2, 3])
    encoded["shape"] = "3"
    np.testing.assert_allclose(_array(encoded), [1, 2, 3])
    with pytest.raises(ValueError, match="one-dimensional"):
        _array({**encoded, "shape": "1, 3"})
    with pytest.raises(ValueError, match="Unsupported"):
        _array({"dtype": "O", "bdata": ""})
    with pytest.raises(ValueError, match="Unexpected"):
        _array({**encoded, "execute": "evil"})
    with pytest.raises(ValueError, match="one-dimensional"):
        _array([[1, 2], [3, 4]])


def _scenario_inputs(start="2025-03-29", end="2026-03-31"):
    index = physical_index(start, end)
    scenarios = pd.concat([pd.DataFrame({"delivery_start_utc": index, "zone": "BE",
                                        "candidate_id": key, "price_eur_mwh": price})
                           for key, price in [("a", 100.0), ("b", 120.0)]], ignore_index=True)
    targets = pd.DataFrame({"timestamp": index, "zone": "BE", "actual": 100.0})
    return scenarios, targets


def test_rolling_selection_uses_exact_365_local_days_including_dst():
    candidates, targets = _scenario_inputs()
    selected, audit = rolling_select(candidates, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-30")
    assert selected.candidate_id.eq("a").all()
    assert selected.expert_oof.eq(True).all()
    days = selected.timestamp.dt.tz_convert("Europe/Paris").dt.date.astype(str)
    assert selected.groupby(days).size().to_dict() == {"2026-03-29": 23, "2026-03-30": 24}
    for row in audit.itertuples():
        train = physical_index(row.training_start, row.training_end)
        assert row.training_days == 365
        assert row.training_hours == len(train)
        assert (pd.Timestamp(row.training_end) - pd.Timestamp(row.training_start)).days == 364
        assert not row.evaluation_label_used
    origins = selected.forecast_origin_utc.dt.tz_convert("Europe/Paris")
    assert origins.dt.hour.eq(8).all()


def test_current_and_future_labels_cannot_change_scenario_for_today():
    candidates, targets = _scenario_inputs()
    reference, audit = rolling_select(candidates, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-29")
    changed = targets.copy()
    changed.loc[changed.timestamp.ge(pd.Timestamp("2026-03-29", tz="Europe/Paris")), "actual"] = 999999.0
    tested, other_audit = rolling_select(candidates, changed, evaluation_start="2026-03-29", evaluation_end="2026-03-29")
    pd.testing.assert_frame_equal(reference, tested)
    pd.testing.assert_frame_equal(audit, other_audit)
    only_past = targets.loc[targets.timestamp.lt(pd.Timestamp("2026-03-29", tz="Europe/Paris"))]
    without_today, _ = rolling_select(candidates, only_past, evaluation_start="2026-03-29", evaluation_end="2026-03-29")
    pd.testing.assert_frame_equal(reference, without_today)


def test_previous_day_labels_can_legitimately_affect_selection():
    candidates, targets = _scenario_inputs()
    targets["actual"] = 110.0  # exact tie -> stable candidate-id ordering
    reference, _ = rolling_select(candidates, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-29")
    targets.loc[targets.timestamp.between(pd.Timestamp("2026-03-28", tz="Europe/Paris"),
                                          pd.Timestamp("2026-03-29", tz="Europe/Paris"), inclusive="left"), "actual"] = 120.0
    changed, _ = rolling_select(candidates, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-29")
    assert reference.candidate_id.eq("a").all()
    assert changed.candidate_id.eq("b").all()


@pytest.mark.parametrize("missing_source", ["actual", "scenario"])
def test_missing_training_hour_refuses_silent_short_window(missing_source):
    candidates, targets = _scenario_inputs()
    if missing_source == "actual":
        targets = targets.drop(index=3)
    else:
        candidates = candidates.drop(index=3)
    with pytest.raises(ValueError, match="incomplete 365-day training"):
        rolling_select(candidates, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-29")


def test_missing_forecast_and_duplicate_candidate_are_rejected():
    candidates, targets = _scenario_inputs()
    duplicate = pd.concat([candidates, candidates.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate scenario"):
        rolling_select(duplicate, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-29")
    missing = candidates.loc[~(candidates.candidate_id.eq("a") &
                               candidates.delivery_start_utc.eq(pd.Timestamp("2026-03-29", tz="Europe/Paris")))]
    with pytest.raises(ValueError, match="incomplete expert forecast"):
        rolling_select(missing, targets, evaluation_start="2026-03-29", evaluation_end="2026-03-29")


def test_scores_include_all_baseline_hours_and_expose_missing_expert_coverage():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
                          "zone": "BE", "actual": [100., 100., 100., 100.],
                          "base": [90., 110., 120., 80.], "expert": [100., 100., 100., np.nan],
                          "guarded": [95., 105., 110., 80.], "storm": [98., 102., 102., np.nan],
                          "weight": [.5, .5, .5, 0.]})
    metrics, daily = score_results(frame)
    scored = metrics.set_index("model")
    assert scored.loc["base", "hours"] == 4
    assert scored.loc["base", "coverage"] == 1
    assert scored.loc["base", "mae"] == 15
    assert scored.loc["base", "bias"] == 0
    assert scored.loc["expert", "coverage"] == .75
    assert scored.loc["guarded", "mae"] == 10
    assert scored.loc["guarded", "activation_share"] == .75
    assert daily.set_index("model").loc["base", "mean_price_error"] == 0


def test_abstain_keeps_730_calendar_days_and_waits_until_training_gap_leaves_window():
    candidates, targets = _scenario_inputs(start="2024-09-10", end="2026-09-09")
    local_days = candidates.delivery_start_utc.dt.tz_convert("Europe/Paris").dt.date.astype(str)
    gap = local_days.between("2024-11-01", "2024-11-20")
    incomplete = candidates.loc[~gap].copy()
    assert len(set(local_days)) == 730
    assert incomplete.delivery_start_utc.dt.tz_convert("Europe/Paris").dt.date.nunique() == 710
    selected, audit = rolling_select(incomplete, targets, evaluation_start="2025-09-10",
                                      evaluation_end="2026-09-09", unavailable_policy="abstain")
    expected = physical_index("2025-09-10", "2026-09-09")
    assert pd.DatetimeIndex(selected.timestamp).equals(expected)
    assert len(selected) == 8760
    assert len(audit) == 365
    assert audit.training_days.eq(365).all()
    assert not audit.evaluation_label_used.any()
    days = selected.timestamp.dt.tz_convert("Europe/Paris").dt.date.astype(str)
    assert days.nunique() == 365
    counts = selected.groupby(days).size()
    assert counts.loc["2025-10-26"] == 25
    assert counts.loc["2026-03-29"] == 23
    before_valid = days.le("2025-11-20")
    assert selected.loc[before_valid, "expert"].isna().all()
    assert selected.loc[before_valid, "expert_oof"].eq(False).all()
    assert selected.loc[before_valid, "expert_unavailable_reason"].eq("incomplete_365_day_training_window").all()
    assert selected.loc[~before_valid, "expert"].eq(100).all()
    assert selected.loc[~before_valid, "expert_oof"].eq(True).all()
    assert audit.loc[audit.day.eq("2025-11-20"), "training_start"].item() == "2024-11-20"
    assert audit.loc[audit.day.eq("2025-11-21"), "training_start"].item() == "2024-11-21"
    for row in audit.itertuples():
        assert row.training_hours == len(physical_index(row.training_start, row.training_end))
        assert (pd.Timestamp(row.training_end) - pd.Timestamp(row.training_start)).days == 364
    # Reporting keeps the missing-expert days as baseline identity rather than
    # silently shortening the 365-day denominator to the available 293 days.
    scored = selected.assign(actual=100., base=100., guarded=100., storm=100., weight=0.)
    metrics, _ = score_results(scored)
    indexed = metrics.set_index("model")
    assert indexed.loc["base", "days"] == 365
    assert indexed.loc["guarded", "hours"] == 8760
    assert indexed.loc["expert", "days"] == 293
    assert indexed.loc["expert", "coverage"] < 1


def test_missing_delivery_inputs_abstain_and_guard_never_carries_yesterdays_expert():
    candidates, targets = _scenario_inputs(start="2025-03-27", end="2026-03-29")
    day = candidates.delivery_start_utc.dt.tz_convert("Europe/Paris").dt.date.astype(str)
    candidates = candidates.loc[day.ne("2026-03-29")]
    selected, audit = rolling_select(candidates, targets, evaluation_start="2026-03-27",
                                      evaluation_end="2026-03-29", unavailable_policy="abstain")
    day = selected.timestamp.dt.tz_convert("Europe/Paris").dt.date.astype(str)
    assert selected.loc[day.eq("2026-03-28"), "expert"].eq(100).all()
    assert selected.loc[day.eq("2026-03-29"), "expert"].isna().all()
    assert selected.loc[day.eq("2026-03-29"), "expert_oof"].eq(False).all()
    assert len(selected.loc[day.eq("2026-03-29")]) == 23
    assert audit.loc[audit.day.eq("2026-03-29"), "reason"].item() == "incomplete_forecast_inputs"
    issued = selected.assign(base=120., actual=100., risk="tight",
                              expert_available=selected.expert.notna())
    local_day = issued.timestamp.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    issued["label_available_at_utc"] = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    result = walkforward_guard(issued, config=GuardConfig(min_history_days=1,
                                                         min_regime_days=1, update_every_days=1))
    assert result.predictions.loc[day.eq("2026-03-28"), "weight"].gt(0).all()
    assert result.predictions.loc[day.eq("2026-03-28"), "guarded"].lt(120).all()
    missing_day = result.predictions.loc[day.eq("2026-03-29")]
    assert missing_day.weight.eq(0).all()
    assert missing_day.guarded.eq(missing_day.base).all()
    assert missing_day.decision.eq("expert_or_risk_unavailable").all()
