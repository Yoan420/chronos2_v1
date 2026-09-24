from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from chronos2_hourly.model_storm_storm_calibration import calibrate_storm_quantiles


ZONE = "Europe/Berlin"


def rows_for_days(start, days, *, error=lambda day, hour: float(day), point=100.):
    first = pd.Timestamp(start).tz_localize(ZONE)
    stop = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize(ZONE)
    rows = {}
    for timestamp in pd.date_range(first, stop, freq="h", inclusive="left").tz_convert("UTC"):
        local = timestamp.tz_convert(ZONE)
        offset = (local.date() - first.date()).days
        rows[timestamp] = {"storm": point, "observed": point + error(offset, local.hour)}
    return rows


def on_day(rows, day):
    return [t for t in rows if t.tz_convert(ZONE).date().isoformat() == day]


def test_default_warmup_exactly_sixty_complete_days_then_centered_interval():
    rows = rows_for_days("2026-01-01", 62)
    before = deepcopy(rows)
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE)
    assert rows == before
    assert len(updates) == len(rows)
    assert all(day["status"] == "warmup" for day in audit["days"][:60])
    assert all(v is None for t in list(rows)[:60 * 24] for v in updates[t].values())
    target = audit["days"][60]
    assert target["history_days"] == 60 and target["status"] == "available"
    assert target["train_last_day"] == "2026-03-01"
    timestamp = on_day(rows, "2026-03-02")[0]
    # Errors 0..59: q10=5.9, q50=29.5, q90=53.1.
    assert updates[timestamp] == pytest.approx({"storm_p10": 76.4, "storm_p90": 123.6})
    assert audit["native_storm_quantiles"] is False
    assert audit["nominal_coverage_verified"] is False


def test_current_and_future_actual_perturbations_do_not_change_current_intervals():
    rows = rows_for_days("2026-01-01", 63)
    current = "2026-03-02"
    changed = deepcopy(rows)
    for timestamp, row in changed.items():
        if timestamp.tz_convert(ZONE).date().isoformat() >= current:
            row["observed"] = None if timestamp.hour % 2 else 1e9
    original, audit_a = calibrate_storm_quantiles(rows, timezone=ZONE)
    perturbed, audit_b = calibrate_storm_quantiles(changed, timezone=ZONE)
    assert {t: original[t] for t in on_day(rows, current)} == {t: perturbed[t] for t in on_day(rows, current)}
    assert audit_a["days"][60] == audit_b["days"][60]


def test_forecast_day_needs_no_actual_values():
    rows = rows_for_days("2026-01-01", 61)
    for timestamp in on_day(rows, "2026-03-02"):
        rows[timestamp].pop("observed")
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE)
    assert audit["days"][-1]["status"] == "available"
    assert all(updates[t]["storm_p10"] is not None for t in on_day(rows, "2026-03-02"))


def test_incomplete_and_nonfinite_training_days_are_excluded_whole():
    rows = rows_for_days("2026-01-01", 64)
    del rows[on_day(rows, "2026-01-02")[4]]
    rows[on_day(rows, "2026-01-03")[4]]["observed"] = float("nan")
    rows[on_day(rows, "2026-01-04")[4]]["storm"] = float("inf")
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE)
    assert audit["days"][-2]["history_days"] == 59
    assert audit["days"][-2]["status"] == "warmup"
    assert audit["days"][-1]["history_days"] == 60
    assert audit["days"][-1]["status"] == "available"
    assert set(audit["incomplete_training_days"]) == {"2026-01-02", "2026-01-03", "2026-01-04"}
    assert all(count == 60 for count in audit["days"][-1]["hour_samples"].values())
    assert len(updates) == len(rows)


def test_calendar_lookback_does_not_reach_across_missing_dates():
    rows = rows_for_days("2026-01-01", 10)
    rows.update(rows_for_days("2026-02-01", 1))
    _, audit = calibrate_storm_quantiles(rows, timezone=ZONE, lookback_days=5, min_history_days=1, min_hour_samples=1)
    assert audit["days"][-1]["history_days"] == 0
    assert audit["days"][-1]["status"] == "warmup"
    assert audit["days"][-1]["train_last_day"] is None
    assert audit["days"][8]["history_days"] == 5
    assert audit["days"][8]["train_first_day"] == "2026-01-04"


def test_large_historical_bias_is_removed_and_point_is_preserved():
    rows = rows_for_days("2026-01-01", 61, error=lambda day, hour: 100000. + day + 10 * hour)
    updates, _ = calibrate_storm_quantiles(rows, timezone=ZONE)
    for timestamp in on_day(rows, "2026-03-02"):
        assert updates[timestamp] == pytest.approx({"storm_p10": 76.4, "storm_p90": 123.6})
        assert updates[timestamp]["storm_p10"] <= rows[timestamp]["storm"] <= updates[timestamp]["storm_p90"]


@pytest.mark.parametrize(("start", "transition", "hours", "hour2_samples"), [
    ("2026-03-28", "2026-03-29", 23, 1),
    ("2026-10-24", "2026-10-25", 25, 3),
])
def test_complete_dst_days_preserve_physical_hours_and_local_hour_pool(start, transition, hours, hour2_samples):
    rows = rows_for_days(start, 3)
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE, min_history_days=1, min_hour_samples=1)
    assert len(on_day(rows, transition)) == hours
    assert len(updates) == len(rows)
    assert audit["incomplete_training_days"] == []
    assert audit["days"][-1]["history_days"] == 2
    assert audit["days"][-1]["hour_samples"]["2"] == hour2_samples
    assert audit["days"][-1]["hour_samples"]["3"] == 2
    assert audit["days"][-1]["status"] == "available"


def test_hour_minimum_can_withhold_only_the_dst_hour():
    rows = rows_for_days("2026-03-28", 3)
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE, min_history_days=2, min_hour_samples=2)
    assert audit["days"][-1]["status"] == "insufficient_hour_samples"
    for timestamp in on_day(rows, "2026-03-30"):
        available = updates[timestamp]["storm_p10"] is not None
        assert available == (timestamp.tz_convert(ZONE).hour != 2)


def test_unsorted_inputs_produce_identical_results_and_chronological_audit():
    rows = rows_for_days("2026-10-24", 3)
    reverse = dict(reversed(list(rows.items())))
    kwargs = {"timezone": ZONE, "min_history_days": 1, "min_hour_samples": 1}
    assert calibrate_storm_quantiles(rows, **kwargs) == calibrate_storm_quantiles(reverse, **kwargs)


def test_partial_target_and_missing_storm_are_not_filled_or_mutated():
    rows = rows_for_days("2026-01-01", 61)
    target = on_day(rows, "2026-03-02")
    del rows[target[-1]]
    rows[target[0]]["storm"] = None
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE)
    assert target[-1] not in updates
    assert updates[target[0]] == {"storm_p10": None, "storm_p90": None}
    assert updates[target[1]]["storm_p10"] is not None
    assert audit["days"][-1]["status"] == "missing_storm_or_nonfinite_interval"
    assert audit["days"][-1]["forecast_hours"] == 23


@pytest.mark.parametrize("field,value", [
    ("lookback_days", 0), ("lookback_days", 1.5), ("min_history_days", -1),
    ("min_hour_samples", True), ("min_hour_samples", 0),
])
def test_protocol_parameters_must_be_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        calibrate_storm_quantiles({}, timezone=ZONE, **{field: value})


def test_naive_timestamps_rejected_and_empty_input_valid():
    with pytest.raises(ValueError, match="timezone-aware"):
        calibrate_storm_quantiles({pd.Timestamp("2026-01-01"): {"storm": 1.}}, timezone=ZONE)
    updates, audit = calibrate_storm_quantiles({}, timezone=ZONE)
    assert updates == {} and audit["days"] == []


def test_hourly_distributions_are_not_pooled_across_hours():
    rows = rows_for_days("2026-01-01", 61, error=lambda day, hour: day * (hour + 1))
    updates, _ = calibrate_storm_quantiles(rows, timezone=ZONE)
    target = on_day(rows, "2026-03-02")
    assert updates[target[0]]["storm_p90"] == pytest.approx(123.6)
    assert updates[target[1]]["storm_p90"] == pytest.approx(147.2)


def test_constant_past_errors_allow_a_data_derived_degenerate_interval():
    rows = rows_for_days("2026-01-01", 61, error=lambda day, hour: 7.)
    updates, audit = calibrate_storm_quantiles(rows, timezone=ZONE)
    assert audit["days"][-1]["status"] == "available"
    assert updates[on_day(rows, "2026-03-02")[0]] == {"storm_p10": 100., "storm_p90": 100.}
