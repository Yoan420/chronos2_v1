import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.prospective_bootstrap import BootstrapPlanningError, plan_bootstrap


TZ = "Europe/Paris"


def _hours(start, end=None):
    return pd.date_range(pd.Timestamp(start, tz=TZ),
                         (pd.Timestamp(end or start) + pd.Timedelta(days=1)).tz_localize(TZ),
                         freq="h", inclusive="left").tz_convert("UTC")


def _history(start="2026-09-02", end=None):
    return pd.DataFrame({"delivery_start_utc": _hours(start, end), "actual": 50.0})


def _targets(start="2026-09-03", end="2026-09-08"):
    return pd.Series(50.0, index=_hours(start, end))


def test_country_missing_auction_does_not_block_other_countries():
    histories = {zone: _history() for zone in ["FR", "DE", "BE", "NL"]}
    targets = {zone: _targets(end="2026-09-07" if zone in {"DE", "NL"} else "2026-09-08")
               for zone in histories}
    result = plan_bootstrap(histories, targets, "2026-09-08")
    assert result["ready_by_zone"]["FR"] == result["ready_by_zone"]["BE"] == [
        f"2026-09-{day:02d}" for day in range(3, 9)]
    assert result["ready_by_zone"]["DE"] == result["ready_by_zone"]["NL"] == [
        f"2026-09-{day:02d}" for day in range(3, 8)]
    assert [row["zone"] for row in result["pending"]] == ["DE", "NL"]
    for row in result["pending"]:
        assert row["delivery_day"] == "2026-09-08"
        assert row["observed_hours"] == 0 and row["expected_hours"] == 24
        assert row["missing_hours_utc"] == [stamp.isoformat() for stamp in _hours("2026-09-08")]
        assert row["reason"] == "observations_unavailable"
    assert result["already_complete"] == []


def test_resume_reuses_completed_country_without_even_requiring_its_snapshot():
    result = plan_bootstrap({"FR": _history(end="2026-09-08"), "DE": _history(end="2026-09-07")},
                            {"DE": _targets()}, "2026-09-08")
    assert result == {"ready_by_zone": {"FR": [], "DE": ["2026-09-08"]},
                      "pending": [], "already_complete": ["FR"]}


def test_no_work_is_a_valid_idempotent_plan():
    assert plan_bootstrap({"FR": _history(end="2026-09-08")}, {}, "2026-09-07") == {
        "ready_by_zone": {"FR": []}, "pending": [], "already_complete": ["FR"]}


def test_first_missing_day_stops_suffix_even_when_later_days_exist():
    target = _targets().drop(_hours("2026-09-05"))
    result = plan_bootstrap({"FR": _history()}, {"FR": target}, "2026-09-08")
    assert result["ready_by_zone"]["FR"] == ["2026-09-03", "2026-09-04"]
    assert len(result["pending"]) == 1
    assert result["pending"][0]["delivery_day"] == "2026-09-05"


@pytest.mark.parametrize("day,previous,expected", [
    ("2026-03-29", "2026-03-28", 23), ("2025-10-26", "2025-10-25", 25),
])
def test_exact_complete_dst_days_are_ready(day, previous, expected):
    target = _targets(day, day)
    assert len(target) == expected
    result = plan_bootstrap({"FR": _history(previous)}, {"FR": target}, day)
    assert result["ready_by_zone"] == {"FR": [day]}
    assert result["pending"] == []


@pytest.mark.parametrize("day,previous,expected", [
    ("2026-03-29", "2026-03-28", 23), ("2025-10-26", "2025-10-25", 25),
])
def test_dst_missing_hour_is_not_interpolated_or_merged(day, previous, expected):
    target = _targets(day, day)
    missing = target.index[3]
    target = target.drop(missing)
    result = plan_bootstrap({"FR": _history(previous)}, {"FR": target}, day)
    assert result["ready_by_zone"] == {"FR": []}
    assert result["pending"] == [{"zone": "FR", "delivery_day": day,
        "observed_hours": expected - 1, "expected_hours": expected,
        "missing_hours_utc": [missing.isoformat()], "reason": "observations_incomplete"}]


def test_nan_and_infinite_labels_count_as_missing_without_mutating_inputs():
    history, target = _history(), _targets()
    target.iloc[:3] = [np.nan, np.inf, -np.inf]
    before_history, before_target = history.copy(deep=True), target.copy(deep=True)
    result = plan_bootstrap({"FR": history}, {"FR": target}, "2026-09-08")
    assert result["pending"][0]["observed_hours"] == 21
    assert result["pending"][0]["missing_hours_utc"] == [stamp.isoformat() for stamp in target.index[:3]]
    pd.testing.assert_frame_equal(history, before_history)
    pd.testing.assert_series_equal(target, before_target)


def test_empty_snapshot_is_pending_not_an_error():
    result = plan_bootstrap({"FR": _history()}, {"FR": pd.Series(dtype=float)}, "2026-09-03")
    assert result["pending"][0]["observed_hours"] == 0


def test_nullable_numeric_observations_are_supported():
    target = _targets().astype("Float64")
    target.iloc[0] = pd.NA
    assert plan_bootstrap({"FR": _history()}, {"FR": target}, "2026-09-03")["pending"][0]["observed_hours"] == 23


@pytest.mark.parametrize("transform", [
    lambda frame: frame.iloc[1:],
    lambda frame: pd.concat([frame, frame.iloc[:1]]),
    lambda frame: frame.assign(delivery_start_utc=frame.delivery_start_utc.dt.tz_localize(None)),
    lambda frame: frame.assign(delivery_start_utc=frame.delivery_start_utc + pd.Timedelta(minutes=15)),
    lambda frame: frame.loc[~frame.delivery_start_utc.isin(_hours("2026-09-04"))],
])
def test_invalid_or_gapped_histories_refused_before_planning(transform):
    with pytest.raises(BootstrapPlanningError, match="FR"):
        plan_bootstrap({"FR": transform(_history(end="2026-09-05"))}, {"FR": _targets()}, "2026-09-08")


@pytest.mark.parametrize("transform", [
    lambda target: pd.concat([target, target.iloc[:1]]),
    lambda target: target.set_axis(target.index.tz_localize(None)),
    lambda target: target.set_axis(target.index + pd.Timedelta(minutes=15)),
    lambda target: target.astype(str).replace("50.0", "not-a-price"),
])
def test_ambiguous_target_snapshots_are_not_treated_as_unpublished(transform):
    with pytest.raises(BootstrapPlanningError, match="FR"):
        plan_bootstrap({"FR": _history()}, {"FR": transform(_targets())}, "2026-09-08")


def test_inputs_may_be_unsorted_without_changing_the_plan():
    result = plan_bootstrap({"FR": _history().iloc[::-1]}, {"FR": _targets().iloc[::-1]}, "2026-09-03")
    assert result["ready_by_zone"] == {"FR": ["2026-09-03"]}


@pytest.mark.parametrize("value", ["2026-9-8", "2026-09-08T00:00Z", "2026-09-08T01:00", "NaT", "2026-02-30", None])
def test_invalid_end_day_refused(value):
    with pytest.raises(BootstrapPlanningError):
        plan_bootstrap({"FR": _history()}, {"FR": _targets()}, value)


@pytest.mark.parametrize("histories,targets", [({}, {}), ({"FR": pd.DataFrame()}, {}), ({"FR": _history()}, {})])
def test_missing_required_inputs_are_errors(histories, targets):
    with pytest.raises(BootstrapPlanningError):
        plan_bootstrap(histories, targets, "2026-09-08")
