from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from marginal_cost_expert.governance import (
    GuardConfig, GovernanceError, calibrate, walkforward_guard,
)


def _frame(days=25, start="2025-01-01", regime="tight", sign=1.0):
    first = pd.Timestamp(start)
    index = pd.date_range(first.tz_localize("Europe/Paris"),
                          (first + pd.Timedelta(days=days)).tz_localize("Europe/Paris"),
                          inclusive="left", freq="h").tz_convert("UTC")
    local_day = index.tz_convert("Europe/Paris").tz_localize(None).normalize()
    published = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize("Europe/Paris")
    return pd.DataFrame({"timestamp": index, "zone": "BE", "base": 100.0,
                         "expert": 100.0 + 10 * sign, "actual": 100.0 + 10 * sign,
                         "risk": regime, "label_available_at_utc": published.tz_convert("UTC"),
                         "expert_oof": True, "expert_available": True})


def _config(**kwargs):
    return replace(GuardConfig(), min_history_days=4, min_regime_days=3,
                   update_every_days=1, **kwargs)


def _origin(day):
    return (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")


def test_cold_start_is_exact_base_then_only_small_grid_weights():
    source = _frame(12)
    untouched = source.copy(deep=True)
    result = walkforward_guard(source, config=_config())
    pd.testing.assert_frame_equal(source, untouched)
    pred = result.predictions
    assert pred.iloc[:4 * 24].weight.eq(0).all()
    assert pred.iloc[:4 * 24].guarded.eq(pred.iloc[:4 * 24].base).all()
    assert pred.iloc[4 * 24:5 * 24].weight.eq(0.25).all()
    assert pred.iloc[5 * 24:].weight.eq(0.5).all()
    assert set(pred.weight) <= {0, 0.1, 0.25, 0.5}
    assert not result.audit["future_non_regression_guarantee"]
    assert not result.audit["production_activation"]


def test_target_of_current_and_future_days_cannot_change_their_first_prediction():
    source = _frame(16)
    config = _config()
    reference = walkforward_guard(source, config=config).predictions
    changed = source.copy()
    changed.loc[changed.timestamp.ge(pd.Timestamp("2025-01-10", tz="Europe/Paris")), "actual"] = -99999.0
    tested = walkforward_guard(changed, config=config).predictions
    mask = reference.timestamp.lt(pd.Timestamp("2025-01-11", tz="Europe/Paris"))
    columns = ["timestamp", "weight", "guarded", "decision", "policy_cutoff_utc", "policy_history_days"]
    pd.testing.assert_frame_equal(reference.loc[mask, columns], tested.loc[mask, columns])
    assert tested.loc[~mask, "weight"].min() == 0


def test_late_labels_are_excluded_and_yesterday_auction_is_allowed():
    source = _frame(10)
    cutoff = _origin("2025-01-11")
    original = calibrate(source, cutoff=cutoff, config=_config())
    assert original.audit["zones"]["BE"]["complete_history_days"] == 10
    assert original.audit["zones"]["BE"]["history_end_day"] == "2025-01-10"
    source.loc[source.timestamp.ge(pd.Timestamp("2025-01-10", tz="Europe/Paris")),
               "label_available_at_utc"] = cutoff.tz_convert("UTC") + pd.Timedelta(seconds=1)
    delayed = calibrate(source, cutoff=cutoff, config=_config())
    assert delayed.audit["zones"]["BE"]["complete_history_days"] == 9


def test_no_oof_or_availability_evidence_fails_closed():
    source = _frame(10)
    no_oof = source.drop(columns="expert_oof")
    assert walkforward_guard(no_oof, config=_config()).predictions.weight.eq(0).all()
    no_time = source.drop(columns="label_available_at_utc")
    result = walkforward_guard(no_time, config=_config())
    assert result.predictions.weight.eq(0).all()
    assert result.policies[-1]["audit"]["label_availability"] == "missing_fail_closed"
    rejected_base = source.assign(baseline_oof=False)
    result = walkforward_guard(rejected_base, config=_config())
    assert result.predictions.weight.eq(0).all()
    assert result.predictions.decision.eq("baseline_evidence_rejected").all()


def test_regime_minimum_counts_distinct_days_not_hourly_rows():
    source = _frame(10)
    cfg = replace(_config(), min_regime_days=11)
    policy = calibrate(source, cutoff=_origin("2025-01-11"), config=cfg)
    assert policy.weights_by_zone["BE"]["tight"] == 0
    assert policy.audit["zones"]["BE"]["regimes"]["tight"]["days"] == 10
    assert policy.audit["zones"]["BE"]["regimes"]["tight"]["hours"] == 240


def test_base_only_prefix_does_not_count_as_expert_oof_warmup():
    source = _frame(16)
    source.loc[:8 * 24 - 1, "expert_oof"] = False
    result = walkforward_guard(source, config=_config())
    assert result.predictions.iloc[:12 * 24].weight.eq(0).all()
    assert result.predictions.iloc[12 * 24:].weight.gt(0).all()
    assert result.policies[12]["audit"]["zones"]["BE"]["complete_history_days"] == 4
    assert result.policies[12]["audit"]["zones"]["BE"]["complete_base_days"] == 12


@pytest.mark.parametrize(("regime", "sign"), [("tight", 1.0), ("surplus", -1.0)])
def test_both_positive_and_negative_physical_adjustments(regime, sign):
    source = _frame(10, regime=regime, sign=sign)
    result = walkforward_guard(source, config=_config())
    active = result.predictions.query("weight > 0")
    assert len(active)
    assert ((active.guarded - active.base) * sign > 0).all()
    assert (active.guarded - active.actual).abs().mean() < (active.base - active.actual).abs().mean()


def test_negative_gain_and_adverse_days_reject_weight():
    source = _frame(10)
    source["actual"] = 90.0
    policy = calibrate(source, cutoff=_origin("2025-01-11"), config=_config())
    assert policy.weights_by_zone["BE"]["tight"] == 0
    source["actual"] = 110.0
    source.loc[:23, "actual"] = 90.0
    cfg = _config(maximum_adverse_day_increase_eur_mwh=0.5)
    policy = calibrate(source, cutoff=_origin("2025-01-11"), config=cfg)
    assert policy.weights_by_zone["BE"]["tight"] == 0
    candidates = policy.audit["zones"]["BE"]["regimes"]["tight"]["candidates"]
    assert all(item["calibration_global_gain"] > 0 for item in candidates)
    assert all(not item["accepted"] for item in candidates)


def test_sources_missing_or_unknown_regime_keep_base_without_nan_arithmetic():
    source = _frame(10)
    source.loc[source.index[-24:-16], "expert"] = np.nan
    source.loc[source.index[-16:-8], "expert_available"] = False
    source.loc[source.index[-8:], "risk"] = "unrecognised"
    pred = walkforward_guard(source, config=_config()).predictions
    assert pred.iloc[-24:].weight.eq(0).all()
    assert pred.iloc[-24:].guarded.eq(100).all()
    assert pred.iloc[-8:].decision.eq("unknown_risk").all()


def test_weekly_policy_is_constant_within_civil_week_and_zone_isolated():
    source = _frame(30)
    other = source.assign(zone="NL", expert=90.0, actual=110.0)
    result = walkforward_guard(pd.concat([source, other], ignore_index=True),
                               config=replace(_config(), update_every_days=7))
    pred = result.predictions
    assert pred.loc[pred.zone.eq("NL"), "weight"].eq(0).all()
    assert pred.loc[pred.zone.eq("BE"), "weight"].max() > 0
    periods = pred.timestamp.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.to_period("W-SUN")
    assert pred.groupby(["zone", periods]).weight.nunique().max() == 1


@pytest.mark.parametrize(("start", "expected_hours"), [("2025-03-28", 95), ("2025-10-24", 97)])
def test_dst_complete_physical_days_and_civil_cutoff(start, expected_hours):
    source = _frame(4, start=start)
    assert len(source) == expected_hours
    end = pd.Timestamp(start) + pd.Timedelta(days=4)
    policy = calibrate(source, cutoff=_origin(end), config=_config())
    audit = policy.audit["zones"]["BE"]
    assert audit["complete_history_days"] == 4
    assert audit["hours"] == expected_hours
    result = walkforward_guard(source, config=replace(_config(), min_history_days=1, min_regime_days=1))
    cuts = pd.to_datetime(result.predictions.policy_cutoff_utc, utc=True).dt.tz_convert("Europe/Paris")
    assert cuts.dt.hour.eq(8).all()


def test_incomplete_day_is_not_silently_counted_and_lookback_is_local_days():
    source = _frame(10).drop(index=30)
    cfg = replace(_config(), lookback_days=5)
    policy = calibrate(source, cutoff=_origin("2025-01-11"), config=cfg)
    assert policy.audit["zones"]["BE"]["complete_history_days"] == 5
    assert policy.audit["zones"]["BE"]["history_start_day"] == "2025-01-06"
    damaged = source.drop(index=source.index[-1])
    policy = calibrate(damaged, cutoff=_origin("2025-01-11"), config=cfg)
    assert policy.audit["zones"]["BE"]["complete_history_days"] == 4
    assert policy.audit["zones"]["BE"]["skipped_incomplete_days"] == 1


def test_storm_chronos_and_current_truth_columns_are_not_inputs():
    source = _frame(9)
    reference = walkforward_guard(source, config=_config()).predictions
    added = source.assign(storm=9999.0, chronos=-9999.0, nuclear_actual=56789.0)
    tested = walkforward_guard(added, config=_config()).predictions
    for column in ("weight", "guarded", "decision", "policy_cutoff_utc"):
        pd.testing.assert_series_equal(reference[column], tested[column])


def test_invalid_schema_timestamps_and_duplicate_hours_raise():
    assert GuardConfig(weights=[0.0, 0.1, 0.25, 0.5]).weights == (0.0, 0.1, 0.25, 0.5)
    with pytest.raises(GovernanceError, match="weights"):
        GuardConfig(weights=(0.0, 0.75))
    source = _frame(2)
    with pytest.raises(GovernanceError, match="Duplicate"):
        walkforward_guard(pd.concat([source, source.iloc[:1]]), config=_config())
    source["timestamp"] = source.timestamp.dt.tz_localize(None)
    with pytest.raises(GovernanceError, match="timezone"):
        walkforward_guard(source, config=_config())


def test_current_day_never_enters_training_even_with_incorrect_early_publication():
    source = _frame(10)
    source["label_available_at_utc"] = pd.Timestamp("2020-01-01", tz="UTC")
    policy = calibrate(source, cutoff=_origin("2025-01-10"), config=_config())
    assert policy.audit["zones"]["BE"]["complete_history_days"] == 9
    assert policy.audit["zones"]["BE"]["history_end_day"] == "2025-01-09"


def test_previous_policy_from_the_future_is_rejected():
    source = _frame(10)
    later = calibrate(source, cutoff=_origin("2025-01-11"), config=_config())
    with pytest.raises(GovernanceError, match="previous_policy"):
        calibrate(source, cutoff=_origin("2025-01-10"), config=_config(), previous_policy=later)
