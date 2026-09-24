from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from nyx_scarcity_zonal.calibration import (
    ZONES, ZonalCalibrationError, apply_zone_offsets, fit_zone_offsets,
)


def day_frame(day="2026-09-10", zone="FR"):
    date = pd.Timestamp(day)
    first = date.tz_localize("Europe/Paris")
    last = (date + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    stamp = pd.date_range(first, last, freq="h", inclusive="left").tz_convert("UTC")
    origin = (date - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    publication = (date - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.DataFrame({"zone": zone, "timestamp_utc": stamp, "forecast_origin_utc": origin,
                         "label_available_at_utc": publication, "_day": day})


def fit(frame, probabilities=None, events=None, **kwargs):
    options = {"cutoff": pd.Timestamp("2026-09-13T06:00Z"), "current_day": "2026-09-14", **kwargs}
    p = np.full(len(frame), .3) if probabilities is None else probabilities
    y = np.zeros(len(frame), dtype=bool) if events is None else events
    return fit_zone_offsets(frame, p, y, **options)


def test_zero_events_gives_finite_shrunk_offset_not_probability_zero():
    frame = day_frame()
    original = frame.copy(deep=True)
    offsets, audit = fit(frame)
    assert -3 < offsets["FR"] < 0
    adjusted = apply_zone_offsets(np.full(24, .3), frame.zone, offsets)
    assert np.all((adjusted > 0) & (adjusted < .3))
    info = audit["zones"]["FR"]
    assert abs(info["gradient_at_solution"]) < 1e-9
    assert info["positive_events"] == 0 and info["negative_events"] == 24
    assert info["objective_after"] < info["objective_before"]
    assert info["effective_loss_weight"] == 24
    assert not audit["independent_hour_assumption"]
    assert not audit["probability_clipping_performed"]
    assert audit["penalty_scale"].startswith("fixed penalty against SUM")
    json.dumps(audit, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)


def test_absent_countries_are_explicit_identity_with_audit():
    offsets, audit = fit(day_frame())
    assert set(offsets) == set(ZONES)
    for zone in ["BE", "DE", "NL"]:
        assert offsets[zone] == 0
        assert audit["zones"][zone]["status"] == "no_calibration_rows_identity"
    np.testing.assert_allclose(apply_zone_offsets([.2, .8], ["BE", "DE"], offsets), [.2, .8])


def test_all_positive_calibration_is_valid_and_increases_probability():
    data = day_frame(zone="DE")
    offsets, audit = fit(data, events=np.ones(len(data), bool))
    assert 0 < offsets["DE"] <= 3
    values = apply_zone_offsets(np.full(24, .3), data.zone, offsets)
    assert np.all((values > .3) & (values < 1))
    assert audit["zones"]["DE"]["negative_events"] == 0


def test_more_events_monotonically_increase_offset():
    data = day_frame()
    values = []
    for positives in [0, 3, 6, 12, 18, 24]:
        y = np.arange(24) < positives
        offsets, _ = fit(data, events=y)
        values.append(offsets["FR"])
    assert np.diff(values).min() > 0


def test_stronger_penalty_shrinks_toward_pooled_zero():
    data = day_frame()
    a, _ = fit(data, penalty=.01)
    b, _ = fit(data, penalty=1)
    c, _ = fit(data, penalty=1000000.)
    assert abs(a["FR"]) > abs(b["FR"]) > abs(c["FR"])
    assert abs(c["FR"]) < 1e-5


def test_declared_sum_loss_not_mean_loss_gives_more_data_more_evidence():
    one = day_frame()
    two = pd.concat([one, day_frame("2026-09-11")], ignore_index=True)
    a, _ = fit(one)
    b, audit = fit(two)
    assert b["FR"] < a["FR"]
    assert audit["zones"]["FR"]["effective_loss_weight"] == 48


@pytest.mark.parametrize("day,n", [("2026-03-28", 24), ("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_complete_civil_day_has_exactly_24_loss_units(day, n):
    data = day_frame(day)
    assert len(data) == n
    offsets, audit = fit(data, probabilities=np.full(n, .4),
                         current_day="2026-10-27", cutoff=pd.Timestamp("2026-10-26T07:00Z"))
    info = audit["zones"]["FR"]
    assert info["physical_day_counts"] == {str(n): 1}
    assert info["effective_loss_weight"] == pytest.approx(24)
    gradient = 24 * expit(logit(.4) + offsets["FR"]) + offsets["FR"]
    assert abs(gradient) < 1e-9


def test_country_offsets_are_independent_shrinkage_to_same_shared_model():
    data = pd.concat([day_frame(zone=z) for z in ZONES], ignore_index=True)
    probabilities = np.full(len(data), .4)
    events = data.zone.eq("BE").to_numpy()
    offsets, _ = fit(data, probabilities, events)
    assert offsets["BE"] > 0
    assert offsets["DE"] == offsets["FR"] == offsets["NL"] < 0
    shifted = apply_zone_offsets(probabilities, data.zone, offsets)
    np.testing.assert_allclose(logit(shifted) - logit(probabilities), data.zone.map(offsets))


@pytest.mark.parametrize("p,y,expected_sign", [(.99, False, -1), (.01, True, 1)])
def test_hard_bound_and_kkt_gradient_are_consistent(p, y, expected_sign):
    data = day_frame()
    offsets, audit = fit(data, np.full(24, p), np.full(24, y), max_abs=.25)
    assert offsets["FR"] == expected_sign * .25
    info = audit["zones"]["FR"]
    assert info["at_bound"] is True
    assert info["gradient_at_solution"] * expected_sign < 0


def test_balanced_correct_shared_probability_yields_zero_offset():
    data = day_frame()
    offsets, _ = fit(data, np.full(24, .5), np.arange(24) < 12)
    assert offsets["FR"] == pytest.approx(0., abs=1e-12)


def test_near_boundary_probabilities_stay_numerically_finite():
    data = day_frame()
    p = np.tile([1e-300, np.nextafter(1., 0.)], 12)
    y = np.tile([False, True], 12)
    offsets, audit = fit(data, p, y)
    adjusted = apply_zone_offsets(p, data.zone, offsets)
    assert np.isfinite(adjusted).all()
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    assert np.isfinite(audit["zones"]["FR"]["objective_after"])


@pytest.mark.parametrize("value", [0., 1., np.nan, np.inf, -.1, 1.1])
def test_invalid_or_boundary_probabilities_rejected_without_hidden_clipping(value):
    p = np.full(24, .2)
    p[0] = value
    with pytest.raises(ZonalCalibrationError, match="strictly inside"):
        fit(day_frame(), p)
    with pytest.raises(ZonalCalibrationError, match="strictly inside"):
        apply_zone_offsets(p, ["FR"] * 24, {z: 0 for z in ZONES})


@pytest.mark.parametrize("column,value", [
    ("label_available_at_utc", pd.Timestamp("2026-09-13T06:00:01Z")),
    ("label_available_at_utc", pd.NaT),
    ("label_available_at_utc", pd.Timestamp("2026-09-08T05:00Z")),
    ("forecast_origin_utc", pd.Timestamp("2026-09-09T05:00Z")),
    ("timestamp_utc", pd.NaT),
    ("_day", "2026-09-09"),
    ("zone", "GB"),
])
def test_invalid_or_unavailable_metadata_raises_instead_of_filtering(column, value):
    data = day_frame()
    data.loc[0, column] = value
    with pytest.raises(ZonalCalibrationError):
        fit(data)


def test_current_and_future_delivery_labels_forbidden_even_if_publication_claims_known():
    for date in ["2026-09-14", "2026-09-15"]:
        data = day_frame(date)
        data["label_available_at_utc"] = pd.Timestamp("2026-09-13T06:00Z")
        with pytest.raises(ZonalCalibrationError, match="Current/future"):
            fit(data)


def test_publication_exactly_at_cutoff_is_known():
    data = day_frame()
    data["label_available_at_utc"] = pd.Timestamp("2026-09-13T06:00Z")
    _, audit = fit(data)
    assert audit["zones"]["FR"]["maximum_label_available_at_utc"] == "2026-09-13T06:00:00+00:00"


@pytest.mark.parametrize("change", ["missing_hour", "duplicate", "not_on_hour", "naive"])
def test_missing_duplicate_or_nonphysical_timestamps_are_rejected(change):
    data = day_frame()
    if change == "missing_hour":
        data = data.iloc[:-1]
    elif change == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif change == "not_on_hour":
        data.loc[0, "timestamp_utc"] += pd.Timedelta(minutes=15)
    else:
        data["timestamp_utc"] = data.timestamp_utc.dt.tz_localize(None)
    with pytest.raises(ZonalCalibrationError):
        fit(data)


def test_numpy_inputs_follow_row_order_without_sort_reassignment():
    data = pd.concat([day_frame(zone="FR"), day_frame(zone="DE")], ignore_index=True)
    p = np.linspace(.01, .7, len(data))
    y = np.arange(len(data)) % 4 == 0
    a, _ = fit(data, p, y)
    order = np.random.default_rng(5).permutation(len(data))
    b, _ = fit(data.iloc[order], p[order], y[order])
    for zone in ZONES:
        assert a[zone] == pytest.approx(b[zone], abs=1e-10)


def test_series_index_alignment_is_not_silently_ignored():
    data = day_frame()
    p = pd.Series(np.full(24, .3), index=np.arange(24)[::-1])
    with pytest.raises(ZonalCalibrationError, match="index alignment"):
        fit(data, p)
    y = pd.Series(np.zeros(24, bool), index=np.arange(24)[::-1])
    with pytest.raises(ZonalCalibrationError, match="index alignment"):
        fit(data, events=y)


@pytest.mark.parametrize("events", [np.full(24, .5), np.full(24, np.nan), np.full(24, "True"), np.zeros((24, 1))])
def test_nonbinary_missing_or_wrongshape_events_rejected(events):
    with pytest.raises(ZonalCalibrationError):
        fit(day_frame(), events=events)


@pytest.mark.parametrize("kwargs", [{"penalty": 0}, {"penalty": -1}, {"penalty": True},
                                    {"penalty": np.inf}, {"max_abs": 0}, {"max_abs": np.nan}])
def test_regularisation_parameters_are_explicit_and_valid(kwargs):
    with pytest.raises(ZonalCalibrationError):
        fit(day_frame(), **kwargs)


def test_wrong_cutoff_or_naive_cutoff_rejected():
    for cutoff in [pd.Timestamp("2026-09-13T07:00Z"), pd.Timestamp("2026-09-13T06:00")]:
        with pytest.raises(ZonalCalibrationError):
            fit(day_frame(), cutoff=cutoff)


def test_empty_schema_valid_calibration_returns_audited_identity():
    data = day_frame().iloc[0:0]
    offsets, audit = fit(data, np.array([], float), np.array([], bool))
    assert offsets == {z: 0. for z in ZONES}
    assert audit["rows"] == 0 and audit["rows_filtered"] == 0
    assert all(v["status"] == "no_calibration_rows_identity" for v in audit["zones"].values())
    json.dumps(audit, allow_nan=False)


def test_apply_requires_complete_known_country_offsets_without_silent_fallback():
    for offsets in [{"FR": 0}, {**{z: 0 for z in ZONES}, "GB": 0}, {z: np.nan for z in ZONES}]:
        with pytest.raises(ZonalCalibrationError):
            apply_zone_offsets([.2], ["FR"], offsets)
    with pytest.raises(ZonalCalibrationError):
        apply_zone_offsets([.2], ["GB"], {z: 0 for z in ZONES})
    with pytest.raises(ZonalCalibrationError):
        apply_zone_offsets([.2], ["FR", "BE"], {z: 0 for z in ZONES})


def test_unrelated_actual_or_storm_columns_do_not_enter_offset_fit():
    data = day_frame()
    a, aa = fit(data.assign(actual=100., benchmark_forecast=200.))
    b, ba = fit(data.assign(actual=[object()] * len(data), benchmark_forecast="not an input"))
    assert a == b and aa == ba


def test_fit_apply_do_not_write_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = day_frame()
    offsets, _ = fit(data)
    apply_zone_offsets(np.full(24, .3), data.zone, offsets)
    assert not list(tmp_path.iterdir())
