from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity.adjustment_analysis import FRACTIONS, build_adjustments


def predictions():
    stamp = pd.date_range("2026-09-12 17:00", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "zone": "DE", "timestamp_utc": stamp, "forecast_origin_utc": stamp - pd.Timedelta(days=1, hours=11),
        "sample": ["evaluation"] * 3 + ["live"], "actual": [110., 200., 70., np.nan],
        "forecast": 100., "benchmark_forecast": 120., "candidate_forecast": [100., 100., 150., 100.],
        "raw_correction": [0., 100., 100., 100.], "bounded_correction": [0., 100., 100., 100.],
        "selected_weight": [0., 0., .5, 0.], "applied_correction": [0., 0., 50., 0.],
        "spike_probability": [.1, .8, .8, .8], "probability_gate": .6,
        "expert_ready": True, "gate_reason": "annual_nonregression_refused"})
    return {name: frame.copy(deep=True) for name in ("hgb_v1", "xgb_unweighted_fixed", "xgb_weighted_fixed",
                                                  "xgb_unweighted_dwt", "xgb_weighted_dwt")}


def test_fixed_proposals_reuse_bounded_corrections_and_keep_governed_outputs():
    source = predictions()
    before = {name: frame.copy(deep=True) for name, frame in source.items()}
    long, summary = build_adjustments(source)
    assert len(long) == 20 and set(long.variant_id) == set(source)
    for label, fraction in FRACTIONS.items():
        np.testing.assert_allclose(long[label], long.forecast + fraction * long.bounded_correction)
    row = long.loc[(long.variant_id == "hgb_v1") & (long.local_day == "2026-09-13")].iloc[0]
    assert row.proposal100 == 200 and row.candidate_forecast == 100
    assert row.selected_weight == 0 and row.applied_correction == 0
    assert summary["refit_performed"] is False and summary["governance_modified"] is False
    assert summary["best_alpha_selected"] is False and summary["quantiles_created"] is False
    assert not any("q10" in name or "q90" in name for name in long.columns)
    json.dumps(summary, allow_nan=False)
    for name in source:
        pd.testing.assert_frame_equal(source[name], before[name])


def test_live_and_future_observations_cannot_change_any_proposed_price():
    original = predictions()
    before, summary = build_adjustments(original)
    changed = deepcopy(original)
    for frame in changed.values():
        frame.loc[3, "actual"] = 5000.
        frame.loc[2, "actual"] = 8000.
    after, second = build_adjustments(changed)
    pd.testing.assert_frame_equal(before[["variant_id", "timestamp_utc", *FRACTIONS]], after[["variant_id", "timestamp_utc", *FRACTIONS]])
    assert summary["live_rows_excluded"] == second["live_rows_excluded"] == 1
    assert second["overall"]["paired_hours"] == 3
    changed_only_live = predictions()
    for frame in changed_only_live.values():
        frame.loc[3, "actual"] = 7000.
    assert build_adjustments(changed_only_live)[1] == summary


def test_common_support_missing_proposal_excludes_hour_from_every_score():
    source = predictions()
    source["xgb_weighted_dwt"].loc[1, ["raw_correction", "bounded_correction", "applied_correction", "candidate_forecast"]] = np.nan
    long, summary = build_adjustments(source)
    own = long.loc[long.variant_id.eq("xgb_weighted_dwt") & long.local_day.eq("2026-09-13")]
    assert own.proposal100.isna().all()
    assert summary["overall"]["paired_hours"] == 2
    assert summary["overall"]["annual"]["nyx"]["hours"] == 2
    for scores in summary["overall"]["annual"]["variants"].values():
        assert all(score["hours"] == 2 for score in scores.values())


def test_missing_observation_is_never_zero_filled():
    source = predictions()
    for frame in source.values():
        frame.loc[1, "actual"] = np.nan
    _, summary = build_adjustments(source)
    assert summary["overall"]["paired_hours"] == 2


def test_all_missing_evaluation_labels_produce_json_safe_empty_metrics():
    source = predictions()
    for frame in source.values():
        frame["actual"] = np.nan
    _, summary = build_adjustments(source)
    assert summary["overall"]["annual"]["nyx"]["mae_eur_mwh"] is None
    assert summary["overall"]["tails"]["top_1_percent"]["threshold_eur_mwh"] is None
    json.dumps(summary, allow_nan=False)


def test_annual_and_intervention_benefit_harm_use_all_common_hours():
    _, summary = build_adjustments(predictions())
    metrics = summary["by_zone"]["DE"]
    assert metrics["annual"]["nyx"]["mae_eur_mwh"] == pytest.approx(140/3)
    assert metrics["annual"]["variants"]["hgb_v1"]["proposal25"]["mae_eur_mwh"] == pytest.approx(140/3)
    impact = metrics["interventions"]["hgb_v1"]["proposal25"]
    assert impact["active_hours"] == 2 and impact["precision_beneficial"] == .5
    assert impact["improved_absolute_error_hours"] == impact["harmed_absolute_error_hours"] == 1
    assert impact["total_benefit_eur_mwh"] == impact["total_harm_eur_mwh"] == 25.
    assert impact["net_gain_eur_mwh"] == 0
    tail = metrics["tails"]["top_1_percent"]
    assert tail["scores"]["nyx"]["hours"] == 1 and tail["used_for_training_or_selection"] is False
    assert tail["scores"]["variants"]["hgb_v1"]["proposal100"]["mae_eur_mwh"] == 0


@pytest.mark.parametrize("column,value,match", [
    ("raw_correction", -1., "nonnegative"),
    ("bounded_correction", 401., "declared clip"),
    ("bounded_correction", 80., "saved raw correction"),
    ("applied_correction", 1., "selected_weight"),
    ("candidate_forecast", 101., "governed price"),
    ("expert_ready", False, "probability gate"),
    ("spike_probability", .6, "probability gate"),
    ("raw_correction", float("inf"), "infinity"),
])
def test_invalid_saved_arithmetic_or_gate_is_rejected(column, value, match):
    source = predictions()
    source["xgb_weighted_fixed"].loc[1, column] = value
    with pytest.raises(ValueError, match=match):
        build_adjustments(source)


def test_explicit_clip_is_respected_without_reclipping_saved_values():
    source = predictions()
    for frame in source.values():
        frame.loc[1, "raw_correction"] = 800.
        frame.loc[1, "bounded_correction"] = 400.
    long, _ = build_adjustments(source)
    assert long.loc[long.local_day.eq("2026-09-13"), "proposal100"].eq(500).all()
    with pytest.raises(ValueError, match="declared clip"):
        build_adjustments(source, correction_clip_eur_mwh=300.)


@pytest.mark.parametrize("clip", [True, 0, -1, "400", np.nan, np.inf])
def test_invalid_clip_parameter(clip):
    with pytest.raises(ValueError, match="finite positive"):
        build_adjustments(predictions(), correction_clip_eur_mwh=clip)


@pytest.mark.parametrize("change", ["missing_row", "duplicate", "actual", "sample", "missing_column"])
def test_identity_shapes_shared_fields_and_schema_validated(change):
    source = predictions()
    frame = source["xgb_weighted_dwt"]
    if change == "missing_row":
        source["xgb_weighted_dwt"] = frame.iloc[:-1]
    elif change == "duplicate":
        source["xgb_weighted_dwt"] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif change == "missing_column":
        source["xgb_weighted_dwt"] = frame.drop(columns="bounded_correction")
    else:
        frame.loc[0, change] = 111. if change == "actual" else "live"
    with pytest.raises(ValueError):
        build_adjustments(source)


def test_order_alignment_does_not_depend_on_incoming_dataframe_order():
    source = predictions()
    expected, _ = build_adjustments(source)
    source["xgb_weighted_dwt"] = source["xgb_weighted_dwt"].iloc[::-1]
    actual, _ = build_adjustments(source)
    pd.testing.assert_frame_equal(expected, actual)


def test_dst_fall_repeated_civil_hour_keeps_two_physical_forecasts():
    source = predictions()
    for frame in source.values():
        frame["timestamp_utc"] = pd.date_range("2025-10-26 00:00", periods=4, freq="h", tz="UTC")
        frame["forecast_origin_utc"] = pd.Timestamp("2025-10-25 06:00", tz="UTC")
    long, _ = build_adjustments(source)
    own = long.loc[long.variant_id.eq("hgb_v1")]
    assert own.local_hour.iloc[:2].tolist() == [2, 2]
    assert own.timestamp_utc.nunique() == 4 and own.local_label.iloc[0] != own.local_label.iloc[1]


def test_only_last_365_civil_days_are_scored():
    source = predictions()
    for frame in source.values():
        frame.loc[0, "timestamp_utc"] = pd.Timestamp("2025-01-01 17:00", tz="UTC")
        frame.loc[0, "forecast_origin_utc"] = pd.Timestamp("2024-12-31 06:00", tz="UTC")
    _, summary = build_adjustments(source)
    assert summary["overall"]["paired_hours"] == 2
    assert summary["windows"]["DE"]["requested_start_day"] == "2025-09-15"
    assert summary["windows"]["DE"]["complete_365_common_support"] is False


def test_no_available_expert_keeps_identity_but_is_retained_in_annual_scores():
    source = predictions()
    for frame in source.values():
        frame["expert_ready"] = False
        frame["spike_probability"] = np.nan
        frame[["raw_correction", "bounded_correction", "applied_correction", "selected_weight"]] = 0.
        frame["candidate_forecast"] = frame.forecast
    long, summary = build_adjustments(source)
    assert long.proposal100.equals(long.forecast)
    assert summary["overall"]["paired_hours"] == 3
    assert summary["overall"]["interventions"]["hgb_v1"]["proposal100"]["active_hours"] == 0
