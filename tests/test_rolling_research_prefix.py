from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.evaluation import build_inference_input, _input_sha256
from chronos2_exogenous.prospective_auxiliary import MARKET_COLUMNS
import chronos2_exogenous.rolling_research_prefix as prefix


TZ = "Europe/Paris"
COVARIATES = (*MARKET_COLUMNS, "local_temperature_fcst", "flowbased_ram_p10_gw")


def _group(day="2024-09-09"):
    delivery = pd.Timestamp(day).date()
    origin = pd.Timestamp(f"{delivery - timedelta(days=1)} 08:00", tz=TZ).tz_convert("UTC")
    future = pd.date_range(pd.Timestamp(delivery, tz=TZ),
        pd.Timestamp(delivery + timedelta(days=1), tz=TZ), freq="h", inclusive="left").tz_convert("UTC")
    context = pd.date_range(end=future[0] - pd.Timedelta(hours=1), periods=2048, freq="h")
    times = context.append(future)
    group = pd.DataFrame({"timestamp": times, "origin_timestamp": origin, "item_id": "FR",
        "feature_available_at_utc": origin - pd.Timedelta(hours=1),
        "phase": ["context"] * 2048 + ["horizon"] * len(future),
        "delivery_day": day, "target": np.arange(len(times), dtype=float) / 8 + 20})
    for i, column in enumerate(COVARIATES):
        group[column] = (np.arange(len(times)) + i).astype(float) / 99
    config = SimpleNamespace(context_length=2048, prediction_length=24, frequency="h",
        cutoff_local_time="08:00", target_columns=("target",),
        lora_config={"r": 16}, known_future_covariates=COVARIATES, past_only_covariates=(),
        covariate_columns=COVARIATES, timestamp_column="timestamp",
        origin_column="origin_timestamp", item_column="item_id",
        feature_available_at_column="feature_available_at_utc", timezone=TZ)
    return group, config


class Pipeline:
    def __init__(self):
        self.calls = []

    def predict_quantiles(self, payloads, **kwargs):
        self.calls.append((payloads, kwargs))
        hours = kwargs["prediction_length"]
        return [np.tile(np.asarray([40., 50., 60.]), (1, hours, 1)) for _ in payloads], None


def test_complete_context_payload_exactly_matches_existing_builder():
    group, config = _group()
    expected, horizon, _actual = build_inference_input(group, config)
    pipeline = Pipeline()
    raw, audit = prefix.predict_prefix_group(group, config, pipeline)
    payloads, kwargs = pipeline.calls[0]
    actual_payload = payloads[0]
    np.testing.assert_array_equal(actual_payload["target"], expected["target"])
    for namespace in ("past_covariates", "future_covariates"):
        assert set(actual_payload[namespace]) == set(expected[namespace])
        for column in actual_payload[namespace]:
            np.testing.assert_array_equal(actual_payload[namespace][column], expected[namespace][column])
    assert audit["input_sha256"] == _input_sha256(expected)
    assert raw.input_sha256.eq(audit["input_sha256"]).all()
    for column in MARKET_COLUMNS:
        np.testing.assert_array_equal(raw[column].to_numpy(), group.iloc[2048:][column].to_numpy(float))
    assert pd.DatetimeIndex(raw.delivery_start_utc).equals(horizon)
    assert raw.actual.isna().all()
    assert kwargs["batch_size"] == 64 and kwargs["cross_learning"] is False
    assert kwargs["context_length"] == 2048


def test_native_context_nan_preserved_without_imputation_or_label_read():
    group, config = _group()
    group.loc[:1903, "flowbased_ram_p10_gw"] = np.nan
    group.loc[:1807, MARKET_COLUMNS[0]] = np.nan
    group["target"] = group.target.astype(object)
    # A whole-frame float conversion would fail; these values must not be read.
    group.loc[group.phase.eq("horizon"), "target"] = "DO_NOT_READ_FUTURE_TARGET"
    before = group.copy(deep=True)
    pipeline = Pipeline()
    raw, audit = prefix.predict_prefix_group(group, config, pipeline)
    payload = pipeline.calls[0][0][0]
    assert np.isnan(payload["past_covariates"]["flowbased_ram_p10_gw"]).sum() == 1904
    assert np.isnan(payload["past_covariates"][MARKET_COLUMNS[0]]).sum() == 1808
    assert all(np.isfinite(values).all() for values in payload["future_covariates"].values())
    assert audit["context_missing_by_column"]["flowbased_ram_p10_gw"] == 1904
    assert audit["context_missing_values"] == 3712
    assert audit["native_nan_mask"] is True and audit["context_imputation"] is False
    assert audit["horizon_observations_read"] is False
    assert audit["neural_in_sample"] is True and audit["neural_oof"] is False
    assert raw.actual.isna().all()
    assert np.isfinite(raw[list(MARKET_COLUMNS)]).all().all()
    pd.testing.assert_frame_equal(group, before)


def test_future_labels_cannot_change_predictions_or_input_hash():
    group, config = _group()
    first, first_audit = prefix.predict_prefix_group(group, config, Pipeline())
    group.loc[group.phase.eq("horizon"), "target"] = np.inf
    second, second_audit = prefix.predict_prefix_group(group, config, Pipeline())
    pd.testing.assert_frame_equal(first, second)
    assert first_audit["input_sha256"] == second_audit["input_sha256"]


@pytest.mark.parametrize("day,hours", [("2024-10-27", 25), ("2025-03-30", 23),
    ("2024-09-03", 24), ("2025-09-02", 24)])
def test_exact_original_boundaries_and_dst(day, hours):
    group, config = _group(day)
    pipeline = Pipeline()
    raw, audit = prefix.predict_prefix_group(group.sample(frac=1, random_state=42), config, pipeline)
    assert len(raw) == hours == audit["horizon_hours"]
    assert raw.delivery_start_utc.is_monotonic_increasing
    assert pipeline.calls[0][1]["prediction_length"] == hours
    assert audit["production_pit_evidence"] is False
    assert audit["promotion_eligible"] is False


@pytest.mark.parametrize("day", ["2024-09-02", "2025-09-03", "2026-09-08"])
def test_refuse_holdout_prospective_or_nonoriginal_prefix(day):
    group, config = _group(day)
    pipeline = Pipeline()
    with pytest.raises(prefix.RollingResearchPrefixError, match="hors du préfixe original"):
        prefix.predict_prefix_group(group, config, pipeline)
    assert not pipeline.calls


@pytest.mark.parametrize("column,row,value,match", [
    ("target", 10, np.nan, "Cible du contexte"),
    ("target", 10, np.inf, "Cible du contexte"),
    ("local_temperature_fcst", 100, np.inf, "contexte.*infinie"),
    ("local_temperature_fcst", 2050, np.nan, "future.*non finie"),
    ("local_temperature_fcst", 2050, np.inf, "future.*non finie"),
])
def test_invalid_values_rejected_before_pipeline(column, row, value, match):
    group, config = _group()
    group.loc[row, column] = value
    pipeline = Pipeline()
    with pytest.raises(prefix.RollingResearchPrefixError, match=match):
        prefix.predict_prefix_group(group, config, pipeline)
    assert not pipeline.calls


@pytest.mark.parametrize("case,match", [("origin", "exactement"), ("late", "après"),
    ("duplicate", "doublon"), ("hole", "suffixe physique"), ("wrong_phase", "phases"),
    ("multiple_items", "seule zone"), ("mixed_origins", "origine unique")])
def test_origin_availability_and_timeline_guards(case, match):
    group, config = _group()
    if case == "origin":
        group["origin_timestamp"] += pd.Timedelta(seconds=1)
    elif case == "late":
        group.loc[10, "feature_available_at_utc"] = group.origin_timestamp.iloc[0] + pd.Timedelta(seconds=1)
    elif case == "duplicate":
        group.loc[10, "timestamp"] = group.timestamp.iloc[9]
    elif case == "hole":
        group.loc[0, "timestamp"] -= pd.Timedelta(hours=1)
    elif case == "wrong_phase":
        group.loc[10, "phase"] = "horizon"
    elif case == "multiple_items":
        group.loc[10, "item_id"] = "DE"
    else:
        group.loc[10, "origin_timestamp"] += pd.Timedelta(days=1)
    pipeline = Pipeline()
    with pytest.raises(prefix.RollingResearchPrefixError, match=match):
        prefix.predict_prefix_group(group, config, pipeline)
    assert not pipeline.calls
