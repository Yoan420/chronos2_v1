"""Synthetic contracts for the corrector-only 2x2 ablation; no real run."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.solar_wind_corrector_interaction as subject
from chronos2_hourly.solar_wind_interaction_features import build_interaction


def _base(raw=None):
    count = len(raw) if raw is not None else 6
    index = pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC",
                          name="delivery_start_utc")
    return pd.DataFrame({"q10": np.full(count, 80.), "q50": np.full(count, 100.),
                         "q90": np.full(count, 130.)}, index=index)


def _panel(zone="DE", timezone="Europe/Berlin", days=20):
    start = pd.Timestamp("2026-01-01", tz=timezone)
    end = pd.Timestamp("2026-01-01") + pd.Timedelta(days=days)
    index = pd.date_range(start, end.tz_localize(timezone), freq="h", inclusive="left").tz_convert("UTC")
    hours = index.tz_convert(timezone).hour.to_numpy()
    return pd.DataFrame({
        f"{zone.lower()}_wind_generation_fcst": 1. + hours / 3,
        f"{zone.lower()}_solar_generation_fcst": np.where((hours >= 8) & (hours <= 17), 6., 0.),
        f"{zone.lower()}_residual_load_fcst": 10. + hours,
    }, index=index)


@pytest.mark.parametrize("upper", [40., 80.])
def test_asymmetric_clip_preserves_lower_cap_and_all_quantile_widths(upper):
    raw = np.array([-200., -40., -12., 0., 22., 40., 55., 80., 200.])
    base = _base(raw)
    before = base.copy(deep=True)
    corrected = subject.apply_variant(base, raw, upper=upper)
    applied = np.clip(raw, -40., upper)
    np.testing.assert_array_equal(corrected.to_numpy(), base.to_numpy() + applied[:, None])
    np.testing.assert_array_equal(corrected.attrs["residual_correction"], applied)
    np.testing.assert_array_equal(corrected.q90 - corrected.q10, base.q90 - base.q10)
    assert (corrected.q10 <= corrected.q50).all() and (corrected.q50 <= corrected.q90).all()
    pd.testing.assert_frame_equal(base, before)


def test_higher_cap_does_not_force_any_upward_correction():
    raw = np.array([-80., -20., 0., 15., 40., 55.])
    base = _base(raw)
    low = subject.apply_variant(base, raw, upper=40.)
    high = subject.apply_variant(base, raw, upper=80.)
    pd.testing.assert_frame_equal(low.iloc[:-1], high.iloc[:-1])
    assert high.q50.iloc[-1] - low.q50.iloc[-1] == 15.
    assert high.attrs["residual_correction"].iloc[0] == -40.


def test_scaling_is_applied_before_asymmetric_clip():
    raw = np.array([-100., 0., 30., 100.])
    base = _base(raw)
    corrected = subject.apply_variant(base, raw, upper=80., scale=.5)
    np.testing.assert_array_equal(corrected.attrs["residual_correction"], [-40., 0., 15., 50.])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_raw_correction_fails_closed(bad):
    raw = np.array([0., 1., bad])
    with pytest.raises(ValueError):
        subject.apply_variant(_base(raw), raw, upper=80.)


@pytest.mark.parametrize("upper", [-40., 0., 60., 100., float("inf")])
def test_unapproved_upper_cap_is_rejected(upper):
    with pytest.raises(ValueError):
        subject.apply_variant(_base(), np.zeros(6), upper=upper)


def test_quantile_crossing_is_not_repaired_silently():
    base = _base()
    base.iloc[0, 0] = 200.
    with pytest.raises(ValueError):
        subject.apply_variant(base, np.zeros(len(base)), upper=80.)


def test_corrector_feature_is_added_once_without_touching_baseline_or_kalman():
    kalman_covariates = _panel()
    feature, _ = build_interaction(kalman_covariates, "DE", "Europe/Berlin")
    features = kalman_covariates.rename(columns=lambda value: f"known_{value}_oracle")
    cov_before, features_before = kalman_covariates.copy(deep=True), features.copy(deep=True)
    augmented = subject.add_corrector_feature(features, feature)
    assert list(augmented) == [*features, *feature]
    pd.testing.assert_frame_equal(augmented[features.columns], features)
    pd.testing.assert_frame_equal(augmented[feature.columns], feature)
    assert not set(feature).intersection(kalman_covariates.columns)
    augmented.iloc[0, 0] += 50.
    pd.testing.assert_frame_equal(kalman_covariates, cov_before)
    pd.testing.assert_frame_equal(features, features_before)


@pytest.mark.parametrize("kind", ["duplicate", "multiple_columns", "missing_hour", "extra_hour", "nonfinite", "outside_bounds"])
def test_invalid_interaction_feature_fails_closed(kind):
    features = _panel()
    feature, _ = build_interaction(features, "DE", "Europe/Berlin")
    if kind == "duplicate":
        features = features.join(feature)
    elif kind == "multiple_columns":
        feature["unexpected_extra_interaction"] = 0.
    elif kind == "missing_hour":
        feature = feature.iloc[:-1]
    elif kind == "extra_hour":
        feature = pd.concat([feature, pd.DataFrame(0., index=[feature.index[-1] + pd.Timedelta(hours=1)], columns=feature.columns)])
    elif kind == "nonfinite":
        feature.iloc[-1, 0] = np.nan
    else:
        feature.iloc[-1, 0] = 1.01
    with pytest.raises(ValueError):
        subject.add_corrector_feature(features, feature)


@pytest.mark.parametrize("zone,timezone", [("DE", "Europe/Berlin"), ("NL", "Europe/Amsterdam")])
def test_interaction_is_prefix_causal_and_does_not_read_actual_price(zone, timezone):
    covariates = _panel(zone, timezone)
    covariates["actual"] = 0.
    feature, audit = build_interaction(covariates, zone, timezone)
    changed = covariates.copy(deep=True)
    changed["actual"] = np.nan
    changed.iloc[-24:, :3] *= 100.
    rerun, _ = build_interaction(changed, zone, timezone)
    pd.testing.assert_frame_equal(feature.iloc[:-24], rerun.iloc[:-24])
    assert audit["prices_or_targets_used"] is False
    assert audit["pit_publication_evidence_verified"] is False
    for row in audit["normalizations"]:
        if row["normalization_training_last_day"] is not None:
            assert row["normalization_training_last_day"] < row["delivery_day"]


@pytest.mark.parametrize("timezone", ["Europe/Berlin", "Europe/Amsterdam"])
@pytest.mark.parametrize("day", [date(2026, 3, 29), date(2026, 9, 22), date(2025, 10, 26)])
def test_training_window_uses_prior_365_civil_days_and_excludes_current_and_future(timezone, day):
    start = pd.Timestamp(day) - pd.Timedelta(days=367)
    stop = pd.Timestamp(day) + pd.Timedelta(days=2)
    index = pd.date_range(start.tz_localize(timezone), stop.tz_localize(timezone),
                          freq="h", inclusive="left").tz_convert("UTC")
    selected = subject.train_window(index, day, timezone)
    lower = (pd.Timestamp(day) - pd.Timedelta(days=365)).date()
    expected = index[(index.tz_convert(timezone).date >= lower) & (index.tz_convert(timezone).date < day)]
    pd.testing.assert_index_equal(selected, expected)
    assert selected[-1].tz_convert(timezone).date() < day
    assert selected[0].tz_convert(timezone).date() == lower


def test_absent_checkpoint_is_not_a_completed_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    assert subject.verify_day_checkpoint(tmp_path / "absent.json", "identity", date(2026, 1, 1), _base().index) is None


def _checkpoint(index):
    return {"identity": "identity", "day": "2026-01-01", "index_ns": index.asi8.tolist(),
            "raw": {"original": [0.] * len(index), "interaction": [50.] * len(index)},
            "audit": {"causality_violations": 0}}


def _store_checkpoint(path, payload):
    path.write_bytes(subject.json_bytes({"payload": payload,
        "sha256": hashlib.sha256(subject.json_bytes(payload)).hexdigest()}))


def test_valid_checkpoint_returns_unchanged_raw_predictions_without_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    path, index = tmp_path / "day.json", _base().index
    payload = _checkpoint(index)
    _store_checkpoint(path, payload)
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    assert subject.verify_day_checkpoint(path, "identity", date(2026, 1, 1), index) == payload
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified


@pytest.mark.parametrize("kind", ["checksum", "identity", "day", "index", "short_raw", "nonfinite", "missing_model"])
def test_checkpoint_integrity_and_identity_are_not_silent_cache_misses(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    path, index = tmp_path / "day.json", _base().index
    payload = _checkpoint(index)
    if kind == "identity":
        payload["identity"] = "other"
    elif kind == "day":
        payload["day"] = "2026-01-02"
    elif kind == "index":
        payload["index_ns"][0] += 1
    elif kind == "short_raw":
        payload["raw"]["original"].pop()
    elif kind == "nonfinite":
        payload["raw"]["interaction"][0] = "nan"
    elif kind == "missing_model":
        del payload["raw"]["interaction"]
    _store_checkpoint(path, payload)
    if kind == "checksum":
        altered = json.loads(path.read_text(encoding="utf-8"))
        altered["payload"]["raw"]["interaction"][0] = 60.
        path.write_text(json.dumps(altered), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        subject.verify_day_checkpoint(path, "identity", date(2026, 1, 1), index)
    assert path.read_bytes() == before


def test_alignment_adds_zero_only_before_calibrated_support_preserving_exact_score():
    feature, _ = build_interaction(_panel(), "DE", "Europe/Berlin")
    before = feature.copy(deep=True)
    prefix = pd.date_range(feature.index[0] - pd.Timedelta(days=12),
                           feature.index[0], freq="h", inclusive="left")
    full = prefix.append(feature.index)
    aligned = subject.align_interaction_prefix(feature, full)
    assert aligned.loc[prefix].eq(0).all().all()
    pd.testing.assert_frame_equal(aligned.loc[feature.index], feature, check_exact=True)
    pd.testing.assert_frame_equal(feature, before)


@pytest.mark.parametrize("kind", ["internal_gap", "forecast_gap"])
def test_alignment_does_not_zero_fill_an_internal_or_future_hour(kind):
    feature, _ = build_interaction(_panel(), "DE", "Europe/Berlin")
    index = feature.index
    if kind == "internal_gap":
        feature = feature.drop(index[24])
    else:
        index = index.append(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)]))
    with pytest.raises(ValueError):
        subject.align_interaction_prefix(feature, index)


def _prepared_identity(zone="DE"):
    return {"run_id": "test-identity-" + zone, "zone": zone, "engine": subject.ENGINE}


def test_validate_never_creates_output_locks_or_fits(monkeypatch, tmp_path):
    output = tmp_path / "must-remain-absent"
    monkeypatch.setattr(subject, "OUTPUT", output)
    monkeypatch.setattr(subject, "prepare", lambda zone, threads: SimpleNamespace(identity=_prepared_identity(zone)))
    def forbidden(*args, **kwargs):
        pytest.fail("Read-only validate must not fit, replay or acquire a write lock")
    for name in ("corrector_controls", "baseline_controls", "replay_correctors", "replay_variant", "exclusive_process_lock"):
        monkeypatch.setattr(subject, name, forbidden)
    result = subject.run(action="validate", zones=("DE", "NL"), threads=2, workers=2)
    assert result == {zone: _prepared_identity(zone) for zone in ("DE", "NL")}
    assert not output.exists()


@pytest.mark.parametrize("kwargs", [
    {"action": "report"}, {"zones": ()}, {"zones": ("FR",)}, {"zones": ("DE", "DE")},
    {"threads": 3}, {"threads": 0}, {"workers": 3}, {"workers": 0},
])
def test_unapproved_scope_is_rejected_before_any_source_is_read(monkeypatch, kwargs):
    monkeypatch.setattr(subject, "prepare", lambda *args, **kw: pytest.fail("Invalid scope must fail before prepare"))
    with pytest.raises(ValueError):
        subject.run(**kwargs)


def _fake_completed(monkeypatch, tmp_path):
    identity = _prepared_identity()
    directory = tmp_path / "de" / identity["run_id"]
    directory.mkdir(parents=True)
    files = ["experiment.json", "baseline_controls.json", "interaction.parquet", "feature_audit.json",
             "corrector_audit.json", "metrics.json", "report.html"]
    files += [name + "/" + item for name in subject.VARIANTS for item in (
        "upstream_history.parquet", "upstream_forecast.parquet", "backtest.parquet", "forecast.parquet", "replay_audit.json")]
    for relative in files:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity) if relative == "experiment.json" else "sealed", encoding="utf-8")
    inventory = {relative: subject.sha(directory / relative) for relative in files}
    (directory / "completion.json").write_text(json.dumps({"identity": identity["run_id"], "annual_complete": True,
        "files": inventory}), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"identity": identity["run_id"], "zone": "DE", "status": "COMPLETE",
        "annual_complete": True, "evaluation_days": 365}), encoding="utf-8")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    monkeypatch.setattr(subject, "prepare", lambda *args, **kwargs: SimpleNamespace(identity=identity))
    monkeypatch.setattr(subject, "exclusive_process_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(subject, "threadpool_limits", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(subject, "corrector_controls", lambda *args, **kwargs: pytest.fail("A complete run must never refit"))
    return directory, inventory


def test_resume_only_reuses_a_complete_verified_inventory(monkeypatch, tmp_path):
    directory, inventory = _fake_completed(monkeypatch, tmp_path)
    assert subject.run(action="run", zones=("DE",)) == {"DE": str(directory / "report.html")}
    assert {name: subject.sha(directory / name) for name in inventory} == inventory


@pytest.mark.parametrize("kind", ["empty_inventory", "omitted_variant", "tampered", "wrong_identity", "partial_status"])
def test_resume_rejects_partial_tampered_or_different_identity(monkeypatch, tmp_path, kind):
    directory, _ = _fake_completed(monkeypatch, tmp_path)
    receipt_path = directory / "completion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if kind == "empty_inventory":
        receipt["files"] = {}
    elif kind == "omitted_variant":
        receipt["files"].pop("interaction_80/forecast.parquet")
    elif kind == "tampered":
        (directory / "interaction_40/backtest.parquet").write_text("altered", encoding="utf-8")
    elif kind == "wrong_identity":
        receipt["identity"] = "other-experiment"
    else:
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["status"], status["evaluation_days"] = "PARTIAL", 1
        status_path.write_text(json.dumps(status), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        subject.run(action="run", zones=("DE",))


@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("future", [False, True])
def test_fit_raw_has_only_past_labels_and_uncapped_predictions(monkeypatch, interaction, future):
    timezone = "Europe/Berlin"
    covariates = _panel(days=21)
    historical_index, future_index = covariates.index[:-24], covariates.index[-24:]
    history = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130.,
                            "actual": np.arange(len(historical_index), dtype=float)}, index=historical_index)
    forecast = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130.}, index=future_index)
    features = covariates.rename(columns=lambda name: "known_" + name + "_oracle")
    feature, _ = build_interaction(covariates, "DE", timezone)
    captured = {}
    class FakeCorrector:
        min_training_rows = 48
        def fit(self, X, y, base, experts):
            captured.update(X=X.copy(), y=y.copy(), base=base.copy(), experts=experts.copy())
            self.feature_columns_ = tuple(X.columns)
        def predict_correction(self, X, base, experts):
            captured["predict_X"] = X.copy()
            result = pd.Series(40., index=base.index)
            result.attrs["raw_correction"] = pd.Series(100., index=base.index)
            return result
    prepared = SimpleNamespace(identity={"zone": "DE", "feature": feature.columns[0]},
        raw_history=history, raw_future=forecast, features=features, interaction=feature,
        factory=FakeCorrector)
    monkeypatch.setattr(subject, "DAY", "2026-01-21")
    day = date(2026, 1, 21 if future else 17)
    raw, audit = subject._fit_raw(prepared, day, interaction=interaction)
    trained = captured["X"].index
    assert all(trained.tz_convert(timezone).date < day)
    pd.testing.assert_series_equal(captured["y"], history.loc[trained, "actual"])
    assert (feature.columns[0] in captured["X"].columns) is interaction
    assert (feature.columns[0] in captured["predict_X"].columns) is interaction
    assert "actual" not in captured["X"]
    np.testing.assert_array_equal(raw, np.full(24, 100.))
    assert audit["causality_violations"] == 0
    assert not set(feature).intersection(covariates)


def test_year_replay_merges_overlay_only_outputs_and_reuses_sealed_cache(monkeypatch, tmp_path):
    """Real replay returns overlay fields only, not actual/Chronos/upstream."""
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    timezone = "Europe/Berlin"
    start, stop = pd.Timestamp("2025-09-22", tz=timezone), pd.Timestamp("2026-09-22", tz=timezone)
    historical_index = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC")
    historical_index = historical_index.rename("delivery_start_utc")
    future_index = pd.date_range(stop, periods=24, freq="h").tz_convert("UTC").rename("delivery_start_utc")
    def upstream(index, observed):
        frame = pd.DataFrame(index=index)
        for q, value in zip(subject.QUANTILES, (80., 100., 130.)):
            frame["chronos2__" + q] = value
            frame["residual_corrected__" + q] = value + 5.
        frame["residual_correction"] = 5.
        frame["forecast_origin_utc"] = index - pd.Timedelta(days=1)
        if observed:
            frame["actual"] = 110.
        return frame
    history, forecast = upstream(historical_index, True), upstream(future_index, False)
    covariates = pd.DataFrame({"de_residual_load_fcst": 10.}, index=historical_index.append(future_index))
    config = {"max_abs_correction": 20., "candidate_kinds": ("linear_bias", "linear_market")}
    covconfig = {"input_columns": ("de_residual_load_fcst",)}
    identity = {**_prepared_identity(), "kalman_config": json.loads(json.dumps(config)),
                "kalman_covariate_config": json.loads(json.dumps(covconfig))}
    prepared = SimpleNamespace(identity=identity, config=config, covconfig=covconfig,
        bundle=SimpleNamespace(covariates=covariates,
            kalman_view=SimpleNamespace(backtest=history)))
    calls = []
    def overlay(source):
        result = pd.DataFrame(index=source.index)
        for q in subject.QUANTILES:
            result["residual_kalman__" + q] = source["residual_corrected__" + q] + 2.
        result["kalman_correction"] = 2.
        result["kalman_raw_correction"] = 2.
        result["kalman_weight"] = 1.
        result["kalman_selected_filter"] = "linear_market"
        assert "actual" not in result and "chronos2__q50" not in result
        return result
    def fake_replay(prefix, **kwargs):
        calls.append(kwargs)
        source = subject.indexed(prefix)
        day = pd.Timestamp(kwargs["evaluation_start_day"]).date()
        selected = source.loc[source.index.tz_convert(timezone).date == day]
        future_source = kwargs["future_upstream"]
        future_frame = None if future_source is None else subject.indexed(future_source)
        assert kwargs["training_lookback_days"] == 365
        assert kwargs["config"] == config and kwargs["covariate_config"] == covconfig
        pd.testing.assert_frame_equal(kwargs["covariates"], covariates)
        assert not any("stress" in column for column in kwargs["covariates"])
        audit = {"causality_violations": 0, "quantile_crossings": 0,
            "future_observations_assimilated": 0, "evaluation_days": 1,
            "evaluation_hours": len(selected), "future_forecast_hours": 0 if future_frame is None else len(future_frame),
            "config": config, "covariate_config": covconfig, "training_lookback_days": 365}
        return SimpleNamespace(predictions=overlay(selected),
            future_predictions=None if future_frame is None else overlay(future_frame), audit=audit)
    monkeypatch.setattr(subject, "replay_kalman_overlay", fake_replay)
    directory = tmp_path / "de" / identity["run_id"]
    backtest, final, audits = subject.replay_variant(prepared, "interaction_80", (history, forecast),
        directory, lambda *args, **kwargs: None)
    assert len(calls) == len(audits) == 365 and len(backtest) == 8760 and len(final) == 24
    assert sum(call["future_upstream"] is not None for call in calls) == 1
    pd.testing.assert_frame_equal(backtest[history.columns], history, check_freq=False)
    pd.testing.assert_frame_equal(final[forecast.columns], forecast, check_freq=False)
    np.testing.assert_array_equal(backtest["residual_kalman__q50"], np.full(8760, 107.))
    assert "actual" not in final
    metrics = subject.evaluate({"baseline": backtest.copy(), "interaction_80": backtest}, timezone)
    assert metrics["slices"]["all"]["interaction_80"]["mae"] == 3.
    monkeypatch.setattr(subject, "replay_kalman_overlay", lambda *args, **kwargs: pytest.fail("Verified days must not refit"))
    reused, reused_future, _ = subject.replay_variant(prepared, "interaction_80", (history, forecast),
        directory, lambda *args, **kwargs: None)
    pd.testing.assert_frame_equal(reused, backtest, check_freq=False)
    pd.testing.assert_frame_equal(reused_future, final, check_freq=False)
    receipt_path = directory / "interaction_80/kalman_daily/2025-09-22.json"
    corrupted = json.loads(receipt_path.read_text(encoding="utf-8"))
    corrupted["payload"]["audit"]["causality_violations"] = 1
    receipt_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt checksum"):
        subject.replay_variant(prepared, "interaction_80", (history, forecast), directory, lambda *args, **kwargs: None)
