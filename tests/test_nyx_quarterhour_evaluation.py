"""Synthetic point forecasts only: no inference, network or production archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_quarterhour.evaluation import (
    EvaluationContractError, ZONES, aggregate_quarterhour_predictions,
    evaluate_predictions, read_sealed_predictions,
)


def grid_frame(start, end, frequency="h"):
    begin = pd.Timestamp(start).tz_localize("Europe/Paris")
    finish = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    times = pd.date_range(begin, finish, freq=frequency, inclusive="left").tz_convert("UTC")
    return pd.MultiIndex.from_product([times, ZONES], names=["timestamp_utc", "zone"]).to_frame(index=False)


def synthetic_hourly(start="2026-06-18", end="2026-07-01"):
    base = grid_frame(start, end)
    local_hour = base.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    base["actual"] = np.where(local_hour.lt(4), -20.0, np.where(local_hour.ge(20), 140.0, 40.0))
    base["nyx_q50"], base["hourly_control"] = base.actual + 10, base.actual + 8
    candidate = base[["timestamp_utc", "zone"]].copy()
    candidate["prediction"] = base.actual + 4
    return base, candidate


def evaluate(base, candidate, **kwargs):
    return evaluate_predictions(
        base, {"native_quarterhour": candidate}, q95_thresholds=dict.fromkeys(ZONES, 100.0),
        start_day="2026-06-18", end_day="2026-07-01", bootstrap_repetitions=80,
        matched_control_verified=True, **kwargs,
    )


@pytest.mark.parametrize("day,quarters", [
    ("2026-03-29", 92), ("2026-03-30", 96), ("2026-10-25", 100),
])
def test_four_points_are_averaged_on_physical_dst_hours_without_quantile_claim(day, quarters):
    frame = grid_frame(day, day, "15min")
    frame["prediction"] = frame.timestamp_utc.dt.minute / 15
    frame["q10"] = frame.prediction - 10
    frame["q90"] = frame.prediction + 10
    original = frame.copy(deep=True)
    result = aggregate_quarterhour_predictions(frame.sample(frac=1, random_state=3), start_day=day, end_day=day)
    pd.testing.assert_frame_equal(frame, original)
    assert len(frame) == quarters * 4
    assert (result.groupby("zone").size() == quarters // 4).all()
    assert result.prediction.eq(1.5).all()
    assert str(result.timestamp_utc.dt.tz) == "UTC"
    assert list(result.columns) == ["timestamp_utc", "zone", "prediction"]
    assert not result.duplicated(["timestamp_utc", "zone"]).any()
    assert result.attrs["quarter_counts_per_zone_by_day"] == {day: quarters}
    assert result.attrs["calibrated_quantiles_produced"] is False


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "inf", "off_grid", "naive", "missing_country", "outside"])
def test_quarter_grid_defects_are_rejected_instead_of_replaced(defect):
    frame = grid_frame("2026-06-18", "2026-06-18", "15min")
    frame["prediction"] = 50.0
    if defect == "missing":
        frame = frame.iloc[1:]
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect in {"nan", "inf"}:
        frame.loc[0, "prediction"] = np.nan if defect == "nan" else np.inf
    elif defect == "off_grid":
        frame.loc[0, "timestamp_utc"] += pd.Timedelta(minutes=1)
    elif defect == "naive":
        frame["timestamp_utc"] = frame.timestamp_utc.dt.tz_localize(None)
    elif defect == "missing_country":
        frame = frame.loc[frame.zone.ne("NL")]
    else:
        extra = frame.iloc[:1].copy()
        extra["timestamp_utc"] += pd.Timedelta(days=1)
        frame = pd.concat([frame, extra], ignore_index=True)
    with pytest.raises(EvaluationContractError):
        aggregate_quarterhour_predictions(frame, start_day="2026-06-18", end_day="2026-06-18")


def test_default_test_population_is_8640_points_and_candidate_is_not_selected_automatically():
    base, candidate = synthetic_hourly(end="2026-09-15")
    alternative = candidate.copy()
    alternative["prediction"] = base.actual + 1
    original = base.copy(deep=True)
    result = evaluate_predictions(
        base, {"native_quarterhour": candidate, "better_recipe": alternative},
        q95_thresholds=dict.fromkeys(ZONES, 100), matched_control_verified=True,
        bootstrap_repetitions=80,
    )
    pd.testing.assert_frame_equal(base, original)
    assert result["status"] == "complete"
    assert result["protocol"]["paired_points_per_family"] == 8640
    assert result["protocol"]["paired_hours_per_zone"] == 2160
    assert result["protocol"]["days"] == 90
    assert result["protocol"]["automatic_model_selection"] is False
    assert result["protocol"]["exploratory_history_already_reviewed"] is True
    assert result["protocol"]["calibrated_quantiles_produced"] is False
    assert result["decision"]["candidate_family"] == "native_quarterhour"
    assert result["decision"]["encouraging"]
    assert result["decision"]["promotion_allowed"] is False
    scores = result["metrics"].loc[lambda f: f.group.eq("overall")].set_index("family")
    assert scores.loc["nyx", "mae"] == pytest.approx(10)
    assert scores.loc["hourly_control", "rmse"] == pytest.approx(8)
    assert scores.loc["native_quarterhour", "bias"] == pytest.approx(4)
    assert scores.loc["better_recipe", "mae"] == pytest.approx(1)
    assert set(result["metrics"].group) >= {"country", "hour", "country_hour", "negative", "train_q95"}
    assert result["regime_checks"].evidence.eq("sufficient").all()
    assert result["decision"]["critical_regimes"]["coverage_complete"]


def test_hourly_comparison_requires_exact_common_population_and_immutable_nyx():
    base, candidate = synthetic_hourly()
    with pytest.raises(EvaluationContractError, match="target grid"):
        evaluate(base, candidate.iloc[1:])
    altered = candidate.copy()
    altered.loc[0, "prediction"] = np.nan
    with pytest.raises(EvaluationContractError, match="no replacement"):
        evaluate(base, altered)
    with pytest.raises(EvaluationContractError, match="immutable NYX"):
        evaluate_predictions(base, {"native_quarterhour": candidate, "nyx": candidate}, q95_thresholds=dict.fromkeys(ZONES, 100))
    with pytest.raises(EvaluationContractError, match="Training q95"):
        evaluate_predictions(base, {"native_quarterhour": candidate}, q95_thresholds={"BE": 100})


def test_missing_or_unverified_control_prevents_encouragement_even_when_nyx_is_beaten():
    base, candidate = synthetic_hourly()
    missing = evaluate(base.drop(columns="hourly_control"), candidate)
    assert missing["status"] == "insufficient_data"
    assert missing["reason"] == "missing_matched_control"
    assert not missing["decision"]["encouraging"]
    assert missing["decision"]["comparisons"]["nyx"]["mae_improvement_at_least_2pct"]
    unverified = evaluate_predictions(
        base, {"native_quarterhour": candidate}, q95_thresholds=dict.fromkeys(ZONES, 100),
        start_day="2026-06-18", end_day="2026-07-01", bootstrap_repetitions=80,
    )
    assert not unverified["decision"]["encouraging"]
    assert "matched_architecture_and_postprocessing_not_verified" in unverified["decision"]["reasons"]
    control = base[["timestamp_utc", "zone"]].copy()
    control["prediction"] = base.hourly_control
    supplied = evaluate_predictions(
        base.drop(columns="hourly_control"), {"native_quarterhour": candidate, "same_chronos_postprocessing": control},
        matched_control_family="same_chronos_postprocessing", matched_control_verified=True,
        q95_thresholds=dict.fromkeys(ZONES, 100), start_day="2026-06-18", end_day="2026-07-01",
        bootstrap_repetitions=80,
    )
    assert supplied["decision"]["encouraging"]
    assert "same_chronos_postprocessing" in supplied["decision"]["comparisons"]
    with pytest.raises(EvaluationContractError, match="supplied twice"):
        evaluate_predictions(base, {"native_quarterhour": candidate, "hourly_control": control}, q95_thresholds=dict.fromkeys(ZONES, 100))


def test_paired_seven_day_bootstrap_matches_raw_hour_resampling_across_dst():
    start, end = "2026-10-19", "2026-11-01"
    base, candidate = synthetic_hourly(start, end)
    days = base.timestamp_utc.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    day_number = (days - pd.Timestamp(start)).dt.days.to_numpy()
    base["nyx_q50"] = base.actual + day_number + 2
    candidate["prediction"] = base.actual + .4 * (day_number + 2)
    result = evaluate_predictions(
        base, {"native_quarterhour": candidate}, q95_thresholds=dict.fromkeys(ZONES, 100),
        matched_control_verified=True, start_day=start, end_day=end, bootstrap_repetitions=60, seed=431,
    )
    assert result["protocol"]["paired_hours_per_zone"] == 337
    rows = result["paired_deltas"].loc[lambda f: f.family.eq("native_quarterhour") & f.baseline.eq("nyx")].set_index("zone")
    # Independently resample actual hourly error arrays, not daily sufficient statistics.
    country = base.zone.eq("FR").to_numpy()
    actual = base.actual.to_numpy()[country]
    eb = base.nyx_q50.to_numpy()[country] - actual
    ec = candidate.prediction.to_numpy()[country] - actual
    labels = day_number[country]
    rng = np.random.default_rng(431)
    mae, rmse = [], []
    for _ in range(60):
        first, second = rng.integers(0, 8, size=2)
        chosen_days = [*range(first, first + 7), *range(second, second + 7)]
        chosen = np.concatenate([np.flatnonzero(labels == d) for d in chosen_days])
        mae.append(np.abs(ec[chosen]).mean() - np.abs(eb[chosen]).mean())
        rmse.append(np.sqrt(np.mean(ec[chosen] ** 2)) - np.sqrt(np.mean(eb[chosen] ** 2)))
    for name, values in (("mae", mae), ("rmse", rmse)):
        for suffix, q in (("low", .025), ("high", .975)):
            column = f"{name}_delta_ci_{suffix}"
            assert rows.loc["FR", column] == pytest.approx(np.quantile(values, q))
            assert rows.loc["all", column] == pytest.approx(rows.loc["FR", column])
    assert rows.loc["all", "n"] == 337 * 4


def test_peak_degradation_overrides_good_global_scores_and_no_regime_means_no_claim():
    base, candidate = synthetic_hourly()
    candidate["prediction"] = base.actual + np.where(base.actual.gt(100), 12.0, 2.0)
    result = evaluate(base, candidate)
    assert all(result["decision"]["comparisons"]["nyx"].values())
    assert all(result["decision"]["comparisons"]["hourly_control"].values())
    assert not result["decision"]["encouraging"]
    assert any("above_train_q95:mae_degradation_over_5pct" in r for r in result["decision"]["reasons"])
    base["actual"], base["nyx_q50"], base["hourly_control"] = 40.0, 50.0, 48.0
    candidate["prediction"] = 44.0
    scarce = evaluate(base, candidate)
    assert not scarce["decision"]["encouraging"]
    assert scarce["regime_checks"].evidence.eq("insufficient").all()
    assert scarce["regime_checks"].mae_relative_degradation.isna().all()
    assert not scarce["decision"]["critical_regimes"]["evidence_available"]


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_sealed_manifest_digest_is_verified_without_rewriting_input(tmp_path, suffix):
    _, forecasts = synthetic_hourly()
    path = tmp_path / ("predictions" + suffix)
    if suffix == ".parquet":
        forecasts.to_parquet(path, index=False)
    else:
        forecasts.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "outputs.manifest.json"
    manifest.write_text(json.dumps({"files": {path.name: digest}}), encoding="utf-8")
    expected = json.loads(manifest.read_text(encoding="utf-8"))["files"][path.name]
    before = path.stat().st_mtime_ns
    restored = read_sealed_predictions(path, expected)
    assert len(restored) == len(forecasts)
    assert path.stat().st_mtime_ns == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(EvaluationContractError, match="size limit"):
        read_sealed_predictions(path, digest, max_bytes=path.stat().st_size - 1)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(EvaluationContractError, match="SHA256 mismatch"):
        read_sealed_predictions(path, digest)


def test_sealed_reader_rejects_reparse_ancestors_before_read(tmp_path, monkeypatch):
    path = tmp_path / "predictions.csv"
    path.write_text("prediction\n1\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = Path.lstat

    def redirected(candidate, *args, **kwargs):
        if candidate == tmp_path:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", redirected)
    with pytest.raises(EvaluationContractError, match="reparse"):
        read_sealed_predictions(path, digest)
