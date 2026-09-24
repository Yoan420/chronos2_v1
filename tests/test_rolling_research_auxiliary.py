from concurrent.futures import Future
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous import prospective_auxiliary as auxiliary
from chronos2_exogenous import rolling_research_auxiliary as replay


TZ = "Europe/Paris"
START, END = "2025-09-09", "2026-09-08"


def raw_support():
    index = pd.date_range("2024-09-09", "2026-09-09", freq="h", inclusive="left", tz=TZ).tz_convert("UTC")
    local = index.tz_convert(TZ)
    median = 50 + np.sin(local.hour * 2 * np.pi / 24)
    frame = pd.DataFrame({"delivery_start_utc": index, "actual": median + 3})
    for quantile, shift in zip(auxiliary.QUANTILES, (-10, 0, 10)):
        frame[f"{auxiliary.RAW_MODEL}__{quantile}"] = median + shift
    for i, column in enumerate(auxiliary.MARKET_COLUMNS):
        frame[column] = 10.0 + i + np.cos(local.hour * 2 * np.pi / 24)
    return frame


def fake_kalman(history, future, *, timezone, config):
    index = pd.DatetimeIndex(future["delivery_start_utc"])
    history_index = pd.DatetimeIndex(history["delivery_start_utc"])
    local_day = index.tz_convert(timezone).date[0]
    days = history_index.tz_convert(timezone).date
    assert len(set(days)) == 365
    assert days[-1] == local_day - timedelta(days=1)
    assert days[0] == local_day - timedelta(days=365)
    assert "actual" not in future
    assert history_index.max() < index.min()
    assert tuple(config.candidate_kinds) == auxiliary.STANDARD_CANDIDATES
    shift = float((history["actual"] - history[f"{auxiliary.RESIDUAL_MODEL}__q50"]).mean())
    predicted = pd.DataFrame(index=index)
    predicted.index.name = "delivery_start_utc"
    for quantile in auxiliary.QUANTILES:
        predicted[f"{auxiliary.KALMAN_MODEL}__{quantile}"] = future[f"{auxiliary.RESIDUAL_MODEL}__{quantile}"].to_numpy() + shift
    predicted["kalman_correction"] = shift
    return {
        "predictions": predicted,
        "daily_audit": {"local_day": str(local_day), "selected_filter": "linear_bias", "selected_weight": 1.0},
        "window_audit": {"training_window_days": 365, "target_observations_assimilated": 0,
                         "training_window_start": str(days[0]), "training_window_end": str(days[-1])},
        "state_audit": [{"local_day": str(local_day), "state_before": shift, "state_after": shift,
                         "target_observations_assimilated": 0}],
        "market_scalers": {}, "covariate_audit": {}, "covariate_columns": list(auxiliary.MARKET_COLUMNS),
    }


@pytest.fixture(scope="module")
def prepared():
    raw = raw_support()
    history = raw.iloc[:-24]
    corrected, audits = auxiliary.build_prequential_residual_history(history)
    return raw, corrected, audits


@pytest.fixture
def fast_replay(monkeypatch, prepared):
    raw, corrected, audits = prepared
    calls = {"prequential": 0, "kalman": 0}
    def prequential(history, **kwargs):
        assert history["actual"].to_numpy() == pytest.approx(raw["actual"].iloc[:-24].to_numpy())
        calls["prequential"] += 1
        return corrected.copy(deep=True), [dict(x) for x in audits]
    def kalman(*args, **kwargs):
        calls["kalman"] += 1
        return fake_kalman(*args, **kwargs)
    monkeypatch.setattr(auxiliary, "build_prequential_residual_history", prequential)
    monkeypatch.setattr(auxiliary, "_fit_kalman_future", kalman)
    return calls


def run(raw, root, **kwargs):
    return replay.run_rolling_research_auxiliary(raw, evaluation_start=START, end_day=END,
                                                output_directory=root, workers=kwargs.pop("workers", 1), **kwargs)


def test_complete_replay_caches_reuse_and_final_actual_not_fit(tmp_path, prepared, fast_replay):
    raw = prepared[0].copy(deep=True)
    untouched = raw.copy(deep=True)
    raw.loc[raw.index[-24:], "actual"] = np.nan
    first, audit = run(raw, tmp_path / "cache", identity={"checkpoint_sha256": "a" * 64, "zone": "FR"})
    assert len(first) == 8760
    assert fast_replay == {"prequential": 1, "kalman": 365}
    assert audit["computed_days"] == 365 and audit["cached_days"] == 0
    assert len(audit["residual_fits"]) == len(audit["daily_audit"]) == len(audit["state_audit"]) == 365
    assert all(f["training_days"] == 365 for f in audit["residual_fits"])
    assert audit["historical_residual_fits"][0]["training_days"] == 0
    assert audit["daily_fit_audits"][0]["historical_corrector_warmup"]["identity_cold_start_days"] == 30
    assert audit["daily_fit_audits"][0]["historical_corrector_warmup"]["expanding_fit_days"] == 335
    assert audit["diagnostic_only"] and not audit["neural_oof"] and not audit["promotion_eligible"]
    assert audit["historical_neural_prefix_in_sample"]
    assert not audit["historical_correctors_all_have_full365"]
    assert first["actual"].iloc[-24:].isna().all()
    json.dumps(audit, allow_nan=False)
    before = {str(p): (p.stat().st_mtime_ns, replay._file_sha(p)) for p in (tmp_path / "cache").rglob("*") if p.is_file()}
    raw.loc[raw.index[-24:], "actual"] = 1e6
    second, reused = run(raw, tmp_path / "cache", identity={"checkpoint_sha256": "a" * 64, "zone": "FR"}, workers=4)
    after = {str(p): (p.stat().st_mtime_ns, replay._file_sha(p)) for p in (tmp_path / "cache").rglob("*") if p.is_file()}
    assert before == after
    assert reused["computed_days"] == 0 and reused["cached_days"] == 365
    assert fast_replay == {"prequential": 2, "kalman": 365}
    pd.testing.assert_frame_equal(first.drop(columns="actual"), second.drop(columns="actual"))
    assert (second["actual"].iloc[-24:] == 1e6).all()
    pd.testing.assert_frame_equal(prepared[0], untouched)
    for day, count in (("2025-10-26", 25), ("2026-03-29", 23)):
        assert (pd.DatetimeIndex(first["delivery_start_utc"]).tz_convert(TZ).strftime("%Y-%m-%d") == day).sum() == count


@pytest.mark.parametrize("bad", ["hour_gap", "day_gap", "duplicate", "missing_market", "history_nan", "extra_day"])
def test_missing_physical_support_is_rejected_before_cache_creation(tmp_path, prepared, bad):
    raw = prepared[0].copy()
    if bad == "hour_gap":
        raw = raw.drop(index=100)
    elif bad == "day_gap":
        raw = raw.drop(index=range(24))
    elif bad == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]])
    elif bad == "missing_market":
        raw = raw.drop(columns=auxiliary.MARKET_COLUMNS[0])
    elif bad == "history_nan":
        raw.loc[100, "actual"] = np.nan
    else:
        extra = raw.iloc[:24].copy()
        extra["delivery_start_utc"] -= pd.Timedelta(days=1)
        raw = pd.concat([extra, raw])
    with pytest.raises((replay.RollingResearchAuxiliaryError, auxiliary.ProspectiveAuxiliaryError)):
        run(raw, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("workers", [0, -1, 9, True, 2.5])
def test_workers_are_bounded(tmp_path, prepared, workers):
    with pytest.raises(replay.RollingResearchAuxiliaryError, match="workers"):
        run(prepared[0], tmp_path / "cache", workers=workers)


def test_evaluation_is_exact365_and_identity_cannot_change(tmp_path, prepared):
    with pytest.raises(replay.RollingResearchAuxiliaryError, match="365 jours calendaires"):
        replay.run_rolling_research_auxiliary(prepared[0], evaluation_start="2025-09-10", end_day=END,
                                             output_directory=tmp_path / "invalid")
    contract = replay._contract(timezone=TZ, identity={"zone": "FR"})
    replay._ensure_contract(tmp_path / "cache", contract)
    with pytest.raises(replay.RollingResearchAuxiliaryError, match="Identite"):
        replay._ensure_contract(tmp_path / "cache", replay._contract(timezone=TZ, identity={"zone": "DE"}))
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "important.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(replay.RollingResearchAuxiliaryError, match="non vide"):
        replay._ensure_contract(other, contract)
    assert (other / "important.txt").read_text(encoding="utf-8") == "keep"


def test_tampered_cache_fails_before_any_additional_fit(tmp_path, prepared, fast_replay):
    run(prepared[0], tmp_path / "cache")
    path = next((tmp_path / "cache" / "days").rglob("audit.json"))
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    calls = dict(fast_replay)
    with pytest.raises(replay.RollingResearchAuxiliaryError, match="SHA divergent"):
        run(prepared[0], tmp_path / "cache")
    assert fast_replay["kalman"] == calls["kalman"]
    assert path.read_bytes() == original + b" "


def test_parallel_path_is_bounded_and_matches_serial(tmp_path, prepared, fast_replay, monkeypatch):
    active = {"submitted": 0, "max_workers": None}
    class ImmediatePool:
        def __init__(self, *, max_workers, mp_context):
            active["max_workers"] = max_workers
            assert mp_context.get_start_method() == "spawn"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, *args, **kwargs):
            active["submitted"] += 1
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
            return future
    monkeypatch.setattr(replay, "ProcessPoolExecutor", ImmediatePool)
    predicted, audit = run(prepared[0], tmp_path / "parallel", workers=4)
    assert active == {"submitted": 365, "max_workers": 4}
    assert audit["computed_days"] == 365
    assert predicted["delivery_start_utc"].is_monotonic_increasing
    assert len(predicted) == 8760
    assert fast_replay["prequential"] == 1


def test_same_day_labels_are_excluded_from_corrector_and_kalman(prepared):
    raw, corrected, fits = prepared
    day = pd.Timestamp(START).date()
    index = pd.DatetimeIndex(raw["delivery_start_utc"])
    local_days = index.tz_convert(TZ).date
    training = raw.loc[local_days < day]
    future = raw.loc[local_days == day].copy()
    fit = auxiliary.fit_residual_corrector(training, target_day=day, require_full_window=True)
    first = auxiliary.apply_residual_corrector(future, fit)
    future["actual"] = -1e9
    second = auxiliary.apply_residual_corrector(future, fit)
    pd.testing.assert_frame_equal(first, second)
    prior_corrected = corrected.loc[pd.DatetimeIndex(corrected["delivery_start_utc"]).tz_convert(TZ).date < day]
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    first_kalman = fake_kalman(prior_corrected, first, timezone=TZ, config=KalmanResidualConfig())
    second_kalman = fake_kalman(prior_corrected, second, timezone=TZ, config=KalmanResidualConfig())
    pd.testing.assert_frame_equal(first_kalman["predictions"], second_kalman["predictions"])


@pytest.mark.parametrize("name", ["recipe", "residual_recipe", "kalman_config", "kalman_configuration"])
def test_declared_identity_parameters_must_match_real_calculation(name):
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    expected = asdict(KalmanResidualConfig() if "kalman" in name else auxiliary.ResidualRecipe())
    accepted = replay._contract(timezone=TZ, identity={name: expected})
    assert accepted["identity"][name] == json.loads(json.dumps(expected))
    changed = dict(expected)
    changed["q_over_r" if "kalman" in name else "ridge_alpha"] = 123.0
    with pytest.raises(replay.RollingResearchAuxiliaryError, match=f"Identite {name}"):
        replay._contract(timezone=TZ, identity={name: changed})
    with pytest.raises(replay.RollingResearchAuxiliaryError, match=f"Identite {name}"):
        replay._contract(timezone=TZ, identity={name: {}})


def test_declared_identity_cannot_use_boolean_for_numeric_recipe():
    recipe = asdict(auxiliary.ResidualRecipe())
    recipe["ridge_alpha"] = True
    with pytest.raises(replay.RollingResearchAuxiliaryError, match="Identite recipe"):
        replay._contract(timezone=TZ, identity={"recipe": recipe})


def test_real_windows_experiment_cache_paths_stay_below_max_path():
    root = (Path(r"C:\Users\BQ6757\chronos2_v1") / "runs" / "experiments"
            / "chronos2_exogenous_rank16_rolling365_research_v1" / "2026-09-08" / "FR" / "auxiliary")
    full_sha = "a" * 64
    path = replay._day_cache_path(root, pd.Timestamp(START).date(), full_sha)
    temporary = replay._partial_cache_path(path)
    assert path.name == full_sha[:24]
    assert len(temporary.name.rsplit("-", 1)[-1]) == 12
    assert max(len(str(folder / filename)) for folder in (path, temporary)
               for filename in ("predictions.parquet", "audit.json", "seal.json")) < 240
    # Truncation affects paths only; conflicting full identities remain detectable.
    other = full_sha[:24] + "b" * 40
    assert replay._day_cache_path(root, pd.Timestamp(START).date(), other) == path
    assert full_sha != other
