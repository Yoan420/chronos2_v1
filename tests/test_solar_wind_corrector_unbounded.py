"""Isolated unbounded-corrector contracts; no scientific fit or live batch."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_corrector_interaction as original
from chronos2_hourly import solar_wind_corrector_reuse as reuse
from chronos2_hourly import solar_wind_corrector_unbounded as subject


def _base():
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC").rename("delivery_start_utc")
    return pd.DataFrame({"q10": -10., "q50": 40., "q90": 120.}, index=index)


def test_uncapped_shift_preserves_extreme_positive_negative_values_and_quantile_widths():
    base = _base()
    before = base.copy(deep=True)
    raw = np.resize(np.array([-400., -90., -41., 0., 81., 120., 400.]), len(base))
    actual = subject.apply_unbounded(base, raw)
    # Independent oracle: addition, not another correction/clipping helper.
    for q in original.QUANTILES:
        np.testing.assert_array_equal(actual[q], before[q].to_numpy() + raw)
    np.testing.assert_array_equal(actual.q50 - base.q50, raw)
    np.testing.assert_array_equal(actual.q90 - actual.q10, before.q90 - before.q10)
    assert (actual.q10 <= actual.q50).all() and (actual.q50 <= actual.q90).all()
    assert np.isfinite(actual.to_numpy()).all()
    pd.testing.assert_frame_equal(base, before)


@pytest.mark.parametrize("kind", ["nan", "positive_infinity", "negative_infinity", "shape", "series_index", "base_nan", "crossed", "duplicate", "overflow"])
def test_unbounded_never_hides_nonfinite_grid_or_quantile_errors(kind):
    base, raw = _base(), np.zeros(24)
    if kind == "nan":
        raw[0] = np.nan
    elif kind == "positive_infinity":
        raw[0] = np.inf
    elif kind == "negative_infinity":
        raw[0] = -np.inf
    elif kind == "shape":
        raw = np.zeros((24, 1))
    elif kind == "series_index":
        raw = pd.Series(np.zeros(24), index=base.index + pd.Timedelta(hours=1))
    elif kind == "base_nan":
        base.iloc[0, 0] = np.nan
    elif kind == "crossed":
        base.iloc[0, 0] = 121.
    elif kind == "duplicate":
        base.index = pd.DatetimeIndex([base.index[0], *base.index[:-1]])
    else:
        base.loc[:, :] = 1e308
        raw[:] = 1e308
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        subject.apply_unbounded(base, raw)


def _source_completions(monkeypatch, tmp_path):
    """Sealed metadata only; no scientific readers are used by the early gate."""
    monkeypatch.setattr(reuse, "OUTPUT", tmp_path / "source")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "new")
    records = {}
    for zone, run_id in subject.SOURCE_RUNS.items():
        directory = reuse.OUTPUT / zone.lower() / run_id
        identity = {"run_id": run_id, "engine": reuse.ENGINE, "zone": zone, "delivery_day": original.DAY,
                    "original_identity": {"run_id": "scientific-" + zone, "zone": zone},
                    "coordinator_files": {"chronos2_hourly/solar_wind_corrector_reuse.py": subject.sha(subject.ROOT / "chronos2_hourly/solar_wind_corrector_reuse.py")}}
        for name in reuse.RESULT_FILES:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subject.json_bytes(identity) if name == "experiment.json" else b"<html>sealed report</html>" if name == "report.html" else ("sealed:" + name).encode())
        completion = {"identity": run_id, "annual_complete": True,
                      "files": {name: subject.sha(directory / name) for name in reuse.RESULT_FILES}}
        status = {"identity": run_id, "zone": zone, "status": "COMPLETE", "annual_complete": True,
                  "evaluation_days": 365, "evaluation_hours": 8760, "future_forecast_hours": 24, "variants": list(reuse.VARIANTS)}
        (directory / "completion.json").write_bytes(subject.json_bytes(completion))
        (directory / "status.json").write_bytes(subject.json_bytes(status))
        records[zone] = (directory, identity, completion, status)
    (reuse.OUTPUT / "latest_DE_NL.json").write_bytes(subject.json_bytes({"engine": reuse.ENGINE, "status": "COMPLETE",
        "annual_complete": True, "workdirs": {z: str(v[0]) for z, v in records.items()},
        "results": {z: str(v[0] / "report.html") for z, v in records.items()}}))
    links = ''.join('<a href="' + (value[0] / "report.html").relative_to(reuse.OUTPUT).as_posix() + '">report</a>' for value in records.values())
    (reuse.OUTPUT / "index.html").write_text("<html>" + links + "</html>", encoding="utf-8")
    return records


def test_predecessor_gate_requires_both_pinned_annual_completions_readonly(monkeypatch, tmp_path):
    records = _source_completions(monkeypatch, tmp_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in reuse.OUTPUT.rglob("*") if path.is_file()}
    sources = subject.predecessor_gate()
    assert set(sources) == {"DE", "NL"}
    assert len(reuse.RESULT_FILES) == 24
    assert subject.SOURCE_RUNS == {"DE": "48d7c14fd8bc2e4f", "NL": "8f246d6bbd54b511"}
    for zone, source in sources.items():
        directory, identity, _, _ = records[zone]
        assert source["identity"] == identity and Path(source["directory"]) == directory
        assert len(source["files"]) == 24
        assert source["completion_sha256"] == subject.sha(directory / "completion.json")
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert not subject.OUTPUT.exists()


@pytest.mark.parametrize("zone", ["DE", "NL"])
@pytest.mark.parametrize("kind", ["missing_status", "running"])
def test_pending_either_country_never_creates_output_or_starts_workers(monkeypatch, tmp_path, zone, kind):
    records = _source_completions(monkeypatch, tmp_path)
    directory, _, _, status = records[zone]
    if kind == "missing_status":
        (directory / "status.json").unlink()
    else:
        status["status"] = "RUNNING"
        (directory / "status.json").write_bytes(subject.json_bytes(status))
    def forbidden(*args, **kwargs):
        pytest.fail("Pending predecessor must not prepare, lock, fit or launch")
    monkeypatch.setattr(original, "prepare", forbidden)
    monkeypatch.setattr(original, "_fit_raw", forbidden)
    monkeypatch.setattr(subject, "exclusive_process_lock", forbidden)
    monkeypatch.setattr(subject, "_pool", forbidden)
    with pytest.raises(subject.PendingPredecessor):
        subject.run(action="run", zones=("DE",))
    assert not subject.OUTPUT.exists()


@pytest.mark.parametrize("kind", ["wrong_identity", "wrong_zone", "partial_year", "wrong_hours", "no_forecast", "not_annual",
                                  "empty_files", "missing_artifact", "extra_artifact", "tamper_file", "missing_completion", "changed_code", "missing_variant", "partial_report"])
def test_gate_rejects_false_completion_and_sha_changes(monkeypatch, tmp_path, kind):
    records = _source_completions(monkeypatch, tmp_path)
    directory, _, completion, status = records["NL"]
    if kind == "wrong_identity":
        status["identity"] = "another-run"
    elif kind == "wrong_zone":
        status["zone"] = "DE"
    elif kind == "partial_year":
        status["evaluation_days"] = 364
    elif kind == "wrong_hours":
        status["evaluation_hours"] = 24
    elif kind == "no_forecast":
        status["future_forecast_hours"] = 0
    elif kind == "not_annual":
        completion["annual_complete"] = False
    elif kind == "empty_files":
        completion["files"] = {}
    elif kind == "missing_artifact":
        del completion["files"]["reconstruction_audit.json"]
    elif kind == "extra_artifact":
        completion["files"]["unapproved.txt"] = "0" * 64
    elif kind == "changed_code":
        identity = records["NL"][1]
        identity["coordinator_files"]["chronos2_hourly/solar_wind_corrector_reuse.py"] = "0" * 64
        (directory / "experiment.json").write_bytes(subject.json_bytes(identity))
        completion["files"]["experiment.json"] = subject.sha(directory / "experiment.json")
    elif kind == "missing_variant":
        status["variants"] = ["cap_80"]
    elif kind == "partial_report":
        (directory / "report.html").write_bytes(b"<html>unfinished")
        completion["files"]["report.html"] = subject.sha(directory / "report.html")
    elif kind == "missing_completion":
        pass
    else:
        (directory / "interaction_80/backtest.parquet").write_bytes(b"changed")
    (directory / "completion.json").write_bytes(subject.json_bytes(completion))
    (directory / "status.json").write_bytes(subject.json_bytes(status))
    if kind == "missing_completion":
        (directory / "completion.json").unlink()
    with pytest.raises(ValueError):
        subject.predecessor_gate()
    assert not subject.OUTPUT.exists()


@pytest.mark.parametrize("kind", ["pointer_absent", "pointer_publishing", "index_absent", "wrong_target", "missing_link"])
def test_gate_waits_for_complete_batch_publication_and_rejects_wrong_links(monkeypatch, tmp_path, kind):
    _source_completions(monkeypatch, tmp_path)
    pointer, index = reuse.OUTPUT / "latest_DE_NL.json", reuse.OUTPUT / "index.html"
    if kind == "pointer_absent":
        pointer.unlink()
    elif kind == "index_absent":
        index.unlink()
    elif kind == "missing_link":
        index.write_text('<html><a href="wrong/report.html">report</a></html>', encoding="utf-8")
    else:
        batch = json.loads(pointer.read_text(encoding="utf-8"))
        if kind == "pointer_publishing":
            batch["status"] = "RUNNING"
        else:
            batch["results"]["NL"] = str(tmp_path / "other/report.html")
        pointer.write_bytes(subject.json_bytes(batch))
    error = subject.PendingPredecessor if kind in ("pointer_absent", "pointer_publishing", "index_absent") else ValueError
    with pytest.raises(error):
        subject.predecessor_gate()
    assert not subject.OUTPUT.exists()


def _kalman_fixture():
    tz = "Europe/Berlin"
    start, stop = pd.Timestamp("2026-09-20", tz=tz), pd.Timestamp("2026-09-22", tz=tz)
    history_index = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = pd.date_range(stop, periods=24, freq="h").tz_convert("UTC").rename(history_index.name)
    def upstream(index, *, future=False):
        frame = pd.DataFrame(index=index)
        for q, value in zip(original.QUANTILES, (80., 100., 130.)):
            frame["chronos2__" + q] = value
            frame["residual_corrected__" + q] = value + 120.
        frame["residual_correction"] = 120.
        frame["forecast_origin_utc"] = index - pd.Timedelta(days=1)
        if not future:
            frame["actual"] = 103.
        return frame
    history, future = upstream(history_index), upstream(future_index, future=True)
    config, covconfig = {"candidate_kinds": ("linear_bias", "linear_market")}, {"input_columns": ("de_residual_load_fcst",)}
    identity = {"zone": "DE", "run_id": "original-science", "kalman_config": json.loads(json.dumps(config)),
                "kalman_covariate_config": json.loads(json.dumps(covconfig))}
    prepared = SimpleNamespace(identity=identity, config=config, covconfig=covconfig,
        bundle=SimpleNamespace(covariates=pd.DataFrame({"de_residual_load_fcst": 10.}, index=history_index.append(future_index)),
                               kalman_view=SimpleNamespace(backtest=history.copy(), forecast=future.copy())))
    return prepared, history, future


def _kalman_output(prepared, history, forecast, day):
    day = str(day)
    selected = history.loc[history.index.tz_convert("Europe/Berlin").date == pd.Timestamp(day).date()]
    last = day == "2026-09-21"
    def overlay(frame):
        frame = frame.copy(deep=True)
        for q in original.QUANTILES:
            frame["residual_kalman__" + q] = frame["residual_corrected__" + q] + 2.
        return frame
    audit = {"causality_violations": 0, "quantile_crossings": 0, "future_observations_assimilated": 0,
        "evaluation_days": 1, "evaluation_hours": len(selected), "future_forecast_hours": 24 if last else 0,
        "training_lookback_days": 365, "config": prepared.config, "covariate_config": prepared.covconfig}
    return {"backtest": overlay(selected), "forecast": overlay(forecast) if last else None, "audit": audit, "elapsed_seconds": .01}


@pytest.mark.parametrize("kind", ["causality", "future_observation", "lookback", "config", "index", "actual", "nonfinite", "crossed", "missing_forecast"])
def test_kalman_daily_contract_rejects_causal_grid_and_future_violations(kind):
    prepared, history, future = _kalman_fixture()
    output = _kalman_output(prepared, history, future, "2026-09-21")
    if kind == "causality":
        output["audit"]["causality_violations"] = 1
    elif kind == "future_observation":
        output["audit"]["future_observations_assimilated"] = 1
    elif kind == "lookback":
        output["audit"]["training_lookback_days"] = 364
    elif kind == "config":
        output["audit"]["config"] = {"candidate_kinds": ("different",)}
    elif kind == "index":
        output["backtest"] = output["backtest"].iloc[1:]
    elif kind == "actual":
        output["backtest"].iloc[0, output["backtest"].columns.get_loc("actual")] = -999.
    elif kind == "nonfinite":
        output["backtest"].iloc[0, output["backtest"].columns.get_loc("residual_kalman__q50")] = np.inf
    elif kind == "crossed":
        output["backtest"].iloc[0, output["backtest"].columns.get_loc("residual_kalman__q10")] = 1e6
    else:
        output["forecast"] = None
    with pytest.raises((ValueError, TypeError, AttributeError)):
        subject._validate_kalman(prepared, "baseline_unbounded", "2026-09-21", history, future,
                                 output["backtest"], output["forecast"], output["audit"])


def _daily_replay(monkeypatch, tmp_path):
    prepared, history, future = _kalman_fixture()
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "new")
    identity = {"zone": "DE", "run_id": "new-synthetic"}
    directory = subject.OUTPUT / "de" / identity["run_id"]
    directory.mkdir(parents=True)
    frames = {name: (history.copy(), future.copy()) for name in subject.VARIANTS}
    calls = []
    def pool(item, identity, tasks, worker, on_result, progress, *, upstreams=None, **options):
        for task in reversed(tasks):
            calls.append((task["variant"], task["day"]))
            hist, fut = upstreams[task["variant"]]
            on_result(task, _kalman_output(item, hist, fut, task["day"]))
    monkeypatch.setattr(subject, "_run_pool_tasks", pool)
    backtests, forecasts = subject._kalman_stage(prepared, identity, directory, frames, lambda *args, **kwargs: None)
    return prepared, identity, directory, frames, calls, backtests, forecasts


def test_daily_kalman_resume_uses_sealed_days_without_new_science(monkeypatch, tmp_path):
    prepared, identity, directory, frames, calls, backtests, forecasts = _daily_replay(monkeypatch, tmp_path)
    assert len(calls) == 4 and calls[0][1] == "2026-09-21"
    monkeypatch.setattr(subject, "_pool", lambda *args, **kwargs: pytest.fail("No pool needed for cached days"))
    def no_new_tasks(prepared, identity, tasks, *args, **kwargs):
        assert tasks == []
    monkeypatch.setattr(subject, "_run_pool_tasks", no_new_tasks)
    resumed_backtests, resumed_forecasts = subject._kalman_stage(prepared, identity, directory, frames, lambda *args, **kwargs: None)
    for name in subject.VARIANTS:
        pd.testing.assert_frame_equal(resumed_backtests[name], backtests[name], check_freq=False)
        pd.testing.assert_frame_equal(resumed_forecasts[name], forecasts[name], check_freq=False)
        assert len(resumed_backtests[name]) == 48 and len(resumed_forecasts[name]) == 24


@pytest.mark.parametrize("kind", ["receipt_checksum", "file_checksum", "identity", "future_file"])
def test_corrupt_daily_checkpoint_never_triggers_silent_recompute(monkeypatch, tmp_path, kind):
    prepared, identity, directory, frames, _, _, _ = _daily_replay(monkeypatch, tmp_path)
    daily = directory / next(iter(subject.VARIANTS)) / "kalman_daily"
    receipt_path = daily / "2026-09-21.json"
    sealed = json.loads(receipt_path.read_text(encoding="utf-8"))
    if kind == "receipt_checksum":
        sealed["payload"]["audit"]["causality_violations"] = 1
    elif kind == "identity":
        sealed["payload"]["identity"] = "another-run"
        sealed["sha256"] = subject._digest(sealed["payload"])
    elif kind == "file_checksum":
        (daily / "2026-09-21.parquet").write_bytes(b"changed")
    else:
        (daily / "forecast.parquet").write_bytes(b"changed forecast")
    receipt_path.write_bytes(subject.json_bytes(sealed))
    monkeypatch.setattr(subject, "_run_pool_tasks", lambda *args, **kwargs: pytest.fail("Corruption must fail closed"))
    with pytest.raises(ValueError):
        subject._kalman_stage(prepared, identity, directory, frames, lambda *args, **kwargs: None)


def _calendar(zone="DE"):
    """730 input days plus a future; only the final 365 are evaluated."""
    tz = original.TIMEZONES[zone]
    start, evaluation, end = (pd.Timestamp(value, tz=tz) for value in ("2024-09-22", "2025-09-22", original.DAY))
    history_index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = pd.date_range(end, periods=24, freq="h").tz_convert("UTC").rename(history_index.name)
    def raw(index, actual):
        frame = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130.}, index=index)
        frame["forecast_origin_utc"] = index - pd.Timedelta(days=1)
        if actual:
            frame["actual"] = 103.
        return frame
    history, future = raw(history_index, True), raw(future_index, False)
    def upstream(frame, delta):
        result = frame.copy(deep=True)
        for q in original.QUANTILES:
            result["chronos2__" + q] = result[q]
            result["residual_corrected__" + q] = result[q] + delta
            result[q] += delta
        result["residual_correction"] = delta
        return result
    baseline_history, baseline_future = upstream(history, 40.), upstream(future, 40.)
    def baseline_overlay(frame):
        result = frame.copy()
        for q in original.QUANTILES:
            result["residual_kalman__" + q] = result["residual_corrected__" + q] + 2.
        return result
    config, covconfig = {"candidate_kinds": ("linear_bias", "linear_market")}, {"input_columns": (zone.lower() + "_residual_load_fcst",)}
    identity = {"zone": zone, "run_id": "scientific-" + zone, "kalman_config": json.loads(json.dumps(config)),
                "kalman_covariate_config": json.loads(json.dumps(covconfig))}
    prepared = SimpleNamespace(identity=identity, raw_history=history, raw_future=future, config=config, covconfig=covconfig,
        bundle=SimpleNamespace(residual_statistics=baseline_history, source_forecast=baseline_future,
            covariates=pd.DataFrame({covconfig["input_columns"][0]: 10.}, index=history_index.append(future_index)),
            kalman_view=SimpleNamespace(backtest=baseline_overlay(baseline_history.loc[history_index >= evaluation.tz_convert("UTC")]),
                                       forecast=baseline_overlay(baseline_future))))
    frames = {name: (upstream(history, -40. if interaction else 80.), upstream(future, -40. if interaction else 80.))
              for name, (interaction, _) in reuse.VARIANTS.items()}
    payloads = {}
    all_index = history_index.append(future_index)
    groups = pd.Series(all_index, index=all_index).groupby(all_index.tz_convert(tz).date, sort=True)
    for day, group in groups:
        grid = group.index
        payloads[str(day)] = {"identity": subject.SOURCE_RUNS[zone], "day": str(day), "index_ns": grid.asi8.tolist(),
            "raw": {"original": [100.] * len(grid), "interaction": [-90.] * len(grid)},
            "audit": {"original": {"generation_source": "synthetic_verified_fit"}, "interaction": {"generation_source": "synthetic_verified_fit"}},
            "origin": None, "baseline_reconstruction": None}
    return prepared, payloads, frames


def _write_corrector_audit(directory, payloads):
    records = []
    for day, payload in payloads.items():
        clipping = {recipe: {"hours": len(values), "raw_gt_40": int((np.asarray(values) > 40).sum()),
            "raw_gt_80": int((np.asarray(values) > 80).sum()), "raw_lt_minus40": int((np.asarray(values) < -40).sum())}
            for recipe, values in payload["raw"].items()}
        records.append({"day": day, **payload["audit"], "clipping": clipping, "baseline_reproduced": True,
                        "origin": payload.get("origin"), "baseline_reconstruction": payload.get("baseline_reconstruction")})
    (directory / "corrector_audit.json").write_bytes(subject.json_bytes(records))


def _import_fixture(monkeypatch, tmp_path, *, all_files=False):
    records = _source_completions(monkeypatch, tmp_path)
    prepared, payloads, frames = _calendar()
    directory, identity, completion, _ = records["DE"]
    identity["original_identity"] = prepared.identity
    identity["execution"] = {"max_workers": 4, "min_free_memory_gb": 3.}
    (directory / "experiment.json").write_bytes(subject.json_bytes(identity))
    _write_corrector_audit(directory, payloads)
    completion["files"]["experiment.json"] = subject.sha(directory / "experiment.json")
    completion["files"]["corrector_audit.json"] = subject.sha(directory / "corrector_audit.json")
    (directory / "completion.json").write_bytes(subject.json_bytes(completion))
    source = subject.predecessor_gate()["DE"]
    monkeypatch.setattr(reuse, "make_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(subject, "_source_artifacts", lambda *args: frames)
    calls = []
    def verify_origin(item, identity, day, grid, payload):
        # Verification itself has dedicated adversarial tests in the reuse suite.
        # This spy proves the new importer invokes it for EVERY day, unchanged.
        assert item is prepared and payload == payloads[str(day)]
        assert payload["index_ns"] == grid.asi8.tolist()
        calls.append(str(day))
    monkeypatch.setattr(reuse, "verify_origin", verify_origin)
    selected = payloads.items() if all_files else [next(iter(payloads.items()))]
    for day, payload in selected:
        path = directory / "corrector_daily" / (day + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(subject.json_bytes({"payload": payload, "sha256": subject._digest(payload)}))
    return prepared, payloads, frames, source, calls


def test_import_all_731_days_preserves_raw_values_and_full_provenance_readonly(monkeypatch, tmp_path):
    prepared, payloads, frames, source, calls = _import_fixture(monkeypatch, tmp_path, all_files=True)
    day = "2025-09-22"
    payload = payloads[day]
    payload["raw"]["original"] = [1.125] * len(payload["index_ns"])
    payload["baseline_reconstruction"] = {"source_generation_source": "daily_prequential_refit", "synthetic_verified_evidence": True}
    grid = subject._source_base(prepared, day).index
    for frame in (prepared.bundle.residual_statistics, frames["cap_80"][0]):
        for q in original.QUANTILES:
            frame.loc[grid, "residual_corrected__" + q] = frame.loc[grid, "chronos2__" + q] + 1.125
    path = Path(source["directory"]) / "corrector_daily" / (day + ".json")
    path.write_bytes(subject.json_bytes({"payload": payload, "sha256": subject._digest(payload)}))
    _write_corrector_audit(Path(source["directory"]), payloads)
    source["files"]["corrector_audit.json"] = subject.sha(Path(source["directory"]) / "corrector_audit.json")
    completion_path = Path(source["directory"]) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"] = source["files"]
    completion_path.write_bytes(subject.json_bytes(completion))
    source["completion_sha256"] = subject.sha(completion_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.parent.iterdir()}
    monkeypatch.setattr(original, "_fit_raw", lambda *args, **kwargs: pytest.fail("No CatBoost refit"))
    imported, audit = subject.load_source_checkpoints(prepared, source)
    assert len(imported) == len(calls) == audit["days"] == 731
    assert imported == payloads and set(calls) == set(payloads)
    assert audit["new_catboost_fits"] == 0 and audit["sources_modified"] is False
    assert audit["existing_reconstructed_baseline_days"] == 1
    assert audit["daily_checkpoints"][day]["baseline_reconstruction"] == payload["baseline_reconstruction"]
    assert audit["daily_checkpoints"][day]["file_sha256"] == subject.sha(path)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert not subject.OUTPUT.exists()


@pytest.mark.parametrize("kind", ["checksum", "grid", "nan", "origin", "scientific_identity", "missing_day", "audit_mismatch", "clipping_mismatch", "duplicate_audit_day"])
def test_source_raw_import_refuses_any_invalid_day_without_refit(monkeypatch, tmp_path, kind):
    prepared, payloads, _, source, _ = _import_fixture(monkeypatch, tmp_path)
    day = next(iter(payloads))
    path = Path(source["directory"]) / "corrector_daily" / (day + ".json")
    sealed = json.loads(path.read_text(encoding="utf-8"))
    if kind == "checksum":
        sealed["payload"]["raw"]["interaction"][0] = 12.
    elif kind == "grid":
        sealed["payload"]["index_ns"][0] += 1
        sealed["sha256"] = subject._digest(sealed["payload"])
    elif kind == "nan":
        sealed["payload"]["raw"]["interaction"][0] = "nan"
        sealed["sha256"] = subject._digest(sealed["payload"])
    elif kind == "origin":
        def reject(*args):
            raise ValueError("invalid full predecessor provenance")
        monkeypatch.setattr(reuse, "verify_origin", reject)
    elif kind == "scientific_identity":
        source["identity"]["original_identity"] = {"changed": True}
    elif kind in ("audit_mismatch", "clipping_mismatch", "duplicate_audit_day"):
        audit_path = Path(source["directory"]) / "corrector_audit.json"
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        if kind == "audit_mismatch":
            records[0]["original"]["generation_source"] = "unapproved"
        elif kind == "clipping_mismatch":
            records[0]["clipping"]["original"]["raw_gt_80"] = 0
        else:
            records[-1]["day"] = records[0]["day"]
        audit_path.write_bytes(subject.json_bytes(records))
        source["files"]["corrector_audit.json"] = subject.sha(audit_path)
    path.write_bytes(subject.json_bytes(sealed))
    if kind == "missing_day":
        path.unlink()
    monkeypatch.setattr(original, "_fit_raw", lambda *args, **kwargs: pytest.fail("Never refit missing donor"))
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_source_checkpoints(prepared, source)
    assert not subject.OUTPUT.exists()


def test_all_731_days_assemble_chronologically_without_clips_or_mutating_sources():
    prepared, payloads, _ = _calendar()
    before = prepared.bundle.residual_statistics.copy(deep=True)
    frames, audit = subject.assemble_unbounded(prepared, dict(reversed(list(payloads.items()))))
    assert len(audit) == 731 and audit[0]["day"] == "2024-09-22" and audit[-1]["day"] == original.DAY
    for variant, expected in (("baseline_unbounded", 200.), ("interaction_unbounded", 10.)):
        history, future = frames[variant]
        pd.testing.assert_index_equal(history.index, before.index)
        assert len(history) == 17520 and len(future) == 24
        np.testing.assert_array_equal(history["residual_corrected__q50"], np.full(len(history), expected))
        np.testing.assert_array_equal(future["residual_corrected__q50"], np.full(24, expected))
        pd.testing.assert_series_equal(history.actual, before.actual)
        assert "actual" not in future
    assert all(a["corrector_clipping_applied"] is False and a["new_catboost_fit_performed"] is False for a in audit)
    pd.testing.assert_frame_equal(prepared.bundle.residual_statistics, before)


def _run_fixture(monkeypatch, tmp_path, *, annual=False):
    _source_completions(monkeypatch, tmp_path)
    monkeypatch.setattr(original, "OUTPUT", tmp_path / "serial")
    monkeypatch.setattr(subject.prior_parallel, "OUTPUT", tmp_path / "parallel")
    sources = subject.predecessor_gate()
    prepared, payloads, audits = {}, {}, {}
    for zone in subject.SOURCE_RUNS:
        if annual:
            item, values, _ = _calendar(zone)
        else:
            item, _, _ = _kalman_fixture()
            item.identity = dict(item.identity, zone=zone, run_id="scientific-" + zone)
            values = {}
        prepared[zone], payloads[zone] = item, values
        audits[zone] = {"days": len(values), "new_catboost_fits": 0, "sources_modified": False,
            "daily_checkpoints": {day: {"file_sha256": "a" * 64, "payload_sha256": subject._digest(payload)} for day, payload in values.items()}}
    monkeypatch.setattr(original, "prepare", lambda zone, **kwargs: prepared[zone])
    monkeypatch.setattr(subject, "load_source_checkpoints", lambda item, source: (payloads[item.identity["zone"]], audits[item.identity["zone"]]))
    # Exact daily-source I/O is tested separately; this isolates run sequencing.
    monkeypatch.setattr(subject, "_verify_source_seals", lambda *args: None)
    monkeypatch.setattr(reuse, "verify_origin", lambda *args: None)
    for name in ("_fit_raw", "corrector_controls"):
        monkeypatch.setattr(original, name, lambda *args, **kwargs: pytest.fail("Unbounded must not fit CatBoost, even in controls"))
    identities = {zone: subject.make_identity(prepared[zone], sources[zone], audits[zone], all_sources=sources) for zone in sources}
    return prepared, payloads, audits, sources, identities


def test_validate_creates_no_outputs_and_pins_both_source_manifests(monkeypatch, tmp_path):
    _, _, _, _, identities = _run_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "exclusive_process_lock", lambda *args: pytest.fail("validate must not lock"))
    monkeypatch.setattr(subject, "_pool", lambda *args, **kwargs: pytest.fail("validate must not start workers"))
    assert subject.run(action="validate", zones=("DE",)) == {"DE": identities["DE"]}
    assert set(identities["DE"]["predecessor_gate"]) == {"DE", "NL"}
    assert identities["DE"]["new_catboost_fits"] == 0
    for recipe in identities["DE"]["variants"].values():
        assert recipe["correction_clip"] is None and recipe["max_abs_correction"] is None and recipe["correction_scale"] == 1.
    assert not subject.OUTPUT.exists()


def test_source_or_coordinator_changes_produce_distinct_identity_and_worker_refuses(monkeypatch, tmp_path):
    prepared, _, audits, sources, identities = _run_fixture(monkeypatch, tmp_path)
    changed = deepcopy(sources)
    changed["NL"]["completion_sha256"] = "0" * 64
    modified = subject.make_identity(prepared["DE"], changed["DE"], audits["DE"], all_sources=changed)
    assert modified["run_id"] != identities["DE"]["run_id"]
    damaged = deepcopy(identities["DE"])
    name = next(iter(damaged["coordinator_files"]))
    damaged["coordinator_files"][name] = "0" * 64
    monkeypatch.setattr(subject, "threadpool_limits", lambda **kwargs: None)
    with pytest.raises(ValueError, match="coordinator"):
        subject._worker_init("DE", damaged)


@pytest.mark.parametrize("namespace", ["serial", "parallel", "source", "new"])
def test_all_inherited_and_new_locks_exclude_duplicate_batch(monkeypatch, tmp_path, namespace):
    _run_fixture(monkeypatch, tmp_path)
    roots = {"serial": original.OUTPUT, "parallel": subject.prior_parallel.OUTPUT, "source": reuse.OUTPUT, "new": subject.OUTPUT}
    with original.exclusive_process_lock(roots[namespace] / "batch.lock"):
        with pytest.raises(ValueError, match="Verrou"):
            subject.run(action="run", zones=("DE",))


def test_intentional_stop_is_paused_without_automatic_restart(monkeypatch, tmp_path):
    _, _, _, _, identities = _run_fixture(monkeypatch, tmp_path)
    subject.OUTPUT.mkdir()
    (subject.OUTPUT / "request_stop.json").write_bytes(subject.json_bytes({"identity": identities["DE"]["run_id"]}))
    monkeypatch.setattr(subject, "assemble_unbounded", lambda *args: pytest.fail("User stop precedes assembly"))
    result = subject.run(action="run", zones=("DE",))
    assert result["status"] == "PAUSED" and result["reason"] == "user_stop"
    state = json.loads((subject.OUTPUT / "de" / identities["DE"]["run_id"] / "status.json").read_text(encoding="utf-8"))
    assert state["automatic_resume"] is False


def test_memory_reserve_pauses_before_submitting_any_worker():
    executor = SimpleNamespace(submit=lambda *args: pytest.fail("No worker below reserve"))
    result = subject.bounded_tasks([1], lambda x: x, executor, lambda *args: None,
                                   lambda: False, lambda: 1., max_workers=4, min_free_memory_gb=3.)
    assert result == {"stopped": True, "reason": "memory_reserve", "completed": 0}


def _completion_fixture(monkeypatch, tmp_path):
    _, _, _, _, identities = _run_fixture(monkeypatch, tmp_path)
    identity = identities["DE"]
    directory = subject.OUTPUT / "de" / identity["run_id"]
    for relative in subject.RESULT_FILES:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subject.json_bytes(identity) if relative == "experiment.json" else b"sealed")
    receipt = {"identity": identity["run_id"], "annual_complete": True,
        "files": {name: subject.sha(directory / name) for name in subject.RESULT_FILES}}
    status = {"identity": identity["run_id"], "zone": "DE", "status": "COMPLETE", "annual_complete": True,
        "evaluation_days": 365, "evaluation_hours": 8760, "future_forecast_hours": 24, "variants": list(subject.VARIANTS)}
    (directory / "completion.json").write_bytes(subject.json_bytes(receipt))
    (directory / "status.json").write_bytes(subject.json_bytes(status))
    return identity, directory, receipt, status


def test_exact_17_artifact_completion_does_not_restart_science(monkeypatch, tmp_path):
    identity, directory, _, _ = _completion_fixture(monkeypatch, tmp_path)
    assert len(subject.RESULT_FILES) == len(set(subject.RESULT_FILES)) == 17
    assert subject._verify_completion(directory, identity)
    monkeypatch.setattr(subject, "_pool", lambda *args, **kwargs: pytest.fail("No refit of complete result"))
    assert subject.run(action="run", zones=("DE",)) == {"DE": str(directory / "report.html")}


@pytest.mark.parametrize("kind", ["24_source_files", "missing_variant", "short_year", "missing_forecast", "changed_file", "wrong_identity"])
def test_final_completion_rejects_wrong_manifest_coverage_or_tampering(monkeypatch, tmp_path, kind):
    identity, directory, receipt, status = _completion_fixture(monkeypatch, tmp_path)
    if kind == "24_source_files":
        receipt["files"] = dict.fromkeys(reuse.RESULT_FILES, "0" * 64)
    elif kind == "missing_variant":
        status["variants"] = ["baseline_unbounded"]
    elif kind == "short_year":
        status["evaluation_days"] = 1
    elif kind == "missing_forecast":
        status["future_forecast_hours"] = 0
    elif kind == "changed_file":
        (directory / "source_audit.json").write_bytes(b"changed")
    else:
        receipt["identity"] = "other"
    (directory / "completion.json").write_bytes(subject.json_bytes(receipt))
    (directory / "status.json").write_bytes(subject.json_bytes(status))
    with pytest.raises(ValueError):
        subject._verify_completion(directory, identity)


def test_mock_annual_both_variants_and_countries_have_separate_forecasts_and_17_artifacts(monkeypatch, tmp_path):
    prepared, _, _, _, identities = _run_fixture(monkeypatch, tmp_path, annual=True)
    def confined(path):
        path = Path(path).absolute()
        assert path.is_relative_to(subject.OUTPUT.absolute())
        return path
    monkeypatch.setattr(subject, "safe_path", confined)  # Redirection guards tested on small fixtures.
    saved_frames = {}
    def light_frame(path, frame):
        path = confined(path)
        saved_frames[path] = frame.copy(deep=True)
        path.write_bytes(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    monkeypatch.setattr(subject, "write_frame", light_frame)
    calls = []
    def reversed_pool(item, identity, tasks, worker, on_result, progress, *, upstreams=None, **options):
        for task in reversed(tasks):
            calls.append((item.identity["zone"], task["variant"], task["day"]))
            history, future = upstreams[task["variant"]]
            on_result(task, _kalman_output(item, history, future, task["day"]))
    monkeypatch.setattr(subject, "_run_pool_tasks", reversed_pool)
    result = subject.run(action="run", zones=("DE", "NL"))
    assert set(result) == {"DE", "NL"} and len(calls) == 2 * 2 * 365
    for zone, identity in identities.items():
        directory = subject.OUTPUT / zone.lower() / identity["run_id"]
        assert subject._verify_completion(directory, identity)
        corrections = json.loads((directory / "unbounded_corrections_audit.json").read_text(encoding="utf-8"))
        assert corrections["days"] == 731 and corrections["new_catboost_fits"] == 0
        assert corrections["corrector_clipping_applied"] is False and corrections["kalman_constraints_unchanged"] is True
        for variant, expected in (("baseline_unbounded", 202.), ("interaction_unbounded", 12.)):
            annual = saved_frames[directory / variant / "backtest.parquet"]
            future = saved_frames[directory / variant / "forecast.parquet"]
            assert len(annual) == 8760 and len(future) == 24
            pd.testing.assert_index_equal(annual.index, prepared[zone].bundle.kalman_view.backtest.index)
            assert "actual" not in future
            np.testing.assert_array_equal(annual["residual_kalman__q50"], np.full(8760, expected))
            np.testing.assert_array_equal(future["residual_kalman__q50"], np.full(24, expected))
    monkeypatch.setattr(subject, "_run_pool_tasks", lambda *args, **kwargs: pytest.fail("Completed batch must not restart workers"))
    assert subject.run(action="run", zones=("DE", "NL")) == result
