"""Chronological CPU qualification for NYX regional price and negative risk.

The CLI obtains the same Saturn banks as the live runner and a separate
official Storm/EPEX reporting snapshot. Storm is never a model input. A failed
qualification publishes an inspectable report but does not unlock live runs.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_cpu_live_features import build_country_features
from chronos2_hourly.nyx_regional_cpu import (
    CANDIDATES, NEGATIVE_METHODS, PROTOCOL, TIMEZONES, ZONES, fit_predict_block,
    fit_predict_negative_block, grid,
)
from chronos2_hourly.nyx_regional_cpu_sources import (
    load_sources, preflight_sources, sync_sources,
)
from chronos2_hourly.process_lock import exclusive_process_lock


ROOT = Path(__file__).resolve().parent
FIRST = "2025-09-24"
CONFIRM_FIRST = "2026-05-06"
STOP = "2026-09-24"
FEATURE_FIRST = "2024-09-24"
SOURCE_FIRST = "2024-09-17"
CONFIG = ROOT / "config" / "nyx_regional_cpu.json"
CANONICAL_RECEIPT = ROOT / "config" / "nyx_regional_cpu_backtest_receipt.json"
REQUIRED_STORM_COUNTRIES = ("FR", "BE", "NL")
SELECTION_WEEKS = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """Replace a complete JSON document within one directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp",
                                        dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _window(first: str, stop: str) -> pd.DatetimeIndex:
    return grid(first, stop, "FR")


def _origin_days() -> list[str]:
    selection = list(pd.date_range(FIRST, CONFIRM_FIRST, freq="7D", inclusive="left"))
    confirmation = list(pd.date_range(CONFIRM_FIRST, STOP, freq="D", inclusive="left"))
    _require(len(selection) == SELECTION_WEEKS and len(confirmation) == 141
             and confirmation[0].date().isoformat() == CONFIRM_FIRST
             and confirmation[-1].date().isoformat() == "2026-09-23",
             "Backtest weekly-selection/daily-confirmation contract changed")
    return [str(day.date()) for day in selection + confirmation]


def _forecast_stop_day(origin: str, number: int) -> str:
    horizon = 7 if number < SELECTION_WEEKS else 1
    return str(min(pd.Timestamp(origin) + pd.Timedelta(days=horizon),
                   pd.Timestamp(STOP)).date())


def _benchmark_series(frame: pd.DataFrame, *, zone: str) -> pd.DataFrame:
    expected = grid(FIRST, STOP, zone)
    _require(isinstance(frame, pd.DataFrame) and set(frame) == {"actual", "storm"}
             and isinstance(frame.index, pd.DatetimeIndex)
             and str(frame.index.tz) == "UTC" and frame.index.is_unique
             and frame.index.is_monotonic_increasing,
             f"{zone}: audited EPEX/Storm UTC benchmark required")
    selected = frame.reindex(expected).astype(float)
    _require(np.isfinite(selected.actual.to_numpy()).all()
             and not np.isinf(selected.storm.to_numpy()).any(),
             f"{zone}: incomplete or infinite official benchmark")
    return selected


def _price_score(points: pd.Series, actual: pd.Series,
                 storm: pd.Series) -> dict:
    mask = np.isfinite(points.to_numpy(float)) & np.isfinite(actual.to_numpy(float)) \
        & np.isfinite(storm.to_numpy(float))
    p, a, s = (value.to_numpy(float)[mask] for value in (points, actual, storm))
    _require(len(p) > 0, "No common price/Storm observations")
    error, benchmark = np.abs(p - a), np.abs(s - a)
    return {"n": int(len(p)), "hours": int(len(p)),
            "rmse": float(np.sqrt(np.mean((p - a)**2))),
            "mae": float(error.mean()),
            "storm_rmse": float(np.sqrt(np.mean((s - a)**2))),
            "storm_mae": float(benchmark.mean()),
            "strict_wins": int((error < benchmark).sum()),
            "ties": int((error == benchmark).sum()),
            "strict_win_rate": float((error < benchmark).mean())}


def _probability_score(points: pd.Series, actual: pd.Series,
                       baseline: pd.Series) -> dict:
    p = points.to_numpy(float)
    b = baseline.to_numpy(float)
    a = actual.to_numpy(float)
    _require(np.isfinite(a).all() and np.isfinite(p).all() and np.isfinite(b).all()
             and ((p >= 0) & (p <= 1)).all() and ((b >= 0) & (b <= 1)).all(),
             "Invalid negative-price probability or observed price")
    event = a < 0.
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = [{"bin": number, "n": int((bins == number).sum()),
                    "mean_probability": float(p[bins == number].mean()),
                    "observed_frequency": float(event[bins == number].mean())}
                   for number in range(10) if (bins == number).any()]
    return {"n": int(len(p)), "hours": int(len(p)),
            "negative_hours": int(event.sum()),
            "observed_frequency": float(event.mean()),
            "mean_probability": float(p.mean()),
            "brier": float(np.mean((p - event)**2)),
            "history_frequency_brier": float(np.mean((b - event)**2)),
            "reliability": reliability}


def _target_alignment(canonical: pd.Series, benchmark_actual: pd.Series,
                      *, zone: str) -> dict:
    """Describe any canonical training-target / EPEX scoring-target drift.

    This is diagnostic only.  Both price candidates and Storm continue to be
    scored against the same EPEX benchmark, and the qualification rule is
    unchanged.  Event disagreement matters independently for P(price < 0).
    """
    _require(isinstance(canonical, pd.Series) and isinstance(benchmark_actual, pd.Series)
             and isinstance(benchmark_actual.index, pd.DatetimeIndex)
             and str(benchmark_actual.index.tz) == "UTC"
             and isinstance(canonical.index, pd.DatetimeIndex)
             and str(canonical.index.tz) == "UTC"
             and canonical.index.is_unique and canonical.index.is_monotonic_increasing,
             f"{zone}: canonical and EPEX prices require ordered physical UTC indexes")
    expected = benchmark_actual.index
    own = canonical.reindex(expected).to_numpy(dtype=float)
    epex = benchmark_actual.to_numpy(dtype=float)
    _require(np.isfinite(own).all() and np.isfinite(epex).all(),
             f"{zone}: canonical/EPEX alignment has missing or nonfinite prices")
    absolute = np.abs(own - epex)
    return {"hours": int(len(expected)), "exact_equal_hours": int(np.sum(own == epex)),
            "different_hours_gt_1e_9_eur_mwh": int(np.sum(absolute > 1e-9)),
            "mean_abs_difference_eur_mwh": float(absolute.mean()),
            "max_abs_difference_eur_mwh": float(absolute.max()),
            "negative_event_disagreements": int(np.sum((own < 0.) != (epex < 0.))),
            "training_target": "canonical Saturn target cache",
            "scoring_target": "EPEX reporting snapshot"}


def _selected_score(frame: pd.DataFrame, candidate: str) -> dict:
    return _price_score(frame[candidate], frame.actual, frame.storm)


def _select_negative_method(frame: pd.DataFrame) -> tuple[str, dict[str, dict]]:
    """Choose an OOF Brier winner with a declared deterministic tie order."""
    scores = {name: _probability_score(frame[name], frame.actual,
        frame.history_frequency) for name in NEGATIVE_METHODS}
    method = min(NEGATIVE_METHODS,
        key=lambda name: (scores[name]["brier"], NEGATIVE_METHODS.index(name)))
    return method, scores


def _source_files(source_audit: dict) -> dict[str, tuple[Path, str]]:
    """Join logical Saturn source names to the paths captured at load time."""
    hashes = source_audit.get("source_sha256", {})
    paths = source_audit.get("source_paths", {})
    _require(isinstance(hashes, dict) and isinstance(paths, dict) and hashes
             and set(hashes) == set(paths),
             "Saturn source audit is missing paths or checksums")
    root = ROOT.resolve()
    files = {name: (Path(paths[name]).resolve(), hashes[name])
             for name in sorted(hashes)}
    _require(all(path.is_relative_to(root) for path, _ in files.values()),
             "Saturn source audit points outside the project")
    return files


def _public_source_hashes(source_audit: dict) -> dict[str, str]:
    root = ROOT.resolve()
    return {path.relative_to(root).as_posix(): digest
            for path, digest in _source_files(source_audit).values()}


def _sources_unchanged(source_audit: dict) -> bool:
    return all(_sha(path) == digest
               for path, digest in _source_files(source_audit).values())


def evaluate_sources(sources: dict, benchmark: dict[str, pd.DataFrame],
                     output: Path, *, threads: int = 2,
                     progress=None) -> dict:
    """Run the fixed annual protocol on already fetched, immutable inputs.

    This callable permits offline tests with injected sources. It never
    synchronizes Saturn or writes the activation configuration.
    """
    _require(set(benchmark) == set(ZONES), "Four official benchmark zones required")
    _require(set(sources) == {"prices", "residual_bank", "nuclear_store"}
             and set(sources["prices"]) == set(ZONES),
             "Four price histories and both Saturn banks required")
    output.mkdir(parents=True, exist_ok=True)
    _require(not any(output.iterdir()), "Backtest output must start empty")
    comparisons = {zone: _benchmark_series(benchmark[zone], zone=zone) for zone in ZONES}
    target_alignment = {zone: _target_alignment(sources["prices"][zone],
        comparisons[zone].actual, zone=zone) for zone in ZONES}
    feature_index = grid(FEATURE_FIRST, STOP, "FR")
    features = {}
    feature_audits = {}
    for zone in ZONES:
        features[zone], feature_audits[zone] = build_country_features(
            zone=zone, delivery_index=feature_index, prices=sources["prices"],
            residual_bank=sources["residual_bank"], nuclear_store=sources["nuclear_store"])
    parts = {zone: [] for zone in ZONES}
    selected: dict[str, str] = {}
    negative_selected: dict[str, str] = {}
    negative_selection_scores: dict[str, dict[str, dict]] = {}
    origins = _origin_days()
    started = time.perf_counter()
    audits = []
    for number, origin in enumerate(origins):
        stop = _forecast_stop_day(origin, number)
        fit_first = str((pd.Timestamp(origin) - pd.Timedelta(days=365)).date())
        x = {zone: features[zone].loc[grid(fit_first, stop, zone)] for zone in ZONES}
        y = {zone: sources["prices"][zone].reindex(grid(fit_first, origin, zone)) for zone in ZONES}
        _require(all(np.isfinite(series.to_numpy(float)).all() for series in y.values()),
                 f"{origin}: canonical training targets incomplete")
        candidates = None if number < SELECTION_WEEKS else selected
        prices, price_audit = fit_predict_block(x, y, sources["prices"],
            origin_day=origin, stop_day=stop, candidate=candidates, threads=threads)
        negatives, negative_audit = fit_predict_negative_block(x, y,
            origin_day=origin, stop_day=stop, threads=threads)
        for zone in ZONES:
            daily = prices[zone].join(negatives[zone])
            daily["history_frequency"] = float(y[zone].lt(0.).mean())
            parts[zone].append(daily)
        audits.append({"origin_day": origin, "stop_day_exclusive": stop,
                       "cadence": "weekly_selection" if number < SELECTION_WEEKS
                                  else "daily_confirmation",
                       "price": price_audit, "negative": negative_audit})
        if progress is not None:
            progress({"phase": "selection_weekly_fit" if number < SELECTION_WEEKS
                                else "confirmation_daily_fit",
                      "completed_origins": number + 1,
                      "total_origins": len(origins), "origin_day": origin,
                      "elapsed_seconds": time.perf_counter() - started})
        if number == SELECTION_WEEKS - 1:
            for zone in ZONES:
                selection = pd.concat(parts[zone]).join(comparisons[zone], how="left")
                _require(selection.index.equals(grid(FIRST, CONFIRM_FIRST, zone)),
                         f"{zone}: incomplete selection forecast grid")
                selected[zone] = min(CANDIDATES,
                    key=lambda name: (_selected_score(selection, name)["rmse"],
                                      CANDIDATES.index(name)))
                negative_selected[zone], negative_selection_scores[zone] = (
                    _select_negative_method(selection))
            if progress is not None:
                progress({"phase": "selection_sealed", "selected": selected.copy(),
                          "negative_selected": negative_selected.copy(),
                          "completed_origins": number + 1,
                          "total_origins": len(origins)})
    _write_json(output / "fit_audits.json", {"fits": audits,
        "selection_weekly_count": SELECTION_WEEKS,
        "confirmation_daily_count": len(origins) - SELECTION_WEEKS})
    result = {"protocol": PROTOCOL, "backtest_protocol": "nyx_regional_cpu_backtest_v2",
              "selection_first_day": FIRST,
              "selection_stop_day_exclusive": CONFIRM_FIRST,
              "confirmation_first_day": CONFIRM_FIRST,
              "stop_day_exclusive": STOP,
              "selection_weekly_origins": origins[:SELECTION_WEEKS],
              "confirmation_daily_origins": origins[SELECTION_WEEKS:],
              "total_origins": len(origins),
              "candidate_order": list(CANDIDATES), "selected": selected,
              "negative_method_order": list(NEGATIVE_METHODS),
              "negative_selected": negative_selected,
              "feature_audits": feature_audits, "countries": {},
              "target_alignment": target_alignment,
              "fit_audits_sha256": _sha(output / "fit_audits.json"),
              "Storm_used_as_model_input": False,
              "future_labels_used_for_fit": False,
              "all_days_have_civil_D_minus_1_08_features": True,
              "causality_passed": True,
              "causality_scope": "physical-grid and query-asof cutoff checks; provider publication vintages unverified",
              "provider_publication_vintages_certified": False,
              "elapsed_seconds": time.perf_counter() - started}
    for zone in ZONES:
        forecast = pd.concat(parts[zone])
        expected = grid(FIRST, STOP, zone)
        _require(forecast.index.equals(expected) and forecast.index.is_unique,
                 f"{zone}: incomplete annual OOF grid")
        annual = forecast.join(comparisons[zone], how="left")
        selection = annual.loc[grid(FIRST, CONFIRM_FIRST, zone)]
        confirm = annual.loc[grid(CONFIRM_FIRST, STOP, zone)]
        common = int(np.isfinite(annual.storm.to_numpy(float)).sum())
        _require(common >= len(expected) - 1,
                 f"{zone}: more than one official Storm hour missing")
        selected_name = selected[zone]
        price_scores = {name: _selected_score(selection, name) for name in CANDIDATES}
        selected_confirmation = _selected_score(confirm, selected_name)
        selected_annual = _selected_score(annual, selected_name)
        negative_method = negative_selected[zone]
        negative_selection = negative_selection_scores[zone][negative_method]
        negative_confirmation = _probability_score(confirm[negative_method],
            confirm.actual, confirm.history_frequency)
        negative_annual = _probability_score(annual[negative_method],
            annual.actual, annual.history_frequency)
        eligible = (selected_confirmation["rmse"] < selected_confirmation["storm_rmse"]
                    and selected_confirmation["strict_win_rate"] > .5)
        output_file = output / f"{zone}.parquet"
        annual.to_parquet(output_file)
        result["countries"][zone] = {
            "hours": int(len(annual)), "storm_common_hours": common,
            "selected": selected_name, "selection_candidates": price_scores,
            "confirmation": selected_confirmation, "annual": selected_annual,
            "negative_selected": negative_method,
            "negative_selection_candidates": negative_selection_scores[zone],
            "negative_selection": negative_selection,
            "negative_confirmation": negative_confirmation,
            "negative_annual": negative_annual,
            "beats_storm_confirmation_rmse_and_win_rate": eligible,
            "oof_sha256": _sha(output_file)}
    result["all_requested_countries_qualify_against_storm"] = all(
        result["countries"][zone]["beats_storm_confirmation_rmse_and_win_rate"]
        for zone in REQUIRED_STORM_COUNTRIES)
    result["passed"] = result["all_requested_countries_qualify_against_storm"]
    result["qualification_blockers"] = [
        f"{zone}: selected CPU candidate does not beat Storm in confirmation RMSE and strict win rate"
        for zone in REQUIRED_STORM_COUNTRIES
        if not result["countries"][zone]["beats_storm_confirmation_rmse_and_win_rate"]]
    _write_json(output / "backtest_receipt.json", result)
    return result


def _fetch_official_benchmark(output: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    from chronos2_hourly.nuclear_reporting_refresh import refresh_nuclear_reporting_sources
    from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN
    from run_nuclear_forecast import load_settings, zone_inputs

    settings = load_settings((ROOT / "config" / "nuclear_forecast.yaml").resolve())
    expected = {zone: grid(FIRST, STOP, zone) for zone in ZONES}
    frames, audits = {}, {}
    for zone in ZONES:
        config, _, _ = zone_inputs(settings, zone)
        observed, folder, audit = refresh_nuclear_reporting_sources(
            config, zone, TIMEZONES[zone], STOP,
            output / "official_benchmark" / zone)
        path = folder / "inputs" / "storm_dashboard_official_statistics.parquet"
        declared = audit["storm_dashboard"]["normalized_artifact_sha256"]
        _require(path.is_file() and _sha(path) == declared,
                 f"{zone}: official Storm snapshot checksum mismatch")
        raw = pd.read_parquet(path)
        _require(set(raw) == {"delivery_start_utc", STORM_DASHBOARD_COLUMN},
                 f"{zone}: Storm snapshot schema mismatch")
        storm = pd.Series(raw[STORM_DASHBOARD_COLUMN].to_numpy(float),
                          index=pd.DatetimeIndex(pd.to_datetime(raw.delivery_start_utc, utc=True)))
        frame = pd.DataFrame({"actual": observed.reindex(expected[zone]),
                              "storm": storm.reindex(expected[zone])},
                             index=expected[zone])
        frames[zone] = _benchmark_series(frame, zone=zone)
        audits[zone] = {"snapshot_sha256": declared,
                        "observed_sha256": audit["observed"]["artifact_sha256"],
                        "source_kind": audit["mode"],
                        "storm_series": audit["storm_dashboard"]["series"]}
    return frames, audits


def _static_preflight() -> dict:
    blockers = []
    if importlib.util.find_spec("tshistory_lite") is None:
        blockers.append("Missing Saturn connector tshistory_lite; install requirements_nyx_regional_cpu.txt in the NYX Python environment")
    for path in (CONFIG, ROOT / "config" / "nuclear_forecast.yaml"):
        if not path.is_file():
            blockers.append(f"Missing configuration: {path.name}")
    try:
        preflight_sources(ROOT)
    except Exception as error:
        blockers.append(f"Source configuration invalid: {error}")
    if CONFIG.is_file():
        try:
            recipe = json.loads(CONFIG.read_text(encoding="utf-8"))
            if recipe.get("protocol") != PROTOCOL:
                blockers.append("CPU recipe protocol differs")
        except (OSError, ValueError) as error:
            blockers.append(f"CPU recipe unreadable: {error}")
    return {"operation": "evaluate", "ready": not blockers,
            "recipe_status": (recipe.get("status", "unknown") if CONFIG.is_file()
                              and "recipe" in locals() else "unknown"),
            "blockers": blockers}


def _activate(result: dict, output: Path, source_audit: dict,
               benchmark_audit: dict) -> None:
    _require(result["passed"], "A failed qualification cannot activate live runs")
    _require(set(result.get("negative_selected", {})) == set(ZONES)
             and all(result["negative_selected"][zone] in NEGATIVE_METHODS for zone in ZONES),
             "Four evaluated negative-probability methods required")
    recorded = {**result,
                "country_metrics": result["countries"],
                "countries": list(ZONES),
                "feature_columns": result["feature_audits"]["FR"]["feature_columns"],
                "source_sha256": _public_source_hashes(source_audit),
                "benchmark_sources": benchmark_audit,
                "code_sha256": {
                    name: _sha(ROOT / name) for name in (
                        "run_nyx_regional_cpu_backtest.py",
                        "run_nyx_regional_cpu.py",
                        "chronos2_hourly/nyx_regional_cpu.py",
                        "chronos2_hourly/nyx_cpu_live_features.py",
                        "chronos2_hourly/nyx_regional_cpu_sources.py")},
                "run_receipt_sha256": _sha(output / "backtest_receipt.json")}
    _write_json(CANONICAL_RECEIPT, recorded)
    recipe = json.loads(CONFIG.read_text(encoding="utf-8"))
    _require(recipe.get("schema_version") == 1 and recipe.get("protocol") == PROTOCOL,
             "CPU activation config changed")
    recipe.update(status="validated", selected=result["selected"],
                   negative_selected=result["negative_selected"],
                  backtest_receipt=str(CANONICAL_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
                  backtest_sha256=_sha(CANONICAL_RECEIPT))
    _write_json(CONFIG, recipe)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "nyx_regional_cpu" / "evaluation")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        info = _static_preflight()
        if args.dry_run:
            print(json.dumps(info, ensure_ascii=False, allow_nan=False), flush=True)
            return 0 if info["ready"] else 2
        _require(info["ready"], "; ".join(info["blockers"]))
        _require(1 <= args.threads <= 8 and 1 <= args.workers <= 8,
                 "1-8 threads/workers required")
        output = args.output.resolve()
        _require(output.is_relative_to(ROOT.resolve()) and output != ROOT.resolve(),
                 "Evaluation output must be inside the project")
        _require(not output.exists() or not any(output.iterdir()),
                 "Evaluation output already contains files")
        output.mkdir(parents=True, exist_ok=True)
        with exclusive_process_lock(ROOT / "runs" / "nyx_regional_cpu" / "evaluation.lock"):
            def status(value: dict) -> None:
                _write_json(output / "status.json", value)
            status({"phase": "saturn_sync", "status": "running"})
            sync_audit = sync_sources(ROOT, STOP, workers=args.workers,
                                      history_start_day=SOURCE_FIRST)
            sources, source_audit = load_sources(ROOT, STOP,
                                                 history_start_day=SOURCE_FIRST)
            status({"phase": "official_benchmark", "status": "running"})
            benchmark, benchmark_audit = _fetch_official_benchmark(output)
            status({"phase": "selection_weekly_fit", "status": "running",
                    "total_origins": len(_origin_days()),
                    "selection_weekly_origins": SELECTION_WEEKS,
                    "confirmation_daily_origins": len(_origin_days()) - SELECTION_WEEKS})
            # Keep the benchmark snapshots; evaluate_sources requires a fresh
            # directory because it seals the OOF outputs there.
            result_dir = output / "results"
            result = evaluate_sources(sources, benchmark, result_dir,
                                      threads=args.threads, progress=lambda item: status(
                                          {"status": "running", **item}))
            _require(_sources_unchanged(source_audit),
                     "Saturn sources changed during evaluation")
            result.update(source_sha256=_public_source_hashes(source_audit),
                          benchmark_sources=benchmark_audit,
                           code_sha256={name: _sha(ROOT / name) for name in (
                               "run_nyx_regional_cpu_backtest.py",
                               "run_nyx_regional_cpu.py",
                               "chronos2_hourly/nyx_regional_cpu.py",
                              "chronos2_hourly/nyx_cpu_live_features.py",
                              "chronos2_hourly/nyx_regional_cpu_sources.py")})
            _write_json(result_dir / "backtest_receipt.json", result)
            if result["passed"]:
                _activate(result, result_dir, source_audit, benchmark_audit)
            status({"phase": "complete", "status": "complete",
                    "qualified": result["passed"],
                    "receipt": str(result_dir / "backtest_receipt.json"),
                    "qualification_blockers": result["qualification_blockers"]})
            print(json.dumps({"status": "complete", "qualified": result["passed"],
                              "selected": result["selected"],
                              "receipt": str(result_dir / "backtest_receipt.json")},
                             ensure_ascii=False, allow_nan=False), flush=True)
            return 0 if result["passed"] else 2
    except Exception as error:
        if not args.dry_run:
            try:
                output = args.output.resolve()
                if output.is_relative_to(ROOT.resolve()):
                    _write_json(output / "status.json",
                                {"phase": "failed", "status": "failed",
                                 "error": f"{type(error).__name__}: {error}"})
            except (OSError, ValueError):
                pass
        print(f"[NYX CPU evaluation] {type(error).__name__}: {error}", file=sys.stderr,
              flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
