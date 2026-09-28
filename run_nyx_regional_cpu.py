#!/usr/bin/env python
"""Run the new Saturn-fed CWE CPU prices and negative-price probabilities."""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_regional_cpu import (
    CANDIDATES, NEGATIVE_METHODS, PROTOCOL, TIMEZONES, ZONES, civil_day, fit_negative_price,
    fit_price_candidates, grid,
)
from chronos2_hourly.nyx_cpu_live_features import FEATURE_COLUMNS, build_country_features
from chronos2_hourly.nyx_regional_cpu_sources import (
    load_sources, preflight_sources, sha256, sync_sources,
)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "nyx_regional_cpu.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filename = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            filename = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(filename, path)
    finally:
        if filename is not None:
            Path(filename).unlink(missing_ok=True)


def _validate_activation(root: Path = ROOT) -> tuple[dict, list[str]]:
    """Refuse manual status flips without the pinned backtest receipt."""
    blockers = []
    try:
        config = _json(root / "config" / "nyx_regional_cpu.json")
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "missing_or_invalid"}, [f"CPU recipe config unavailable: {exc}"]
    if (config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL
            or config.get("status") not in {"pending_backtest", "validated"}):
        blockers.append("CPU recipe config schema/status invalid")
    if config.get("status") != "validated":
        blockers.append("CPU recipe awaits the complete chronological backtest")
        return config, blockers
    selected = config.get("selected")
    if (not isinstance(selected, dict) or set(selected) != set(ZONES)
            or any(value not in CANDIDATES for value in selected.values())):
        blockers.append("Exactly one evaluated candidate per country required")
        return config, blockers
    negative_selected = config.get("negative_selected")
    if (not isinstance(negative_selected, dict) or set(negative_selected) != set(ZONES)
            or any(value not in NEGATIVE_METHODS for value in negative_selected.values())):
        blockers.append("Exactly one evaluated negative-probability method per country required")
        return config, blockers
    relative = config.get("backtest_receipt")
    digest = config.get("backtest_sha256")
    if not isinstance(relative, str) or not relative or not isinstance(digest, str) or len(digest) != 64:
        blockers.append("Pinned backtest receipt path and SHA-256 required")
        return config, blockers
    receipt = (root / relative).resolve()
    if not receipt.is_relative_to(root.resolve()) or not receipt.is_file():
        blockers.append("Pinned backtest receipt missing or outside repository")
        return config, blockers
    if sha256(receipt) != digest:
        blockers.append("Backtest receipt SHA-256 mismatch")
        return config, blockers
    try:
        evidence = _json(receipt)
        if (evidence.get("protocol") != PROTOCOL or evidence.get("passed") is not True
                or evidence.get("selected") != selected
                or evidence.get("negative_selected") != negative_selected
                or evidence.get("feature_columns") != list(FEATURE_COLUMNS)
                or evidence.get("countries") != list(ZONES)):
            blockers.append("Backtest receipt did not validate this selection and feature schema")
        if (evidence.get("causality_passed") is not True
                or evidence.get("future_labels_used_for_fit") is True
                or evidence.get("Storm_used_as_model_input") is True
                or evidence.get("storm_used_as_model_input") is True):
            blockers.append("Backtest causal/Storm input checks did not pass")
        code = evidence.get("code_sha256")
        required_code = ("run_nyx_regional_cpu_backtest.py",
            "run_nyx_regional_cpu.py",
            "chronos2_hourly/nyx_regional_cpu.py",
            "chronos2_hourly/nyx_cpu_live_features.py",
            "chronos2_hourly/nyx_regional_cpu_sources.py")
        if (not isinstance(code, dict) or not set(required_code) <= set(code)
                or any(not isinstance(code.get(name), str) or len(code[name]) != 64
                       or sha256(root / name) != code[name] for name in required_code)):
            blockers.append("Backtest model, feature or source code SHA-256 mismatch")
        metrics = evidence.get("country_metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(ZONES):
            blockers.append("Backtest receipt lacks four-country metrics")
        else:
            for zone in ZONES:
                item = metrics[zone]
                confirmation = item.get("confirmation") if isinstance(item, dict) else None
                negative = item.get("negative_confirmation") if isinstance(item, dict) else None
                if (not isinstance(item, dict) or item.get("selected") != selected[zone]
                        or item.get("negative_selected") != negative_selected[zone]
                        or not isinstance(item.get("hours"), int) or item["hours"] < 8759
                        or not isinstance(item.get("storm_common_hours"), int)
                        or item["storm_common_hours"] < 8759
                        or not isinstance(confirmation, dict)
                        or not isinstance(confirmation.get("hours"), int)
                        or confirmation["hours"] <= 0
                        or not isinstance(negative, dict)
                        or not isinstance(negative.get("hours"), int)
                        or negative["hours"] <= 0):
                    blockers.append(f"{zone}: backtest/confirmation support incomplete")
                    continue
                scores = item.get("negative_selection_candidates")
                if (not isinstance(scores, dict) or set(scores) != set(NEGATIVE_METHODS)
                        or any(not isinstance(scores[name], dict)
                               or not isinstance(scores[name].get("brier"), (int, float))
                               or not np.isfinite(scores[name]["brier"])
                               or not 0 <= scores[name]["brier"] <= 1
                               for name in NEGATIVE_METHODS)):
                    blockers.append(f"{zone}: negative Brier selection scores missing or invalid")
                    continue
                winner = min(NEGATIVE_METHODS, key=lambda name:
                    (scores[name]["brier"], NEGATIVE_METHODS.index(name)))
                if winner != negative_selected[zone]:
                    blockers.append(f"{zone}: negative method is not the selection Brier winner")
    except (OSError, ValueError, TypeError) as exc:
        blockers.append(f"Backtest receipt invalid: {exc}")
    return config, blockers


def plan(*, delivery_day: str, countries: tuple[str, ...], output: Path,
         inspect_sources: bool) -> dict:
    day = civil_day(delivery_day)
    if not countries or len(countries) != len(set(countries)) or not set(countries) <= set(ZONES):
        raise ValueError("Unique FR/DE/BE/NL countries required")
    target = output.resolve()
    config, blockers = _validate_activation()
    if target.exists():
        blockers.append(f"Output exists; choose a new directory: {target}")
    sources = None
    if inspect_sources:
        if importlib.util.find_spec("tshistory_lite") is None:
            blockers.append("tshistory_lite is required for live Saturn synchronization")
        try:
            from run_nuclear_forecast import delivery_date
            delivery_date(delivery_day)  # the normal D-1 08:00 cutoff must have occurred
            sources = preflight_sources(ROOT)
            if not sources["ready"]:
                blockers.append("Local Saturn source cache or audit missing")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            blockers.append(f"Saturn source preflight failed: {exc}")
    return {"operation": "forecast", "protocol": PROTOCOL,
            "delivery_day": str(day.date()), "countries": list(countries),
            "output": str(target), "recipe_status": config.get("status", "missing_or_invalid"),
            "selected": config.get("selected", {}),
            "negative_selected": config.get("negative_selected", {}),
            "feature_source": "production Saturn nuclear/residual banks plus canonical price caches",
            "requires_saturn_sync": True, "source_preflight": sources,
            "ready": not blockers, "blockers": blockers}


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Existing forecast artifact: {path}")
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"Existing temporary forecast artifact: {temporary}")
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=True)
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False, float_format="%.17g")
        else:
            raise ValueError("CSV or Parquet output required")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_html(path: Path, frame: pd.DataFrame, zone: str, day: str,
                candidate: str, negative_method: str) -> None:
    if path.exists():
        raise ValueError(f"Existing forecast report: {path}")
    rows = frame[["timestamp_local", "price_eur_mwh", "p_negative",
                  "is_negative_predicted"]].copy()
    rows["price_eur_mwh"] = rows["price_eur_mwh"].map(lambda x: f"{x:.2f}")
    rows["p_negative"] = rows["p_negative"].map(lambda x: f"{100*x:.1f} %")
    table = rows.to_html(index=False, escape=True, classes="forecast")
    title = f"NYX regional CPU – {zone} – {day}"
    document = ("<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>body{{font:16px system-ui;max-width:980px;"
        "margin:2rem auto;padding:0 1rem;color:#17202a}table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #ccd;padding:.55rem;text-align:right}td:first-child,th:first-child{text-align:left}"
        "tr:nth-child(even){background:#f6f8fa}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p>Prévision ponctuelle : {html.escape(candidate)}. "
        f"Méthode de probabilité : {html.escape(negative_method)}. "
        "Probabilité estimée d’un prix horaire négatif (seuil d’alerte 50 %).</p>"
        f"{table}</body></html>")
    path.write_text(document, encoding="utf-8")


def _operational_negative_probability(points: pd.DataFrame, actual: pd.Series,
                                      method: str) -> pd.Series:
    """Apply the sealed method using only prices before the delivery day."""
    if method not in NEGATIVE_METHODS:
        raise ValueError(f"Unknown negative-probability method: {method}")
    if method == "history_frequency":
        values = actual.to_numpy(float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Complete historical prices required for frequency probability")
        chosen = pd.Series(float((values < 0.).mean()), index=points.index)
    else:
        chosen = points[method].astype(float)
    values = chosen.to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0.) & (values <= 1.)).all():
        raise ValueError("Selected negative-price probabilities are invalid")
    return chosen.rename("p_negative")


def _save_model(path: Path, estimator) -> dict:
    if path.exists():
        raise ValueError(f"Model already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp.cbm")
    if temporary.exists():
        raise ValueError(f"Model temporary file already exists: {temporary}")
    try:
        estimator.save_model(str(temporary), format="cbm")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError("Empty CatBoost model")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def run_forecast(selection: dict, negative_selection: dict, *, delivery_day: str,
                 countries: tuple[str, ...],
                 output: Path, threads: int = 2, workers: int = 2) -> dict:
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("1-8 CPU threads required")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("1-8 Saturn workers required")
    if any(negative_selection.get(zone) not in NEGATIVE_METHODS for zone in countries):
        raise ValueError("An evaluated negative-probability method is required per country")
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    started = pd.Timestamp.now(tz="UTC").isoformat()

    def status(phase: str, state: str = "RUNNING", **extra):
        payload = {"status": state, "phase": phase, "protocol": PROTOCOL,
                   "delivery_day": delivery_day, "countries": list(countries),
                   "started_utc": started, "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                   **extra}
        _atomic_json(status_path, payload)
        print(json.dumps({"status": state, "phase": phase, **extra}, default=str), flush=True)

    try:
        status("sync_saturn")
        sync_audit = sync_sources(ROOT, delivery_day, workers=workers)
        sources, source_audit = load_sources(ROOT, delivery_day)
        status("build_features")
        day = civil_day(delivery_day)
        train_index = grid(day - pd.Timedelta(days=365), day, "FR")
        forecast_index = grid(day, day + pd.Timedelta(days=1), "FR")
        full_index = train_index.append(forecast_index)
        receipts = {}
        for zone in countries:
            features, feature_audit = build_country_features(zone=zone,
                delivery_index=full_index, prices=sources["prices"],
                residual_bank=sources["residual_bank"], nuclear_store=sources["nuclear_store"])
            if list(features.columns) != list(FEATURE_COLUMNS):
                raise ValueError("CPU feature schema changed")
            actual = sources["prices"][zone].reindex(grid(day - pd.Timedelta(days=365), day, zone))
            if not np.isfinite(actual.to_numpy(float)).all():
                raise ValueError(f"{zone}: incomplete target history")
            selected = selection[zone]
            negative_method = negative_selection[zone]
            status("fit_models", zone=zone, candidate=selected,
                   negative_method=negative_method)
            price = fit_price_candidates(features, actual, sources["prices"][zone],
                zone=zone, delivery_day=delivery_day, selected=selected, threads=threads)
            negative = fit_negative_price(features, actual, zone=zone,
                delivery_day=delivery_day, threads=threads)
            if not price.predictions.index.equals(negative.predictions.index):
                raise ValueError(f"{zone}: price and probability grids differ")
            probability = _operational_negative_probability(
                negative.predictions, actual, negative_method)
            frame = pd.DataFrame({"price_eur_mwh": price.predictions[selected],
                "p_negative_raw": negative.predictions.p_negative_raw,
                "p_negative_calibrated": negative.predictions.p_negative,
                "p_negative": probability,
                "is_negative_predicted": probability >= .5,
                "timestamp_local": [t.isoformat() for t in price.predictions.index.tz_convert(TIMEZONES[zone])]},
                index=price.predictions.index)
            frame.index.name = "timestamp_utc"
            zone_dir = output / "zones" / zone
            zone_dir.mkdir(parents=True, exist_ok=False)
            basename = f"forecast_{zone.lower()}_{delivery_day}_nyx_regional_cpu"
            parquet_path, csv_path, html_path = (zone_dir / f"{basename}.{suffix}"
                                                 for suffix in ("parquet", "csv", "html"))
            status("publish", zone=zone)
            _write_frame(parquet_path, frame)
            csv_frame = frame.copy()
            csv_frame.insert(0, "timestamp_utc", [t.isoformat() for t in frame.index])
            _write_frame(csv_path, csv_frame)
            _write_html(html_path, frame, zone, delivery_day, selected,
                        negative_method)
            saved = {name: _save_model(zone_dir / f"price_{name}.cbm", estimator)
                     for name, estimator in price.models.items()}
            if negative.model is not None:
                saved["negative"] = _save_model(zone_dir / "negative_price.cbm", negative.model)
            zone_receipt = {"status": "COMPLETE", "protocol": PROTOCOL, "zone": zone,
                "delivery_day": delivery_day, "selected_candidate": selected,
                "negative_selected": negative_method,
                "forecast_hours": len(frame), "feature_audit": feature_audit,
                "price_audit": price.audit, "negative_audit": negative.audit,
                "source_sha256": source_audit["source_sha256"],
                "models": saved, "outputs": {path.name: sha256(path)
                    for path in (parquet_path, csv_path, html_path)},
                "current_or_future_actual_used": False, "Storm_used": False}
            receipt_path = zone_dir / "receipt.json"
            _atomic_json(receipt_path, zone_receipt)
            receipts[zone] = {"path": str(receipt_path), "sha256": sha256(receipt_path),
                              "report": str(html_path), "csv": str(csv_path)}
        global_receipt = {"status": "COMPLETE", "protocol": PROTOCOL,
            "delivery_day": delivery_day, "countries": list(countries),
            "selected_candidates": {zone: selection[zone] for zone in countries},
            "negative_selected": {zone: negative_selection[zone] for zone in countries},
            "recipe_config_sha256": sha256(CONFIG), "source_sync": sync_audit,
            "source_audit": source_audit, "zones": receipts,
            "current_or_future_actual_used": False, "Storm_used": False}
        _atomic_json(output / "receipt.json", global_receipt)
        status("complete", "COMPLETE", completed_countries=list(countries),
               receipt=str(output / "receipt.json"))
        return global_receipt
    except BaseException as exc:
        status("failed", "FAILED", error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc())
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--countries", "--zones", nargs="+", default=list(ZONES), dest="countries")
    parser.add_argument("--output", "--output-dir", type=Path, default=None, dest="output")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        countries = tuple(zone.upper() for zone in args.countries)
        output = args.output or ROOT / "runs" / "nyx_regional_cpu" / args.delivery_day
        dry = plan(delivery_day=args.delivery_day, countries=countries, output=output,
                   inspect_sources=args.preflight or not args.dry_run)
        if args.dry_run or args.preflight:
            print(json.dumps(dry, ensure_ascii=False, default=str))
            return 0 if args.dry_run or dry["ready"] else 2
        if not dry["ready"]:
            print(json.dumps(dry, ensure_ascii=False, default=str))
            return 2
        receipt = run_forecast(dry["selected"], dry["negative_selected"],
            delivery_day=args.delivery_day,
            countries=countries, output=output, threads=args.threads, workers=args.workers)
        print(json.dumps({"status": "COMPLETE", "receipt": str(output / "receipt.json"),
                          "countries": receipt["countries"]}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "FAILED", "operation": "forecast",
                          "recipe_status": "unknown", "ready": False,
                          "blockers": [f"{type(exc).__name__}: {exc}"]}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
