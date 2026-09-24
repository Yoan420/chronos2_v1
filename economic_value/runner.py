"""Snapshot/evaluate/report lifecycle for the separate economic-value lab."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import uuid

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock


LOGGER = logging.getLogger("economic_value")
ZONES = {"FR", "DE", "BE", "NL"}
MODELS = {"autonomous", "kalman", "nuclear_autonomous", "nuclear_kalman"}
INPUT_FILES = {"panel.parquet", "config.json", "data_audit.json", "reference_audit.json"}
RESULT_FILES = {"rows.parquet", "metrics.parquet", "daily.parquet", "breakdowns.parquet"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _json(path: Path, value) -> None:
    temp = path.with_name(".tmp_" + uuid.uuid4().hex + ".json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temp.replace(path)


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _output(root: Path, value: str | Path) -> Path:
    path = _path(root, value)
    allowed = (root / "runs/experiments").resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("Economic outputs must be a dedicated child of runs/experiments.")
    return path


def _date(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD, without time or timezone.")
    datetime.strptime(value, "%Y-%m-%d")
    return value


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected Economic Value schema_version: 1.")
    if config.get("evaluation_days") != 365 or config.get("timezone") != "Europe/Paris" or config.get("cutoff_time") != "08:00":
        raise ValueError("Exactly 365 calendar days and strict D-1 08:00 Europe/Paris are required.")
    for key, allowed in (("zones", ZONES), ("models", MODELS)):
        selected = config.get(key)
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected)-allowed:
            raise ValueError(f"Choose unique supported {key}: {sorted(allowed)}.")
    reference = config.get("reference", {})
    if (reference.get("kind") != "lagged_day_ahead_proxy" or reference.get("executable") is not False
            or reference.get("historical_publication_proof") is not False):
        raise ValueError("Current lab supports only the explicitly non-executable previous-day proxy.")
    if config.get("order_execution_enabled") is not False or config.get("production_modified") is not False:
        raise ValueError("This lab never activates a model or places an order.")
    portfolio = config["portfolio"]
    capacity = portfolio.get("capacity_mw")
    if isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Portfolio capacity_mw must be finite and strictly positive.")
    if portfolio.get("allocation") != "equal_fixed":
        raise ValueError("Only equal fixed allocation is implemented; no redistribution from missing countries.")
    for key in ("delivery_day", "end_day"):
        _date(config.get(key))
    strategy = config["strategy"]
    for key in ("signal_threshold_eur_mwh", "transaction_cost_eur_mwh", "slippage_eur_mwh"):
        value = strategy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if strategy.get("confidence_filter") != "none":
        raise ValueError("Confidence is stratified, not a candidate-only trading filter without benchmark quantiles.")


def load_config(path: Path, **overrides) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Configuration YAML illisible : {path}. {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The economic configuration must be a YAML mapping.")
    for key in ("zones", "models", "delivery_day", "end_day"):
        if overrides.get(key) is not None:
            config[key] = overrides[key]
    if overrides.get("portfolio_mw") is not None:
        config["portfolio"]["capacity_mw"] = overrides["portfolio_mw"]
    validate_config(config)
    return config


def engine_config(config: dict) -> dict:
    result = deepcopy(config["strategy"])
    result.update(timezone=config["timezone"], evaluation_days=config["evaluation_days"],
                  portfolio_capacity_mw=config["portfolio"]["capacity_mw"],
                  zone_capacity_mw={zone: config["portfolio"]["capacity_mw"] / len(config["zones"]) for zone in config["zones"]})
    return result


def _validate_panel(panel: pd.DataFrame, config: dict) -> dict:
    keys = ["timestamp_utc", "zone", "model"]
    if not set(keys).issubset(panel) or panel.duplicated(keys).any():
        raise ValueError("Unique physical timestamp/country/model rows are required.")
    if set(panel.zone) != set(config["zones"]) or set(panel.model) != set(config["models"]):
        raise ValueError("Selected countries/model alternatives do not match the panel.")
    if any(pd.Timestamp(t).tzinfo is None for t in panel.timestamp_utc):
        raise ValueError("UTC-aware physical timestamps are mandatory.")
    local_days = pd.DatetimeIndex(panel.timestamp_utc).tz_convert(config["timezone"]).tz_localize(None).normalize()
    origins = (local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(config["timezone"]).tz_convert("UTC")
    if not pd.DatetimeIndex(panel.forecast_origin_utc).equals(origins):
        raise ValueError("Forecast origin must be exactly D-1 08:00 civil for every delivery interval.")
    if not panel["sample"].isin(["evaluation", "live"]).all():
        raise ValueError("Only evaluation or live sample labels are supported.")
    if not panel.duration_hours.eq(1).all():
        raise ValueError("This report adapter is hourly: duration_hours must equal one.")
    sample = panel.loc[panel["sample"].eq("evaluation")].copy()
    stamps = pd.DatetimeIndex(sample.timestamp_utc).tz_convert(config["timezone"])
    first, last = stamps.min().date().isoformat(), stamps.max().date().isoformat()
    start = pd.Timestamp(first).tz_localize(config["timezone"])
    stop = (pd.Timestamp(last) + pd.Timedelta(days=1)).tz_localize(config["timezone"])
    if (pd.Timestamp(last)-pd.Timestamp(first)).days != 364:
        raise ValueError("Evaluation must retain exactly 365 calendar days including unavailable hours.")
    if config.get("end_day") is not None and last != config["end_day"]:
        raise ValueError("The panel does not end on the explicitly requested evaluation date.")
    expected = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC")
    for zone in config["zones"]:
        for model in config["models"]:
            block = sample.loc[sample.zone.eq(zone) & sample.model.eq(model)].sort_values("timestamp_utc")
            if not pd.DatetimeIndex(block.timestamp_utc).equals(expected):
                raise ValueError(f"{zone}/{model}: shortened, duplicated or shifted evaluation support.")
    # Candidate comparisons must use the same observations and reference signals.
    for name in ("actual", "benchmark_forecast", "reference_price"):
        if panel.groupby(["timestamp_utc", "zone"])[name].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Candidate alternatives disagree on shared {name}; do not rank unlike benchmarks.")
    return {"evaluation_start": first, "evaluation_end": last, "evaluation_days": 365,
            "expected_hours_per_zone_model": len(expected)}


def audit_inputs(config: dict, *, root: Path) -> tuple[pd.DataFrame, dict, dict]:
    from .data import load_report_panel, load_reference_proxy
    validate_config(config)
    panel, data_audit = load_report_panel(root, zones=config["zones"], models=config["models"],
                                         delivery_day=config.get("delivery_day"), end_day=config.get("end_day"))
    references, reference_audit = load_reference_proxy(root, panel)
    panel = panel.merge(references, on=["timestamp_utc", "zone"], how="left", validate="many_to_one")
    coverage = _validate_panel(panel, config)
    return panel, {**data_audit, **coverage}, reference_audit


def _engine_seals() -> dict:
    return {name: digest(Path(__file__).with_name(name)) for name in ("data.py", "engine.py", "runner.py")}


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = _output(root, config["output_root"])
    with exclusive_process_lock(output / "prepare.lock"):
        LOGGER.info("Gel des rapports locaux, observations et prix de reference de la veille.")
        panel, data_audit, reference_audit = audit_inputs(config, root=root)
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        snapshot = output / "snapshots" / token
        snapshot.mkdir(parents=True, exist_ok=False)
        panel.to_parquet(snapshot / "panel.parquet", index=False)
        for name, value in (("config", config), ("data_audit", data_audit), ("reference_audit", reference_audit)):
            _json(snapshot / f"{name}.json", value)
        manifest = {"schema_version": 1, "status": "prepared", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    **{key: data_audit[key] for key in ("evaluation_start", "evaluation_end", "evaluation_days", "expected_hours_per_zone_model")},
                    "zones": config["zones"], "models": config["models"], "cutoff_time": "08:00", "timezone": config["timezone"],
                    "source_code_sha256": _engine_seals(),
                    "snapshot_files": {name: digest(snapshot / name) for name in sorted(INPUT_FILES)},
                    "diagnostic_only": True, "executable_reference": False, "production_modified": False,
                    "activation_performed": False, "orders_placed": False,
                    "forecast_and_reference_pit_certified": False, "baseline_neural_oof_certified": False}
        _json(snapshot / "manifest.json", manifest)
        _json(snapshot / "status.json", {"status": "prepared", "snapshot": str(snapshot)})
        _json(output / "latest_prepared.json", {"snapshot": str(snapshot)})
        LOGGER.info("Snapshot prepare : %s", snapshot)
        return snapshot


def _verify(snapshot: Path, manifest: dict, *, results=False) -> None:
    expected = manifest["result_files" if results else "snapshot_files"]
    if set(expected) != (RESULT_FILES if results else INPUT_FILES):
        raise ValueError("Economic snapshot manifest is incomplete.")
    for name, sha in expected.items():
        if Path(name).name != name or digest(snapshot / name) != sha:
            raise ValueError(f"Economic snapshot checksum mismatch: {name}.")


def read_snapshot(snapshot: Path, *, root: Path) -> tuple[Path, dict, dict, pd.DataFrame]:
    snapshot = _output(root, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    _verify(snapshot, manifest)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    panel = pd.read_parquet(snapshot / "panel.parquet")
    coverage = _validate_panel(panel, config)
    if any(manifest.get(key) != value for key, value in coverage.items()):
        raise ValueError("Snapshot calendar manifest disagrees with its sealed inputs.")
    if manifest.get("zones") != config["zones"] or manifest.get("models") != config["models"]:
        raise ValueError("Snapshot selection differs from its sealed configuration.")
    return snapshot, config, manifest, panel


def evaluate(snapshot: Path, *, root: Path) -> Path:
    from .engine import simulate
    snapshot, config, manifest, panel = read_snapshot(snapshot, root=root)
    with exclusive_process_lock(snapshot / "evaluation.lock"):
        if (snapshot / "results_manifest.json").exists():
            LOGGER.info("Resultats deja calcules : verification puis regeneration HTML seule.")
            path = report(snapshot, root=root)
            _publish_completion(root, config, manifest, snapshot, path)
            return path
        if manifest["source_code_sha256"] != _engine_seals():
            raise ValueError("Evaluation code changed since Prepare; create a new snapshot.")
        _json(snapshot / "status.json", {"status": "running", "snapshot": str(snapshot), "stage": "simulation"})
        try:
            LOGGER.info("Evaluation des trois strategies, a capacite et regle identiques.")
            policy = engine_config(config)
            policy.update(evaluation_start_day=manifest["evaluation_start"], evaluation_end_day=manifest["evaluation_end"])
            result = simulate(panel, policy)
            # Preserve all breakdowns in one safe Parquet schema, without relying
            # on user-provided names as output paths. Group labels span ints/strings.
            breakdowns = pd.concat([table.assign(group=table["group"].astype(str), breakdown=name)
                                    for name, table in result.breakdowns.items()], ignore_index=True)
            for name, frame in (("rows", result.rows), ("metrics", result.metrics), ("daily", result.daily), ("breakdowns", breakdowns)):
                temp = snapshot / f".tmp_{name}_{uuid.uuid4().hex}.parquet"
                frame.to_parquet(temp, index=False)
                temp.replace(snapshot / f"{name}.parquet")
            audit = {**manifest, "status": "completed_hypothetical_diagnostic", "engine": result.audit,
                     "portfolio_capacity_mw": config["portfolio"]["capacity_mw"], "zone_capacity_mw": engine_config(config)["zone_capacity_mw"],
                     "rule": config["strategy"], "reference_kind": "lagged_day_ahead_proxy",
                     "reference_warning": "Le prix de la veille concerne une autre livraison : ce n'est pas un prix auquel traiter pour le jour prédit.",
                     "confidence_warning": "Heuristique issue du signal et des intervalles, pas une probabilité de profit calibrée.",
                     "market_resolution_warning": "Proxy horaire : ne reproduit pas les produits quart-horaires ni les mécanismes d'exécution réels.",
                     "data": json.loads((snapshot / "data_audit.json").read_text(encoding="utf-8")),
                     "reference": json.loads((snapshot / "reference_audit.json").read_text(encoding="utf-8")),
                     "methodology_sources": ["https://web.stanford.edu/~wfsharpe/art/sr/sr.htm",
                                               "https://alo.mit.edu/publications/page/18/",
                                               "https://www.epexspot.com/en/new-15-minute-products-market-coupling"],
                     "result_files": {name: digest(snapshot / name) for name in sorted(RESULT_FILES)}}
            _json(snapshot / "results_manifest.json", audit)
            path = report(snapshot, root=root)
            _publish_completion(root, config, manifest, snapshot, path)
            return path
        except Exception as exc:
            _json(snapshot / "status.json", {"status": "failed", "snapshot": str(snapshot), "error": f"{type(exc).__name__}: {exc}"})
            raise


def _publish_completion(root: Path, config: dict, manifest: dict, snapshot: Path, report_path: Path) -> None:
    _json(snapshot / "status.json", {"status": "completed", "snapshot": str(snapshot), "report": str(report_path)})
    _json(_output(root, config["output_root"]) / "latest.json", {"snapshot": str(snapshot), "report": str(report_path),
                                                               "evaluation_end": manifest["evaluation_end"]})


def report(snapshot: Path, *, root: Path) -> Path:
    from .reporting import render_report
    snapshot, config, manifest, panel = read_snapshot(snapshot, root=root)
    result = json.loads((snapshot / "results_manifest.json").read_text(encoding="utf-8"))
    _verify(snapshot, result, results=True)
    for key in ("evaluation_start", "evaluation_end", "zones", "models", "snapshot_files", "source_code_sha256"):
        if result.get(key) != manifest[key]:
            raise ValueError("Result audit and frozen input manifest disagree.")
    destination = snapshot / "economic_value_report.html"
    with exclusive_process_lock(snapshot / "report.lock"):
        temporary = snapshot / f".tmp_report_{uuid.uuid4().hex}.html"
        flat = pd.read_parquet(snapshot / "breakdowns.parquet")
        breakdowns = {name: table.drop(columns="breakdown") for name, table in flat.groupby("breakdown")}
        render_report(pd.read_parquet(snapshot / "rows.parquet"), pd.read_parquet(snapshot / "metrics.parquet"),
                      pd.read_parquet(snapshot / "daily.parquet"), breakdowns, result, temporary)
        temporary.replace(destination)
    LOGGER.info("Rapport : %s", destination)
    return destination


def resolve_snapshot(root: Path, config: dict, run_directory: Path | None, *, prepared=False) -> Path:
    if run_directory is not None:
        return _output(root, run_directory)
    pointer = _output(root, config["output_root"]) / ("latest_prepared.json" if prepared else "latest.json")
    if not pointer.is_file():
        raise ValueError("Aucun snapshot disponible. Lancez d'abord -Action Run ou Prepare.")
    return _output(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
