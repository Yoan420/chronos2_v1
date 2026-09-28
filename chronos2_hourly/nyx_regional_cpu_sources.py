"""Saturn data acquisition for the prospective regional CPU model.

The live update reuses the nuclear Kalman source synchronizers. A read-only
preflight inspects local structure; it cannot prove a future Saturn response.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ZONES = ("FR", "DE", "BE", "NL")
TARGET_LOOKBACK_DAYS = 372  # 365 fit days plus D-7 price features.


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _settings(project_root: Path):
    from run_nuclear_forecast import load_settings
    config = (project_root / "config" / "nuclear_forecast.yaml").resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    settings = load_settings(config)
    if Path(settings["project_root"]).resolve() != project_root.resolve():
        raise ValueError("Nuclear source config points to a different project")
    return settings


def source_paths(project_root: Path) -> dict[str, Path]:
    from run_nuclear_forecast import zone_inputs
    settings = _settings(project_root)
    paths = {"residual_bank": Path(settings["residual_bank"]),
             "nuclear_store": Path(settings["nuclear_store"])}
    for zone in ZONES:
        _, _, zone_paths = zone_inputs(settings, zone)
        paths[f"target_{zone}"] = Path(zone_paths["target"])
    return paths


def preflight_sources(project_root: Path) -> dict:
    """Check local source structure without network, writes or model fitting."""
    paths = source_paths(project_root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    missing_audits = [name for name in ("residual_bank", "nuclear_store")
                      if not paths[name].with_name(paths[name].name + ".audit.json").is_file()]
    return {"source_paths": {name: str(path) for name, path in paths.items()},
            "missing_source_files": missing, "missing_source_audits": missing_audits,
            "requires_saturn_sync": True,
            "ready": not missing and not missing_audits}


def sync_sources(project_root: Path, delivery_day: str, *, workers: int = 2,
                 history_start_day: str | None = None) -> dict:
    """Use existing production Saturn synchronizers, then hash every input.

    This operation contacts Saturn and updates the same shared caches as the
    nuclear Kalman pipeline. It must only be called by a real run, never by a
    dry-run or preflight.
    """
    from chronos2_hourly.nuclear_residual_inputs import ensure_residual_bank
    from chronos2_modular.common import build_zone_configs
    from chronos2_modular.saturn import sync_saturn_data
    from run_nuclear_forecast import delivery_date, sync_source, zone_inputs

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("1-8 Saturn workers required")
    settings = _settings(project_root)
    day = delivery_date(delivery_day)
    default_start = (day - pd.Timedelta(days=TARGET_LOOKBACK_DAYS)).date()
    start = pd.Timestamp(history_start_day).date() if history_start_day else default_start
    if start > default_start or start >= day.date():
        raise ValueError("history_start_day must cover the normal 372-day history")
    nuclear = sync_source(settings, day, workers)
    if not nuclear["complete"]:
        raise ValueError("Saturn nuclear forecast source incomplete")
    # The regular nuclear run needs 730 days. Annual backtests need seven more
    # days so the first training day's D-7 price/forecast features exist.
    if history_start_day:
        from chronos2_hourly.nuclear_sources import audit_nuclear_store, build_materialize_command
        from chronos2_hourly.process_lock import exclusive_process_lock
        import subprocess
        import sys
        path = Path(settings["nuclear_store"])
        with exclusive_process_lock(path.with_suffix(".sync.lock")):
            extended = audit_nuclear_store(path, str(start), str(day.date()))
            missing = [pd.Timestamp(value) for value in extended["missing_days"]]
            while missing:
                chunk = [missing.pop(0)]
                while (missing and len(chunk) < int(settings.get("sync_chunk_days", 31))
                       and missing[0] == chunk[-1] + pd.Timedelta(days=1)):
                    chunk.append(missing.pop(0))
                command = build_materialize_command(chunk[0], chunk[-1], path,
                    sys.executable, min(workers, 32), settings["incomplete_dst_policy"])
                command.append("--merge-existing")
                subprocess.run(command, check=True, cwd=project_root)
            nuclear = audit_nuclear_store(path, str(start), str(day.date()))
            if not nuclear["complete"]:
                raise ValueError("Extended nuclear Saturn history incomplete: "
                                 + " | ".join(nuclear["blockers"][:3]))
    residual = ensure_residual_bank(path=settings["residual_bank"],
        seed_path=settings["residual_bank_seed"], start_day=start, end_day=day.date(),
        workers=workers, allow_sync=True)
    if not residual["complete"]:
        raise ValueError("Saturn residual forecast source incomplete")
    target_manifest = {}
    for zone in ZONES:
        raw_config, config_path, _ = zone_inputs(settings, zone)
        config = deepcopy(raw_config)
        config["data"]["start"] = str(start)
        zones = build_zone_configs(config, [zone], None, None)
        manifest = sync_saturn_data(zones, config, config_path.parent,
                                    skip_pit=True, as_of=pd.Timestamp.now(tz="UTC"))
        target_manifest[zone] = json.loads(manifest.to_json(orient="records", date_format="iso"))
    paths = source_paths(project_root)
    return {"nuclear": {"complete": nuclear["complete"],
                         "expected_hours": nuclear["expected_hours"]},
            "residual": {"complete": residual["complete"],
                           "expected_hours": residual["expected_hours"]},
            "target_sync": target_manifest,
            "history_start_day": str(start),
            "paths": {key: {"path": str(path), "sha256": sha256(path)}
                      for key, path in paths.items()}}


def _load_target(path: Path, *, first_day: pd.Timestamp, delivery_day: pd.Timestamp,
                 zone: str) -> pd.Series:
    from chronos2_hourly.nyx_regional_cpu import grid
    from run_nuclear_forecast import validate_target_cache
    validate_target_cache(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
    values = pd.Series(frame.value.to_numpy(float), index=index, name=f"price_{zone}")
    expected = grid(first_day, delivery_day, zone)
    actual = values.reindex(expected)
    if not np.isfinite(actual.to_numpy(float)).all():
        missing = expected[~np.isfinite(actual.to_numpy(float))]
        raise ValueError(f"{zone}: canonical Saturn target is missing {len(missing)} "
                         f"historical hours, first {missing[0] if len(missing) else 'unknown'}")
    return values.loc[values.index < expected[-1] + pd.Timedelta(hours=1)]


def load_sources(project_root: Path, delivery_day: str, *,
                 history_start_day: str | None = None) -> tuple[dict, dict]:
    """Read complete synced source banks; no network and no future labels."""
    from chronos2_hourly.nuclear_residual_inputs import audit_residual_bank
    from chronos2_hourly.nuclear_sources import audit_nuclear_store
    from chronos2_hourly.nyx_regional_cpu import civil_day
    day = civil_day(delivery_day)
    default_first = day - pd.Timedelta(days=TARGET_LOOKBACK_DAYS)
    first = pd.Timestamp(history_start_day) if history_start_day else default_first
    if first.tzinfo is not None or first != first.normalize() or first > default_first:
        raise ValueError("history_start_day must cover the normal 372-day history")
    paths = source_paths(project_root)
    before = {name: sha256(path) for name, path in paths.items()}
    nuclear_audit = audit_nuclear_store(paths["nuclear_store"], str(first.date()), str(day.date()))
    residual_audit = audit_residual_bank(paths["residual_bank"], str(first.date()), str(day.date()))
    if not nuclear_audit["complete"]:
        raise ValueError("Nuclear Saturn bank failed read-only audit: "
                         + " | ".join(nuclear_audit["blockers"][:3]))
    if not residual_audit["complete"]:
        raise ValueError("Residual Saturn bank failed read-only audit: "
                         + " | ".join(residual_audit["blockers"][:3]))
    prices = {zone: _load_target(paths[f"target_{zone}"], first_day=first,
                                 delivery_day=day, zone=zone) for zone in ZONES}
    residual = pd.read_parquet(paths["residual_bank"])
    nuclear = pd.read_parquet(paths["nuclear_store"])
    if any(sha256(path) != before[name] for name, path in paths.items()):
        raise ValueError("Saturn source changed during read")
    return {"prices": prices, "residual_bank": residual, "nuclear_store": nuclear}, {
        "source_sha256": before, "source_paths": {name: str(path) for name, path in paths.items()},
        "nuclear_source_complete": True, "residual_source_complete": True,
        "target_history_first_day": str(first.date()),
        "target_history_last_day": str((day - pd.Timedelta(days=1)).date()),
        "future_actuals_loaded": False}
