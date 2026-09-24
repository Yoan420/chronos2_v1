#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_modular.common import deep_get, load_yaml, resolve_path
from chronos2_modular.data import (
    PIT_AVAILABILITY_ALIASES,
    PIT_DELIVERY_ALIASES,
    PIT_REVISION_ALIASES,
    PIT_VALUE_ALIASES,
    first_matching_column,
)
from chronos2_modular.saturn import fetch_saturn_series

LOGGER = logging.getLogger("build_extended_exogenous_inputs")


def _parse_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _origin_clock(config: Mapping[str, Any]) -> tuple[int, int]:
    raw = str(deep_get(config, "data.forecast_origin_local_time", "08:00"))
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if not match:
        raise ValueError("forecast_origin_local_time invalide.")
    return int(match.group(1)), int(match.group(2))


def _cutoff_utc(
    delivery_utc: pd.Series,
    timezone: str,
    config: Mapping[str, Any],
) -> pd.Series:
    local = delivery_utc.dt.tz_convert(timezone)
    hour, minute = _origin_clock(config)
    cutoff = (
        local.dt.normalize()
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=hour, minutes=minute)
    )
    return cutoff.dt.tz_convert("UTC")


def _read_raw_pit(path: Path, alias: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()

    delivery = first_matching_column(frame, None, PIT_DELIVERY_ALIASES)
    available = first_matching_column(frame, None, PIT_AVAILABILITY_ALIASES)
    revision = first_matching_column(frame, None, PIT_REVISION_ALIASES)
    value = first_matching_column(
        frame, None, (alias, *PIT_VALUE_ALIASES)
    )
    if delivery is None:
        raise KeyError(f"{path}: colonne livraison introuvable.")
    if available is None and revision is None:
        raise KeyError(f"{path}: colonne snapshot/révision introuvable.")

    excluded = {x for x in (delivery, available, revision) if x}
    if value is None:
        candidates = [
            c for c in frame.columns
            if c not in excluded
            and pd.to_numeric(frame[c], errors="coerce").notna().mean() > 0.8
        ]
        if len(candidates) == 1:
            value = candidates[0]
    if value is None:
        raise KeyError(f"{path}: colonne valeur introuvable.")

    snapshot_source = frame[available] if available else frame[revision]
    revision_source = frame[revision] if revision else snapshot_source
    out = pd.DataFrame(
        {
            "value_time_utc": _parse_utc(frame[delivery]),
            "snapshot_time_utc": _parse_utc(snapshot_source),
            "revision_time_utc": _parse_utc(revision_source),
            "value": pd.to_numeric(frame[value], errors="coerce"),
        }
    )
    return out.dropna(
        subset=[
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
        ]
    )


def build_uncertainty_metrics(
    frame: pd.DataFrame,
    *,
    timezone: str,
    config: Mapping[str, Any],
    max_revisions: int,
) -> pd.DataFrame:
    data = frame.copy()
    data["cutoff_utc"] = _cutoff_utc(
        data["value_time_utc"], timezone, config
    )
    data = data.loc[
        data["snapshot_time_utc"] <= data["cutoff_utc"]
    ].sort_values(
        ["value_time_utc", "snapshot_time_utc", "revision_time_utc"]
    )

    rows: list[dict[str, Any]] = []
    for delivery, group in data.groupby("value_time_utc", sort=True):
        tail = group.tail(max_revisions)
        values = tail["value"].dropna()
        if values.empty:
            continue
        latest = tail.iloc[-1]
        latest_value = float(values.iloc[-1])
        previous_value = (
            float(values.iloc[-2]) if len(values) >= 2 else latest_value
        )
        cutoff = latest["cutoff_utc"]
        age = (
            cutoff - latest["snapshot_time_utc"]
        ).total_seconds() / 3600.0
        rows.append(
            {
                "value_time_utc": delivery,
                "snapshot_time_utc": cutoff,
                "revision_time_utc": cutoff,
                "revision_std": (
                    float(values.std(ddof=0)) if len(values) >= 2 else 0.0
                ),
                "revision_abs_delta": abs(latest_value - previous_value),
                "revision_age_hours": max(float(age), 0.0),
                "revision_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _write_metric(
    frame: pd.DataFrame,
    metric: str,
    path: Path,
) -> None:
    out = frame[
        [
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            metric,
        ]
    ].rename(columns={metric: "value"})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def build_uncertainty_files(
    config: Mapping[str, Any],
    config_dir: Path,
    zone: str,
) -> None:
    settings = deep_get(
        config, "exogenous_extensions.forecast_uncertainty", {}
    ) or {}
    if not settings.get("enabled", False):
        return

    root = resolve_path(
        deep_get(config, "data.pit_vintage_dir", "data/pit/vintages"),
        resolve_path(deep_get(config, "data.project_root", "."), config_dir),
    )
    pit_files = deep_get(config, "data.pit_files", {}) or {}
    timezone = str(deep_get(config, f"zones.{zone}.timezone", "Europe/Paris"))
    max_revisions = int(settings.get("max_revisions", 6))
    default_metrics = list(
        settings.get(
            "metrics",
            ["revision_std", "revision_abs_delta", "revision_age_hours"],
        )
    )

    frames: dict[str, pd.DataFrame] = {}
    for source, raw in (settings.get("sources", {}) or {}).items():
        raw = raw or {}
        input_name = pit_files.get(source, raw.get("pit_file"))
        if not input_name:
            raise KeyError(f"Fichier PIT non configuré pour {source}.")
        metrics = build_uncertainty_metrics(
            _read_raw_pit(root / str(input_name), source),
            timezone=timezone,
            config=config,
            max_revisions=max_revisions,
        )
        frames[source] = metrics
        prefix = str(raw.get("output_prefix", source))
        for metric in raw.get("metrics", default_metrics):
            alias = f"{prefix}_{metric}"
            _write_metric(metrics, metric, root / f"{alias}.parquet")
            LOGGER.info("[UNCERTAINTY] %s -> %s", source, alias)

    for group_name, raw in (
        settings.get("aggregate_groups", {}) or {}
    ).items():
        members = [
            str(x) for x in (raw or {}).get("sources", [])
            if str(x) in frames
        ]
        if not members:
            continue
        merged = None
        for member in members:
            current = frames[member].rename(
                columns={
                    "revision_std": f"{member}__revision_std",
                    "revision_abs_delta": f"{member}__revision_abs_delta",
                    "revision_age_hours": f"{member}__revision_age_hours",
                }
            )
            keep = [
                "value_time_utc",
                "snapshot_time_utc",
                "revision_time_utc",
                f"{member}__revision_std",
                f"{member}__revision_abs_delta",
                f"{member}__revision_age_hours",
            ]
            merged = current[keep] if merged is None else merged.merge(
                current[keep],
                on=[
                    "value_time_utc",
                    "snapshot_time_utc",
                    "revision_time_utc",
                ],
                how="outer",
            )
        metrics_map = {
            "revision_std_mean": merged.filter(
                like="__revision_std"
            ).mean(axis=1),
            "revision_std_max": merged.filter(
                like="__revision_std"
            ).max(axis=1),
            "revision_abs_delta_mean": merged.filter(
                like="__revision_abs_delta"
            ).mean(axis=1),
            "revision_abs_delta_max": merged.filter(
                like="__revision_abs_delta"
            ).max(axis=1),
            "revision_age_hours_max": merged.filter(
                like="__revision_age_hours"
            ).max(axis=1),
        }
        base = merged[
            [
                "value_time_utc",
                "snapshot_time_utc",
                "revision_time_utc",
            ]
        ].copy()
        for metric in (raw or {}).get(
            "metrics",
            [
                "revision_std_mean",
                "revision_std_max",
                "revision_abs_delta_mean",
            ],
        ):
            out = base.copy()
            out[metric] = metrics_map[metric]
            alias = f"{group_name}_{metric}"
            _write_metric(out, metric, root / f"{alias}.parquet")
            LOGGER.info("[UNCERTAINTY] groupe %s -> %s", group_name, alias)


def update_neighbour_prices(
    config: Mapping[str, Any],
    config_dir: Path,
    zone: str,
    local_only: bool,
) -> None:
    settings = deep_get(
        config, "exogenous_extensions.neighbour_price_sources", {}
    ) or {}
    if not settings.get("enabled", False):
        return
    project_root = resolve_path(
        deep_get(config, "data.project_root", "."), config_dir
    )
    output = resolve_path(
        settings.get("output_dir", "data/derived/neighbour_prices"),
        project_root,
    )
    output.mkdir(parents=True, exist_ok=True)
    timezone = str(deep_get(config, f"zones.{zone}.timezone", "Europe/Paris"))
    now = pd.Timestamp.now(tz=timezone)
    start = (
        now
        - pd.DateOffset(
            years=int(deep_get(config, "data.historical_years", 4))
        )
        - pd.Timedelta(days=8)
    )
    end = now.normalize() + pd.DateOffset(days=1) - pd.Timedelta(hours=1)

    for country, raw in (
        settings.get("countries", {}) or {}
    ).items():
        alias = str(raw.get("alias", f"{country.lower()}_price_da"))
        path = output / f"{alias}.csv.gz"
        if local_only:
            if not path.exists():
                raise FileNotFoundError(
                    f"Prix voisin absent en mode local : {path}"
                )
            continue
        series = fetch_saturn_series(
            str(raw["series"]),
            start,
            end,
            timezone,
            str(deep_get(config, "data.saturn_url", "")),
            str(deep_get(config, "data.saturn_author", "")),
        ).rename("value")
        series.to_frame().reset_index(names="timestamp").to_csv(
            path, index=False, compression="gzip"
        )
        LOGGER.info(
            "[NEIGHBOUR] %s | %s | %d valeurs", country, path, len(series)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--zone", default="FR")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    path = Path(args.config).expanduser().resolve()
    config = load_yaml(path)
    update_neighbour_prices(
        config, path.parent, args.zone.upper(), args.local_only
    )
    build_uncertainty_files(
        config, path.parent, args.zone.upper()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
