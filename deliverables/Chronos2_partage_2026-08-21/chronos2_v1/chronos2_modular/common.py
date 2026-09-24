from __future__ import annotations

import logging
import math
import os
import random
import re
import ssl
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

LOGGER = logging.getLogger("chronos2_modular")
SCRIPT_VERSION = "2.3.5-hourly-fixed-lag96"
DEFAULT_QUANTILES = tuple(round(value / 10, 1) for value in range(1, 10))
SUPPORTED_FUTURE_LAG_HOURS = (24, 48, 96, 168)
SUPPORTED_FUTURE_LAG_STRATEGIES = frozenset(
    f"lag{hours}" for hours in SUPPORTED_FUTURE_LAG_HOURS
)
KNOWN_FUTURE_COLUMN_PATTERN = re.compile(
    r"^known_(.+)_("
    + "|".join(
        [
            *(f"lag{hours}" for hours in SUPPORTED_FUTURE_LAG_HOURS),
            "persistence",
            "oracle",
        ]
    )
    + r")$"
)
CALENDAR_COLUMNS = (
    "known_hour_sin",
    "known_hour_cos",
    "known_dow_sin",
    "known_dow_cos",
    "known_doy_sin",
    "known_doy_cos",
    "known_is_weekend",
)


@dataclass(frozen=True)
class SeriesSpec:
    alias: str
    series: str | None = None
    enabled: bool = True
    description: str = ""
    source: str = "auto"
    file: str | None = None
    pit_file: str | None = None
    timestamp_col: str | None = None
    value_col: str | None = None
    availability_col: str | None = None
    revision_col: str | None = None
    fill_method: str = "ffill"
    fill_limit: int = 3
    minimum_coverage: float = 0.05
    include_base_context: bool = True
    known_future: bool = False
    future_strategies: tuple[str, ...] = field(default_factory=tuple)
    # Appended to preserve the positional signature of legacy SeriesSpec
    # callers while allowing an explicit contract for naive Saturn indices.
    naive_timezone: str | None = None
    # Opt-in repair for local-naive covariates whose second autumn DST fold
    # was dropped by the source.  Keep this field last for positional
    # compatibility with legacy SeriesSpec callers.
    incomplete_dst_policy: str = "raise"


@dataclass
class ZoneConfig:
    zone: str
    timezone: str
    target: SeriesSpec
    covariates: dict[str, SeriesSpec]
    include_calendar: bool = True


@dataclass
class ZoneData:
    zone: str
    timezone: str
    frequency: str
    target: pd.Series
    covariates: pd.DataFrame
    model_context_covariates: pd.DataFrame
    known_future_columns: list[str]
    coverage: pd.DataFrame
    input_manifest: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class ModelRuntime:
    pipeline: Any
    model_id: str
    device: str
    dtype: torch.dtype


@dataclass
class ZoneRunResult:
    zone: str
    metrics_native: dict[str, Any]
    metrics_baseline: dict[str, Any] | None
    backtest_native: pd.DataFrame
    backtest_baseline: pd.DataFrame | None
    metrics_by_horizon: pd.DataFrame
    metrics_by_hour: pd.DataFrame
    forecast_native: pd.DataFrame
    forecast_baseline: pd.DataFrame | None
    zone_data: ZoneData
    output_dir: Path


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Le YAML doit contenir un dictionnaire à la racine.")
    return payload


def deep_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def parse_future_lag_hours(strategy: str) -> int | None:
    """Return the validated physical-hour lag encoded by a strategy."""
    normalized = str(strategy).strip().lower()
    if normalized not in SUPPORTED_FUTURE_LAG_STRATEGIES:
        return None
    return int(normalized.removeprefix("lag"))


def parse_series_spec(
    alias: str,
    raw: Mapping[str, Any],
    default_fill: int,
) -> SeriesSpec:
    future_raw = raw.get("future", raw.get("future_strategies", []))

    if isinstance(future_raw, Mapping):
        strategies = future_raw.get("strategies", [])
        known_future = bool(future_raw.get("known_future", False))
    else:
        strategies = future_raw
        known_future = False

    if isinstance(strategies, str):
        strategies = [strategies]

    strategies_tuple = tuple(
        str(item).lower() for item in (strategies or [])
    )
    allowed = {
        *SUPPORTED_FUTURE_LAG_STRATEGIES,
        "persistence",
        "oracle",
    }
    unknown = sorted(set(strategies_tuple) - allowed)
    if unknown:
        raise ValueError(
            f"{alias}: stratégies futures inconnues : {unknown}"
        )

    if "oracle" in strategies_tuple and not known_future:
        raise ValueError(
            f"{alias}: la stratégie 'oracle' exige "
            "future.known_future: true."
        )

    incomplete_dst_policy = str(
        raw.get("incomplete_dst_policy", "raise")
    ).strip().lower()
    allowed_dst_policies = {"raise", "duplicate"}
    if incomplete_dst_policy not in allowed_dst_policies:
        raise ValueError(
            f"{alias}: incomplete_dst_policy inconnue "
            f"'{incomplete_dst_policy}'. Valeurs autorisées : "
            "raise, duplicate."
        )

    return SeriesSpec(
        alias=alias,
        series=raw.get("series"),
        enabled=bool(raw.get("enabled", True)),
        description=str(raw.get("description", "")),
        source=str(raw.get("source", "auto")).lower(),
        file=raw.get("file"),
        pit_file=raw.get("pit_file"),
        timestamp_col=raw.get("timestamp_col"),
        naive_timezone=(
            str(raw.get("naive_timezone")).strip()
            if raw.get("naive_timezone") not in (None, "")
            else None
        ),
        value_col=raw.get("value_col"),
        availability_col=raw.get("availability_col"),
        revision_col=raw.get("revision_col"),
        fill_method=str(raw.get("fill_method", "ffill")).lower(),
        fill_limit=int(raw.get("fill_limit", default_fill)),
        minimum_coverage=float(raw.get("minimum_coverage", 0.05)),
        include_base_context=bool(
            raw.get("include_base_context", True)
        ),
        known_future=known_future,
        future_strategies=strategies_tuple,
        incomplete_dst_policy=incomplete_dst_policy,
    )


def build_zone_configs(
    config: Mapping[str, Any],
    requested_zones: Sequence[str] | None,
    include_covariates: Sequence[str] | None,
    exclude_covariates: Sequence[str] | None,
) -> list[ZoneConfig]:
    zones_raw = config.get("zones", {})
    if not isinstance(zones_raw, Mapping):
        raise ValueError("zones doit être un dictionnaire YAML.")

    selected = (
        [zone.upper() for zone in requested_zones]
        if requested_zones
        else [
            str(zone).upper()
            for zone, raw in zones_raw.items()
            if bool(raw.get("enabled", True))
        ]
    )

    include = {value.lower() for value in include_covariates or []}
    exclude = {value.lower() for value in exclude_covariates or []}
    default_fill = int(
        deep_get(config, "data.default_fill_limit", 3)
    )

    result: list[ZoneConfig] = []

    for zone in selected:
        if zone not in zones_raw:
            raise KeyError(f"Zone absente du YAML : {zone}")

        raw = zones_raw[zone]
        target_raw = raw.get("target")
        if not isinstance(target_raw, Mapping):
            raise ValueError(f"{zone}: target doit être défini.")

        target = parse_series_spec("target", target_raw, default_fill)
        if not target.series and not target.file:
            raise ValueError(f"{zone}: target exige series ou file.")
        if target.incomplete_dst_policy != "raise":
            raise ValueError(
                f"{zone}/target: incomplete_dst_policy=duplicate est "
                "interdit pour la cible. La timeline cible doit fournir "
                "les deux folds DST physiques sans copie ni imputation."
            )

        covariates: dict[str, SeriesSpec] = {}

        for alias, item in (raw.get("covariates") or {}).items():
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"{zone}/{alias}: la définition doit être un mapping YAML."
                )

            alias_lower = str(alias).lower()
            spec = parse_series_spec(alias_lower, item, default_fill)

            enabled = spec.enabled
            if include and alias_lower not in include:
                enabled = False
            if alias_lower in exclude:
                enabled = False

            spec = SeriesSpec(**{**asdict(spec), "enabled": enabled})

            unsafe_observed_lags = [
                strategy
                for strategy in spec.future_strategies
                if (
                    (lag_hours := parse_future_lag_hours(strategy))
                    is not None
                    and lag_hours < 48
                )
            ]
            if (
                enabled
                and alias_lower == "fr_net_exports"
                and spec.series
                and spec.series.endswith(".obs")
                and unsafe_observed_lags
            ):
                raise ValueError(
                    "fr_net_exports: "
                    f"{', '.join(unsafe_observed_lags)} d'une série "
                    "observée est "
                    "non causal au cutoff D-1. Utilisez une série "
                    "scheduled/forecast PIT ou un lag >= 48 h avec "
                    "include_base_context: false."
                )

            if enabled:
                covariates[alias_lower] = spec

        result.append(
            ZoneConfig(
                zone=zone,
                timezone=str(raw.get("timezone", "UTC")),
                target=target,
                covariates=covariates,
                include_calendar=bool(raw.get("include_calendar", True)),
            )
        )

    return result


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_huggingface_ssl() -> None:
    ca_value = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
    )
    if not ca_value:
        return

    ca_path = Path(ca_value).expanduser()
    if not ca_path.is_file():
        LOGGER.warning("Certificat configuré mais introuvable : %s", ca_path)
        return

    try:
        import httpx
        from huggingface_hub import close_session, set_client_factory
    except ImportError:
        return

    context = ssl.create_default_context(cafile=str(ca_path))

    def factory() -> httpx.Client:
        return httpx.Client(
            verify=context,
            follow_redirects=True,
            timeout=httpx.Timeout(180.0, connect=30.0),
        )

    close_session()
    set_client_factory(factory)
    LOGGER.info("SSL Hugging Face configuré avec %s", ca_path)


def resolve_device(requested: str) -> tuple[str, torch.dtype]:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible.")
        device = "cuda"
    elif requested == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    return device, dtype


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan
