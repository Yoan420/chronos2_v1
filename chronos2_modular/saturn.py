from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .common import (
    LOGGER,
    SeriesSpec,
    ZoneConfig,
    deep_get,
    resolve_path,
)


VINTAGE_COLUMNS = (
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "value",
    "downloaded_at_utc",
)


@dataclass(frozen=True)
class SaturnSyncResult:
    zone: str
    alias: str
    series: str
    kind: str
    path: str
    status: str
    rows_before: int
    rows_downloaded: int
    rows_after: int
    first_revision_utc: str | None
    last_revision_utc: str | None
    sync_as_of_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_") or "series"


def cache_path_for_series(
    cache_root: Path,
    zone: str,
    spec: SeriesSpec,
) -> Path:
    identity = (
        spec.series
        or spec.file
        or spec.pit_file
        or spec.alias
    )
    digest = hashlib.sha1(
        str(identity).encode("utf-8")
    ).hexdigest()[:10]

    return (
        cache_root
        / zone.lower()
        / f"{sanitize_filename(spec.alias)}__{digest}.csv.gz"
    )


def resolve_pit_path(
    spec: SeriesSpec,
    config: Mapping[str, Any],
    config_dir: Path,
) -> Path:
    pit_root = resolve_path(
        deep_get(config, "data.pit_vintage_dir", "."),
        config_dir,
    )

    pit_files = deep_get(config, "data.pit_files", {})
    configured_name = None

    if isinstance(pit_files, Mapping):
        configured_name = pit_files.get(spec.alias)

    candidate = spec.pit_file or configured_name

    if not candidate:
        raise KeyError(
            f"{spec.alias}: aucun fichier PIT défini. "
            "Ajoute data.pit_files ou pit_file."
        )

    candidate_path = Path(str(candidate)).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = pit_root / candidate_path

    return candidate_path.resolve()


def is_pit_spec(
    spec: SeriesSpec,
    config: Mapping[str, Any],
) -> bool:
    if spec.source == "pit_parquet":
        return True

    pit_files = deep_get(config, "data.pit_files", {})
    return (
        isinstance(pit_files, Mapping)
        and spec.alias in pit_files
    )


def _as_utc(
    value: Any,
    *,
    naive_timezone: str = "UTC",
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(naive_timezone)
    return timestamp.tz_convert("UTC")


def _as_zone(
    value: Any,
    timezone: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def _has_values(raw: Any) -> bool:
    if raw is None:
        return False
    try:
        return len(raw) > 0
    except TypeError:
        return False


def _normalized_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def coerce_saturn_to_series(
    raw: Any,
    name: str,
) -> pd.Series:
    if isinstance(raw, pd.Series):
        return raw.rename(name)

    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] == 1:
            return raw.iloc[:, 0].rename(name)

        normalized = {
            _normalized_column_name(column): column
            for column in raw.columns
        }
        for candidate in (
            "value",
            "price",
            "observation",
            name,
        ):
            column = normalized.get(
                _normalized_column_name(candidate)
            )
            if column is not None:
                return raw[column].rename(name)

    raise ValueError(f"Objet Saturn ambigu pour {name}.")


def normalize_saturn_series(
    raw: Any,
    name: str,
    timezone: str,
) -> pd.Series:
    series = coerce_saturn_to_series(raw, name)

    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(
                series.index,
                errors="coerce",
            )
        )
    except (TypeError, ValueError):
        index = pd.DatetimeIndex(
            pd.to_datetime(
                series.index,
                errors="coerce",
                utc=True,
            )
        ).tz_convert(timezone)

    valid = ~index.isna()
    series = series.loc[valid]
    index = index[valid]

    if index.tz is None:
        try:
            index = index.tz_localize(
                timezone,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        except Exception:
            index = index.tz_localize(
                timezone,
                ambiguous=True,
                nonexistent="shift_forward",
            )
    else:
        index = index.tz_convert(timezone)

    result = pd.Series(
        pd.to_numeric(
            series.to_numpy(),
            errors="coerce",
        ),
        index=index,
        name=name,
        dtype=float,
    ).sort_index()

    if result.index.duplicated().any():
        result = result.groupby(level=0).last()

    return result


def create_saturn_client(
    saturn_url: str,
    author: str,
) -> Any:
    if not saturn_url or saturn_url.lower() == "none":
        raise ValueError("data.saturn_url est absent ou invalide.")
    if not author:
        raise ValueError(
            "data.saturn_author est absent. Définis-le dans le YAML "
            "ou via la variable SATURN_AUTHOR."
        )

    try:
        import tshistory_lite
    except ImportError as exc:
        raise ImportError(
            "tshistory_lite est requis pour Saturn."
        ) from exc

    return tshistory_lite.Client(
        uri=saturn_url,
        author=author,
    )


def fetch_saturn_series_from_client(
    client: Any,
    series_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone: str,
    *,
    revision_date: pd.Timestamp | None = None,
) -> pd.Series:
    date_kwargs = (
        {
            "from_value_date": start,
            "to_value_date": end,
        },
        {
            "from_value": start,
            "to_value": end,
        },
        {
            "start": start,
            "end": end,
        },
    )

    errors: list[str] = []
    raw = None

    for kwargs in date_kwargs:
        if revision_date is not None:
            kwargs = {
                **kwargs,
                "revision_date": _as_utc(revision_date),
            }
        try:
            raw = client.get(series_name, **kwargs)
            if _has_values(raw):
                break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if not _has_values(raw) and revision_date is None:
        try:
            raw = client.get(series_name, start, end)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if not _has_values(raw):
        suffix = " | ".join(errors[-4:])
        raise RuntimeError(
            f"Saturn indisponible ou vide pour {series_name}. "
            f"{suffix}"
        )

    return normalize_saturn_series(
        raw,
        series_name,
        timezone,
    )


def fetch_saturn_series(
    series_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone: str,
    saturn_url: str,
    author: str,
    *,
    revision_date: pd.Timestamp | None = None,
) -> pd.Series:
    client = create_saturn_client(
        saturn_url,
        os.getenv("SATURN_AUTHOR") or author,
    )
    return fetch_saturn_series_from_client(
        client,
        series_name,
        start,
        end,
        timezone,
        revision_date=revision_date,
    )


def fetch_saturn_history(
    client: Any,
    series_name: str,
    *,
    from_insertion_date: pd.Timestamp,
    to_insertion_date: pd.Timestamp,
    from_value_date: pd.Timestamp,
    to_value_date: pd.Timestamp,
    diffmode: bool = True,
) -> Mapping[Any, Any]:
    history_method = getattr(client, "history", None)
    if history_method is None:
        raise RuntimeError(
            "Cette version de tshistory_lite ne fournit pas history(). "
            "Impossible de construire un historique point-in-time "
            "efficace."
        )

    kwargs = {
        "from_insertion_date": _as_utc(from_insertion_date),
        "to_insertion_date": _as_utc(to_insertion_date),
        "from_value_date": from_value_date,
        "to_value_date": to_value_date,
    }
    try:
        history = history_method(
            series_name,
            **kwargs,
            diffmode=diffmode,
        )
    except TypeError as exc:
        if not diffmode:
            raise
        LOGGER.warning(
            "history(diffmode=True) non supporté pour %s (%s). "
            "Téléchargement des snapshots complets.",
            series_name,
            exc,
        )
        history = history_method(series_name, **kwargs)

    if history is None:
        return {}
    if not isinstance(history, Mapping):
        raise TypeError(
            f"history({series_name}) doit retourner un mapping "
            "revision -> série."
        )
    return history


def history_to_vintage_frame(
    history: Mapping[Any, Any],
    series_name: str,
    timezone: str,
    *,
    retrieved_at_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    retrieved_at = _as_utc(
        retrieved_at_utc or pd.Timestamp.now(tz="UTC")
    )
    frames: list[pd.DataFrame] = []

    for revision, raw in history.items():
        if raw is None:
            continue
        revision_utc = _as_utc(revision)
        series = normalize_saturn_series(
            raw,
            series_name,
            timezone,
        )
        if series.empty:
            continue

        frame = pd.DataFrame(
            {
                "delivery_utc": series.index.tz_convert("UTC"),
                "availability_utc": revision_utc,
                "revision_utc": revision_utc,
                "value": series.to_numpy(dtype=float),
                "retrieved_at_utc": retrieved_at,
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=VINTAGE_COLUMNS)

    return normalize_vintage_frame(
        pd.concat(frames, ignore_index=True)
    )


def normalize_vintage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "delivery_utc",
        "availability_utc",
        "revision_utc",
        "value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"Colonnes PIT absentes : {missing}. "
            f"Colonnes reçues : {list(frame.columns)}"
        )

    normalized = frame.copy()
    if "retrieved_at_utc" not in normalized:
        normalized["retrieved_at_utc"] = pd.NaT

    for column in (
        "delivery_utc",
        "availability_utc",
        "revision_utc",
        "retrieved_at_utc",
    ):
        normalized[column] = pd.to_datetime(
            normalized[column],
            errors="coerce",
            utc=True,
        )

    normalized["value"] = pd.to_numeric(
        normalized["value"],
        errors="coerce",
    )
    normalized = normalized.dropna(
        subset=[
            "delivery_utc",
            "availability_utc",
            "revision_utc",
        ]
    )
    normalized = normalized.sort_values(
        [
            "revision_utc",
            "delivery_utc",
            "availability_utc",
            "retrieved_at_utc",
        ],
        na_position="first",
    )
    normalized = normalized.drop_duplicates(
        subset=["revision_utc", "delivery_utc"],
        keep="last",
    )
    return normalized.loc[:, VINTAGE_COLUMNS].reset_index(
        drop=True
    )


def read_vintage_store(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=VINTAGE_COLUMNS)
    return normalize_vintage_frame(pd.read_parquet(path))


def _atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp.parquet"
    )
    try:
        frame.to_parquet(temporary, index=False)
        pd.read_parquet(temporary, columns=["delivery_utc"]).head(1)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_latest_store(
    path: Path,
    alias: str,
    timezone: str,
) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float, name=alias)

    frame = pd.read_csv(path, low_memory=False)
    if "timestamp" not in frame or "value" not in frame:
        raise KeyError(
            f"Cache Saturn invalide : {path}. "
            "Colonnes attendues : timestamp, value."
        )

    index = pd.DatetimeIndex(
        pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
            utc=True,
        )
    ).tz_convert(timezone)
    series = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
        index=index,
        name=alias,
        dtype=float,
    )
    series = series.loc[~series.index.isna()].sort_index()
    return series.loc[~series.index.duplicated(keep="last")]


def _atomic_write_latest(
    series: pd.Series,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp.gz"
    )
    try:
        (
            series.rename("value")
            .to_frame()
            .reset_index(names="timestamp")
            .to_csv(
                temporary,
                index=False,
                compression="gzip",
            )
        )
        pd.read_csv(temporary, nrows=1)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sync_latest_series(
    client: Any,
    *,
    zone: str,
    alias: str,
    series_name: str,
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone: str,
    sync_as_of_utc: pd.Timestamp,
    overlap_days: int = 7,
    full: bool = False,
) -> SaturnSyncResult:
    existing = (
        pd.Series(dtype=float, name=alias)
        if full
        else _read_latest_store(path, alias, timezone)
    )
    rows_before = int(len(existing))
    fetch_start = _as_zone(start, timezone)
    fetch_end = _as_zone(end, timezone)

    if not existing.empty:
        overlap_start = (
            existing.index.max()
            - pd.Timedelta(days=max(0, overlap_days))
        )
        fetch_start = max(fetch_start, overlap_start)

    downloaded = fetch_saturn_series_from_client(
        client,
        series_name,
        fetch_start,
        fetch_end,
        timezone,
        revision_date=sync_as_of_utc,
    ).rename(alias)

    merged = (
        downloaded.copy()
        if existing.empty
        else pd.concat([existing, downloaded])
    ).sort_index()
    merged = merged.loc[
        ~merged.index.duplicated(keep="last")
    ]
    _atomic_write_latest(merged, path)

    return SaturnSyncResult(
        zone=zone,
        alias=alias,
        series=series_name,
        kind="latest",
        path=str(path),
        status="updated",
        rows_before=rows_before,
        rows_downloaded=int(len(downloaded)),
        rows_after=int(len(merged)),
        first_revision_utc=None,
        last_revision_utc=None,
        sync_as_of_utc=str(_as_utc(sync_as_of_utc)),
    )


def sync_vintage_series(
    client: Any,
    *,
    zone: str,
    alias: str,
    series_name: str,
    path: Path,
    revision_start: pd.Timestamp,
    revision_end: pd.Timestamp,
    value_start: pd.Timestamp,
    value_end: pd.Timestamp,
    timezone: str,
    overlap_days: int = 2,
    chunk_days: int = 90,
    retries: int = 3,
    full: bool = False,
) -> SaturnSyncResult:
    existing = (
        pd.DataFrame(columns=VINTAGE_COLUMNS)
        if full
        else read_vintage_store(path)
    )
    rows_before = int(len(existing))

    fetch_start = _as_utc(revision_start)
    fetch_end = _as_utc(revision_end)

    if not existing.empty:
        last_revision = existing["revision_utc"].max()
        incremental_start = last_revision - pd.Timedelta(
            days=max(0, overlap_days)
        )
        fetch_start = max(fetch_start, incremental_start)

    if fetch_start > fetch_end:
        return SaturnSyncResult(
            zone=zone,
            alias=alias,
            series=series_name,
            kind="forecast_vintages",
            path=str(path),
            status="up_to_date",
            rows_before=rows_before,
            rows_downloaded=0,
            rows_after=rows_before,
            first_revision_utc=(
                str(existing["revision_utc"].min())
                if not existing.empty
                else None
            ),
            last_revision_utc=(
                str(existing["revision_utc"].max())
                if not existing.empty
                else None
            ),
            sync_as_of_utc=str(fetch_end),
        )

    retrieved_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []
    current = fetch_start
    chunk_delta = pd.Timedelta(days=max(1, chunk_days))

    while current <= fetch_end:
        chunk_end = min(current + chunk_delta, fetch_end)
        history = None

        for attempt in range(max(1, retries)):
            try:
                history = fetch_saturn_history(
                    client,
                    series_name,
                    from_insertion_date=current,
                    to_insertion_date=chunk_end,
                    from_value_date=value_start,
                    to_value_date=value_end,
                    diffmode=True,
                )
                break
            except Exception:
                if attempt + 1 >= max(1, retries):
                    raise
                delay = min(2**attempt, 8)
                LOGGER.warning(
                    "[Saturn] %s | nouvelle tentative dans %ss.",
                    alias,
                    delay,
                )
                time.sleep(delay)

        frame = history_to_vintage_frame(
            history or {},
            series_name,
            timezone,
            retrieved_at_utc=retrieved_at,
        )
        if not frame.empty:
            frames.append(frame)

        LOGGER.info(
            "[Saturn/PIT] %s | révisions %s -> %s | %d lignes",
            alias,
            current,
            chunk_end,
            len(frame),
        )

        if chunk_end >= fetch_end:
            break
        current = chunk_end

    downloaded = (
        normalize_vintage_frame(pd.concat(frames, ignore_index=True))
        if frames
        else pd.DataFrame(columns=VINTAGE_COLUMNS)
    )

    if downloaded.empty and existing.empty:
        raise RuntimeError(
            f"Aucune révision Saturn reçue pour {series_name} entre "
            f"{fetch_start} et {fetch_end}."
        )

    if downloaded.empty:
        merged = existing
        status = "up_to_date"
    else:
        merged = (
            downloaded.copy()
            if existing.empty
            else normalize_vintage_frame(
                pd.concat([existing, downloaded], ignore_index=True)
            )
        )
        _atomic_write_parquet(merged, path)
        status = "updated"

    return SaturnSyncResult(
        zone=zone,
        alias=alias,
        series=series_name,
        kind="forecast_vintages",
        path=str(path),
        status=status,
        rows_before=rows_before,
        rows_downloaded=int(len(downloaded)),
        rows_after=int(len(merged)),
        first_revision_utc=(
            str(merged["revision_utc"].min())
            if not merged.empty
            else None
        ),
        last_revision_utc=(
            str(merged["revision_utc"].max())
            if not merged.empty
            else None
        ),
        sync_as_of_utc=str(fetch_end),
    )


def _configured_timestamp(
    value: Any,
    timezone: str,
) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    return _as_zone(value, timezone)


def _target_sync_end(
    config: Mapping[str, Any],
    timezone: str,
    sync_as_of_utc: pd.Timestamp,
) -> pd.Timestamp:
    configured = _configured_timestamp(
        deep_get(config, "data.end"),
        timezone,
    )
    if configured is not None:
        return configured

    policy = str(
        deep_get(
            config,
            "data.target_end_policy",
            "current_day_end",
        )
    ).lower()
    frequency = str(deep_get(config, "data.frequency", "h"))
    as_of_local = sync_as_of_utc.tz_convert(timezone)

    if policy == "current_day_end":
        offset = pd.tseries.frequencies.to_offset(frequency)
        return as_of_local.normalize() + pd.DateOffset(days=1) - offset
    if policy == "now":
        return as_of_local.floor(frequency)
    return as_of_local.ceil(frequency)


def sync_saturn_data(
    zone_configs: Sequence[ZoneConfig],
    config: Mapping[str, Any],
    config_dir: Path,
    *,
    full: bool = False,
    as_of: Any | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Actualise les caches Saturn latest et les historiques PIT.

    ``as_of`` est un plafond global d'ingestion. La sélection de la révision
    utilisable pour chaque livraison reste effectuée dans ``data.py`` avec
    ``forecast_origin_local_time``.
    """
    if not zone_configs:
        return pd.DataFrame()

    sync_as_of_utc = _as_utc(
        as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
    )
    project_root = resolve_path(
        deep_get(config, "data.project_root", "."),
        config_dir,
    )
    cache_root = resolve_path(
        deep_get(config, "data.cache_dir", "data/cache"),
        project_root,
    )
    sync_config = deep_get(config, "data.saturn_sync", {})
    if not isinstance(sync_config, Mapping):
        raise ValueError("data.saturn_sync doit être un mapping YAML.")

    saturn_url = str(deep_get(config, "data.saturn_url", ""))
    author = os.getenv("SATURN_AUTHOR") or str(
        deep_get(config, "data.saturn_author", "")
    )
    connector = client or create_saturn_client(saturn_url, author)

    historical_years = int(
        deep_get(config, "data.historical_years", 4)
    )
    revision_overlap_days = int(
        sync_config.get("revision_overlap_days", 2)
    )
    latest_overlap_days = int(
        sync_config.get("latest_overlap_days", 7)
    )
    chunk_days = int(sync_config.get("history_chunk_days", 90))
    retries = int(sync_config.get("retries", 3))
    lookahead_days = int(
        sync_config.get("value_lookahead_days", 7)
    )
    global_source = str(
        deep_get(config, "data.source", "auto")
    ).lower()

    results: list[SaturnSyncResult] = []
    destinations: dict[Path, str] = {}

    for zone in zone_configs:
        as_of_local = sync_as_of_utc.tz_convert(zone.timezone)
        configured_start = _configured_timestamp(
            deep_get(config, "data.start"),
            zone.timezone,
        )
        value_start = configured_start or (
            as_of_local - pd.DateOffset(years=historical_years)
        )
        configured_end = _configured_timestamp(
            deep_get(config, "data.end"),
            zone.timezone,
        )
        value_end = configured_end or (
            as_of_local.ceil(
                str(deep_get(config, "data.frequency", "h"))
            )
            + pd.DateOffset(days=lookahead_days)
        )
        revision_start_raw = sync_config.get(
            "initial_revision_start",
            value_start,
        )
        revision_start = _as_utc(
            revision_start_raw,
            naive_timezone=zone.timezone,
        )

        specs = [zone.target, *zone.covariates.values()]
        for spec in specs:
            if not spec.enabled or spec.file or not spec.series:
                continue

            source = (
                spec.source
                if spec.source != "auto"
                else global_source
            )
            pit = is_pit_spec(spec, config)

            if not pit and source not in {"auto", "saturn"}:
                continue

            if pit:
                path = resolve_pit_path(spec, config, config_dir)
            else:
                path = cache_path_for_series(
                    cache_root,
                    zone.zone,
                    spec,
                )

            previous_series = destinations.get(path)
            if previous_series is not None:
                if previous_series != spec.series:
                    raise ValueError(
                        f"Deux séries Saturn ciblent le même fichier {path}: "
                        f"{previous_series} et {spec.series}."
                    )
                continue
            destinations[path] = spec.series

            LOGGER.info(
                "[Saturn] %s/%s | %s | as-of=%s",
                zone.zone,
                spec.alias,
                spec.series,
                sync_as_of_utc,
            )

            if pit:
                result = sync_vintage_series(
                    connector,
                    zone=zone.zone,
                    alias=spec.alias,
                    series_name=spec.series,
                    path=path,
                    revision_start=revision_start,
                    revision_end=sync_as_of_utc,
                    value_start=value_start,
                    value_end=value_end,
                    timezone=zone.timezone,
                    overlap_days=revision_overlap_days,
                    chunk_days=chunk_days,
                    retries=retries,
                    full=full,
                )
            else:
                end = (
                    _target_sync_end(
                        config,
                        zone.timezone,
                        sync_as_of_utc,
                    )
                    if spec.alias == "target"
                    else value_end
                )
                result = sync_latest_series(
                    connector,
                    zone=zone.zone,
                    alias=spec.alias,
                    series_name=spec.series,
                    path=path,
                    start=value_start,
                    end=end,
                    timezone=zone.timezone,
                    sync_as_of_utc=sync_as_of_utc,
                    overlap_days=latest_overlap_days,
                    full=full,
                )

            results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(
        [result.to_dict() for result in results]
    ).sort_values(["zone", "kind", "alias"]).reset_index(
        drop=True
    )
