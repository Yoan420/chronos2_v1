from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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


# The canonical UTC/CDH targets can lag complete auction days although Saturn's
# official multi-source hourly formulas are already published.  These four
# completion contracts were audited point-in-time at 2026-09-01 06:00Z: each
# supplied a finite, gap-free 48-hour suffix and matched the canonical target
# over its preceding 168 available hours to floating-point noise.  Admission
# comes from this explicit central source contract; because each official
# formula can itself fall back to the canonical series, the 168-hour runtime
# match is a compatibility/continuity guard, not an independent lineage proof.
# The formulas are local-civil, not intrinsically safe on a 25-hour autumn DST
# day; strict target normalisation below therefore rejects a missing fold
# instead of inventing it.
# Spain is deliberately absent: its official formula matched only 30/168 hours
# (MAE 4.60 EUR/MWh, maximum difference 28.52 EUR/MWh).
AUDITED_EQUIVALENT_TARGET_FALLBACKS: Mapping[str, Mapping[str, Any]] = {
    "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh": {
        "series": "power.price.da.fr.bzn.hourly.entsoe.eurmwh",
        "naive_timezone": "Europe/Paris",
        "request_padding_hours": 3,
        "nocache": True,
        "validation_hours": 168,
        "atol_eur_mwh": 1e-9,
    },
    "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh": {
        "series": "power.price.da.de_lu.bzn.hourly.entsoe.eurmwh",
        "naive_timezone": "Europe/Berlin",
        "request_padding_hours": 3,
        "nocache": True,
        "validation_hours": 168,
        "atol_eur_mwh": 1e-9,
    },
    "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh": {
        "series": "power.price.da.be.bzn.hourly.entsoe.eurmwh",
        "naive_timezone": "Europe/Brussels",
        "request_padding_hours": 3,
        "nocache": True,
        "validation_hours": 168,
        "atol_eur_mwh": 1e-9,
    },
    "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh": {
        "series": "power.price.da.nl.bzn.hourly.entsoe.eurmwh",
        "naive_timezone": "Europe/Amsterdam",
        "request_padding_hours": 3,
        "nocache": True,
        "validation_hours": 168,
        "atol_eur_mwh": 1e-9,
    },
}


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
    equivalent_fallback_series: str | None = None
    equivalent_fallback_rows: int = 0
    equivalent_fallback_validation_paired_hours: int = 0
    equivalent_fallback_validation_max_abs_difference_eur_mwh: float | None = None
    equivalent_fallback_validation_tolerance_eur_mwh: float | None = None
    equivalent_fallback_internal_rows: int = 0
    equivalent_fallback_value_times_utc: tuple[str, ...] = ()

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
    if spec.naive_timezone:
        # Changing the interpretation of naive timestamps must not silently
        # reuse a cache produced with another timezone contract.
        identity = (
            f"{identity}|naive_timezone={spec.naive_timezone}"
        )
    # Preserve every legacy strict-cache path.  Only an opt-in repaired cache
    # needs a distinct identity so it can never be reused under the default
    # strict contract (or conversely).
    if spec.incomplete_dst_policy != "raise":
        identity = (
            f"{identity}|incomplete_dst_policy="
            f"{spec.incomplete_dst_policy}"
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
    *,
    naive_timezone: str | None = None,
    incomplete_dst_policy: str = "raise",
) -> pd.Series:
    """Normalise a Saturn series while preserving physical DST products.

    Saturn series without timezone metadata are classified from their DST
    shape.  A grid containing a nonexistent local spring label is interpreted
    as UTC-naive; otherwise it is interpreted as local civil time.  The caller
    may pass an explicit ``naive_timezone`` when the inspected interval does
    not cross a DST transition and is therefore intrinsically ambiguous.

    For a local-naive source, each ambiguous autumn timestamp must normally
    occur exactly twice.  The first occurrence is assigned to the DST fold and
    the second to standard time.  ``incomplete_dst_policy='duplicate'`` is an
    explicit covariate-only escape hatch.  The narrower
    ``'duplicate_zero_only'`` policy copies a singleton only when its value is
    finite and zero to numerical tolerance; any non-zero or malformed group is
    rejected.  Every repair is exposed in ``Series.attrs['dst_repairs']``.
    """
    dst_policy = str(incomplete_dst_policy).strip().lower()
    if dst_policy not in {"raise", "duplicate", "duplicate_zero_only"}:
        raise ValueError(
            f"{name}: incomplete_dst_policy inconnue '{dst_policy}'. "
            "Valeurs autorisées : raise, duplicate, duplicate_zero_only."
        )

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
    repairs: list[dict[str, Any]] = []

    if index.tz is None:
        # Saturn payloads may be returned in descending or otherwise
        # non-monotonic order.  DST disambiguation must operate on delivery
        # order, while preserving the order within an autumn duplicate pair.
        deltas = np.diff(index.asi8)
        forward_steps = int((deltas > 0).sum())
        backward_steps = int((deltas < 0).sum())
        directional_steps = forward_steps + backward_steps
        order_coherence = (
            max(forward_steps, backward_steps) / directional_steps
            if directional_steps
            else 1.0
        )
        if backward_steps > forward_steps:
            index = index[::-1]
            series = series.iloc[::-1]
        order = np.argsort(index.asi8, kind="stable")
        index = index.take(order)
        series = series.iloc[order]

        timezone_hint = (
            "auto"
            if naive_timezone is None
            else str(naive_timezone).strip() or "auto"
        )
        if timezone_hint.lower() == "auto":
            # A continuous UTC-naive grid contains civil labels that do not
            # exist on the spring transition day.  A genuine local-naive grid
            # skips those labels.  Outside a transition the two conventions
            # cannot be inferred, so the market timezone is the conservative
            # default used by Saturn price series.
            local_probe = index.tz_localize(
                timezone,
                ambiguous=True,
                nonexistent="NaT",
            )
            source_timezone = (
                "UTC" if bool(local_probe.isna().any()) else timezone
            )
        else:
            source_timezone = timezone_hint
        if source_timezone.upper() == "UTC":
            index = index.tz_localize("UTC")
        else:
            try:
                probe = index.tz_localize(
                    source_timezone,
                    ambiguous="NaT",
                    nonexistent="raise",
                )
                ambiguous_mask = probe.isna()
                if bool(ambiguous_mask.any()) and order_coherence < 0.8:
                    raise ValueError(
                        f"{name}: ordre source insuffisant pour associer "
                        "les deux folds DST sans échanger leurs prix."
                    )
                ambiguous_timestamps = index[ambiguous_mask].unique()
                repair_positions: list[int] = []
                repaired_timestamps: list[pd.Timestamp] = []
                repaired_values: list[float] = []
                for timestamp in ambiguous_timestamps:
                    positions = np.flatnonzero(index == timestamp)
                    if len(positions) == 1 and dst_policy in {
                        "duplicate",
                        "duplicate_zero_only",
                    }:
                        position = int(positions[0])
                        numeric_value = pd.to_numeric(
                            pd.Series([series.iloc[position]]),
                            errors="coerce",
                        ).iloc[0]
                        value = float(numeric_value)
                        if dst_policy == "duplicate_zero_only" and (
                            not np.isfinite(value) or abs(value) > 1e-12
                        ):
                            raise ValueError(
                                f"{name}: l'heure DST ambiguë {timestamp} "
                                "est singleton mais sa valeur n'est pas un "
                                f"zero fini prouve (valeur={numeric_value!r})."
                            )
                        repair_positions.append(position)
                        repaired_timestamps.append(pd.Timestamp(timestamp))
                        repaired_values.append(value)
                    elif len(positions) != 2:
                        raise ValueError(
                            f"{name}: l'heure DST ambiguë {timestamp} "
                            f"apparaît {len(positions)} fois au lieu de 2."
                        )

                if repair_positions:
                    repeat_counts = np.ones(len(index), dtype=int)
                    repeat_counts[repair_positions] = 2
                    expanded_positions = np.repeat(
                        np.arange(len(index)),
                        repeat_counts,
                    )
                    index = index.take(expanded_positions)
                    series = series.iloc[expanded_positions]
                    LOGGER.warning(
                        "%s: réparation DST opt-in "
                        "incomplete_dst_policy=%s; %s heure(s) "
                        "automnale(s) singleton dupliquée(s) sur les deux "
                        "folds: %s",
                        name,
                        dst_policy,
                        len(repaired_timestamps),
                        ", ".join(
                            str(timestamp)
                            for timestamp in repaired_timestamps
                        ),
                    )

                probe = index.tz_localize(
                    source_timezone,
                    ambiguous="NaT",
                    nonexistent="raise",
                )
                ambiguous_mask = probe.isna()
                ambiguous_flags = np.zeros(len(index), dtype=bool)
                for timestamp in index[ambiguous_mask].unique():
                    positions = np.flatnonzero(index == timestamp)
                    if len(positions) != 2:
                        raise ValueError(
                            f"{name}: l'heure DST ambiguë {timestamp} "
                            f"apparaît {len(positions)} fois au lieu de 2."
                        )
                    ambiguous_flags[positions[0]] = True
                    ambiguous_flags[positions[1]] = False
                index = index.tz_localize(
                    source_timezone,
                    ambiguous=ambiguous_flags,
                    nonexistent="raise",
                )
                for timestamp, value in zip(
                    repaired_timestamps,
                    repaired_values,
                ):
                    local_mask = index.tz_localize(None) == timestamp
                    physical_hours = index[local_mask].tz_convert("UTC")
                    repairs.append(
                        {
                            "policy": dst_policy,
                            "local_timestamp": timestamp.isoformat(),
                            "duplicated_value": float(value),
                            "physical_hours_utc": [
                                item.isoformat() for item in physical_hours
                            ],
                        }
                    )
            except Exception as exc:
                if isinstance(exc, ValueError) and str(exc).startswith(
                    (
                        f"{name}: l'heure DST ambiguë",
                        f"{name}: ordre source insuffisant",
                    )
                ):
                    raise
                raise ValueError(
                    f"{name}: timestamps naïfs non désambiguïsables dans "
                    f"{source_timezone}; fournissez des instants UTC ou "
                    "des offsets explicites pour chaque heure DST."
                ) from exc
        index = index.tz_convert(timezone)
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

    if result.index.duplicated().any() and dst_policy == "duplicate_zero_only":
        duplicates = result.index[result.index.duplicated(keep=False)].unique()
        raise ValueError(
            f"{name}: duplicate physique inattendu sous "
            "incomplete_dst_policy=duplicate_zero_only: "
            f"{[item.isoformat() for item in duplicates[:4]]}."
        )
    if result.index.duplicated().any():
        result = result.groupby(level=0).last()

    result.attrs["dst_repairs"] = repairs

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
    naive_timezone: str | None = None,
    incomplete_dst_policy: str = "raise",
    nocache: bool = False,
    live: bool = False,
    allow_empty: bool = False,
    request_padding_hours: int = 0,
) -> pd.Series:
    query_start = pd.Timestamp(start)
    query_end = pd.Timestamp(end)
    padding_hours = int(request_padding_hours)
    if not 0 <= padding_hours <= 48:
        raise ValueError(
            f"{series_name}: request_padding_hours invalide "
            f"({padding_hours}; attendu entre 0 et 48)."
        )
    if padding_hours:
        # Some Saturn local-civil formulas are filtered server-side as if their
        # naive labels were UTC.  A small symmetric over-fetch avoids dropping
        # boundary hours; the caller still reindexes the exact physical grid.
        padding = pd.Timedelta(hours=padding_hours)
        query_start -= padding
        query_end += padding

    date_dialects = (
        (
            "from_value_date/to_value_date",
            {
                "from_value_date": query_start,
                "to_value_date": query_end,
            },
        ),
        (
            "from_value/to_value",
            {
                "from_value": query_start,
                "to_value": query_end,
            },
        ),
        (
            "start/end",
            {
                "start": query_start,
                "end": query_end,
            },
        ),
    )

    errors: list[str] = []
    raw = None
    resolved_dialect: str | None = None
    resolved_kwargs: dict[str, Any] | None = None

    for dialect, kwargs in date_dialects:
        if revision_date is not None:
            kwargs = {
                **kwargs,
                "revision_date": _as_utc(revision_date),
            }
        if nocache:
            kwargs = {**kwargs, "nocache": True}
        if live:
            kwargs = {**kwargs, "live": True}
        try:
            raw = client.get(series_name, **kwargs)
        except TypeError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            # A dialect fallback is safe only when the client explicitly says
            # that one of the keyword names is unsupported.  A TypeError from
            # inside Saturn (or from payload processing) is a terminal error:
            # retrying it with another date convention can hide the cause or
            # silently change the requested interval.
            if "unexpected keyword argument" in str(exc).lower():
                continue
            break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            break
        else:
            # An exception-free response proves that this client accepted the
            # dialect.  Empty is a valid Saturn answer for an unpublished
            # interval and must not trigger probing of other signatures.
            resolved_dialect = dialect
            resolved_kwargs = dict(kwargs)
            break

    if (
        resolved_dialect is not None
        and resolved_kwargs is not None
        and not _has_values(raw)
        and not nocache
    ):
        # Saturn's HTTP/cache layer may transiently retain an empty payload
        # while the auction has already propagated.  Retry the *same* proven
        # keyword dialect once with nocache; never fall through to a legacy
        # signature after a valid empty response.
        retry_kwargs = {**resolved_kwargs, "nocache": True}
        try:
            retry_raw = client.get(series_name, **retry_kwargs)
        except Exception as exc:
            errors.append(
                f"reprise nocache {type(exc).__name__}: {exc}"
            )
        else:
            raw = retry_raw

    if not _has_values(raw):
        if allow_empty and resolved_dialect is not None:
            return pd.Series(
                dtype=float,
                index=pd.DatetimeIndex([], tz=timezone),
                name=series_name,
            )

        cutoff = (
            str(_as_utc(revision_date))
            if revision_date is not None
            else "latest"
        )
        if resolved_dialect is not None:
            detail = (
                f"reponse vide (dialecte={resolved_dialect})"
            )
        else:
            detail = " | ".join(errors[-4:]) or "aucune reponse exploitable"
        raise RuntimeError(
            f"Saturn indisponible ou vide pour {series_name}; "
            f"plage={start} -> {end}; cutoff={cutoff}. {detail}"
        )

    return normalize_saturn_series(
        raw,
        series_name,
        timezone,
        naive_timezone=naive_timezone,
        incomplete_dst_policy=incomplete_dst_policy,
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
    naive_timezone: str | None = None,
    incomplete_dst_policy: str = "raise",
    nocache: bool = False,
    live: bool = False,
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
        naive_timezone=naive_timezone,
        incomplete_dst_policy=incomplete_dst_policy,
        nocache=nocache,
        live=live,
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
    downloaded_at_utc: pd.Timestamp | None = None,
    naive_timezone: str | None = None,
    incomplete_dst_policy: str = "raise",
) -> pd.DataFrame:
    downloaded_at = _as_utc(
        downloaded_at_utc or pd.Timestamp.now(tz="UTC")
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
            naive_timezone=naive_timezone,
            incomplete_dst_policy=incomplete_dst_policy,
        )
        if series.empty:
            continue

        frame = pd.DataFrame(
            {
                "value_time_utc": series.index.tz_convert("UTC"),
                "snapshot_time_utc": revision_utc,
                "revision_time_utc": revision_utc,
                "value": series.to_numpy(dtype=float),
                "downloaded_at_utc": downloaded_at,
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
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"Colonnes PIT absentes : {missing}. "
            f"Colonnes reçues : {list(frame.columns)}"
        )

    normalized = frame.copy()
    if "downloaded_at_utc" not in normalized:
        normalized["downloaded_at_utc"] = pd.NaT

    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "downloaded_at_utc",
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
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
        ]
    )
    normalized = normalized.sort_values(
        [
            "revision_time_utc",
            "value_time_utc",
            "snapshot_time_utc",
            "downloaded_at_utc",
        ],
        na_position="first",
    )
    normalized = normalized.drop_duplicates(
        subset=["revision_time_utc", "value_time_utc"],
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
        pd.read_parquet(temporary, columns=["value_time_utc"]).head(1)
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
    naive_timezone: str | None = None,
    incomplete_dst_policy: str = "raise",
    overlap_days: int = 7,
    full: bool = False,
    require_contiguous_hourly: bool = False,
    equivalent_target_fallback: Mapping[str, Any] | None = None,
) -> SaturnSyncResult:
    if (
        str(alias).lower() == "target"
        and incomplete_dst_policy != "raise"
    ):
        raise ValueError(
            f"{zone}/target: incomplete_dst_policy=duplicate est interdit "
            "pour la cible."
        )

    existing = (
        pd.Series(dtype=float, name=alias)
        if full
        else _read_latest_store(path, alias, timezone)
    )
    rows_before = int(len(existing))
    audited_fallback_config = AUDITED_EQUIVALENT_TARGET_FALLBACKS.get(
        str(series_name)
    )
    fallback_config = (
        audited_fallback_config
        if equivalent_target_fallback is None
        else equivalent_target_fallback
    )
    # An empty canonical response may continue only along the narrow path
    # whose safety can be re-proved below: strict target continuity, an
    # existing canonical cache, and one of the centrally audited equivalents
    # with at least the operational 168-hour validation contract.  Explicit
    # caller-provided mappings remain usable for non-empty gap repair but do
    # not gain this empty-response exemption.
    allow_empty_canonical = bool(
        str(alias).casefold() == "target"
        and require_contiguous_hourly
        and incomplete_dst_policy == "raise"
        and not existing.empty
        and equivalent_target_fallback is None
        and audited_fallback_config is not None
        and int(audited_fallback_config.get("validation_hours", 0)) >= 168
    )
    fetch_start = _as_zone(start, timezone)
    fetch_end = _as_zone(end, timezone)
    requested_start = fetch_start

    if not existing.empty:
        # Historical PIT replays may run after the shared latest cache has
        # already advanced beyond their cutoff.  Anchoring the overlap on the
        # cache maximum would then invert the Saturn request (start > end),
        # which returns an empty series.  Cap the anchor at the replay end so
        # the target is refreshed on a valid, causal interval.
        overlap_anchor = min(existing.index.max(), fetch_end)
        effective_overlap_days = max(0, overlap_days)
        if allow_empty_canonical and audited_fallback_config is not None:
            validation_hours = int(
                audited_fallback_config.get("validation_hours", 168)
            )
            # The cache maximum may itself belong to a previously published
            # fallback suffix.  Fetch enough extra overlap to retain 168 fully
            # canonical hours even when the formula currently lags by up to
            # two auction days.
            validation_days = (validation_hours + 23) // 24
            effective_overlap_days = max(
                effective_overlap_days,
                validation_days + 2,
            )
        overlap_start = (
            overlap_anchor
            - pd.Timedelta(days=effective_overlap_days)
        )
        if allow_empty_canonical:
            # The live runners deliberately request only D-2 onward.  That is
            # sufficient for ordinary incremental target refreshes but not for
            # the 168-hour compatibility guard of an official completion
            # source.  Read the extra canonical overlap only on this centrally
            # audited strict-target path; otherwise a late requested start
            # makes the canonical download empty and the guard accidentally
            # compares against a stale completion suffix already in the cache.
            fetch_start = overlap_start
        else:
            fetch_start = max(fetch_start, overlap_start)

    continuity_start = (
        requested_start
        if existing.empty
        else pd.Timestamp(existing.index.min())
    )
    if require_contiguous_hourly and continuity_start <= fetch_end:
        expected_before = pd.date_range(
            start=continuity_start,
            end=fetch_end,
            freq="h",
        )
        finite_existing = existing.loc[np.isfinite(existing.to_numpy(dtype=float))].index
        missing_before = expected_before.difference(finite_existing)
        if len(missing_before):
            first_missing = pd.Timestamp(missing_before[0])
            # A sync-only live start may sit after the end of an older cache.
            # Always bridge from immediately before the earliest missing
            # physical hour; otherwise a failed/interrupted run can append new
            # values after a permanent internal hole.
            repair_start = max(
                continuity_start,
                first_missing - pd.Timedelta(hours=1),
            )
            if repair_start < fetch_start:
                LOGGER.warning(
                    "[Saturn] %s/%s | cache cible discontinu: "
                    "%s heure(s) absente(s), reprise depuis %s.",
                    zone,
                    alias,
                    len(missing_before),
                    repair_start,
                )
                fetch_start = repair_start

    downloaded = fetch_saturn_series_from_client(
        client,
        series_name,
        fetch_start,
        fetch_end,
        timezone,
        revision_date=sync_as_of_utc,
        naive_timezone=naive_timezone,
        incomplete_dst_policy=incomplete_dst_policy,
        allow_empty=allow_empty_canonical,
    ).rename(alias)
    if downloaded.empty and not allow_empty_canonical:
        raise RuntimeError(
            f"{zone}/{alias}: reponse Saturn vide interdite; "
            f"plage={fetch_start} -> {fetch_end}; "
            f"cutoff={_as_utc(sync_as_of_utc)}. Une reponse canonique "
            "vide n'est eligible qu'en cible stricte avec cache existant "
            "et fallback central audite sur 168 h."
        )

    if require_contiguous_hourly and fallback_config and not downloaded.empty:
        finite_current = downloaded.loc[np.isfinite(downloaded.to_numpy(dtype=float))]
        if not finite_current.empty:
            current_grid = pd.date_range(
                finite_current.index.min(), finite_current.index.max(), freq="h"
            )
            current_gaps = current_grid.difference(finite_current.index)
            if len(current_gaps):
                # Internal holes need an authoritative overlap before the first
                # hole, even when the caller requested only D-2.  Fetch that
                # small missing prefix once, at exactly the same PIT cutoff.
                validation_start = current_gaps[0] - pd.Timedelta(
                    hours=int(fallback_config.get("validation_hours", 168))
                )
                prefix_end = finite_current.index.min() - pd.Timedelta(hours=1)
                if validation_start <= prefix_end:
                    prefix = fetch_saturn_series_from_client(
                        client, series_name, validation_start, prefix_end, timezone,
                        revision_date=sync_as_of_utc,
                        naive_timezone=naive_timezone,
                        incomplete_dst_policy=incomplete_dst_policy,
                        nocache=True,
                    ).rename(alias)
                    prefix = prefix.loc[
                        (prefix.index >= validation_start) & (prefix.index <= prefix_end)
                    ]
                    downloaded = pd.concat([prefix, downloaded]).sort_index()
                    downloaded = downloaded.loc[~downloaded.index.duplicated(keep="last")]

    merged = (
        downloaded.copy()
        if existing.empty
        else pd.concat([existing, downloaded])
    ).sort_index()
    merged = merged.loc[
        ~merged.index.duplicated(keep="last")
    ]
    fallback_series: str | None = None
    fallback_rows = 0
    fallback_paired_hours = 0
    fallback_maximum_difference: float | None = None
    fallback_tolerance: float | None = None
    fallback_internal_rows = 0
    fallback_value_times_utc: tuple[str, ...] = ()
    if require_contiguous_hourly and continuity_start <= fetch_end:
        expected_after = pd.date_range(
            start=continuity_start,
            end=fetch_end,
            freq="h",
        )
        finite_merged = merged.loc[np.isfinite(merged.to_numpy(dtype=float))].index
        missing_after = expected_after.difference(finite_merged)
        finite_downloaded = downloaded.loc[np.isfinite(downloaded.to_numpy(dtype=float))]
        fallback_tail_index: pd.DatetimeIndex | None = None
        validation_index: pd.DatetimeIndex | None = None
        validation_source: pd.Series | None = None

        if not finite_downloaded.empty:
            canonical_start = pd.Timestamp(finite_downloaded.index.min())
            canonical_end = pd.Timestamp(finite_downloaded.index.max())
            canonical_grid = pd.date_range(
                start=canonical_start,
                end=canonical_end,
                freq="h",
            )
            missing_in_download = canonical_grid.difference(
                finite_downloaded.index
            )
            if len(missing_in_download) and not fallback_config:
                examples = [str(value) for value in missing_in_download[:8]]
                raise ValueError(
                    f"{zone}/target: download canonique courant discontinu; "
                    f"{len(missing_in_download)} heure(s) absente(s) entre "
                    f"{canonical_start} et {canonical_end}. Exemples: "
                    f"{examples}. Le cache existant n'a pas ete remplace."
                )

            # A current canonical download is the only authoritative anchor
            # for refreshing an already-populated fallback tail.  Validate
            # against its own last 168 hours, then replace every later hour up
            # to fetch_end; never compare against stale fallback values that
            # happen to be present in the merged cache.
            if (canonical_end < fetch_end or len(missing_in_download)) and fallback_config:
                suffix_index = pd.date_range(
                    start=canonical_end + pd.Timedelta(hours=1),
                    end=fetch_end,
                    freq="h",
                )
                fallback_tail_index = missing_in_download.union(suffix_index)
                validation_hours_for_index = int(
                    fallback_config.get("validation_hours", 168)
                )
                first_repair = pd.Timestamp(fallback_tail_index[0])
                validation_index = pd.date_range(
                    end=first_repair - pd.Timedelta(hours=1),
                    periods=validation_hours_for_index,
                    freq="h",
                )
                if len(missing_in_download):
                    # Prove the full overlap before the first gap and every
                    # fresh canonical value between/after internal gaps.  Cached
                    # completion values never prove their own equivalence.
                    validation_index = validation_index.union(
                        finite_downloaded.index[finite_downloaded.index >= first_repair]
                    )
                    fallback_internal_rows = int(len(missing_in_download))
                validation_source = downloaded
        elif len(missing_after) and fallback_config:
            # With an empty canonical response there is no new authoritative
            # anchor.  Preserve the original fail-closed behaviour: only a
            # genuinely missing contiguous suffix may be completed, using the
            # existing canonical cache for the 168-hour equivalence proof.
            first_missing = pd.Timestamp(missing_after[0])
            missing_suffix = pd.date_range(
                start=first_missing,
                end=fetch_end,
                freq="h",
            )
            if missing_after.equals(missing_suffix):
                validation_hours_for_index = int(
                    fallback_config.get("validation_hours", 168)
                )
                fallback_tail_index = missing_after
                validation_index = pd.date_range(
                    end=first_missing - pd.Timedelta(hours=1),
                    periods=validation_hours_for_index,
                    freq="h",
                )
                validation_source = merged

        if fallback_tail_index is not None and fallback_config:
            if str(alias).casefold() != "target":
                raise ValueError(
                    "Un fallback equivalent est autorise uniquement pour la cible."
                )
            fallback_series = str(fallback_config.get("series", "")).strip()
            fallback_naive_timezone = str(
                fallback_config.get("naive_timezone", "UTC")
            ).strip()
            fallback_request_padding_hours = int(
                fallback_config.get("request_padding_hours", 0)
            )
            fallback_nocache = bool(fallback_config.get("nocache", False))
            validation_hours = int(
                fallback_config.get("validation_hours", 168)
            )
            fallback_tolerance = float(
                fallback_config.get("atol_eur_mwh", 1e-9)
            )
            if not fallback_series:
                raise ValueError(f"{zone}/target: serie fallback equivalente absente.")
            if validation_hours < 24:
                raise ValueError(
                    f"{zone}/target: validation fallback insuffisante "
                    f"({validation_hours} h < 24 h)."
                )
            if not np.isfinite(fallback_tolerance) or not (
                0.0 <= fallback_tolerance <= 1e-6
            ):
                raise ValueError(
                    f"{zone}/target: tolerance fallback invalide."
                )
            if validation_index is None or validation_source is None:
                raise AssertionError("Validation fallback non initialisee.")
            if len(validation_index) < validation_hours:
                raise ValueError(
                    f"{zone}/target: fenetre de validation fallback invalide."
                )

            fallback_start = pd.Timestamp(validation_index[0])
            fallback_downloaded = fetch_saturn_series_from_client(
                client,
                fallback_series,
                fallback_start,
                fetch_end,
                timezone,
                revision_date=sync_as_of_utc,
                naive_timezone=fallback_naive_timezone,
                incomplete_dst_policy="raise",
                request_padding_hours=fallback_request_padding_hours,
                nocache=fallback_nocache,
            ).sort_index()
            paired = pd.concat(
                [
                    validation_source.reindex(validation_index).rename(
                        "canonical"
                    ),
                    fallback_downloaded.reindex(validation_index).rename(
                        "fallback"
                    ),
                ],
                axis=1,
            )
            paired = paired.loc[np.isfinite(paired.to_numpy(dtype=float)).all(axis=1)]
            fallback_paired_hours = int(len(paired))
            if fallback_paired_hours != len(validation_index):
                raise ValueError(
                    f"{zone}/target: recouvrement fallback insuffisant "
                    f"({fallback_paired_hours} h != {len(validation_index)} h). "
                    "Le cache existant n'a pas ete remplace."
                )
            differences = (paired["canonical"] - paired["fallback"]).abs()
            fallback_maximum_difference = float(differences.max())
            if fallback_maximum_difference > fallback_tolerance:
                raise ValueError(
                    f"{zone}/target: fallback {fallback_series} non "
                    "equivalent a la cible canonique "
                    f"(ecart max={fallback_maximum_difference:.12g} "
                    f"EUR/MWh > {fallback_tolerance:.12g}). "
                    "Le cache existant n'a pas ete remplace."
                )
            fallback_tail = pd.to_numeric(
                fallback_downloaded.reindex(fallback_tail_index),
                errors="coerce",
            )
            if not np.isfinite(fallback_tail.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"{zone}/target: fallback {fallback_series} incomplet "
                    "sur les heures manquantes. Le cache existant n'a pas ete "
                    "remplace."
                )
            supplement = pd.Series(
                fallback_tail.to_numpy(dtype=float),
                index=fallback_tail_index,
                name=alias,
            )
            # Replace exactly the absent canonical hours and the suffix.  Keep
            # canonical islands between gaps, and newer values beyond cutoff.
            replace_mask = merged.index.isin(fallback_tail_index)
            merged = pd.concat(
                [merged.loc[~replace_mask], supplement]
            ).sort_index()
            merged = merged.loc[
                ~merged.index.duplicated(keep="last")
            ]
            fallback_rows = int(len(fallback_tail_index))
            fallback_value_times_utc = tuple(
                value.isoformat() for value in fallback_tail_index.tz_convert("UTC")
            )
            finite_merged = merged.loc[np.isfinite(merged.to_numpy(dtype=float))].index
            missing_after = expected_after.difference(finite_merged)
            LOGGER.warning(
                "[Saturn] %s/target | %s heure(s) remplacee(s) depuis %s "
                "au cutoff %s apres validation de %s h (ecart max %.3g).",
                zone,
                fallback_rows,
                fallback_series,
                sync_as_of_utc,
                fallback_paired_hours,
                fallback_maximum_difference,
            )
        if len(missing_after):
            examples = [str(value) for value in missing_after[:8]]
            raise ValueError(
                f"{zone}/{alias}: Saturn n'a pas comble le cache cible "
                f"horaire; {len(missing_after)} heure(s) absente(s) entre "
                f"{continuity_start} et {fetch_end}. Exemples: {examples}. "
                "Le cache existant n'a pas ete remplace."
            )
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
        equivalent_fallback_series=(
            fallback_series if fallback_rows else None
        ),
        equivalent_fallback_rows=fallback_rows,
        equivalent_fallback_validation_paired_hours=fallback_paired_hours,
        equivalent_fallback_validation_max_abs_difference_eur_mwh=(
            fallback_maximum_difference
        ),
        equivalent_fallback_validation_tolerance_eur_mwh=(
            fallback_tolerance if fallback_rows else None
        ),
        equivalent_fallback_internal_rows=fallback_internal_rows,
        equivalent_fallback_value_times_utc=fallback_value_times_utc,
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
    naive_timezone: str | None = None,
    incomplete_dst_policy: str = "raise",
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

    requested_start = _as_utc(revision_start)
    fetch_start = requested_start
    fetch_end = _as_utc(revision_end)

    if not existing.empty:
        last_revision = existing["revision_time_utc"].max()
        if fetch_end < last_revision:
            # A PIT replay asks what was known at an older cutoff.  A cache
            # containing newer revisions does not prove that this historical
            # insertion window was ever downloaded.  Query the requested
            # window exactly instead of applying the normal latest-tail
            # optimisation, then merge the returned vintages atomically.
            LOGGER.info(
                "[Saturn/PIT] %s | resynchronisation historique exacte "
                "%s -> %s (cache plus recent: %s)",
                alias,
                requested_start,
                fetch_end,
                last_revision,
            )
        else:
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
                str(existing["revision_time_utc"].min())
                if not existing.empty
                else None
            ),
            last_revision_utc=(
                str(existing["revision_time_utc"].max())
                if not existing.empty
                else None
            ),
            sync_as_of_utc=str(fetch_end),
        )

    downloaded_at = pd.Timestamp.now(tz="UTC")
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
            downloaded_at_utc=downloaded_at,
            naive_timezone=naive_timezone,
            incomplete_dst_policy=incomplete_dst_policy,
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
            str(merged["revision_time_utc"].min())
            if not merged.empty
            else None
        ),
        last_revision_utc=(
            str(merged["revision_time_utc"].max())
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
    full_target: bool = False,
    skip_pit: bool = False,
    as_of: Any | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Actualise les caches Saturn latest et les historiques PIT.

    ``as_of`` est un plafond global d'ingestion. La sélection de la révision
    utilisable pour chaque livraison reste effectuée dans ``data.py`` avec
    ``forecast_origin_local_time``.

    ``full_target`` reconstruit uniquement le cache latest de la cible et
    laisse les historiques PIT en mode incrémental. ``skip_pit`` permet de ne
    pas contacter Saturn pour les historiques PIT pendant cette opération.
    ``full`` conserve son comportement historique et reconstruit toutes les
    séries traitées.
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

            if (
                spec.alias == "target"
                and spec.incomplete_dst_policy != "raise"
            ):
                raise ValueError(
                    f"{zone.zone}/target: "
                    "incomplete_dst_policy=duplicate est interdit pour "
                    "la cible."
                )

            source = (
                spec.source
                if spec.source != "auto"
                else global_source
            )
            pit = is_pit_spec(spec, config)

            if not pit and source not in {"auto", "saturn"}:
                continue

            if pit and skip_pit:
                LOGGER.info(
                    "[Saturn/PIT] %s/%s | actualisation ignorée.",
                    zone.zone,
                    spec.alias,
                )
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
                    naive_timezone=spec.naive_timezone,
                    incomplete_dst_policy=spec.incomplete_dst_policy,
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
                    naive_timezone=spec.naive_timezone,
                    incomplete_dst_policy=spec.incomplete_dst_policy,
                    overlap_days=latest_overlap_days,
                    full=bool(
                        full
                        or (
                            full_target
                            and spec is zone.target
                        )
                    ),
                    require_contiguous_hourly=(
                        spec.alias == "target"
                        and str(
                            deep_get(
                                config,
                                "data.target_input_resolution",
                                "hourly",
                            )
                        ).strip().lower()
                        == "hourly"
                    ),
                )

            results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(
        [result.to_dict() for result in results]
    ).sort_values(["zone", "kind", "alias"]).reset_index(
        drop=True
    )
