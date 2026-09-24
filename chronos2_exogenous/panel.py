"""Origin-aware training panels built from causal exogenous banks.

Each ``(origin_timestamp, item_id)`` group contains a fixed number of physical
UTC context hours followed by the complete local delivery day.  Consequently
the horizon is 23, 24 or 25 hours at DST transitions.  The full panel always
keeps those transition days for physical evaluation; :meth:`OriginPanel.for_fit`
selects only the horizon length supported by a particular trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from .feature_bank import (
    ExogenousBank,
    ExogenousBankError,
    cutoff_by_delivery_hour,
    delivery_utc_index,
)


SUPPORTED_LAYOUTS = frozenset({"per_zone", "cwe_wide"})
DEFAULT_CWE_ZONES = ("FR", "DE", "BE", "NL")


class OriginPanelError(ValueError):
    """Raised when a panel would mix origins, omit hours or leak information."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_target_cache(path: str | Path, *, zone: str) -> pd.Series:
    """Load one canonical ``target__*.csv.gz`` cache as a strict UTC series."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise OriginPanelError(f"{zone}: cache cible absent: {source}.")
    try:
        raw = pd.read_csv(source, usecols=["timestamp", "value"])
        timestamps = pd.to_datetime(
            raw["timestamp"], utc=True, errors="raise", format="mixed"
        )
        values = pd.to_numeric(raw["value"], errors="coerce").astype(float)
    except Exception as exc:
        raise OriginPanelError(f"{zone}: cache cible illisible: {source}.") from exc
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise OriginPanelError(f"{zone}: timestamps cibles invalides ou dupliques.")
    if bool(np.isinf(values.to_numpy(dtype=float)).any()):
        raise OriginPanelError(f"{zone}: cible infinie interdite.")
    code = str(zone).strip().upper()
    result = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(timestamps),
        name=f"target_{code.casefold()}",
    ).sort_index()
    result.attrs.update(
        {
            "zone": code,
            "source_path": str(source),
            "source_sha256": _sha256_file(source),
        }
    )
    return result


@dataclass(frozen=True)
class OriginPanel:
    """Complete physical panel plus fit/evaluation selection metadata."""

    frame: pd.DataFrame
    audit: Mapping[str, Any]

    def for_fit(self, *, horizon_hours: int = 24, copy: bool = True) -> pd.DataFrame:
        """Return whole groups whose civil horizon matches ``horizon_hours``."""

        if int(horizon_hours) <= 0:
            raise OriginPanelError("horizon_hours doit etre positif.")
        selected = self.frame.loc[
            self.frame["horizon_hours"].eq(int(horizon_hours))
        ]
        return selected.copy() if copy else selected

    def for_evaluation(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the unfiltered panel, including reserved 23/25-hour days."""

        return self.frame.copy() if copy else self.frame


def _target_series(value: pd.Series, *, zone: str) -> pd.Series:
    if not isinstance(value, pd.Series) or not isinstance(value.index, pd.DatetimeIndex):
        raise OriginPanelError(f"{zone}: la cible doit etre une Series datetime.")
    if value.index.tz is None or value.index.has_duplicates:
        raise OriginPanelError(f"{zone}: timeline cible aware et unique requise.")
    numeric = pd.to_numeric(value, errors="coerce").astype(float)
    if bool(np.isinf(numeric.to_numpy(dtype=float)).any()):
        raise OriginPanelError(f"{zone}: cible infinie interdite.")
    result = pd.Series(
        numeric.to_numpy(dtype=float),
        index=value.index.tz_convert("UTC"),
        name=f"target_{zone.casefold()}",
    ).sort_index()
    result.attrs.update(value.attrs)
    return result


def _localise_day(value: str | pd.Timestamp, timezone: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert(timezone).tz_localize(None)
    return day.normalize()


def _normalise_targets(
    targets: Mapping[str, pd.Series], zones: Sequence[str]
) -> dict[str, pd.Series]:
    normalised = {str(zone).strip().upper(): value for zone, value in targets.items()}
    missing = sorted(set(zones).difference(normalised))
    if missing:
        raise OriginPanelError(f"Cibles absentes: {missing}.")
    return {zone: _target_series(normalised[zone], zone=zone) for zone in zones}


def _normalise_banks(
    banks: ExogenousBank | Mapping[str, ExogenousBank],
    zones: Sequence[str],
) -> dict[str, ExogenousBank]:
    if isinstance(banks, ExogenousBank):
        return {zone: banks for zone in zones}
    normalised = {str(zone).strip().upper(): bank for zone, bank in banks.items()}
    missing = sorted(set(zones).difference(normalised))
    if missing:
        raise OriginPanelError(f"Banques exogenes absentes: {missing}.")
    invalid = [zone for zone in zones if not isinstance(normalised[zone], ExogenousBank)]
    if invalid:
        raise OriginPanelError(f"Banques exogenes invalides: {invalid}.")
    return {zone: normalised[zone] for zone in zones}


def _zone_local_feature_frame(bank: ExogenousBank, zone: str) -> pd.DataFrame:
    """Give zone-local weather fields a common schema for shared LoRA."""

    code = zone.casefold()
    frame = bank.for_consumer("chronos")
    renames = {
        f"{code}_temperature_fcst": "local_temperature_fcst",
        f"{code}_wind_generation_fcst": "local_wind_generation_fcst",
        f"{code}_solar_generation_fcst": "local_solar_generation_fcst",
    }
    for column in frame.columns:
        prefix = f"weather_{code}_"
        if str(column).startswith(prefix):
            renames[str(column)] = "weather_local_" + str(column)[len(prefix) :]
    result = frame.rename(columns=renames)
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise OriginPanelError(f"{zone}: features locales dupliquees: {duplicates}.")
    return result


def _cwe_feature_frame(banks: Mapping[str, ExogenousBank], zones: Sequence[str]) -> pd.DataFrame:
    """Merge zone banks and prove that common columns are identical."""

    result: pd.DataFrame | None = None
    for zone in zones:
        candidate = banks[zone].for_consumer("chronos")
        if result is None:
            result = candidate.copy()
            continue
        union_index = result.index.union(candidate.index).sort_values()
        result = result.reindex(union_index)
        candidate = candidate.reindex(union_index)
        for column in candidate:
            if column in result:
                if not result[column].equals(candidate[column]):
                    raise OriginPanelError(
                        f"Feature commune {column!r} incoherente entre banques CWE."
                    )
            else:
                result[column] = candidate[column]
    if result is None:  # pragma: no cover - zones are validated non-empty.
        raise OriginPanelError("Aucune banque CWE.")
    return result


def _origin_window(
    delivery_day: pd.Timestamp,
    *,
    context_length: int,
    timezone: str,
) -> tuple[pd.Timestamp, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    horizon = delivery_utc_index(delivery_day, delivery_day, timezone=timezone)
    context = pd.date_range(
        end=horizon[0] - pd.Timedelta(hours=1),
        periods=int(context_length),
        freq="h",
    )
    window = context.append(horizon)
    cutoffs = cutoff_by_delivery_hour(horizon, timezone=timezone)
    if len(set(cutoffs)) != 1:
        raise OriginPanelError(f"{delivery_day.date()}: cutoff non constant.")
    return pd.Timestamp(cutoffs[0]), context, horizon, window


def _base_rows(
    *,
    origin: pd.Timestamp,
    delivery_day: pd.Timestamp,
    context: pd.DatetimeIndex,
    horizon: pd.DatetimeIndex,
    item_id: str,
) -> pd.DataFrame:
    window = context.append(horizon)
    horizon_hours = len(horizon)
    relative = np.arange(-len(context), horizon_hours, dtype=int)
    return pd.DataFrame(
        {
            "origin_timestamp": pd.DatetimeIndex([origin] * len(window)),
            "timestamp": window,
            "item_id": str(item_id),
            # The bank has proven each underlying timestamp <= origin.  This
            # column records when the complete row was assembled/available;
            # it never claims that the future label was available.
            "feature_available_at_utc": pd.DatetimeIndex([origin] * len(window)),
            "phase": np.where(relative < 0, "context", "horizon"),
            "delivery_day": str(delivery_day.date()),
            "horizon_hours": int(horizon_hours),
            "eligible_for_24h_fit": bool(horizon_hours == 24),
            "reserved_for_evaluation": True,
            "relative_step": relative,
        }
    )


def _require_finite(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = frame.loc[:, list(columns)].isna().sum()
    bad = {str(column): int(count) for column, count in missing.items() if count}
    if bad:
        raise OriginPanelError(f"{label}: valeurs manquantes: {bad}.")


def build_origin_panel(
    banks: ExogenousBank | Mapping[str, ExogenousBank],
    targets: Mapping[str, pd.Series],
    *,
    delivery_days: Sequence[str | pd.Timestamp],
    context_length: int,
    layout: str = "per_zone",
    zones: Sequence[str] = DEFAULT_CWE_ZONES,
    timezone: str = "Europe/Paris",
    require_horizon_targets: bool = True,
    require_complete_covariates: bool = True,
) -> OriginPanel:
    """Build the origin-aware table consumed by ``lora_finetune``.

    ``per_zone`` creates one item per zone with a generic ``target`` column and
    generic zone-local weather names.  ``cwe_wide`` creates a single ``CWE``
    item with ``target_fr`` ... ``target_nl`` columns.
    """

    if layout not in SUPPORTED_LAYOUTS:
        raise OriginPanelError(
            f"Layout inconnu {layout!r}; disponibles={sorted(SUPPORTED_LAYOUTS)}."
        )
    if isinstance(context_length, bool) or int(context_length) <= 0:
        raise OriginPanelError("context_length doit etre un entier positif.")
    selected_zones = tuple(dict.fromkeys(str(zone).strip().upper() for zone in zones))
    if not selected_zones:
        raise OriginPanelError("Au moins une zone cible est requise.")
    days = tuple(
        dict.fromkeys(_localise_day(value, timezone) for value in delivery_days)
    )
    if not days:
        raise OriginPanelError("Au moins une origine est requise.")
    if tuple(sorted(days)) != days:
        raise OriginPanelError("Les delivery_days doivent etre uniques et croissants.")
    target_map = _normalise_targets(targets, selected_zones)
    bank_map = _normalise_banks(banks, selected_zones)
    target_audit = {
        zone: {
            "source_path": target_map[zone].attrs.get("source_path"),
            "source_sha256": target_map[zone].attrs.get("source_sha256"),
            "first_timestamp": target_map[zone].index.min().isoformat(),
            "last_timestamp": target_map[zone].index.max().isoformat(),
        }
        for zone in selected_zones
    }
    bank_audit = {
        zone: {
            "source_hashes": dict(bank_map[zone].audit.get("source_hashes", {})),
            "source_audit_hashes": dict(
                bank_map[zone].audit.get("source_audit_hashes", {})
            ),
            "historical_evidence_manifest_hashes": dict(
                bank_map[zone].audit.get(
                    "historical_evidence_manifest_hashes", {}
                )
            ),
            "historical_evidence_manifest_paths": dict(
                bank_map[zone].audit.get(
                    "historical_evidence_manifest_paths", {}
                )
            ),
            "historical_source_contract_hashes": dict(
                bank_map[zone].audit.get(
                    "historical_source_contract_hashes", {}
                )
            ),
            "forecast_origin_timezone": str(
                bank_map[zone].audit.get("timezone", timezone)
            ),
            "source_cutoff_timezones": {
                str(name): str(details.get("source_cutoff_timezone", timezone))
                for name, details in dict(
                    bank_map[zone].audit.get("sources", {})
                ).items()
                if isinstance(details, Mapping)
            },
            "production_pit_evidence": dict(
                bank_map[zone].audit.get("production_pit_evidence", {})
            ),
            "historical_backtest_pit_evidence": dict(
                bank_map[zone].audit.get(
                    "historical_backtest_pit_evidence", {}
                )
            ),
            "historical_backtest_ready": bool(
                bank_map[zone].audit.get("historical_backtest_ready")
            ),
            "historical_backtest_blockers": list(
                bank_map[zone].audit.get(
                    "historical_backtest_blockers", []
                )
            ),
            "production_ready": bool(bank_map[zone].audit.get("production_ready")),
            "production_blockers": list(
                bank_map[zone].audit.get("production_blockers", [])
            ),
        }
        for zone in selected_zones
    }

    frames: list[pd.DataFrame] = []
    excluded_fit_days: list[str] = []
    horizon_histogram: dict[str, int] = {}
    wide_features = (
        _cwe_feature_frame(bank_map, selected_zones)
        if layout == "cwe_wide"
        else None
    )
    local_features = (
        {zone: _zone_local_feature_frame(bank_map[zone], zone) for zone in selected_zones}
        if layout == "per_zone"
        else {}
    )
    expected_feature_columns: tuple[str, ...] | None = None
    if layout == "per_zone":
        schemas = {zone: tuple(local_features[zone].columns) for zone in selected_zones}
        expected_feature_columns = schemas[selected_zones[0]]
        mismatched = [
            zone for zone in selected_zones if schemas[zone] != expected_feature_columns
        ]
        if mismatched:
            raise OriginPanelError(
                "Le layout per_zone exige le meme schema apres renommage local; "
                f"zones incompatibles={mismatched}."
            )

    for delivery_day in days:
        origin, context, horizon, window = _origin_window(
            delivery_day,
            context_length=int(context_length),
            timezone=timezone,
        )
        horizon_histogram[str(len(horizon))] = horizon_histogram.get(
            str(len(horizon)), 0
        ) + 1
        if len(horizon) != 24:
            excluded_fit_days.append(str(delivery_day.date()))
        if layout == "cwe_wide":
            assert wide_features is not None
            features = wide_features.reindex(window)
            block = _base_rows(
                origin=origin,
                delivery_day=delivery_day,
                context=context,
                horizon=horizon,
                item_id="CWE",
            )
            target_columns: list[str] = []
            for zone in selected_zones:
                column = f"target_{zone.casefold()}"
                block[column] = target_map[zone].reindex(window).to_numpy(dtype=float)
                target_columns.append(column)
            feature_columns = tuple(map(str, features.columns))
            collisions = sorted(set(feature_columns).intersection(block.columns))
            if collisions:
                raise OriginPanelError(f"Features reservees interdites: {collisions}.")
            for number, column in enumerate(feature_columns):
                block[column] = features.iloc[:, number].to_numpy(dtype=float)
            _require_finite(
                block.iloc[: len(context)], target_columns, label=f"{delivery_day.date()}/context"
            )
            if require_horizon_targets:
                _require_finite(
                    block.iloc[len(context) :],
                    target_columns,
                    label=f"{delivery_day.date()}/horizon",
                )
            if require_complete_covariates:
                _require_finite(
                    block,
                    feature_columns,
                    label=f"{delivery_day.date()}/covariates",
                )
            frames.append(block)
        else:
            for zone in selected_zones:
                features = local_features[zone].reindex(window)
                block = _base_rows(
                    origin=origin,
                    delivery_day=delivery_day,
                    context=context,
                    horizon=horizon,
                    item_id=zone,
                )
                block["target"] = target_map[zone].reindex(window).to_numpy(dtype=float)
                feature_columns = tuple(map(str, features.columns))
                collisions = sorted(set(feature_columns).intersection(block.columns))
                if collisions:
                    raise OriginPanelError(
                        f"Features reservees interdites: {collisions}."
                    )
                for number, column in enumerate(feature_columns):
                    block[column] = features.iloc[:, number].to_numpy(dtype=float)
                _require_finite(
                    block.iloc[: len(context)], ["target"], label=f"{delivery_day.date()}/{zone}/context"
                )
                if require_horizon_targets:
                    _require_finite(
                        block.iloc[len(context) :],
                        ["target"],
                        label=f"{delivery_day.date()}/{zone}/horizon",
                    )
                if require_complete_covariates:
                    _require_finite(
                        block,
                        feature_columns,
                        label=f"{delivery_day.date()}/{zone}/covariates",
                    )
                frames.append(block)

    panel = pd.concat(frames, ignore_index=True)
    if bool((panel["feature_available_at_utc"] > panel["origin_timestamp"]).any()):
        raise OriginPanelError("feature_available_at_utc depasse une origine.")
    keys = ["origin_timestamp", "item_id", "timestamp"]
    if panel.duplicated(keys).any():
        raise OriginPanelError("Le panel contient des lignes origine/item/heure dupliquees.")
    group_horizons = panel.loc[panel["phase"].eq("horizon")].groupby(
        ["origin_timestamp", "item_id"], sort=False
    ).size()
    declared = panel.groupby(["origin_timestamp", "item_id"], sort=False)[
        "horizon_hours"
    ].first()
    if not group_horizons.equals(declared):
        raise OriginPanelError("Le nombre de lignes horizon differe de horizon_hours.")
    group_contexts = panel.loc[panel["phase"].eq("context")].groupby(
        ["origin_timestamp", "item_id"], sort=False
    ).size()
    if not bool(group_contexts.eq(int(context_length)).all()):
        raise OriginPanelError("Le nombre de lignes contexte est incoherent.")

    metadata_columns = {
        "origin_timestamp",
        "timestamp",
        "item_id",
        "feature_available_at_utc",
        "phase",
        "delivery_day",
        "horizon_hours",
        "eligible_for_24h_fit",
        "reserved_for_evaluation",
        "relative_step",
        "target",
        *(f"target_{zone.casefold()}" for zone in selected_zones),
    }
    feature_columns = [column for column in panel if column not in metadata_columns]
    audit: dict[str, Any] = {
        "schema_version": 1,
        "layout": layout,
        "zones": list(selected_zones),
        "context_length": int(context_length),
        "origins": len(days),
        "origin_items": int(panel.groupby(["origin_timestamp", "item_id"]).ngroups),
        "rows": len(panel),
        "horizon_day_counts": horizon_histogram,
        "fit_horizon_hours": 24,
        "fit_origins": int(sum(length == 24 for length in map(int, [len(delivery_utc_index(day, day, timezone=timezone)) for day in days]))),
        "fit_excluded_dst_days": excluded_fit_days,
        "evaluation_origins_reserved": len(days),
        "feature_columns": feature_columns,
        "target_sources": target_audit,
        "exogenous_banks": bank_audit,
        "production_pit_evidence": {
            zone: bool(bank_map[zone].audit.get("production_ready"))
            for zone in selected_zones
        },
        "historical_backtest_pit_evidence": {
            zone: bool(bank_map[zone].audit.get("historical_backtest_ready"))
            for zone in selected_zones
        },
        "historical_backtest_ready": bool(
            all(
                bank_map[zone].audit.get("historical_backtest_ready")
                for zone in selected_zones
            )
        ),
        "production_ready": bool(
            all(bank_map[zone].audit.get("production_ready") for zone in selected_zones)
        ),
    }
    return OriginPanel(frame=panel, audit=audit)


def write_origin_panel(
    panel: OriginPanel,
    path: str | Path,
    *,
    audit_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Atomically publish a Parquet plus its source/hash audit JSON."""

    output = Path(path).expanduser().resolve()
    audit_output = (
        Path(audit_path).expanduser().resolve()
        if audit_path is not None
        else output.with_suffix(output.suffix + ".audit.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = output.with_name(f".{output.name}.{token}.tmp.parquet")
    temporary_audit = audit_output.with_name(f".{audit_output.name}.{token}.tmp")
    try:
        panel.frame.to_parquet(temporary, index=False)
        verified = pd.read_parquet(temporary)
        if len(verified) != len(panel.frame) or list(verified) != list(panel.frame):
            raise OriginPanelError("Verification du panel Parquet en echec.")
        payload = dict(panel.audit)
        payload["panel_path"] = str(output)
        payload["panel_sha256"] = _sha256_file(temporary)
        temporary_audit.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        os.replace(temporary_audit, audit_output)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_audit.unlink(missing_ok=True)
    return output, audit_output


__all__ = [
    "DEFAULT_CWE_ZONES",
    "OriginPanel",
    "OriginPanelError",
    "SUPPORTED_LAYOUTS",
    "build_origin_panel",
    "load_target_cache",
    "write_origin_panel",
]
