"""Strict civil-cutoff PIT selection for the isolated nuclear experiment.

The incumbent preparation path is deliberately untouched. Its generic PIT
reader subtracts physical 24-hour durations from timezone-aware midnights;
here D-1 08:00 is constructed in civil dates before timezone localisation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_modular.common import SeriesSpec, ZoneConfig, ZoneData, resolve_path
from chronos2_modular.data import first_matching_column, prepare_zone_data, read_series_file
from chronos2_modular.saturn import is_pit_spec, resolve_pit_path


class NuclearPreparationError(ValueError):
    """An isolated forecast input cannot satisfy its causal contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware_utc(values: Any, *, label: str) -> pd.DatetimeIndex:
    """Do not guess a timezone for an ambiguous PIT identity timestamp."""
    if isinstance(getattr(values, "dtype", None), pd.DatetimeTZDtype):
        result = pd.DatetimeIndex(values).tz_convert("UTC")
    else:
        parsed = [pd.Timestamp(value) for value in values]
        if any(pd.isna(value) or value.tzinfo is None for value in parsed):
            raise NuclearPreparationError(f"{label}: timestamps PIT absents ou sans fuseau explicite.")
        result = pd.DatetimeIndex(pd.to_datetime(parsed, utc=True))
    if result.hasnans:
        raise NuclearPreparationError(f"{label}: timestamp PIT absent.")
    return result


def _civil_cutoffs(index: pd.DatetimeIndex, *, timezone: str, runtime_as_of: Any) -> pd.DatetimeIndex:
    runtime = _aware_utc([runtime_as_of], label="runtime_as_of")[0]
    # Drop the zone BEFORE calendar subtraction. This is crucial for the
    # Monday immediately after either DST transition (08:00, not 07/09:00).
    local_days = index.tz_convert(timezone).tz_localize(None).normalize()
    civil = local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    cutoffs = civil.tz_localize(timezone).tz_convert("UTC")
    return pd.DatetimeIndex(cutoffs.where(cutoffs <= runtime, runtime))


def _strict_selection(
    frame: pd.DataFrame,
    spec: SeriesSpec,
    *,
    timezone: str,
    runtime_as_of: Any,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select the latest eligible vintage, including a latest missing value.

    Supports canonical narrow stores and the audited wide residual-load bank.
    No interpolation, older-finite substitution, or timezone guessing occurs.
    """
    if not spec.known_future or tuple(spec.future_strategies) != ("oracle",):
        raise NuclearPreparationError(f"{spec.alias}: une prevision connue a D-1 avec strategie oracle est requise.")
    if spec.fill_method != "none":
        raise NuclearPreparationError(f"{spec.alias}: fill_method=none requis; aucune imputation implicite.")
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    delivery_col = spec.timestamp_col or "value_time_utc"
    snapshot_col = spec.availability_col or "snapshot_time_utc"
    revision_col = spec.revision_col or "revision_time_utc"
    value_col = spec.value_col or (spec.alias if spec.alias in frame.columns else "value")
    missing = {delivery_col, snapshot_col, revision_col, value_col}.difference(frame.columns)
    if missing:
        raise NuclearPreparationError(f"{spec.alias}: colonnes PIT manquantes: {sorted(missing)}.")
    delivery = _aware_utc(frame[delivery_col], label=f"{spec.alias}/{delivery_col}")
    snapshots = _aware_utc(frame[snapshot_col], label=f"{spec.alias}/{snapshot_col}")
    revisions = _aware_utc(frame[revision_col], label=f"{spec.alias}/{revision_col}")
    if not delivery.equals(delivery.floor("h")):
        raise NuclearPreparationError(f"{spec.alias}: la livraison PIT doit identifier des heures physiques HH:00.")
    cutoffs = _civil_cutoffs(delivery, timezone=timezone, runtime_as_of=runtime_as_of)
    normalized = pd.DataFrame({
        "timestamp": delivery,
        "snapshot": snapshots,
        "revision": revisions,
        "cutoff": cutoffs,
        "value": pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float),
    })
    eligible = normalized.loc[(snapshots <= cutoffs) & (revisions <= cutoffs)]
    keys = ["timestamp", "snapshot", "revision"]
    ties = eligible.loc[eligible.duplicated(keys, keep=False)]
    if not ties.empty and ties.groupby(keys, dropna=False)["value"].nunique(dropna=False).gt(1).any():
        raise NuclearPreparationError(f"{spec.alias}: valeurs contradictoires pour une meme identite PIT.")
    selected = eligible.sort_values(keys, kind="stable").drop_duplicates("timestamp", keep="last")
    series = pd.Series(selected["value"].to_numpy(), index=pd.DatetimeIndex(selected["timestamp"]), name=spec.alias)
    series.index.name = "timestamp"
    return series, {
        "alias": spec.alias, "raw_rows": len(frame), "eligible_rows": len(eligible),
        "selected_rows": len(selected), "delivery_column": delivery_col,
        "availability_column": snapshot_col, "revision_column": revision_col,
        "value_column": value_col, "cutoff_violations": 0,
        "first_selected_utc": None if series.empty else series.index[0].isoformat(),
        "last_selected_utc": None if series.empty else series.index[-1].isoformat(),
        "selection_rule": "latest snapshot/revision <= min(civil D-1 08:00, runtime_as_of)",
        "fill_or_interpolation": "none in selection; upstream derived values retain their source audit",
    }


def _require_complete(series: pd.Series, expected: pd.DatetimeIndex, *, alias: str, timezone: str) -> None:
    finite = np.isfinite(series.reindex(expected).to_numpy(dtype=float))
    missing = expected[~finite]
    if len(missing):
        days = missing.tz_convert(timezone).strftime("%Y-%m-%d").unique()
        preview = ", ".join(timestamp.isoformat() for timestamp in missing[:6])
        raise NuclearPreparationError(
            f"{alias}: historique/futur PIT incomplet: {len(missing)}/{len(expected)} heures "
            f"absentes ou non finies sur {len(days)} jour(s); premiers jours: {', '.join(days[:6])}; "
            f"premieres heures UTC: {preview}. Resynchroniser les entrees nucleaires auditees; "
            "aucune valeur observee ni ancien vintage ne sera substitue."
        )


def _select_target_context(
    zone: ZoneConfig,
    config: Mapping[str, Any],
    config_dir: Path,
    required_hours: pd.DatetimeIndex,
    delivery_day: pd.Timestamp,
) -> tuple[pd.Series, dict[str, Any]]:
    """Keep exactly the historical prices consumed by the first-to-last run.

    The full operational cache often spans four years. Unused earlier prices
    must not change a covariate coverage denominator for this 730-day replay.
    The first origin still has all ``context_length`` physical observations.
    """
    if not zone.target.file:
        raise NuclearPreparationError("target: un snapshot fichier canonique explicite est requis.")
    resolution = str(config.get("data", {}).get("target_input_resolution", "hourly"))
    if resolution != "hourly":
        raise NuclearPreparationError("target: le laboratoire nucleaire exige une cible canonique horaire.")
    context_raw = config.get("model", {}).get("context_length", 2048)
    if isinstance(context_raw, bool) or not isinstance(context_raw, (int, np.integer)) or context_raw < 1:
        raise NuclearPreparationError("model.context_length doit etre un entier positif.")
    source_path = resolve_path(zone.target.file, config_dir)
    source_sha = _sha256(source_path)
    # The reusable reader consolidates duplicate dates. Validate the canonical
    # physical identity before calling it, so it cannot silently average two
    # conflicting prices or infer a timezone for an ambiguous autumn label.
    raw_frame = (pd.read_parquet(source_path) if source_path.suffix.lower() in {".parquet", ".pq"}
                 else pd.read_csv(source_path, low_memory=False))
    if not isinstance(raw_frame.index, pd.RangeIndex):
        raw_frame = raw_frame.reset_index()
    timestamp_col = first_matching_column(raw_frame, zone.target.timestamp_col,
                                          ("timestamp", "datetime", "date_time", "date", "time", "index", "unnamed: 0"))
    if timestamp_col is None:
        raise NuclearPreparationError("target: colonne temporelle du cache canonique introuvable.")
    raw_index = _aware_utc(raw_frame[timestamp_col], label="target/timestamp")
    if raw_index.has_duplicates or not raw_index.equals(raw_index.floor("h")):
        raise NuclearPreparationError("target: grille physique horaire non unique ou non alignee.")
    raw = read_series_file(source_path, zone.target, zone.timezone)
    if _sha256(source_path) != source_sha:
        raise NuclearPreparationError("target: le snapshot a change pendant sa lecture.")
    index = _aware_utc(raw.index, label="target/timestamp")
    if index.has_duplicates or not index.equals(index.floor("h")):
        raise NuclearPreparationError("target: grille physique horaire non unique ou non alignee.")
    target_index = pd.date_range(
        required_hours[0] - pd.Timedelta(hours=int(context_raw)),
        delivery_day.tz_localize(zone.timezone).tz_convert("UTC"),
        freq="h", inclusive="left",
    )
    selected = pd.Series(raw.to_numpy(), index=index, name="target").reindex(target_index)
    _require_complete(selected, target_index, alias="target", timezone=zone.timezone)
    return selected, {
        "source_path": str(source_path), "source_sha256": source_sha,
        "selection_rule": "configured replay days plus context_length physical hours before first origin; no future target",
        "context_length": int(context_raw), "source_rows": len(raw), "selected_rows": len(selected),
        "first_selected_utc": selected.index[0].isoformat(),
        "last_selected_utc": selected.index[-1].isoformat(),
    }


def prepare_nuclear_zone_data(
    zone: ZoneConfig,
    config: Mapping[str, Any],
    config_dir: Path,
    output_dir: Path,
) -> ZoneData:
    """Prepare every enabled PIT covariate with an audited civil cutoff.

    Input files and the caller's configuration remain unchanged. Only selected
    UTC CSVs and their audit are written below the isolated ``output_dir``.
    """
    cloned = deepcopy(dict(config))
    runtime_raw = cloned.get("data", {}).get("runtime_as_of")
    if runtime_raw in (None, ""):
        raise NuclearPreparationError("data.runtime_as_of explicite avec fuseau est requis.")
    runtime = _aware_utc([runtime_raw], label="runtime_as_of")[0]
    if str(cloned.get("data", {}).get("forecast_origin_local_time", "08:00")) != "08:00":
        raise NuclearPreparationError("Le laboratoire nucleaire exige le cutoff civil D-1 08:00.")
    if str(cloned.get("data", {}).get("revision_policy", "latest_before_asof")) != "latest_before_asof":
        raise NuclearPreparationError("revision_policy=latest_before_asof requis.")
    delivery_day = runtime.tz_convert(zone.timezone).tz_localize(None).normalize() + pd.Timedelta(days=1)
    start_day = delivery_day - pd.Timedelta(days=730)
    experiment = cloned.get("nuclear_experiment", {})
    if experiment.get("mode", "full") == "incremental":
        raw_start = experiment.get("raw_history_start_day")
        if raw_start is None:
            from .nuclear_incremental import prepare_incremental_settings
            _, _, raw_start = prepare_incremental_settings(cloned, delivery_day.date())
        start_day = pd.Timestamp(raw_start)
        if (pd.isna(start_day) or start_day.tzinfo is not None or start_day != start_day.normalize()
                or not delivery_day - pd.Timedelta(days=1095) <= start_day <= delivery_day - pd.Timedelta(days=730)):
            raise NuclearPreparationError("Le support historique incremental doit couvrir 730 a 1095 jours civils.")
    expected = pd.date_range(
        start_day.tz_localize(zone.timezone).tz_convert("UTC"),
        (delivery_day + pd.Timedelta(days=1)).tz_localize(zone.timezone).tz_convert("UTC"),
        freq="h", inclusive="left",
    )
    output = Path(output_dir).resolve()
    selected_root = output / "selected_pit"
    prepared: list[tuple[SeriesSpec, Path, pd.Series, dict[str, Any]]] = []
    for alias, spec in zone.covariates.items():
        if not spec.enabled:
            continue
        if not is_pit_spec(spec, cloned):
            raise NuclearPreparationError(f"{alias}: toutes les entrees nucleaires doivent etre PIT auditees.")
        source_path = resolve_pit_path(spec, cloned, Path(config_dir))
        input_sha = _sha256(source_path)
        selected, audit = _strict_selection(
            pd.read_parquet(source_path), spec, timezone=zone.timezone, runtime_as_of=runtime,
        )
        if _sha256(source_path) != input_sha:
            raise NuclearPreparationError(f"{alias}: le fichier PIT a change pendant sa lecture.")
        _require_complete(selected, expected, alias=alias, timezone=zone.timezone)
        safe_alias = re.sub(r"[^A-Za-z0-9_-]", "_", alias)
        suffix = hashlib.sha256(alias.encode("utf-8")).hexdigest()[:10]
        selected_path = selected_root / f"{safe_alias}_{suffix}.csv"
        if selected_path == source_path:
            raise NuclearPreparationError(f"{alias}: la sortie ne doit jamais remplacer la source PIT.")
        audit.update(source_path=str(source_path), source_sha256=input_sha,
                     expected_hours=len(expected), covered_hours=len(expected), missing_hours=0,
                     selected_path=str(selected_path))
        prepared.append((spec, selected_path, selected, audit))
    if not prepared:
        raise NuclearPreparationError("Aucune covariable PIT active pour le laboratoire nucleaire.")
    selected_target, target_audit = _select_target_context(
        zone, cloned, Path(config_dir), expected, delivery_day,
    )
    # Preflight all sources BEFORE writing any selected input or model request.
    selected_root.mkdir(parents=True, exist_ok=True)
    covariates: dict[str, SeriesSpec] = {}
    audits = []
    for spec, path, selected, audit in prepared:
        selected.rename("value").to_csv(path, index=True, index_label="timestamp")
        audit["selected_sha256"] = _sha256(path)
        covariates[spec.alias] = replace(
            spec, file=str(path), timestamp_col="timestamp", value_col="value",
            fill_method="none", fill_limit=0,
        )
        zone_raw = cloned.get("zones", {}).get(zone.zone, {})
        if spec.alias in zone_raw.get("covariates", {}):
            zone_raw["covariates"][spec.alias].update(
                file=str(path), timestamp_col="timestamp", value_col="value", fill_method="none", fill_limit=0,
            )
        audits.append(audit)
    target_path = selected_root / "target_context.csv"
    if target_path == Path(target_audit["source_path"]):
        raise NuclearPreparationError("target: la sortie ne doit jamais remplacer le snapshot source.")
    selected_target.rename("value").to_csv(target_path, index=True, index_label="timestamp")
    target_audit.update(selected_path=str(target_path), selected_sha256=_sha256(target_path))
    isolated_target = replace(zone.target, file=str(target_path), timestamp_col="timestamp", value_col="value")
    zone_raw = cloned.get("zones", {}).get(zone.zone, {})
    if "target" in zone_raw:
        zone_raw["target"].update(file=str(target_path), timestamp_col="timestamp", value_col="value")
    cloned.setdefault("data", {}).update(require_all_covariates=True, fail_below_minimum_coverage=True)
    isolated_zone = replace(zone, target=isolated_target, covariates=covariates)
    result = prepare_zone_data(isolated_zone, cloned, Path(config_dir), False, output)
    payload = {
        "schema_version": 1, "zone": zone.zone, "timezone": zone.timezone,
        "runtime_as_of_utc": runtime.isoformat(), "forecast_origin_local_time": "08:00",
        "delivery_day": delivery_day.date().isoformat(), "required_start_day": start_day.date().isoformat(),
        "required_end_day": delivery_day.date().isoformat(), "expected_hours": len(expected),
        "selection_timezone_semantics": "civil calendar D-1, then localise 08:00; never subtract physical 24h",
        "provider_revision_timestamp_available": False, "production_pit_evidence": False,
        "target_context": target_audit, "inputs": audits,
    }
    audit_path = output / "pit_selection_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result.diagnostics["nuclear_pit_selection"] = {"audit_path": str(audit_path), **payload}
    return result
