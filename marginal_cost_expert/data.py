"""Fail-closed, price-free input adapter for the marginal-cost experiment.

No network access, fitting, interpolation or modification of source artifacts.
Output contains only physical inputs and fuel costs, never electricity labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class MarginalCostDataError(ValueError):
    pass


REQUIRED_FIELDS = ("demand_mw", "must_run_mw", "ccgt_available_mw", "ocgt_available_mw",
                   "ttf_eur_mwh_th", "eua_eur_tco2")
OPTIONAL_FIELDS = ("coal_available_mw", "coal_eur_mwh_th")
ALLOWED_INFORMATION = {"day_ahead_forecast", "capacity_forecast", "market_observation_known_before_cutoff"}
FORBIDDEN = re.compile(r"(?i)(?:chronos|mkonline|storm|kalman|residual_corrected|(?:^|[/_.])(?:actual|target|q10|q50|q90)(?:$|[/_.])|power\..*price)")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise MarginalCostDataError("Explicit source path required")
    raw = Path(value)
    return (raw if raw.is_absolute() else root / raw).resolve()


def _utc(values: Any, label: str) -> pd.DatetimeIndex:
    try:
        # Do not let utc=True silently turn ambiguous naive labels into UTC.
        original = pd.DatetimeIndex(values)
        if original.tz is None or original.isna().any():
            raise ValueError("naive or missing timestamps")
        return original.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise MarginalCostDataError(f"{label}: explicit timezone-aware timestamps required") from exc


def expected_hours(start_day: str, end_day: str, timezone: str = "Europe/Paris") -> pd.DatetimeIndex:
    first, last = pd.Timestamp(start_day), pd.Timestamp(end_day)
    if first.tzinfo or last.tzinfo or first != first.normalize() or last != last.normalize() or first > last:
        raise MarginalCostDataError("Inclusive local start/end dates required")
    return pd.date_range(first.tz_localize(timezone), (last + pd.Timedelta(days=1)).tz_localize(timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def _cutoffs(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    days = index.tz_convert(timezone).tz_localize(None).normalize()
    return (days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(timezone).tz_convert("UTC")


def _scale(unit: str, field: str) -> float:
    if field.endswith("_mw"):
        scales = {"MW": 1.0, "GW": 1000.0}
    elif field == "eua_eur_tco2":
        scales = {"EUR/tCO2": 1.0, "EUR/tCO2e": 1.0}
    else:
        scales = {"EUR/MWh_th": 1.0}
    if unit not in scales:
        raise MarginalCostDataError(f"Unit {unit!r} incompatible with {field}")
    return scales[unit]


def _load_source(name: str, spec: Mapping[str, Any], *, root: Path, expected: pd.DatetimeIndex,
                 timezone: str, allow_missing_hours: bool = False) -> tuple[pd.Series, dict[str, Any]]:
    info = spec.get("information_type")
    if info not in ALLOWED_INFORMATION:
        raise MarginalCostDataError(f"{name}: explicit forecast/capacity/fuel information_type required")
    source = _path(spec.get("path"), root)
    value_col = str(spec.get("value_column", "value"))
    identity = " ".join([name, source.name, value_col, str(spec.get("series", ""))])
    if FORBIDDEN.search(identity):
        raise MarginalCostDataError(f"{name}: forbidden electricity-price or model-derived input")
    if info != "market_observation_known_before_cutoff" and re.search(r"(?:^|[._])(?:obs|observed|realized|realised)(?:$|[._])", str(spec.get("series", ""))):
        raise MarginalCostDataError(f"{name}: realised data cannot be declared a forecast")
    audit_path = _path(spec.get("audit_path", str(source) + ".audit.json"), root)
    if not source.is_file() or not audit_path.is_file():
        raise MarginalCostDataError(f"{name}: source or audit absent: {source}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_hash = audit.get("sha256") or audit.get("output_sha256") or audit.get("parquet_sha256")
    actual_hash = _hash(source)
    if not expected_hash or expected_hash != actual_hash:
        raise MarginalCostDataError(f"{name}: source checksum mismatch")
    if spec.get("series"):
        declared = audit.get("series")
        valid = spec["series"] == declared or (isinstance(declared, dict) and spec["series"] in declared.values())
        if not valid:
            raise MarginalCostDataError(f"{name}: series identity absent from source audit")
    if audit.get("cutoff_time") not in (None, "08:00") or audit.get("cutoff_timezone", timezone) != timezone:
        raise MarginalCostDataError(f"{name}: source cutoff contract differs from D-1 08:00 civil")
    if audit.get("causality_violations", 0) != 0:
        raise MarginalCostDataError(f"{name}: source declares causality violations")
    frame = pd.read_parquet(source)
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    timestamp = str(spec.get("timestamp_column", "value_time_utc"))
    snapshot = str(spec.get("snapshot_column", "snapshot_time_utc"))
    revision = str(spec.get("revision_column", "revision_time_utc"))
    needed = {timestamp, snapshot, revision, value_col}
    if needed.difference(frame):
        raise MarginalCostDataError(f"{name}: required PIT columns missing: {sorted(needed.difference(frame))}")
    delivery = _utc(frame[timestamp], name + "/delivery")
    if not delivery.equals(delivery.floor("h")):
        raise MarginalCostDataError(f"{name}: source is not an hourly physical grid")
    snap = _utc(frame[snapshot], name + "/snapshot")
    rev = _utc(frame[revision], name + "/revision")
    cutoff = _cutoffs(delivery, timezone)
    numeric = pd.to_numeric(frame[value_col], errors="coerce")
    if (frame[value_col].notna() & numeric.isna()).any() or np.isinf(numeric.to_numpy(float)).any():
        raise MarginalCostDataError(f"{name}: malformed or infinite physical values")
    raw = pd.DataFrame({"delivery": delivery, "snapshot": snap, "revision": rev,
                        "value": numeric.to_numpy(float)})
    eligible = raw.loc[(snap <= cutoff) & (rev <= cutoff) & delivery.isin(expected)]
    if allow_missing_hours:
        # A truly absent historical day may be kept as NaN. A row supplied as
        # that day's input with only a post-cutoff vintage is a causality error,
        # not an ordinary missing-day exemption. Later revisions alongside a
        # valid earlier vintage remain harmless and are excluded normally.
        late_only = delivery[((snap > cutoff) | (rev > cutoff)) & delivery.isin(expected)].difference(
            pd.DatetimeIndex(eligible.delivery))
        if len(late_only):
            raise MarginalCostDataError(f"{name}: post-cutoff-only vintage; missing-data option cannot bypass causality")
    keys = ["delivery", "snapshot", "revision"]
    if eligible.groupby(keys, dropna=False).value.nunique(dropna=False).gt(1).any():
        raise MarginalCostDataError(f"{name}: conflicting values for one PIT identity")
    selected = eligible.sort_values(keys, kind="stable").drop_duplicates("delivery", keep="last").set_index("delivery")
    values = selected.value.reindex(expected)
    finite = np.isfinite(values.to_numpy(float))
    missing = expected[~finite]
    if len(missing) and not allow_missing_hours:
        raise MarginalCostDataError(f"{name}: {len(missing)} missing/nonfinite PIT hours; first={missing[0]}")
    source_time_column = spec.get("source_value_time_column")
    maximum_age = None
    if source_time_column:
        if source_time_column not in frame:
            raise MarginalCostDataError(f"{name}: source-value timestamp absent")
        times = _utc(frame[source_time_column], name + "/source_value_time")
        ages = (cutoff - times).total_seconds() / 3600
        if ((times > cutoff) & delivery.isin(expected)).any():
            raise MarginalCostDataError(f"{name}: fuel observation after forecast cutoff")
        maximum_age = float(np.max(ages[delivery.isin(expected)]))
        if spec.get("maximum_age_hours") is not None and maximum_age > float(spec["maximum_age_hours"]):
            raise MarginalCostDataError(f"{name}: source observation too stale ({maximum_age:.1f}h)")
    return values.astype(float), {
        "path": str(source), "audit_path": str(audit_path), "sha256": actual_hash,
        "audit_sha256": _hash(audit_path), "series": spec.get("series", audit.get("series")),
        "information_type": info, "unit": spec.get("unit"), "selected_hours": len(values),
        "finite_hours": int(finite.sum()), "missing_hours": len(missing),
        "missing_days": sorted(set(missing.tz_convert(timezone).strftime("%Y-%m-%d"))),
        "maximum_source_age_hours": maximum_age,
        "provider_revision_timestamp_available": audit.get("provider_revision_timestamp_available", False),
        "revision_time_semantics": audit.get("revision_time_semantics", "unspecified"),
        "source_approximation": audit.get("approximation"),
        "source_fill_or_interpolation": audit.get("fill_or_interpolation"),
        "production_pit_evidence": audit.get("production_pit_evidence") is True,
    }


def load_zonal_inputs(config: Mapping[str, Any], *, project_root: str | Path,
                      start_day: str, end_day: str, zones: Sequence[str] | None = None
                      ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble engine fields from named sources and explicit physical sums.

    Configuration shape: ``sources[name]`` defines a PIT file, value column,
    unit and information type; ``zones[FR].inputs[field].terms`` is a list of
    ``{source: name, coefficient: 1}``. Constants require a written assumption.
    Sources default to GW->MW only when GW is explicitly declared. There is
    never inferred scaling, interpolation or an observed-data replacement.
    """
    root = Path(project_root).resolve()
    allow_missing_hours = config.get("allow_missing_hours", False)
    if not isinstance(allow_missing_hours, bool):
        raise MarginalCostDataError("allow_missing_hours must be a boolean")
    timezone = str(config.get("timezone", "Europe/Paris"))
    expected = expected_hours(start_day, end_day, timezone)
    definitions = config.get("sources", {})
    zonal = config.get("zones", {})
    selected_zones = list(zones) if zones is not None else list(zonal)
    if not selected_zones or len(set(selected_zones)) != len(selected_zones):
        raise MarginalCostDataError("Nonempty unique zone list required")
    loaded: dict[str, pd.Series] = {}
    source_audits: dict[str, Any] = {}
    assumptions: list[dict[str, Any]] = []
    output = []
    for zone in selected_zones:
        if zone not in zonal:
            raise MarginalCostDataError(f"{zone}: zone input mapping absent")
        zone_config = zonal[zone]
        inputs = zone_config.get("inputs", {})
        if set(REQUIRED_FIELDS).difference(inputs) or set(inputs).difference(REQUIRED_FIELDS + OPTIONAL_FIELDS):
            raise MarginalCostDataError(f"{zone}: exact physical-field allowlist violated")
        if ("coal_available_mw" in inputs) != ("coal_eur_mwh_th" in inputs):
            raise MarginalCostDataError(f"{zone}: coal capacity requires a coal fuel cost and vice versa")
        basis = zone_config.get("demand_basis", "gross")
        if basis not in {"gross", "residual"}:
            raise MarginalCostDataError(f"{zone}: unknown demand basis")
        subtracted = set()
        if basis == "residual":
            proof = zone_config.get("residual_definition", {})
            if not proof.get("evidence") or not proof.get("already_subtracted_components"):
                raise MarginalCostDataError(f"{zone}: residual-demand formula evidence required")
            subtracted = set(proof["already_subtracted_components"])
        block = pd.DataFrame({"delivery_start_utc": expected, "zone": zone, "demand_basis": basis})
        for field, expression in inputs.items():
            if "constant" in expression:
                if expression.get("terms") or not str(expression.get("assumption", "")).strip():
                    raise MarginalCostDataError(f"{zone}/{field}: constant needs explicit assumption, without sources")
                value = float(expression["constant"]) * _scale(str(expression.get("unit", "")), field)
                if not np.isfinite(value):
                    raise MarginalCostDataError(f"{zone}/{field}: nonfinite assumption")
                block[field] = value
                assumptions.append({"zone": zone, "field": field, "value_engine_units": value,
                                    "reason": expression["assumption"]})
                continue
            terms = expression.get("terms", [])
            if not terms:
                raise MarginalCostDataError(f"{zone}/{field}: physical source terms required")
            result = np.zeros(len(expected))
            for term in terms:
                source_name = term.get("source")
                if source_name not in definitions:
                    raise MarginalCostDataError(f"{zone}/{field}: unknown source {source_name}")
                spec = definitions[source_name]
                if field == "must_run_mw" and spec.get("component") in subtracted:
                    raise MarginalCostDataError(f"{zone}: double subtraction of {spec['component']}")
                if field == "must_run_mw" and spec.get("information_type") == "capacity_forecast":
                    if not term.get("assumption"):
                        raise MarginalCostDataError(f"{zone}: availability is not must-run production; explicit approximation required")
                    assumptions.append({"zone": zone, "field": field, "source": source_name, "reason": term["assumption"]})
                elif term.get("assumption"):
                    assumptions.append({"zone": zone, "field": field, "source": source_name, "reason": term["assumption"]})
                if source_name not in loaded:
                    loaded[source_name], source_audits[source_name] = _load_source(
                        source_name, spec, root=root, expected=expected, timezone=timezone,
                        allow_missing_hours=allow_missing_hours)
                coefficient = float(term.get("coefficient", 1.0))
                if not np.isfinite(coefficient):
                    raise MarginalCostDataError("Nonfinite physical coefficient")
                result += loaded[source_name].to_numpy() * coefficient * _scale(str(spec.get("unit", "")), field)
            block[field] = result
        for field in inputs:
            if field == "demand_mw" and basis == "residual":
                continue
            if (block[field] < 0).any():
                raise MarginalCostDataError(f"{zone}/{field}: negative physical input")
        output.append(block)
    result = pd.concat(output, ignore_index=True).sort_values(["delivery_start_utc", "zone"]).reset_index(drop=True)
    engine_fields = [c for c in REQUIRED_FIELDS + OPTIONAL_FIELDS if c in result]
    incomplete = result[engine_fields].isna().any(axis=1)
    audit = {"schema_version": 1, "start_day": start_day, "end_day": end_day, "zones": selected_zones,
             "physical_hours_per_zone": len(expected), "rows": len(result), "source_audits": source_audits,
             "assumptions": assumptions, "electricity_price_inputs": False, "labels_loaded": False,
             "allow_missing_hours": allow_missing_hours, "incomplete_physical_rows": int(incomplete.sum()),
             "incomplete_days_by_zone": {
                 z: sorted(set(pd.DatetimeIndex(result.loc[incomplete & result.zone.eq(z), "delivery_start_utc"])
                               .tz_convert(timezone).strftime("%Y-%m-%d"))) for z in selected_zones},
             "cutoff": "D-1 08:00 civil", "timezone": timezone,
             "zone_definitions": {z: {"demand_basis": zonal[z].get("demand_basis", "gross"),
                                      "residual_definition": zonal[z].get("residual_definition"),
                                      "excluded_segments": zonal[z].get("excluded_segments", [])} for z in selected_zones},
             "production_pit_evidence": False, "promotion_eligible": False,
             "research_only_reason": "Historical as-of evidence and simplified physical assumptions require independent qualification"}
    return result, audit
