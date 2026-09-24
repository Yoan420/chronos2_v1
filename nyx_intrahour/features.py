"""Causal hourly summaries of declared native quarter-hour forecast vintages.

This research feature builder neither acquires data nor interpolates it. A source
declaration is required, but cannot itself establish production point-in-time
evidence. All delivery keys and output indices are physical instants in UTC.
"""
from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd


TIMEZONE = "Europe/Paris"
_TIME_COLUMNS = ("value_time_utc", "snapshot_time_utc", "revision_time_utc")
_REQUIRED_COLUMNS = ("source_alias", *_TIME_COLUMNS, "value")
_FEATURE_NAMES = ("mean_gw", "std_gw", "range_gw", "ramp_gw_per_hour",
                  "max_deviation_gw", "min_deviation_gw")
_COUNTERS = ("expected_hours", "complete_hours", "varying_hours", "expected_quarters",
             "selected_quarters", "absent_quarters", "nonfinite_quarters",
             "missing_quarters", "late_rows_excluded")


class IntrahourFeatureError(ValueError):
    """The source or vintage table violates the intrahour research contract."""


def _sources(sources: list[dict]) -> list[dict[str, Any]]:
    if not isinstance(sources, list) or not sources:
        raise IntrahourFeatureError("sources must be a nonempty list of source manifests.")
    result: list[dict[str, Any]] = []
    aliases: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise IntrahourFeatureError("Each source manifest must be a mapping.")
        alias = source.get("alias")
        if not isinstance(alias, str) or re.fullmatch(r"[a-z0-9_]+", alias) is None:
            raise IntrahourFeatureError("Source alias must match [a-z0-9_]+.")
        if alias in aliases:
            raise IntrahourFeatureError(f"Duplicate source alias: {alias}.")
        aliases.add(alias)
        if source.get("zone") not in {"BE", "DE", "FR", "NL", "ES"}:
            raise IntrahourFeatureError(f"{alias}: unsupported source zone.")
        if source.get("driver") not in {"residual_load", "load", "wind", "solar", "nuclear"}:
            raise IntrahourFeatureError(f"{alias}: only forecast fundamental drivers are accepted.")
        if source.get("unit") not in {"GW", "MW"}:
            raise IntrahourFeatureError(f"{alias}: source unit must be GW or MW.")
        if not isinstance(source.get("series"), str) or not source["series"].strip():
            raise IntrahourFeatureError(f"{alias}: nonempty series identifier required.")
        if source.get("native_resolution_minutes") != 15:
            raise IntrahourFeatureError(f"{alias}: native_resolution_minutes must be 15; hourly sources are not accepted.")
        if source.get("is_forecast") is not True:
            raise IntrahourFeatureError(f"{alias}: is_forecast must be True.")
        if source.get("interpolation") != "none":
            raise IntrahourFeatureError(f"{alias}: interpolation must be 'none'.")
        evidence = source.get("native_resolution_evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise IntrahourFeatureError(f"{alias}: nonempty native_resolution_evidence required.")
        result.append({key: source[key] for key in (
            "alias", "zone", "driver", "unit", "series", "native_resolution_minutes",
            "is_forecast", "interpolation", "native_resolution_evidence")})
        result[-1]["native_resolution_minutes"] = 15
    return result


def _day(value: str, label: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise IntrahourFeatureError(f"{label} must be a YYYY-MM-DD civil date.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise IntrahourFeatureError(f"{label} must be a valid civil date.") from exc


def _vintages(vintages: pd.DataFrame, aliases: set[str]) -> tuple[pd.DataFrame, int]:
    if not isinstance(vintages, pd.DataFrame):
        raise IntrahourFeatureError("vintages must be a pandas DataFrame.")
    if vintages.columns.has_duplicates:
        raise IntrahourFeatureError("Duplicate vintage column names are ambiguous.")
    missing = set(_REQUIRED_COLUMNS).difference(vintages.columns)
    if missing:
        raise IntrahourFeatureError(f"Missing vintage columns: {', '.join(sorted(missing))}.")
    frame = vintages.loc[:, _REQUIRED_COLUMNS].copy()
    if not frame.source_alias.map(lambda value: isinstance(value, str) and value in aliases).all():
        raise IntrahourFeatureError("Every vintage source_alias must identify a declared source.")
    for column in _TIME_COLUMNS:
        if frame.empty:
            frame[column] = pd.Series(index=frame.index, dtype="datetime64[ns, UTC]")
            continue
        try:
            index = pd.DatetimeIndex(frame[column])
        except (TypeError, ValueError) as exc:
            raise IntrahourFeatureError(f"{column}: valid timezone-aware timestamps required.") from exc
        if index.tz is None or index.hasnans:
            raise IntrahourFeatureError(f"{column}: timezone-aware, nonmissing timestamps required.")
        frame[column] = index.tz_convert("UTC")
    value_time = frame.value_time_utc
    if not value_time.eq(value_time.dt.floor("15min")).all():
        raise IntrahourFeatureError("value_time_utc must be aligned to exact quarters 00/15/30/45.")
    try:
        numeric = pd.to_numeric(frame.value, errors="raise")
        if np.iscomplexobj(numeric):
            raise ValueError("Complex values are not forecast levels.")
        frame["value"] = numeric.astype(float)
    except (TypeError, ValueError) as exc:
        raise IntrahourFeatureError("Forecast values must be numeric or missing.") from exc
    identity = ["source_alias", *_TIME_COLUMNS]
    tied = frame.loc[frame.duplicated(identity, keep=False)]
    if not tied.empty and tied.groupby(identity, dropna=False).value.nunique(dropna=False).gt(1).any():
        raise IntrahourFeatureError("Conflicting values for the same source, quarter and vintage timestamps.")
    before = len(frame)
    frame = frame.drop_duplicates(identity, keep="first")
    return frame, before-len(frame)


def build_hourly_features(
    vintages: pd.DataFrame,
    sources: list[dict],
    start_day: str,
    end_day: str,
) -> tuple[pd.DataFrame, dict]:
    """Return hourly forecast features and a JSON-safe retrospective audit.

    Civil delivery dates are inclusive. For each quarter of D, both vintage
    timestamps must be <= D-1 08:00 Europe/Paris. Choose the latest revision,
    then snapshot timestamp; a latest NaN/inf never falls back to an older value.
    Exact duplicate identities with conflicting values raise; identical copies
    are deduplicated. Aware timestamps are normalized to UTC without interpreting
    naive timestamps. Missing or nonfinite quarters invalidate all six features
    for that source/hour. ``std_gw`` uses ddof=0; signed min/max deviations are
    measured from the hourly mean; ramp is (last-first)/0.75 hours.

    Top-level expected_hours/expected_quarters describe the shared physical
    grid, complete_hours requires all sources, and missing_quarters sums across
    sources. Per-source counters expose the corresponding denominators.
    """
    declarations = _sources(sources)
    first, last = _day(start_day, "start_day"), _day(end_day, "end_day")
    if first > last:
        raise IntrahourFeatureError("start_day must not be after end_day.")
    frame, duplicate_count = _vintages(vintages, {s["alias"] for s in declarations})
    frame["_day"] = frame.value_time_utc.dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")
    in_window = frame._day.between(start_day, end_day)
    outside_rows = int((~in_window).sum())
    groups = {key: group for key, group in frame.loc[in_window].groupby(["source_alias", "_day"], sort=False)}
    by_source = {s["alias"]: dict.fromkeys(_COUNTERS, 0) for s in declarations}
    daily_audits: list[dict] = []
    outputs: list[pd.DataFrame] = []
    day = first
    while day <= last:
        day_text = day.isoformat()
        # Localize each civil boundary separately: adding 24 hours to an aware
        # midnight is wrong on a 23/25-hour delivery day.
        begin = pd.Timestamp(day).tz_localize(TIMEZONE).tz_convert("UTC")
        finish = pd.Timestamp(day+timedelta(days=1)).tz_localize(TIMEZONE).tz_convert("UTC")
        quarters = pd.date_range(begin, finish, freq="15min", inclusive="left")
        hours = pd.date_range(begin, finish, freq="1h", inclusive="left", name="timestamp_utc")
        cutoff = (pd.Timestamp(day-timedelta(days=1))+pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")
        output = pd.DataFrame(index=hours)
        output["forecast_origin_utc"] = cutoff
        daily = {"day": day_text, "forecast_origin_utc": cutoff.isoformat(),
                 "expected_hours": len(hours), "expected_quarters": len(quarters), "sources": {}}
        complete_all = np.ones(len(hours), dtype=bool)
        for source in declarations:
            alias = source["alias"]
            part = groups.get((alias, day_text))
            selected = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
            late = 0
            if part is not None:
                admissible = part.snapshot_time_utc.le(cutoff) & part.revision_time_utc.le(cutoff)
                late = int((~admissible).sum())
                chosen = part.loc[admissible].sort_values(["revision_time_utc", "snapshot_time_utc"], kind="stable").drop_duplicates("value_time_utc", keep="last")
                selected = chosen.set_index("value_time_utc").value
            values = selected.reindex(quarters).to_numpy(float)
            if source["unit"] == "MW":
                values = values/1000.0
            blocks = values.reshape(len(hours), 4)
            finite = np.isfinite(blocks)
            complete = finite.all(axis=1)
            feature_values = np.full((len(hours), len(_FEATURE_NAMES)), np.nan)
            valid = blocks[complete]
            if len(valid):
                mean = valid.mean(axis=1)
                low, high = valid.min(axis=1), valid.max(axis=1)
                feature_values[complete] = np.column_stack((mean, valid.std(axis=1, ddof=0),
                                                           high-low, (valid[:, -1]-valid[:, 0])/0.75,
                                                           high-mean, low-mean))
            for i, name in enumerate(_FEATURE_NAMES):
                output[f"feature_intrahour_{alias}__{name}"] = feature_values[:, i]
            output[f"data_{alias}__complete"] = complete
            complete_all &= complete
            counts = {"expected_hours": len(hours), "complete_hours": int(complete.sum()),
                      "varying_hours": int((feature_values[:, 2]>0).sum()),
                      "expected_quarters": len(quarters), "selected_quarters": len(selected),
                      "absent_quarters": len(quarters)-len(selected),
                      "nonfinite_quarters": int((~np.isfinite(selected.to_numpy(float))).sum()),
                      "missing_quarters": int((~finite).sum()), "late_rows_excluded": late}
            daily["sources"][alias] = counts
            for key in _COUNTERS:
                by_source[alias][key] += counts[key]
        daily["complete_hours"] = int(complete_all.sum())
        daily_audits.append(daily)
        outputs.append(output)
        day += timedelta(days=1)
    result = pd.concat(outputs)
    audit = {"schema_version": 1, "identity": "nyx_intrahour_retrospective",
             "production_pit_evidence": False, "retrospective": True,
             "native_resolution_independently_verified": False,
             "start_day": start_day, "end_day": end_day, "days": len(daily_audits),
             "timezone": TIMEZONE, "cutoff_rule": "D-1 08:00 Europe/Paris",
             "selection_order": ["revision_time_utc", "snapshot_time_utc"],
             "expected_hours": len(result), "complete_hours": sum(d["complete_hours"] for d in daily_audits),
             "expected_quarters": len(result)*4,
             "expected_source_quarters": len(result)*4*len(declarations),
             "missing_quarters": sum(s["missing_quarters"] for s in by_source.values()),
             "late_rows_excluded": sum(s["late_rows_excluded"] for s in by_source.values()),
             "identical_duplicates_removed": duplicate_count, "input_rows": len(vintages),
             "out_of_window_rows": outside_rows,
             "sources": declarations, "by_source": by_source, "by_day": daily_audits,
             "feature_contract": {"output_unit": "GW", "std_ddof": 0,
                                  "ramp_formula": "(q45-q00)/0.75 hours",
                                  "deviation_reference": "arithmetic hourly mean",
                                  "incomplete_policy": "all six features NaN unless four finite native quarters",
                                  "flat_profile_policy": "valid; varying_hours counts strict range >0",
                                  "interpolation": "none"}}
    return result, audit
