"""Descriptive physical counterexamples and stratification, never causality.

Thresholds are fixed from the first warmup calendar days without using prices.
All comparisons exclude warmup and live rows. This module does not fit a model,
alter predictions, certify historical publication, or use realised generation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["zone", "timestamp_utc", "forecast_origin_utc"]
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
PHYSICAL = {
    "solar_local_gw": "solarx_local_solar_gw", "solar_peer_gw": "solarx_peer_solar_gw",
    "solar_drop_gwph": "solarx_local_solar_drop_1h_gwph", "peer_solar_drop_gwph": "solarx_peer_solar_drop_1h_gwph",
    "residual_gw": "solarx_local_residual_gw", "wind_gw": "solarx_local_wind_gw",
    "pressure_proxy": "solarx_local_pressure_proxy",
}


def _clean(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _aware(series, name):
    if series.isna().any() or any(pd.Timestamp(v).tzinfo is None for v in series):
        raise ValueError(f"{name}: complete timezone-aware timestamps required")
    return pd.to_datetime(series, utc=True)


def _number(frame, name):
    if name not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[name], errors="raise").replace([np.inf, -np.inf], np.nan)


def _median(values):
    known = values.dropna()
    return float(known.median()) if len(known) else np.nan


def _records(frame):
    columns = ["zone", "delivery_start_local", "delivery_end_local", "timestamp_utc", "forecast_origin_utc",
               "season", "hour", *PHYSICAL, "actual", "baseline", "candidate", "risk_probability", "alert", "drop_regime"]
    return frame[[c for c in columns if c in frame]].to_dict("records")


def analyse(panel: pd.DataFrame, predictions: pd.DataFrame, config: dict) -> dict:
    """Return JSON-safe descriptive analysis from saved hourly artefacts only."""
    required = {*KEYS, "sample", "actual", "forecast"}
    if not isinstance(panel, pd.DataFrame) or panel.columns.has_duplicates or required.difference(panel):
        raise ValueError("Unique-column panel with sample, physical identity, actual and baseline required")
    # Explicit sample membership is required; no potentially live row is
    # silently reclassified from its date or from the presence of an actual.
    data = panel.loc[panel["sample"].eq("evaluation")].copy().reset_index(drop=True)
    if data.empty:
        return {"status": "insufficient_evaluation_data", "event_rows": [], "counterexamples": {},
                "regime_rates": [], "warmup_thresholds": [], "matched_strata": {"summary": [], "rows": []}}
    for key in KEYS[1:]:
        data[key] = _aware(data[key], key)
    if data.duplicated(KEYS).any() or not data.zone.isin(TIMEZONES).all():
        raise ValueError("Unique supported zone/origin/delivery identities required")
    if not data.timestamp_utc.eq(data.timestamp_utc.dt.floor("h")).all():
        raise ValueError("Physical hourly timestamps required")
    for key, name in PHYSICAL.items():
        data[key] = _number(data, name)
    data["actual"], data["baseline"] = _number(data, "actual"), _number(data, "forecast")
    data["candidate"], data["risk_probability"], data["alert"] = np.nan, np.nan, False
    primary = config.get("primary_variant", "governed")
    if predictions is not None and not predictions.empty:
        if {*KEYS, "variant", "candidate_forecast"}.difference(predictions) or predictions.columns.has_duplicates:
            raise ValueError("Predictions require unique columns, identity, variant and candidate_forecast")
        chosen = predictions.loc[predictions.variant.eq(primary)].copy()
        for key in KEYS[1:]:
            chosen[key] = _aware(chosen[key], "prediction/"+key)
        if chosen.duplicated(KEYS).any():
            raise ValueError("Duplicate primary-variant physical identities")
        fields = [c for c in ("candidate_forecast", "risk_probability", "alert", "actual", "forecast") if c in chosen]
        # Live predictions are never consulted by the matching/analysis.
        chosen = chosen.merge(data[KEYS], on=KEYS, how="inner", validate="one_to_one")
        matched = data[KEYS].merge(chosen[KEYS+fields], on=KEYS, how="left", validate="one_to_one")
        for supplied, expected in (("actual", "actual"), ("forecast", "baseline")):
            if supplied in matched:
                value = _number(matched, supplied)
                known = value.notna()
                if not np.allclose(value[known], data.loc[known, expected], equal_nan=True):
                    raise ValueError(f"Primary predictions disagree with panel {supplied}")
        data["candidate"] = _number(matched, "candidate_forecast")
        data["risk_probability"] = _number(matched, "risk_probability")
        if "alert" in matched:
            data["alert"] = matched.alert.fillna(False).astype(bool)
    local = data.timestamp_utc.dt.tz_convert("Europe/Paris")
    data["day"] = local.dt.strftime("%Y-%m-%d")
    data["hour"] = local.dt.hour
    data["season"] = local.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    data["delivery_start_local"] = [t.tz_convert(TIMEZONES[z]).isoformat() for z, t in zip(data.zone, data.timestamp_utc)]
    data["delivery_end_local"] = [(t+pd.Timedelta(hours=1)).tz_convert(TIMEZONES[z]).isoformat() for z, t in zip(data.zone, data.timestamp_utc)]
    warmup_days = int(config.get("minimum_training_days", 120))
    if warmup_days < 1:
        raise ValueError("Positive warmup days required")
    start = pd.Timestamp(data.day.min())
    end = start+pd.Timedelta(days=warmup_days-1)
    data["is_warmup"] = pd.to_datetime(data.day).le(end)
    thresholds, threshold_records = {}, []
    for zone in sorted(data.zone.unique()):
        warm = data.loc[data.zone.eq(zone) & data.is_warmup]
        positive = warm.loc[warm.solar_drop_gwph.gt(0), "solar_drop_gwph"].dropna()
        row = {"zone": zone, "start_day": str(start.date()), "end_day": str(end.date()),
               "n_rows": len(warm), "n_positive_drop": len(positive),
               "solar_drop_q90_gwph": float(positive.quantile(.9)) if len(positive) else np.nan,
               "residual_median_gw": _median(warm.residual_gw), "wind_median_gw": _median(warm.wind_gw),
               "pressure_median_proxy": _median(warm.pressure_proxy)}
        thresholds[zone] = row
        threshold_records.append(row)
    data["drop_threshold"] = data.zone.map({z: row["solar_drop_q90_gwph"] for z, row in thresholds.items()})
    data["drop_regime"] = np.select(
        [data.solar_drop_gwph.isna(), data.solar_drop_gwph.le(1e-9), data.drop_threshold.isna(),
         data.solar_drop_gwph.ge(data.drop_threshold)],
        ["unknown", "no_drop", "unknown", "large_drop"], default="smaller_positive_drop")
    data["risk_regime"] = np.select([data.risk_probability.isna(), data.alert], ["unavailable", "alert"], default="no_alert")
    data["spike"] = data.actual.ge(float(config.get("business_spike", 300.))).astype(float).where(data.actual.notna())
    comparison = data.loc[~data.is_warmup].copy()
    rate_rows = []
    for keys, part in comparison.groupby(["zone", "season", "hour", "drop_regime", "risk_regime"], sort=True):
        known = part.spike.dropna()
        rate_rows.append(dict(zip(("zone", "season", "hour", "drop_regime", "risk_regime"), keys),
                             n=len(part), n_known_prices=len(known), n_spikes=int(known.sum()),
                             spike_rate=float(known.mean()) if len(known) else np.nan,
                             mean_residual_gw=part.residual_gw.mean(), mean_wind_gw=part.wind_gw.mean(),
                             mean_pressure_proxy=part.pressure_proxy.mean()))
    control_specs = (("residual_gw", "residual_median_gw"), ("wind_gw", "wind_median_gw"), ("pressure_proxy", "pressure_median_proxy"))
    for control, threshold_name in control_specs:
        median = comparison.zone.map({z: row[threshold_name] for z, row in thresholds.items()})
        comparison[control+"_stratum"] = np.where(comparison[control].isna() | median.isna(), "unknown",
                                                   np.where(comparison[control].le(median), "low", "high"))
    strata = ["zone", "season", "hour", *(control+"_stratum" for control, _ in control_specs)]
    valid = comparison.drop_regime.ne("unknown") & comparison.spike.notna()
    for control, _ in control_specs:
        valid &= comparison[control+"_stratum"].ne("unknown")
    matched_rows = []
    for keys, part in comparison.loc[valid].groupby(strata, sort=True):
        large = part.loc[part.drop_regime.eq("large_drop")]
        other = part.loc[~part.drop_regime.eq("large_drop")]
        if large.empty or other.empty:
            continue
        row = {**dict(zip(strata, keys)), "n_large_drop": len(large), "n_other": len(other),
               "n_spikes_large_drop": int(large.spike.sum()), "n_spikes_other": int(other.spike.sum()),
               "spike_rate_large_drop": large.spike.mean(), "spike_rate_other": other.spike.mean(),
               "overlap_weight": min(len(large), len(other))}
        row["rate_difference"] = row["spike_rate_large_drop"]-row["spike_rate_other"]
        for control, _ in control_specs:
            row["mean_"+control+"_large_drop"] = large[control].mean()
            row["mean_"+control+"_other"] = other[control].mean()
        matched_rows.append(row)
    summaries = []
    for zone in sorted(data.zone.unique()):
        rows = [row for row in matched_rows if row["zone"] == zone]
        weight = sum(row["overlap_weight"] for row in rows)
        left = sum(row["overlap_weight"]*row["spike_rate_large_drop"] for row in rows)/weight if weight else np.nan
        right = sum(row["overlap_weight"]*row["spike_rate_other"] for row in rows)/weight if weight else np.nan
        summaries.append({"zone": zone, "n_matched_strata": len(rows), "overlap_weight": weight,
                          "n_large_drop": sum(row["n_large_drop"] for row in rows),
                          "n_other": sum(row["n_other"] for row in rows),
                          "weighted_spike_rate_large_drop": left, "weighted_spike_rate_other": right,
                          "weighted_rate_difference": left-right})
    without_spike = comparison.loc[comparison.drop_regime.eq("large_drop") & comparison.spike.eq(0)]
    without_drop = comparison.loc[comparison.drop_regime.eq("no_drop") & comparison.spike.eq(1)]
    examples = lambda frame, column: frame.sort_values(["zone", column, "timestamp_utc"], ascending=[True, False, True]).groupby("zone", sort=True).head(3)
    event = data.loc[data.zone.isin(["DE", "BE"]) & data.day.eq("2026-09-14") & data.hour.between(14, 22)].sort_values(["zone", "timestamp_utc"])
    result = {
        "status": "complete", "protocol": {
            "descriptive_only": True, "causal_identification": False, "model_fit_performed": False,
            "primary_variant": primary, "warmup_days": warmup_days, "threshold_quantile": .9,
            "threshold_definition": "Per-zone q90 of strictly positive forecast solar 1h drops in first warmup calendar days; no prices used",
            "comparison_scope": "Post-warmup rows explicitly labelled sample=evaluation; live rows excluded everywhere",
            "matched_strata": "Exact local hour and meteorological season plus local residual-load, wind, pressure below/above own warmup median; only strata containing both large-drop and other observations",
            "weighting": "Per-stratum min(n_large_drop,n_other), same weights for both rates; descriptive common support, not propensity matching or a causal effect",
            "counterexample_selection": "Up to three per country; large-drop/no-spike ranked by forecast drop, spike/no-drop by price; counts include every qualifying post-warmup row",
            "event_window": "2026-09-14, DE and BE, start-labelled hourly intervals 14:00 through 22:00 Europe/Berlin or Europe/Brussels, both UTC+02:00",
            "limitations": ["Retrospective association, not causal attribution or independent validation of the initiating event.",
                "No realised solar or solar forecast errors; solar inputs are saved day-ahead forecasts.",
                "Pressure uses incomplete generation/available-Pmax supply proxies, not observed scarcity or network capacity.",
                "No realised cross-border flows, binding network constraints, or certified contemporaneous publication evidence.",
                "Residual load already includes renewable forecasts: this is conditional incremental information, not removal of all confounding.",
                "Coarse control bins retain within-bin differences; warmup medians may extrapolate poorly across seasons; empty common support is reported, not filled.",
                "Hourly prices can average native quarter-hours and do not describe quarter-hour maxima."]},
        "coverage": {"evaluation_rows": len(data), "warmup_rows": int(data.is_warmup.sum()),
                     "post_warmup_rows": len(comparison), "known_post_warmup_prices": int(comparison.spike.notna().sum()),
                     "post_warmup_matching_eligible_rows": int(valid.sum()),
                     "large_drop_without_spike_rows": len(without_spike), "spike_without_drop_rows": len(without_drop),
                     "event_rows": len(event), "primary_candidate_rows": int(data.candidate.notna().sum())},
        "warmup_thresholds": threshold_records, "event_rows": _records(event),
        "counterexamples": {"large_drop_without_spike": _records(examples(without_spike, "solar_drop_gwph")),
                            "spike_without_drop": _records(examples(without_drop, "actual"))},
        "regime_rates": rate_rows, "matched_strata": {"summary": summaries, "rows": matched_rows},
    }
    return _clean(result)
