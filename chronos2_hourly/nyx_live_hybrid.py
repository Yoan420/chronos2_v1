"""Isolated rolling NYX/Test2 hybrid for an unchanged DE/NL pair and new BE/FR pair.

DE/NL delegates to the pinned numerical recipes. BE/FR is an explicit separately
trained extension: own-country wind/solar/residual-load forecasts, the other
member of BE<->FR as neighbour, and country_is_fr (never an alias for Germany).
Neither models nor routing calibration are transferred across these pairs.

No I/O, provider call, production activation or process launch occurs here.
Optional checkpoint callbacks belong to the caller, which must verify their
SHA/identity before returning cached data. Prediction-only checkpoints precede
attachment of historical scoring labels. Current-day actuals are forbidden.
Publication PIT and label availability remain caller-owned audit obligations.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Mapping

import numpy as np
import pandas as pd

from . import solar_wind_interaction_features as original_interaction
from . import solar_wind_scarcity_regime as regime
from . import solar_wind_scarcity_hybrid as original_hybrid
from .solar_wind_scarcity_ablation import DAILY_PEAK_COLUMNS


PROTOCOL_VERSION = "nyx_live_hybrid_v1"
PAIRS = (("DE", "NL"), ("BE", "FR"))
QUANTILES = ("q10", "q50", "q90")
TZ = "Europe/Berlin"  # Same civil-day boundaries for these four CWE countries.
EXTENSION_TZ = {"BE": "Europe/Brussels", "FR": "Europe/Paris"}


def _pair(pair):
    result = tuple(pair)
    if result not in PAIRS:
        raise ValueError("Pair must be explicitly ordered DE/NL or BE/FR; no country recoding")
    return result


def _day(value):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("Finite civil origin day required")
    return parsed.date()


def _grid(first, stop):
    return pd.date_range(pd.Timestamp(first).tz_localize(TZ), pd.Timestamp(stop).tz_localize(TZ),
                         freq="h", inclusive="left").tz_convert("UTC")


def _extension_components(frame: pd.DataFrame, zone: str, timezone: str):
    """Explicit BE/FR implementation of the same past-only component formulae."""
    if zone not in EXTENSION_TZ or timezone != EXTENSION_TZ[zone]:
        raise ValueError("BE/FR extension requires its proper country and IANA timezone")
    values = regime._validated_source(frame, zone, timezone)
    index = frame.index
    local_days = index.tz_convert(timezone).date
    days = list(dict.fromkeys(local_days))
    boundaries = np.r_[0, np.flatnonzero(local_days[1:] != local_days[:-1]) + 1, len(index)]
    names = (
        "wind_gw", "solar_gw", "residual_load_gw", "wind_ratio", "solar_ratio", "low_wind",
        "low_solar", "joint_deficit", "residual_stress", "deficit_stress", "rl_ramp_previous",
        "rl_ramp_next1", "rl_ramp_next2", "wind_drop_next2", "solar_drop_next2", "stress_next1",
        "stress_next2", "daily_peak_residual_stress", "daily_peak_deficit_stress",
    )
    result = pd.DataFrame(np.nan, index=index.copy(), columns=names)
    audit = []
    for number, day in enumerate(days):
        first = max(0, number - 365)
        left, right = int(boundaries[number]), int(boundaries[number + 1])
        record = {"delivery_day": str(day), "physical_hours": right - left,
                  "history_complete_days": number - first,
                  "history_first_day": str(days[first]) if number else None,
                  "history_last_day": str(days[number - 1]) if number else None,
                  "status": "excluded_feature_warmup"}
        if number - first < 14:
            audit.append(record)
            continue
        history = values[int(boundaries[first]):left]
        positive_solar = history[history[:, 1] > 0, 1]
        if not len(positive_solar):
            raise ValueError(f"{zone} {day}: no positive solar history")
        wind_scale = float(np.quantile(history[:, 0], .75))
        solar_scale = float(np.quantile(positive_solar, .75))
        r50, r90 = np.quantile(history[:, 2], [.5, .9])
        span = float(r90 - r50)
        if (not np.isfinite([wind_scale, solar_scale, r50, r90, span]).all()
                or min(wind_scale, solar_scale, span) <= 0):
            raise ValueError(f"{zone} {day}: invalid normalization scale")
        wind, solar, residual = values[left:right].T
        low_wind = np.clip(1 - wind / wind_scale, 0., 1.)
        low_solar = np.clip(1 - solar / solar_scale, 0., 1.)
        deficit = low_wind * low_solar
        stress = (residual - r50) / span
        deficit_stress = deficit * np.maximum(stress, 0.)
        positions = np.arange(right - left)
        previous = np.maximum(positions - 1, 0)
        next1 = np.minimum(positions + 1, right - left - 1)
        next2 = np.minimum(positions + 2, right - left - 1)
        matrix = np.column_stack([
            wind, solar, residual, wind / wind_scale, solar / solar_scale,
            low_wind, low_solar, deficit, stress, deficit_stress,
            (residual - residual[previous]) / span, (residual[next1] - residual) / span,
            (residual[next2] - residual) / span, (wind - wind[next2]) / wind_scale,
            (solar - solar[next2]) / solar_scale, stress[next1], stress[next2],
            np.full(len(positions), stress.max()), np.full(len(positions), deficit_stress.max()),
        ])
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite BE/FR components; no imputation")
        result.iloc[left:right] = matrix
        record.update(status="ready", wind_q75_gw=wind_scale, solar_positive_q75_gw=solar_scale,
                      residual_q50_gw=float(r50), residual_q90_gw=float(r90), residual_span_gw=span)
        audit.append(record)
    return result, audit


def build_pair_interaction(covariates: pd.DataFrame, zone: str, timezone: str):
    """One corrector interaction; exact original for DE/NL, explicit extension BE/FR."""
    if zone in ("DE", "NL"):
        return original_interaction.build_interaction(covariates, zone, timezone)
    components, daily = _extension_components(covariates, zone, timezone)
    # The corrector's original interaction is clipped; Test2 separately retains
    # unbounded residual stress. These are intentionally different features.
    score = components.joint_deficit.to_numpy() * np.clip(components.residual_stress.to_numpy(), 0., 1.)
    warmup = components.wind_gw.isna().to_numpy()
    score[warmup] = 0.0
    name = f"{zone.lower()}_low_wind_solar_stress"
    return pd.DataFrame({name: score}, index=covariates.index.copy()), {
        "protocol_version": PROTOCOL_VERSION + "_interaction_BE_FR_extension",
        "zone": zone, "timezone": timezone, "feature_column": name,
        "formula": "low_wind*low_solar*clip((RL-past_q50)/(past_q90-past_q50),0,1)",
        "normalization_rule": "complete forecast civil days [D-365,D); first14 days zero warmup",
        "source_columns_read": [f"{zone.lower()}_{x}" for x in
                                ("wind_generation_fcst", "solar_generation_fcst", "residual_load_fcst")],
        "source_units": "GW", "normalizations": daily,
        "prices_or_targets_used": False, "imputation_performed": False,
        "country_aliasing_performed": False, "pit_publication_evidence_verified": False,
    }


def build_pair_features(covariates: Mapping[str, pd.DataFrame], pair):
    """Return exactly the original DE/NL features or separately defined BE/FR features."""
    pair = _pair(pair)
    if set(covariates) != set(pair):
        raise ValueError("Exactly the two named countries' own frozen covariate frames are required")
    if pair == ("DE", "NL"):
        return regime.build_features(covariates, timezone=TZ)
    components, normalizations = {}, {}
    for zone in pair:
        components[zone], normalizations[zone] = _extension_components(covariates[zone], zone, EXTENSION_TZ[zone])
    index = covariates["BE"].index
    if not index.equals(covariates["FR"].index):
        raise ValueError("BE/FR source grids differ")
    local = index.tz_convert(TZ)
    result = {}
    for zone in pair:
        neighbor_zone = "FR" if zone == "BE" else "BE"
        own, neighbor = components[zone], components[neighbor_zone]
        frame = pd.concat([own.add_prefix("own_"), neighbor.add_prefix("other_")], axis=1)
        frame["regional_max_residual_stress"] = np.maximum(own.residual_stress, neighbor.residual_stress)
        frame["regional_mean_residual_stress"] = (own.residual_stress + neighbor.residual_stress) / 2
        frame["regional_joint_deficit"] = own.joint_deficit * neighbor.joint_deficit
        frame["country_is_fr"] = float(zone == "FR")
        for name, value, period in (("hour", local.hour, 24), ("weekday", local.dayofweek, 7),
                                    ("annual", local.dayofyear - 1, 365.25)):
            frame[name + "_sin"] = np.sin(2 * np.pi * value / period)
            frame[name + "_cos"] = np.cos(2 * np.pi * value / period)
        frame.loc[own.wind_gw.isna(), :] = np.nan
        result[zone] = frame.loc[:, sorted(frame.columns)]
    return result, {
        "protocol_version": PROTOCOL_VERSION + "_features_BE_FR_extension", "pair": list(pair),
        "neighbor_mapping": {"BE": "FR", "FR": "BE"}, "country_indicator": "country_is_fr",
        "feature_names": list(result["BE"].columns), "normalizations": normalizations,
        "country_aliasing_performed": False, "models_transferred_from_DE_NL": False,
        "normalization_rule": "past365 complete civil forecast days, excluding D",
        "same_day_leads_rule": "same daily forecast profile only; never the next civil day",
        "excluded_feature_warmup_days": 14, "stress_upper_clipped": False,
        "pit_publication_evidence_verified": False, "prices_or_targets_used": False,
    }


def _baseline(frame: pd.DataFrame, *, allow_actual: bool):
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Nonempty baseline with unique columns required")
    if not allow_actual and "actual" in frame:
        raise ValueError("Current baseline must not contain actual labels")
    prefixes = [p for p in ("nyx__", "residual_kalman__") if all(p + q in frame for q in QUANTILES)]
    if not prefixes:
        raise ValueError("Final interaction40 baseline q10/q50/q90 required")
    result = frame[[prefixes[0] + q for q in QUANTILES]].copy()
    result.columns = ["nyx__" + q for q in QUANTILES]
    for p in prefixes[1:]:
        if not np.array_equal(result.to_numpy(), frame[[p + q for q in QUANTILES]].to_numpy()):
            raise ValueError("Ambiguous baseline aliases have different quantiles")
    if allow_actual:
        if "actual" not in frame:
            raise ValueError("Historical baseline actual labels required")
        result["actual"] = frame.actual.to_numpy()
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Finite baseline and historical labels required")
    if np.any(np.diff(result[["nyx__" + q for q in QUANTILES]].to_numpy(dtype=float), axis=1) < 0):
        raise ValueError("Baseline quantiles are crossed")
    return result


def _checked_panel(frame, pair, first, stop, *, labels):
    pair = _pair(pair)
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Nonempty country panel required")
    if not labels and "actual" in frame:
        raise ValueError("Actual labels forbidden in current or cached prediction panel")
    required = [f"{prefix}__{q}" for prefix in ("nyx", "test2") for q in QUANTILES]
    required += ["own_joint_deficit", "own_residual_stress", "nyx_daily_peak_gap"]
    if labels:
        required.append("actual")
    if not set(["zone", "fit_origin", *required]).issubset(frame.columns):
        raise ValueError("Missing paired model predictions, physical features, or origin")
    if (not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC"
            or frame.index.hasnans or set(frame.zone) != set(pair)):
        raise ValueError("Exact pair and UTC-indexed predictions required")
    if not pd.MultiIndex.from_arrays([frame.index, frame.zone]).is_unique:
        raise ValueError("Duplicate country/hour identity")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite paired predictions or features")
    for prefix in ("nyx", "test2"):
        if np.any(np.diff(frame[[f"{prefix}__{q}" for q in QUANTILES]].to_numpy(dtype=float), axis=1) < 0):
            raise ValueError("Input quantiles must already be ordered")
    if not frame.own_joint_deficit.between(0, 1).all():
        raise ValueError("Joint deficit must lie in [0,1]")
    days = np.asarray(frame.index.tz_convert(TZ).date)
    origins = pd.DatetimeIndex(pd.to_datetime(frame.fit_origin))
    if origins.hasnans or np.any(np.asarray(origins.date) > days):
        raise ValueError("Model origin follows delivery target")
    expected = _grid(first, stop)
    for zone in pair:
        country = frame.loc[frame.zone == zone]
        if not country.index.equals(expected):
            raise ValueError(f"{zone}: incomplete, unordered or mismatched physical-hour panel")
        d = country.index.tz_convert(TZ).date
        gaps = country.groupby(d)["nyx__q50"].transform("max") - country.nyx__q50
        if not np.allclose(gaps, country.nyx_daily_peak_gap, atol=1e-8, rtol=0):
            raise ValueError("Peak gap must come from the same complete NYX delivery-day curve")
    return days


def _pair_metrics(actual, prediction, zones, pair):
    output = {}
    for name in ("pooled", *pair):
        errors = (prediction - actual) if name == "pooled" else (prediction - actual)[zones == name]
        output[name] = {"rows": len(errors), "mae": float(np.abs(errors).mean()),
                        "rmse": float(np.sqrt(np.square(errors).mean()))}
    if not all(np.isfinite([v["mae"], v["rmse"]]).all() for v in output.values()):
        raise ValueError("Nonfinite validation metrics")
    return output


def _pair_support(mask, days, zones, pair):
    return {"hours": int(mask.sum()), "days": len(set(days[mask])), "by_country": {
        z: {"hours": int((mask & (zones == z)).sum()), "days": len(set(days[mask & (zones == z)]))}
        for z in pair}}


def select_pair_rule(past_panel, pair, origin_day):
    pair, origin = _pair(pair), _day(origin_day)
    if pair == ("DE", "NL"):
        return original_hybrid.select_rule(past_panel, origin)
    if not isinstance(past_panel.index, pd.DatetimeIndex) or str(past_panel.index.tz) != "UTC":
        raise ValueError("UTC history required")
    days = np.asarray(past_panel.index.tz_convert(TZ).date)
    if np.any(days >= origin):
        raise ValueError("Origin/future historical labels forbidden in routing calibration")
    first = origin - timedelta(days=90)
    frame = past_panel.loc[days >= first].copy()
    days = _checked_panel(frame, pair, first, origin, labels=True)
    zones = frame.zone.to_numpy()
    actual, baseline, candidate = (frame[c].to_numpy(dtype=float) for c in ("actual", "nyx__q50", "test2__q50"))
    base_scores = _pair_metrics(actual, baseline, zones, pair)
    scores, best = [], None
    for number, rule in enumerate(original_hybrid.RULES):
        mask = original_hybrid._mask(frame, rule)
        support = _pair_support(mask, days, zones, pair)
        metrics = _pair_metrics(actual, np.where(mask, candidate, baseline), zones, pair)
        gain = base_scores["pooled"]["mae"] - metrics["pooled"]["mae"]
        failed = []
        if support["hours"] < 20: failed.append("pooled_support_hours_below_20")
        if support["days"] < 5: failed.append("pooled_support_days_below_5")
        if gain < .02 - 1e-12: failed.append("pooled_mae_gain_below_0.02")
        for zone in pair:
            if support["by_country"][zone]["hours"] < 5: failed.append(zone + "_support_hours_below_5")
            if support["by_country"][zone]["days"] < 2: failed.append(zone + "_support_days_below_2")
            if metrics[zone]["mae"] > base_scores[zone]["mae"] + .05 + 1e-12:
                failed.append(zone + "_mae_degradation_above_0.05")
            if metrics[zone]["rmse"] > base_scores[zone]["rmse"] + 1e-12:
                failed.append(zone + "_rmse_worse_than_nyx")
        score = {"rule_index": number, "rule": dict(rule), "support": support, "metrics": metrics,
                 "pooled_mae_gain": gain, "eligible": not failed, "failed_guards": failed}
        scores.append(score)
        if score["eligible"] and (best is None or metrics["pooled"]["mae"] < best["metrics"]["pooled"]["mae"] - 1e-12):
            best = score
    return {"protocol_version": PROTOCOL_VERSION + "_routing_BE_FR_extension", "pair": list(pair),
            "origin_day": str(origin), "mode": "hybrid" if best else "nyx",
            "rule": best["rule"] if best else None, "window_days": 90, "window_start": str(first),
            "window_end_exclusive": str(origin), "rows": len(frame), "candidate_scores": scores,
            "support": best["support"] if best else _pair_support(np.zeros(len(frame), bool), days, zones, pair),
            "baseline_metrics": base_scores, "selected_metrics": best["metrics"] if best else base_scores,
            "pooled_mae_gain": best["pooled_mae_gain"] if best else 0.,
            "selected_rule_index": best["rule_index"] if best else None,
            "selection_reason": "best_eligible_pooled_mae" if best else "no_rule_passed_all_guards",
            "transferred_DE_NL_rule": False, "pit_publication_evidence_verified": False,
            "independent_validation": False}


def apply_pair_rule(current_panel, policy, pair):
    pair = _pair(pair)
    if pair == ("DE", "NL"):
        return original_hybrid.apply_rule(current_panel, policy)
    if (policy.get("protocol_version") != PROTOCOL_VERSION + "_routing_BE_FR_extension"
            or policy.get("pair") != list(pair) or policy.get("mode") not in ("nyx", "hybrid")):
        raise ValueError("A separately learned BE/FR routing policy is required")
    origin = _day(policy["origin_day"])
    days = np.asarray(current_panel.index.tz_convert(TZ).date)
    if min(days) != origin or max(days) >= origin + timedelta(days=7):
        raise ValueError("Current block must span 1-7 full days starting at its policy origin")
    _checked_panel(current_panel, pair, origin, max(days) + timedelta(days=1), labels=False)
    if policy["mode"] == "hybrid":
        rule = original_hybrid._validate_rule(policy["rule"])
        selected = original_hybrid._mask(current_panel, rule)
    else:
        if policy.get("rule") is not None: raise ValueError("NYX fallback cannot carry an active rule")
        selected = np.zeros(len(current_panel), dtype=bool)
    result = current_panel.copy(deep=True)
    result["selected_test2"] = selected
    result["selected_model"] = np.where(selected, "Test2", "NYX")
    result["reason"] = np.where(selected, "test2_selected_by_nyx_and_physical_rule", "nyx_rule_conditions_not_met")
    if policy["mode"] == "nyx": result["reason"] = "nyx_fallback_no_eligible_past_rule"
    for q in QUANTILES:
        result["hybrid__" + q] = np.where(selected, current_panel["test2__" + q], current_panel["nyx__" + q])
    return result


def fit_test2_block(history, current, features, pair, *, origin_day, threads=1, iterations=120, seed=20260923):
    """Fresh paired Test2 fit using only labels strictly earlier than the origin."""
    pair, origin = _pair(pair), _day(origin_day)
    if threads != 1 or not 1 <= iterations <= 120:
        raise ValueError("One thread and at most120 trees are required")
    if any(set(mapping) != set(pair) for mapping in (history, current, features)):
        raise ValueError("Every input mapping must contain exactly this pair")
    train_x, train_y, train_days, test_x, parts = [], [], [], [], []
    for zone in pair:
        # Selection precedes label access, even when the caller holds a longer
        # annual evaluation panel containing later outcomes.
        days = np.asarray(history[zone].index.tz_convert(TZ).date)
        selected = history[zone].loc[(days >= origin - timedelta(days=365)) & (days < origin)]
        past = _baseline(selected, allow_actual=True)
        target = _baseline(current[zone], allow_actual=False)
        if not isinstance(target.index, pd.DatetimeIndex) or str(target.index.tz) != "UTC":
            raise ValueError("Current Test2 targets require UTC timestamps")
        target_days = target.index.tz_convert(TZ).date
        if (min(target_days) != origin or max(target_days) >= origin + timedelta(days=7)
                or not target.index.equals(_grid(origin, max(target_days) + timedelta(days=1)))):
            raise ValueError("Current Test2 requires 1-7 complete days starting at its origin")
        x_train = features[zone].loc[past.index].copy()
        x_test = features[zone].loc[target.index].copy()
        x_train["baseline_p50"] = past.nyx__q50.to_numpy()
        x_test["baseline_p50"] = target.nyx__q50.to_numpy()
        x_train = x_train.drop(columns=list(DAILY_PEAK_COLUMNS))
        x_test = x_test.drop(columns=list(DAILY_PEAK_COLUMNS))
        train_x.append(x_train); test_x.append(x_test)
        train_y.append((past.actual - past.nyx__q50).to_numpy(dtype=float))
        train_days.extend(past.index.tz_convert(TZ).date)
        target["zone"] = zone
        for name in ("own_joint_deficit", "own_residual_stress"):
            target[name] = features[zone].loc[target.index, name].to_numpy()
        parts.append(target)
    quantiles, probability, numerical_audit = regime.fit_predict(
        pd.concat(train_x), np.concatenate(train_y), pd.concat(test_x), train_days,
        origin_day=origin, threads=1, iterations=iterations, seed=seed,
    )
    panel = pd.concat(parts)
    for position, q in enumerate(QUANTILES):
        panel["test2__" + q] = panel.nyx__q50.to_numpy() + quantiles[:, position]
    panel["spike_probability"] = probability
    panel["fit_origin"] = str(origin)
    panel["nyx_daily_peak_gap"] = panel.groupby([panel.zone, panel.index.tz_convert(TZ).date])["nyx__q50"].transform("max") - panel.nyx__q50
    return panel, {"protocol_version": PROTOCOL_VERSION, "pair": list(pair), "origin_day": str(origin),
                   "variant": "regime_hour_local", "separately_fitted_pair": True,
                   "feature_country_indicator": "country_is_nl" if pair == ("DE", "NL") else "country_is_fr",
                   "dropped_columns": list(DAILY_PEAK_COLUMNS), "numerical_recipe_audit": numerical_audit}


def _validate_checkpoint(panel, audit, current, features, pair, origin, stop):
    _checked_panel(panel, pair, origin, stop, labels=False)
    expected_zones = np.concatenate([np.repeat(z, len(current[z])) for z in pair])
    if not np.array_equal(panel.zone.to_numpy(), expected_zones):
        raise ValueError("Checkpoint country ordering differs from canonical paired ordering")
    if audit.get("pair") != list(pair) or audit.get("origin_day") != str(origin):
        raise ValueError("Checkpoint pair/origin identity mismatch")
    if not (panel.fit_origin == str(origin)).all():
        raise ValueError("Checkpoint prediction fit origin mismatch")
    if "spike_probability" not in panel or not panel.spike_probability.between(0, 1).all():
        raise ValueError("Checkpoint spike probabilities invalid")
    for zone in pair:
        sub = panel.loc[panel.zone == zone]
        base = _baseline(current[zone], allow_actual=False)
        for q in QUANTILES:
            if not np.array_equal(sub["nyx__" + q], base["nyx__" + q]):
                raise ValueError("Checkpoint baseline changed")
        for name in ("own_joint_deficit", "own_residual_stress"):
            if not np.array_equal(sub[name], features[zone].loc[sub.index, name]):
                raise ValueError("Checkpoint physical features changed")


def run_pair_pipeline(history_by_zone, forecast_by_zone, covariates_by_zone, *, pair, delivery_day,
                      threads=1, iterations=120, seed=20260923,
                      load_checkpoint=None, save_checkpoint=None):
    """Pure orchestrator:184 days OOF Test2,93 days rolling hybrid,one delivery day.

    Baseline history must cover exactly [delivery-365,delivery), and forecast
    exactly that delivery day with NO actual column. Features need sufficient
    earlier forecast context for their 14-day normalization warmup. Weekly
    model origins start delivery-184; the first routing origin is +91 days.
    A fresh model and routing policy are fitted for the delivery day. These
    are date-relative origins, not reuse of fitted22-September coefficients.

    load_checkpoint(stage,key) -> (prediction_panel,audit) or None;
    save_checkpoint(stage,key,prediction_panel,audit). Stages are test2/hybrid;
    key is origin ISO date. Caller namespaces/checksums by pair and identity.
    No current actual price is accepted or requested.
    """
    pair, delivery = _pair(pair), _day(delivery_day)
    if threads != 1 or not 1 <= iterations <= 120:
        raise ValueError("One thread and at most120 trees are required")
    if any(set(mapping) != set(pair) for mapping in (history_by_zone, forecast_by_zone, covariates_by_zone)):
        raise ValueError("Exactly the requested country pair is required throughout")
    history, forecast = {}, {}
    for zone in pair:
        history[zone] = _baseline(history_by_zone[zone], allow_actual=True)
        forecast[zone] = _baseline(forecast_by_zone[zone], allow_actual=False)
        if not history[zone].index.equals(_grid(delivery - timedelta(days=365), delivery)):
            raise ValueError("Baseline requires exact365 civil days, preserving DST hours")
        if not forecast[zone].index.equals(_grid(delivery, delivery + timedelta(days=1))):
            raise ValueError("Baseline forecast requires one complete physical delivery day")
    features, feature_audit = build_pair_features(covariates_by_zone, pair)
    first_test2 = delivery - timedelta(days=184)
    blocks = []
    origin = first_test2
    while origin < delivery:
        blocks.append((origin, min(origin + timedelta(days=7), delivery), False))
        origin += timedelta(days=7)
    blocks.append((delivery, delivery + timedelta(days=1), True))
    predictions, audits = [], []
    future_panel = None
    for origin, stop, is_forecast in blocks:
        current = {}
        for zone in pair:
            source = forecast[zone] if is_forecast else history[zone]
            days = np.asarray(source.index.tz_convert(TZ).date)
            current[zone] = source.loc[(days >= origin) & (days < stop)].drop(columns="actual", errors="ignore").copy()
        cached = load_checkpoint("test2", str(origin)) if load_checkpoint else None
        if cached is None:
            panel, audit = fit_test2_block(history, current, features, pair, origin_day=origin,
                                          threads=threads, iterations=iterations, seed=seed)
            _validate_checkpoint(panel, audit, current, features, pair, origin, stop)
            if save_checkpoint: save_checkpoint("test2", str(origin), panel.copy(deep=True), audit)
        else:
            panel, audit = cached
            _validate_checkpoint(panel, audit, current, features, pair, origin, stop)
            panel = panel.copy(deep=True)
        audits.append(audit)
        if is_forecast:
            future_panel = panel
        else:
            panel = panel.copy(deep=True)
            panel["actual"] = np.concatenate([history[z].loc[panel.loc[panel.zone == z].index, "actual"].to_numpy() for z in pair])
            predictions.append(panel)
    # Canonical country-first/hourly order, identical to the frozen reference.
    all_history = pd.concat(predictions)
    all_history = pd.concat([all_history.loc[all_history.zone == z].sort_index() for z in pair])
    first_hybrid = first_test2 + timedelta(days=91)
    routed, policies = [], []
    future_hybrid = None
    for origin, stop, is_forecast in blocks:
        if origin < first_hybrid: continue
        days = np.asarray(all_history.index.tz_convert(TZ).date)
        past = all_history.loc[(days >= origin - timedelta(days=90)) & (days < origin)].copy()
        target = future_panel if is_forecast else all_history
        target_days = np.asarray(target.index.tz_convert(TZ).date)
        current = target.loc[(target_days >= origin) & (target_days < stop)].drop(columns="actual", errors="ignore").copy()
        policy = select_pair_rule(past, pair, origin)
        expected = apply_pair_rule(current, policy, pair)
        audit = {"protocol_version": PROTOCOL_VERSION, "pair": list(pair), "origin_day": str(origin), "policy": policy}
        cached = load_checkpoint("hybrid", str(origin)) if load_checkpoint else None
        if cached is None:
            output = expected
            if save_checkpoint: save_checkpoint("hybrid", str(origin), output.copy(deep=True), audit)
        else:
            output, saved_audit = cached
            if saved_audit != audit:
                raise ValueError("Cached routing policy differs from strict past-only recalculation")
            pd.testing.assert_frame_equal(output, expected, check_exact=True)
            output = output.copy(deep=True)
        policies.append(audit)
        if is_forecast:
            future_hybrid = output
        else:
            output["actual"] = np.concatenate([history[z].loc[output.loc[output.zone == z].index, "actual"].to_numpy() for z in pair])
            routed.append(output)
    historical_hybrid = pd.concat(routed)
    historical_hybrid = pd.concat([historical_hybrid.loc[historical_hybrid.zone == z].sort_index() for z in pair])
    return {"historical_test2": all_history, "historical_hybrid": historical_hybrid,
            "forecast_test2": future_panel, "forecast_hybrid": future_hybrid,
            "feature_audit": feature_audit, "fit_audits": audits, "policies": policies,
            "protocol": {"version": PROTOCOL_VERSION, "pair": list(pair), "delivery_day": str(delivery),
                         "test2_start_day": str(first_test2), "hybrid_start_day": str(first_hybrid),
                         "test2_historical_days": 184, "hybrid_historical_days": 93,
                         "test2_origins": len(blocks), "routing_origins": len(policies),
                         "threads": threads, "iterations": iterations, "seed": seed,
                         "recipe_DE_NL": "unchanged numerical Test2 and routing; date-relative fresh fits",
                         "extension_BE_FR": "separately trained BE<->FR with country_is_fr and own wind",
                         "production_modified": False, "pit_publication_evidence_verified": False,
                         "independent_validation": False}}


__all__ = ["PROTOCOL_VERSION", "build_pair_interaction", "build_pair_features", "fit_test2_block",
           "select_pair_rule", "apply_pair_rule", "run_pair_pipeline"]
