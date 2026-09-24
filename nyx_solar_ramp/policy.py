"""Chronological pooled risk/median experts and a prior-OOS zonal governor.

The source archive is an as-of reconstruction, not publication certification.
All calibrations are chronologically held out; no contemporaneous realised
price or post-coupling quantity is included in a feature matrix.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger(__name__)
VARIANTS = ("control", "local", "regional", "ramps", "interactions")


def feature_sets(groups):
    controls = groups["calendar"] + groups["controls"] + groups.get("baseline", [])
    names = {"control": controls}
    for key, addition, previous in (("local", "local_solar", "control"), ("regional", "regional_solar", "local"),
                                    ("ramps", "ramps", "regional"), ("interactions", "interactions", "ramps")):
        names[key] = names[previous] + groups[addition]
    for columns in names.values():
        if len(columns) != len(set(columns)) or any(not c.startswith("solarx_") or any(x in c for x in ("actual", "label", "storm")) for c in columns):
            raise ValueError("An explicit unique non-label solar feature allowlist is required.")
    return names


def prepare_panel(panel, config):
    data = panel.copy(deep=True).reset_index(drop=True)
    required = {"zone", "timestamp_utc", "forecast_origin_utc", "forecast", "actual", "q10", "q90", "solarx_eligible"}
    if required.difference(data) or data.empty:
        raise ValueError("Incomplete solar panel.")
    for name in ("timestamp_utc", "forecast_origin_utc"):
        if data[name].isna().any() or any(pd.Timestamp(v).tzinfo is None for v in data[name]):
            raise ValueError("Aware complete delivery/origin timestamps required.")
        data[name] = pd.to_datetime(data[name], utc=True)
    if data.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Duplicate physical delivery hour.")
    local = data.timestamp_utc.dt.tz_convert(config["timezone"]).dt.tz_localize(None)
    civil = local.dt.normalize()
    expected = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize(config["timezone"]).dt.tz_convert("UTC")
    if not data.forecast_origin_utc.eq(expected).all():
        raise ValueError("Only day-ahead D-1 08h civil origins are supported.")
    data["local_day"] = civil.dt.strftime("%Y-%m-%d")
    # This conservative lag is still an assumption, never publication evidence.
    known = (civil + pd.Timedelta(days=1 + config["label_delay_days"])).dt.tz_localize(config["timezone"]).dt.tz_convert("UTC")
    if "label_available_at_utc" in data:
        prior = pd.to_datetime(data.label_available_at_utc, utc=True)
        known = known.where(prior.isna() | known.ge(prior), prior)
    data["training_label_available_at_utc"] = known
    data["training_label_eligible"] = data.actual.notna()
    if "label_eligible" in data:
        data["training_label_eligible"] &= data.label_eligible.fillna(False).astype(bool)
    if "feature_available_at_utc" in data:
        available = pd.to_datetime(data.feature_available_at_utc, utc=True)
        data["solarx_eligible"] &= available.notna() & available.le(data.forecast_origin_utc)
    numeric = data[["actual", "forecast", "q10", "q90"]].to_numpy(float)
    if np.isinf(numeric).any() or not np.isfinite(numeric[:, 1:]).all():
        raise ValueError("Finite baseline and finite-or-missing observations required.")
    if not ((data.q10 <= data.forecast) & (data.forecast <= data.q90)).all():
        raise ValueError("Baseline quantiles must be ordered.")
    data["spike_label"] = data.actual.ge(config["business_spike"]).astype(float).where(data.actual.notna())
    keys = pd.MultiIndex.from_arrays([data.zone, data.timestamp_utc])
    lag = pd.MultiIndex.from_arrays([data.zone, data.timestamp_utc - pd.Timedelta(hours=1)])
    prior_price = pd.Series(data.actual.to_numpy(), index=keys).reindex(lag).to_numpy()
    data["ramp_label"] = (data.actual - prior_price).ge(config["price_ramp_threshold"]).astype(float).where(np.isfinite(prior_price) & data.actual.notna())
    historical = data.loc[data.get("sample", pd.Series("evaluation", index=data.index)).ne("live"), "local_day"]
    start, end = pd.Timestamp(historical.min()), pd.Timestamp(historical.max())
    final_start = end - pd.Timedelta(days=config["final_days"] - 1)
    selection_start = final_start - pd.Timedelta(days=config["selection_days"])
    day = pd.to_datetime(data.local_day)
    data["evaluation_phase"] = np.select([day < start + pd.Timedelta(days=config["minimum_training_days"]),
        day < selection_start, day < final_start, day <= end],
        ["warmup", "exploration", "selection", "final_diagnostic"], default="live_historical")
    return data


def _matrix(frame, names):
    return np.column_stack([frame[names].to_numpy(float), *(frame.zone.eq(z).to_numpy(float) for z in ("FR", "DE", "BE", "NL"))])


def _logits(prob):
    p = np.clip(prob, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def _alert_threshold(probabilities, labels, budget):
    negative = np.asarray(probabilities)[np.asarray(labels) == 0]
    if len(negative) < 24:
        return 1.0
    # Strict > avoids a mass of tied probabilities breaching the false-alarm budget.
    return float(np.quantile(negative, 1-budget, method="higher"))


def _fit(past, names, config):
    days = sorted(past.local_day.unique())
    if len(days) < config["minimum_training_days"]:
        return None
    cal_days = config["calibration_days"]
    first_cal = days[-cal_days]
    second_cal = days[-max(1, cal_days//2)]
    train = past.loc[past.local_day < first_cal]
    calibration = past.loc[(past.local_day >= first_cal) & (past.local_day < second_cal)]
    threshold_data = past.loc[past.local_day >= second_cal]
    if min(len(train), len(calibration), len(threshold_data)) < 100 or train.spike_label.nunique() < 2:
        return None
    opts = dict(max_iter=config["max_iter"], max_leaf_nodes=config["max_leaf_nodes"],
                min_samples_leaf=config["min_samples_leaf"], l2_regularization=config["l2_regularization"],
                learning_rate=config["learning_rate"], random_state=config["seed"], early_stopping=False)
    classifier = HistGradientBoostingClassifier(**opts).fit(_matrix(train, names), train.spike_label)
    median = HistGradientBoostingRegressor(loss="quantile", quantile=.5, **opts).fit(
        _matrix(train, names), train.actual-train.forecast)
    raw = classifier.predict_proba(_matrix(calibration, names))[:, 1]
    calibrator = None
    if calibration.spike_label.value_counts().reindex([0, 1], fill_value=0).min() >= 5:
        calibrator = LogisticRegression(C=.1, random_state=config["seed"]).fit(_logits(raw), calibration.spike_label)
    prob = classifier.predict_proba(_matrix(threshold_data, names))[:, 1]
    if calibrator is not None:
        prob = calibrator.predict_proba(_logits(prob))[:, 1]
    correction = np.clip(median.predict(_matrix(threshold_data, names)), -config["correction_clip"], config["correction_clip"])
    errors = threshold_data.actual.to_numpy()-threshold_data.forecast.to_numpy()-correction
    thresholds, intervals, statistical = {}, {}, {}
    for z in ("FR", "DE", "BE", "NL"):
        mask = threshold_data.zone.eq(z).to_numpy()
        thresholds[z] = _alert_threshold(prob[mask], threshold_data.loc[mask, "spike_label"], config["false_alert_budget"])
        intervals[z] = tuple(np.quantile(errors[mask], [.1, .9])) if mask.sum() >= 48 else None
        zone_train = train.loc[train.zone.eq(z), "actual"]
        statistical[z] = float(zone_train.quantile(config["statistical_quantile"])) if len(zone_train) else np.nan
    return {"classifier": classifier, "median": median, "calibrator": calibrator,
            "thresholds": thresholds, "intervals": intervals, "statistical": statistical, "names": names,
            "train_end": train.local_day.max(), "calibration_start": first_cal,
            "calibration_end": calibration.local_day.max(), "threshold_start": second_cal,
            "threshold_end": threshold_data.local_day.max(), "maximum_label_time": past.training_label_available_at_utc.max(),
            "train_rows": len(train), "calibration_rows": len(calibration), "threshold_rows": len(threshold_data)}


def _blank(data, variant):
    result = data.copy()
    result["variant"] = variant
    result["candidate_forecast"], result["candidate_q10"], result["candidate_q90"] = result.forecast, result.q10, result.q90
    for name in ("risk_probability", "alert_threshold", "statistical_spike_label", "spike_threshold"):
        result[name] = np.nan
    result["alert"], result["expert_ready"], result["interval_ready"] = False, False, False
    result["applied_correction"], result["bounded_correction"], result["selected_weight"] = 0., 0., 0.
    result["gate_reason"], result["fold_id"], result["probability_calibration"] = "baseline", -1, "unavailable"
    return result


def _block_upper(values, *, seed, block_days):
    if isinstance(values, pd.Series):
        values = values.copy()
        values.index = pd.to_datetime(values.index)
        values = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="D"))
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(199, int(np.ceil(len(values)/block_days))))
    indices = ((starts[:, :, None] + np.arange(block_days)) % len(values)).reshape(199, -1)[:, :len(values)]
    sampled = values[indices]
    valid = np.isfinite(sampled)
    means = np.divide(np.nansum(sampled, axis=1), valid.sum(axis=1),
                      out=np.full(len(sampled), np.nan), where=valid.sum(axis=1) > 0)
    return float(np.quantile(means[np.isfinite(means)], .95)) if np.isfinite(means).sum() >= 190 else np.inf


def govern(proposal, config):
    """Use only prior issued OOS proposals whose labels precede the current origin."""
    out = _blank(proposal.drop(columns=["variant"]), "governed")
    for name in ("risk_probability", "alert_threshold", "alert", "expert_ready", "fold_id", "spike_threshold", "statistical_spike_label", "probability_calibration"):
        out[name] = proposal[name]
    out["gate_reason"] = "no_prior_oos_non_regression_evidence"
    records = []
    for (day, zone), current in proposal.groupby(["local_day", "zone"], sort=True):
        origin = current.forecast_origin_utc.iloc[0]
        start = (pd.Timestamp(day)-pd.Timedelta(days=config["governance_days"])).strftime("%Y-%m-%d")
        past = proposal.loc[proposal.zone.eq(zone) & proposal.local_day.ge(start) & proposal.local_day.lt(day)
                            & proposal.training_label_available_at_utc.le(origin) & proposal.expert_ready
                            & proposal.training_label_eligible & proposal.actual.notna()].copy()
        positive = past.bounded_correction.clip(lower=0).where(past.alert, 0.)
        intervened = positive.gt(0)
        chosen, quantiles = 0., None
        if (past.local_day.nunique() >= config["governance_min_days"]
                and past.loc[intervened, "local_day"].nunique() >= config["governance_min_changed_days"]
                and intervened.sum() >= config["governance_min_alert_rows"]
                and (past.spike_label.eq(1) & intervened).sum() >= 5):
            best_loss = 0.
            for weight in config["candidate_weights"][1:]:
                candidate = past.forecast + weight*positive
                delta = (past.actual-candidate).abs()-(past.actual-past.forecast).abs()
                daily = delta.groupby(past.local_day).mean()
                nonspike = delta.loc[past.spike_label.eq(0)]
                tail = delta.loc[past.spike_label.eq(1)]
                upper = _block_upper(daily, seed=config["seed"], block_days=config["block_days"])
                if (upper <= config["mae_tolerance"] and delta.mean() < best_loss
                        and len(nonspike) and nonspike.mean() <= config["mae_tolerance"]
                        and len(tail) and tail.mean() < 0):
                    chosen, best_loss = float(weight), float(delta.mean())
                    quantiles = np.quantile((past.actual-candidate).loc[intervened], [.1, .9])
        active = current.alert & current.expert_ready & current.bounded_correction.gt(0)
        correction = chosen*current.bounded_correction.clip(lower=0).where(active, 0.)
        out.loc[current.index, "bounded_correction"] = current.bounded_correction.clip(lower=0).where(active, 0.)
        out.loc[current.index, "selected_weight"] = chosen
        out.loc[current.index, "applied_correction"] = correction
        out.loc[current.index, "candidate_forecast"] = current.forecast+correction
        changed = current.index[correction > 0]
        if len(changed) and quantiles is not None:
            point = out.loc[changed, "candidate_forecast"]
            out.loc[changed, "candidate_q10"] = point+min(float(quantiles[0]), 0.)
            out.loc[changed, "candidate_q90"] = point+max(float(quantiles[1]), 0.)
            out.loc[changed, "interval_ready"] = True
            out.loc[changed, "gate_reason"] = "prior_oos_zonal_mae_and_tail_guard_passed"
        records.append({"day": day, "zone": zone, "origin": str(origin), "weight": chosen,
                        "past_oos_rows": len(past), "past_alert_rows": int(intervened.sum()),
                        "max_label_time": str(past.training_label_available_at_utc.max()), "changed_hours": len(changed)})
    return out, records


def run_policy(panel, groups, config, *, on_checkpoint=None):
    data = prepare_panel(panel, config)
    sets = feature_sets(groups)
    outputs, folds, models = {"baseline": _blank(data, "baseline")}, [], {}
    for variant in VARIANTS:
        outputs[variant] = _blank(data, variant)
    days = sorted(data.local_day.unique())
    with threadpool_limits(limits=config["threads"]):
        for number, start in enumerate(range(0, len(days), config["refit_days"])):
            block = days[start:start+config["refit_days"]]
            current = data.loc[data.local_day.isin(block)]
            cutoff = current.forecast_origin_utc.min()
            first = (pd.Timestamp(block[0])-pd.Timedelta(days=config["training_window_days"])).strftime("%Y-%m-%d")
            past = data.loc[data.local_day.ge(first) & data.local_day.lt(block[0]) & data.training_label_available_at_utc.le(cutoff)
                            & data.training_label_eligible & data.actual.notna() & data.solarx_eligible]
            targets = current.loc[current.solarx_eligible]
            for variant, names in sets.items():
                if not config["enabled"]:
                    continue
                fitted = _fit(past, names, config)
                if fitted is None or targets.empty:
                    continue
                out = outputs[variant]
                prob = fitted["classifier"].predict_proba(_matrix(targets, names))[:, 1]
                if fitted["calibrator"] is not None:
                    prob = fitted["calibrator"].predict_proba(_logits(prob))[:, 1]
                correction = np.clip(fitted["median"].predict(_matrix(targets, names)), -config["correction_clip"], config["correction_clip"])
                out.loc[targets.index, "risk_probability"] = np.clip(prob, 1e-6, 1-1e-6)
                out.loc[targets.index, "alert_threshold"] = targets.zone.map(fitted["thresholds"])
                out.loc[targets.index, "alert"] = prob > targets.zone.map(fitted["thresholds"]).to_numpy()
                out.loc[targets.index, "expert_ready"] = True
                out.loc[targets.index, "bounded_correction"] = correction
                out.loc[targets.index, "applied_correction"] = correction
                out.loc[targets.index, "selected_weight"] = 1.
                out.loc[targets.index, "candidate_forecast"] = targets.forecast.to_numpy()+correction
                out.loc[targets.index, "gate_reason"] = "raw_median_challenger_diagnostic"
                out.loc[targets.index, "fold_id"] = number
                out.loc[targets.index, "probability_calibration"] = "heldout_platt" if fitted["calibrator"] is not None else "raw_insufficient_calibration_events"
                out.loc[targets.index, "spike_threshold"] = targets.zone.map(fitted["statistical"])
                out.loc[targets.index, "statistical_spike_label"] = targets.actual.ge(targets.zone.map(fitted["statistical"])).astype(float).where(targets.actual.notna())
                for zone, interval in fitted["intervals"].items():
                    indices = targets.index[targets.zone.eq(zone)]
                    if interval is not None:
                        point = out.loc[indices, "candidate_forecast"]
                        out.loc[indices, "candidate_q10"] = point+min(float(interval[0]), 0.)
                        out.loc[indices, "candidate_q90"] = point+max(float(interval[1]), 0.)
                        out.loc[indices, "interval_ready"] = True
                    else:
                        out.loc[indices, "candidate_forecast"] = out.loc[indices, "forecast"]
                        out.loc[indices, ["bounded_correction", "applied_correction", "selected_weight"]] = 0.
                        out.loc[indices, "gate_reason"] = "risk_only_no_interval_calibration"
                record = {key: value for key, value in fitted.items() if key not in {"classifier", "median", "calibrator"}}
                record.update(variant=variant, fold_id=number, fit_origin_utc=str(cutoff), predict_first_day=block[0],
                              predict_last_day=block[-1], calibration_method="platt" if fitted["calibrator"] is not None else "raw_unvalidated")
                folds.append(record)
                models[variant] = fitted
            LOGGER.info("[Solar] Fold %d/%d | %s -> %s | past=%d", number+1, int(np.ceil(len(days)/config["refit_days"])), block[0], block[-1], len(past))
            if on_checkpoint is not None:
                on_checkpoint(number, outputs, folds)
    outputs["governed"], governance = govern(outputs["interactions"], config)
    for frame in outputs.values():
        if len(frame) != len(data) or not np.allclose(frame.candidate_forecast, frame.forecast+frame.applied_correction, rtol=0, atol=1e-9):
            raise ValueError("Prediction coverage/arithmetic changed.")
        if not ((frame.candidate_q10 <= frame.candidate_forecast) & (frame.candidate_forecast <= frame.candidate_q90)).all():
            raise ValueError("Crossed candidate quantiles.")
    return pd.concat(outputs.values(), ignore_index=True), folds, governance, models
