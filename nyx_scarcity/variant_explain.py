"""Out-of-sample TreeSHAP for the isolated XGBoost classifier variants.

This module never fits a model, reads labels, reloads a later model, or writes
files. The daily prediction callback supplies the very state used at that
origin. Exact TreeSHAP explains the classifier's raw log-odds, not a price or
the final governed correction. A one-dimensional Platt fit on *raw margins*
allows the same contributions to be transformed to calibrated log-odds.

The sample is calendar-based and declared before outcomes are inspected:
all hours of the last seven evaluation days and the optional live day, plus
19:00 every seventh earlier evaluation day. Its global importance therefore
describes this selected sample, not uniformly sampled annual performance.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


class VariantExplainError(ValueError):
    """An explanation cannot truthfully reproduce the issued classifier."""


LONG_COLUMNS = ["sample_id", "variant_id", "zone", "timestamp_utc", "forecast_origin_utc",
                "feature", "feature_value", "shap_value_raw", "shap_value_calibrated"]
OBSERVATION_COLUMNS = [
    "sample_id", "variant_id", "zone", "timestamp_utc", "forecast_origin_utc",
    "sampling_rule", "status", "skip_reason", "target_kind", "fit_day", "fit_cutoff_utc",
    "model_id", "base_value_raw", "base_value_calibrated", "raw_margin", "calibrated_margin",
    "prob_raw", "prob_calibrated", "platt_slope", "platt_intercept",
    "reconstruction_error_raw", "reconstruction_error_calibrated",
]
OBSERVATION_NUMERIC = ["base_value_raw", "base_value_calibrated", "raw_margin", "calibrated_margin",
                       "prob_raw", "prob_calibrated", "platt_slope", "platt_intercept",
                       "reconstruction_error_raw", "reconstruction_error_calibrated"]
OBSERVATION_DATES = ["timestamp_utc", "forecast_origin_utc", "fit_cutoff_utc"]


def _observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable schema, including unavailable explanations with explicit NaNs."""
    result = frame.reindex(columns=OBSERVATION_COLUMNS).copy()
    for name in OBSERVATION_NUMERIC:
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
    for name in OBSERVATION_DATES:
        result[name] = pd.to_datetime(result[name], utc=True, format="mixed")
    for name in set(OBSERVATION_COLUMNS) - set(OBSERVATION_NUMERIC) - set(OBSERVATION_DATES):
        result[name] = result[name].fillna("").astype(str)
    return result


def _day(value: str, name: str) -> str:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise VariantExplainError(f"{name}: YYYY-MM-DD required.") from exc
    if not isinstance(value, str) or parsed.tzinfo is not None or parsed.strftime("%Y-%m-%d") != value:
        raise VariantExplainError(f"{name}: YYYY-MM-DD required.")
    return value


def _utc(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any() or any(pd.Timestamp(v).tzinfo is None for v in values):
        raise VariantExplainError(f"{name}: finite timezone-aware timestamps required.")
    return pd.to_datetime(values, utc=True, format="mixed")


def _flag(values: pd.Series, name: str) -> pd.Series:
    if not values.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise VariantExplainError(f"{name}: explicit boolean flags required.")
    return values.fillna(False).astype(bool)


def _feature_contract(state: Mapping[str, Any], p: Mapping[str, Any]):
    features = state.get("features")
    _validate_feature_names(features)
    if list(p.get("feature_columns", [])) != features:
        raise VariantExplainError("SHAP and issued policy feature order differ.")
    zones = state.get("zones")
    if (not isinstance(zones, (list, tuple)) or not zones or len(set(zones)) != len(zones)
            or any(z not in {"FR", "DE", "BE", "NL"} for z in zones)):
        raise VariantExplainError("Explicit unique model countries required.")
    names = features + [f"zone__{z}" for z in zones]
    if state.get("model_feature_names") != names or len(set(names)) != len(names):
        raise VariantExplainError("Model feature names must preserve raw columns followed by country indicators.")
    return features, tuple(zones), names


def _validate_feature_names(features):
    if (not isinstance(features, list) or not features or len(set(features)) != len(features)
            or any(not isinstance(v, str) or any(token in v.lower() for token in
                   ("actual", "storm", "benchmark", "label", "candidate_", "eligible", "available_at"))
                   or v in {"forecast", "q10", "q90", "zone", "timestamp_utc", "forecast_origin_utc"}
                   for v in features)):
        raise VariantExplainError("Explicit unique feature allowlist must exclude labels, Storm and metadata.")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _state_origin(state: Mapping[str, Any], frame: pd.DataFrame) -> pd.Timestamp:
    fit_day = _day(state.get("fit_day"), "fit_day")
    fit_cutoff = pd.Timestamp(state.get("fit_cutoff"))
    if pd.isna(fit_cutoff) or fit_cutoff.tzinfo is None:
        raise VariantExplainError("The fitted state needs its timezone-aware fit_cutoff.")
    fit_cutoff = fit_cutoff.tz_convert("UTC")
    expected = (pd.Timestamp(fit_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    if fit_cutoff != expected or not frame.forecast_origin_utc.ge(fit_cutoff).all():
        raise VariantExplainError("A model fitted after an explained origin is forbidden.")
    for key in ("model_training_end_day", "calibration_end_day"):
        if state.get(key) is not None and _day(state[key], key) >= fit_day:
            raise VariantExplainError("Explained deliveries must be outside the fitted/calibration sample.")
    if state.get("max_label_available_at_utc") is not None:
        known = pd.Timestamp(state["max_label_available_at_utc"])
        if pd.isna(known) or known.tzinfo is None or known.tz_convert("UTC") > fit_cutoff:
            raise VariantExplainError("Fitted state claims labels unavailable at its own origin.")
    return fit_cutoff


def _platt(state: Mapping[str, Any]):
    calibrator = state.get("calibrator")
    if state.get("calibration_input") != "raw_margin":
        raise VariantExplainError("Affine SHAP chaining requires Platt fitted on raw_margin, without clipping.")
    if (not isinstance(calibrator, LogisticRegression) or not hasattr(calibrator, "coef_")
            or np.asarray(calibrator.coef_).shape != (1, 1)
            or np.asarray(calibrator.intercept_).shape != (1,)
            or not np.array_equal(np.asarray(calibrator.classes_), [0, 1])):
        raise VariantExplainError("A fitted binary one-dimensional LogisticRegression calibrator is required.")
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    if not np.isfinite([slope, intercept]).all():
        raise VariantExplainError("Non-finite Platt parameters.")
    return calibrator, slope, intercept


def explain_xgb(state: Mapping[str, Any], frame: pd.DataFrame, parameters: Mapping[str, Any],
                *, tolerance: float = 2e-5) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Explain only the callback's current admissible OOS rows, without fitting.

    Returned contributions are raw/calibrated classifier log-odds. The bias is
    stored separately in observations; adding it to every feature contribution
    reconstructs each issued raw margin and the Platt calibrated margin.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise VariantExplainError("XGBoost runtime unavailable; true TreeSHAP cannot be fabricated.") from exc
    if frame.empty:
        raise VariantExplainError("No callback rows to explain.")
    frame = frame.reset_index(drop=True)
    timestamps = _utc(frame.timestamp_utc, "timestamp_utc")
    origins = _utc(frame.forecast_origin_utc, "forecast_origin_utc")
    civil = timestamps.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
    if origins.nunique() != 1 or not origins.eq(expected).all():
        raise VariantExplainError("TreeSHAP accepts one current delivery origin, never a mixed future/history panel.")
    frame = frame.assign(timestamp_utc=timestamps, forecast_origin_utc=origins)
    if not np.isfinite(tolerance) or not 1e-7 <= tolerance <= 1e-3:
        raise VariantExplainError("Bounded positive numerical reconstruction tolerance required.")
    features, zones, names = _feature_contract(state, parameters)
    if set(features).difference(frame) or not frame.zone.isin(zones).all():
        raise VariantExplainError("Callback rows do not match the fitted feature/country contract.")
    fit_cutoff = _state_origin(state, frame)
    calibrator, slope, intercept = _platt(state)
    classifier = state.get("classifier")
    if isinstance(classifier, xgb.XGBClassifier):
        booster = classifier.get_booster()
        try:
            iteration_range = (0, int(classifier.best_iteration) + 1)
        except AttributeError:
            iteration_range = (0, 0)
    elif isinstance(classifier, xgb.Booster):
        booster, iteration_range = classifier, (0, 0)
    else:
        raise VariantExplainError("Exact XGBoost TreeSHAP requires an actual XGBClassifier or Booster.")
    saved_config = json.loads(booster.save_config())["learner"]
    if (saved_config["objective"]["name"] != "binary:logistic"
            or saved_config["gradient_booster"]["name"] != "gbtree"):
        raise VariantExplainError("Only binary logistic tree boosters are qualified for this explanation.")
    if booster.num_features() != len(names) or (booster.feature_names is not None and booster.feature_names != names):
        raise VariantExplainError("Booster feature identity/order diverges from the declared matrix.")
    X = np.column_stack([frame[features].to_numpy(float), *(frame.zone.eq(z).to_numpy(float) for z in zones)])
    if np.isinf(X).any():
        raise VariantExplainError("Infinite explanatory features are forbidden; optional NaNs remain missing.")
    matrix = xgb.DMatrix(X, feature_names=booster.feature_names, nthread=int(parameters.get("threads", 1)))
    common = {"iteration_range": iteration_range, "validate_features": True}
    contributions = np.asarray(booster.predict(matrix, pred_contribs=True, approx_contribs=False, **common), dtype=float)
    raw = np.asarray(booster.predict(matrix, output_margin=True, **common), dtype=float).reshape(-1)
    probability = np.asarray(booster.predict(matrix, **common), dtype=float).reshape(-1)
    if contributions.shape != (len(frame), len(names) + 1) or not np.isfinite(contributions).all():
        raise VariantExplainError("Unexpected or non-finite binary TreeSHAP output.")
    raw_sum = contributions.sum(axis=1)
    if not np.allclose(raw_sum, raw, atol=tolerance, rtol=tolerance):
        raise VariantExplainError("TreeSHAP contributions do not reconstruct the issued raw margin.")
    if not np.allclose(expit(raw), probability, atol=1e-6, rtol=1e-6):
        raise VariantExplainError("The raw margin does not reconstruct the classifier probability.")
    calibrated = intercept + slope * raw
    cal_values = slope * contributions[:, :-1]
    cal_base = intercept + slope * contributions[:, -1]
    cal_sum = cal_base + cal_values.sum(axis=1)
    cal_probability = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    if (not np.allclose(cal_sum, calibrated, atol=tolerance, rtol=tolerance)
            or not np.allclose(calibrator.decision_function(raw.reshape(-1, 1)), calibrated, atol=1e-10, rtol=1e-10)
            or not np.allclose(expit(calibrated), cal_probability, atol=1e-10, rtol=1e-10)):
        raise VariantExplainError("TreeSHAP/Platt contributions do not reconstruct calibrated log-odds/probabilities.")
    identity = hashlib.sha256(bytes(booster.save_raw(raw_format="ubj")))
    identity.update(json.dumps({"features": names, "fit_cutoff": str(fit_cutoff), "slope": slope,
                                "intercept": intercept, "iteration_range": iteration_range}, sort_keys=True).encode())
    observations = frame[["sample_id", "variant_id", "zone", "timestamp_utc", "forecast_origin_utc", "sampling_rule"]].copy()
    observations = observations.assign(
        status="explained", skip_reason="", target_kind=str(state.get("target_kind", parameters.get("target_kind", "unspecified_classifier_event"))),
        fit_day=state["fit_day"], fit_cutoff_utc=fit_cutoff, model_id=identity.hexdigest(),
        base_value_raw=contributions[:, -1], base_value_calibrated=cal_base,
        raw_margin=raw, calibrated_margin=calibrated, prob_raw=probability, prob_calibrated=cal_probability,
        platt_slope=slope, platt_intercept=intercept,
        reconstruction_error_raw=np.abs(raw_sum - raw),
        reconstruction_error_calibrated=np.abs(cal_sum - calibrated),
    )
    repeated = observations.loc[observations.index.repeat(len(names)), LONG_COLUMNS[:5]].reset_index(drop=True)
    long = repeated.assign(feature=np.tile(names, len(frame)), feature_value=X.reshape(-1),
                           shap_value_raw=contributions[:, :-1].reshape(-1),
                           shap_value_calibrated=cal_values.reshape(-1))
    audit = {"xgboost_version": xgb.__version__, "method": "exact_tree_shap_xgboost_pred_contribs",
             "approx_contribs": False, "reconstruction_atol": tolerance, "reconstruction_rtol": tolerance,
             "iteration_range": list(iteration_range), "calibration_input": "raw_margin",
             "maximum_raw_reconstruction_error": float(np.abs(raw_sum - raw).max()),
             "maximum_calibrated_reconstruction_error": float(np.abs(cal_sum - calibrated).max())}
    return long[LONG_COLUMNS], observations[OBSERVATION_COLUMNS], audit


class ShapCollector:
    """Bounded deterministic OOS sample; root runner owns all artifact writes."""

    def __init__(self, variant_id: str, evaluation_start_day: str, evaluation_end_day: str, *,
                 live_day: str | None = None, timezone: str = "Europe/Paris", sample_days_last: int = 7,
                 historical_stride_days: int = 7, historical_hour: int = 19):
        if not isinstance(variant_id, str) or not variant_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in variant_id):
            raise VariantExplainError("A simple explicit variant identifier is required.")
        self.variant_id = variant_id
        self.start = _day(evaluation_start_day, "evaluation_start_day")
        self.end = _day(evaluation_end_day, "evaluation_end_day")
        if self.start > self.end or (pd.Timestamp(self.end) - pd.Timestamp(self.start)).days > 365:
            raise VariantExplainError("Bounded chronological evaluation range required.")
        self.live = _day(live_day, "live_day") if live_day else None
        if self.live and self.live != (pd.Timestamp(self.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"):
            raise VariantExplainError("The optional live day must immediately follow evaluation.")
        if timezone != "Europe/Paris":
            raise VariantExplainError("The qualified callback timezone is Europe/Paris.")
        for name, value, maximum, minimum in (("sample_days_last", sample_days_last, 31, 1),
                                              ("historical_stride_days", historical_stride_days, 365, 1),
                                              ("historical_hour", historical_hour, 23, 0)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise VariantExplainError(f"{name}: integer in [{minimum}, {maximum}] required.")
        self.timezone, self.last_days, self.stride, self.hour = timezone, sample_days_last, historical_stride_days, historical_hour
        self._long: list[pd.DataFrame] = []
        self._observations: list[pd.DataFrame] = []
        self._audits: list[dict] = []
        self._seen: set[tuple] = set()

    def _current(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"zone", "timestamp_utc", "forecast_origin_utc"}
        if frame.columns.has_duplicates or required.difference(frame):
            raise VariantExplainError("Current callback rows need unique columns and country/delivery/origin metadata.")
        current = frame.copy(deep=False).reset_index(drop=True)
        current = current.assign(timestamp_utc=_utc(current.timestamp_utc, "timestamp_utc"),
                                 forecast_origin_utc=_utc(current.forecast_origin_utc, "forecast_origin_utc"))
        civil = current.timestamp_utc.dt.tz_convert(self.timezone).dt.tz_localize(None)
        expected = (civil.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize(self.timezone).dt.tz_convert("UTC")
        if (not current.forecast_origin_utc.eq(expected).all() or current.forecast_origin_utc.nunique() != 1
                or not current.timestamp_utc.eq(current.timestamp_utc.dt.floor("h")).all()
                or current.duplicated(["zone", "timestamp_utc"]).any()
                or not current.zone.isin(["FR", "DE", "BE", "NL"]).all()):
            raise VariantExplainError("Explain one current D-1 08:00 origin at a time, not a future/history panel.")
        day = civil.dt.strftime("%Y-%m-%d")
        in_evaluation = day.ge(self.start) & day.le(self.end)
        recent = in_evaluation & day.ge((pd.Timestamp(self.end) - pd.Timedelta(days=self.last_days - 1)).strftime("%Y-%m-%d"))
        sparse = in_evaluation & ((civil.dt.normalize() - pd.Timestamp(self.start)).dt.days % self.stride).eq(0) & civil.dt.hour.eq(self.hour)
        live = day.eq(self.live) if self.live else pd.Series(False, index=current.index)
        selected = recent | sparse | live
        current = current.loc[selected].copy()
        current["sampling_rule"] = np.select([live.loc[selected], recent.loc[selected]], ["live_day_all_hours", "last_evaluation_days_all_hours"], default="earlier_weekly_fixed_hour")
        current["variant_id"] = self.variant_id
        current["sample_id"] = [f"{self.variant_id}|{z}|{t.isoformat()}" for z, t in zip(current.zone, current.timestamp_utc)]
        return current

    def observe(self, state: Mapping[str, Any] | None, frame: pd.DataFrame, parameters: Mapping[str, Any]) -> None:
        """Daily on_predict callback; outcomes and error sizes never select rows."""
        if frame.empty:
            return
        _validate_feature_names(parameters.get("feature_columns"))
        current = self._current(frame)
        if current.empty:
            return
        keys = list(zip(current.zone, current.timestamp_utc))
        if self._seen.intersection(keys):
            raise VariantExplainError("An explained delivery cannot be revisited using another fitted state.")
        self._seen.update(keys)
        valid = pd.Series(True, index=current.index)
        for name in ("_features_valid", "feature_eligible", "forecast_eligible"):
            if name in current:
                valid &= _flag(current[name], name)
        if "feature_available_at_utc" in current:
            available = pd.to_datetime(current.feature_available_at_utc, utc=True, format="mixed")
            if any(pd.Timestamp(v).tzinfo is None for v in current.feature_available_at_utc.dropna()):
                raise VariantExplainError("Feature availability metadata requires explicit timezone.")
            valid &= available.notna() & available.le(current.forecast_origin_utc)
        required = parameters.get("required_feature_columns", parameters.get("feature_columns", []))
        if (not isinstance(required, list) or set(required).difference(current)
                or not set(required).issubset(parameters["feature_columns"])):
            raise VariantExplainError("Missing required feature contract in callback.")
        if required:
            valid &= current[required].notna().all(axis=1)
        pending = current.loc[valid]
        skipped = current.loc[~valid].copy()
        skipped["skip_reason"] = "features_unavailable_at_origin"
        if state is None:
            skipped = pd.concat([skipped, pending.assign(skip_reason="no_fitted_state_at_origin")])
            pending = pending.iloc[0:0]
        elif not state.get("classifier").__class__.__module__.startswith("xgboost"):
            skipped = pd.concat([skipped, pending.assign(skip_reason="non_xgboost_classifier_no_fabricated_shap")])
            pending = pending.iloc[0:0]
        if not skipped.empty:
            skipped = skipped.assign(status="not_explained").reindex(columns=OBSERVATION_COLUMNS)
            self._observations.append(_observations(skipped))
        if not pending.empty:
            long, observations, audit = explain_xgb(state, pending, parameters)
            self._long.append(long)
            self._observations.append(_observations(observations))
            self._audits.append(audit)

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return full sampled contributions and observation identities, no I/O."""
        long = pd.concat(self._long, ignore_index=True) if self._long else pd.DataFrame(columns=LONG_COLUMNS)
        observed = pd.concat(self._observations, ignore_index=True) if self._observations else pd.DataFrame(columns=OBSERVATION_COLUMNS)
        return long, observed

    def summary(self) -> dict:
        """Small report payload; full Parquet export is the root runner's job."""
        long, observed = self.frames()
        explained = observed.loc[observed.status.eq("explained")]
        skip_counts = Counter(observed.loc[observed.status.ne("explained"), "skip_reason"].dropna())
        importance = []
        if not long.empty:
            grouped = long.assign(raw=long.shap_value_raw.abs(), calibrated=long.shap_value_calibrated.abs()).groupby("feature", sort=True)
            global_frame = grouped.agg(mean_abs_shap_raw=("raw", "mean"), mean_abs_shap_calibrated=("calibrated", "mean"))
            importance = global_frame.sort_values(["mean_abs_shap_calibrated", "mean_abs_shap_raw"], ascending=False).reset_index().to_dict("records")
        local_cases = []
        if not explained.empty:
            civil = explained.timestamp_utc.dt.tz_convert(self.timezone)
            focus = civil.dt.strftime("%Y-%m-%d").isin([self.end, self.live]) & civil.dt.hour.eq(self.hour)
            for row in explained.loc[focus].sort_values(["timestamp_utc", "zone"]).to_dict("records"):
                values = long.loc[long.sample_id.eq(row["sample_id"]), ["feature", "feature_value", "shap_value_raw", "shap_value_calibrated"]]
                row["contributions"] = values.assign(_importance=values.shap_value_calibrated.abs()).sort_values("_importance", ascending=False).drop(columns="_importance").to_dict("records")
                local_cases.append(row)
        result = {
            "variant_id": self.variant_id,
            "status": "complete" if len(explained) and not skip_counts else ("partial" if len(explained) else "unavailable"),
            "method": "exact_tree_shap_xgboost_pred_contribs" if len(explained) else "no_tree_shap_available",
            "explained_output": "Raw classifier log-odds and affine Platt-calibrated classifier log-odds; not EUR/MWh, final price, or final governed correction.",
            "sampling": {"selection_uses_outcomes": False, "evaluation_start_day": self.start, "evaluation_end_day": self.end,
                         "sample_days_last": self.last_days, "historical_stride_days": self.stride, "historical_hour_local": self.hour,
                         "live_day": self.live, "timezone": self.timezone,
                         "global_importance_scope": "Mean absolute contributions over the explicitly selected callback sample, not the uniform annual population.",
                         "selected_rows_received": len(observed), "explained_rows": len(explained), "skipped_by_reason": dict(skip_counts)},
            "global_importance": importance, "local_cases": local_cases, "paths": {},
            "audit": {"no_refit": True, "no_external_data_reads": True, "no_writes": True, "labels_used_in_explanation": False,
                      "approx_contribs": False, "calibration_input": "raw_margin", "contributions_unit": "classifier_log_odds",
                      "not_causal_effects": True, "not_final_price_contributions": True,
                      "out_of_sample_scope": "The daily prediction callback supplies its own fitted state; model fit cutoff checked against each explained origin. Training data are not independently re-audited here.",
                      "xgboost_versions": sorted({a["xgboost_version"] for a in self._audits}),
                      "reconstruction_verified": bool(len(explained)),
                      "reconstruction_atol": 2e-5, "reconstruction_rtol": 2e-5,
                      "maximum_raw_reconstruction_error": max((a["maximum_raw_reconstruction_error"] for a in self._audits), default=None),
                      "maximum_calibrated_reconstruction_error": max((a["maximum_calibrated_reconstruction_error"] for a in self._audits), default=None),
                      "unique_fitted_models": int(explained.model_id.nunique())},
        }
        # JSON-safe summaries without non-standard NaNs; Parquet preserves missing features.
        return _json_safe(result)
