"""Predeclared, past-only governance for an isolated fundamental expert.

This module never discovers features, loads Storm, fits the expert, or changes
production. ``risk`` must be computed upstream from information available at
the forecast cutoff. Historical base/expert predictions must be genuinely
out-of-sample; explicit flags document that caller contract, not a PIT proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class GovernanceError(ValueError):
    """Invalid governance contract; no implicit repairs are performed."""


@dataclass(frozen=True)
class GuardConfig:
    schema_version: int = 1
    policy_version: str = "physical_regime_prequential_v1"
    timezone: str = "Europe/Paris"
    weights: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    regimes: tuple[str, ...] = ("tight", "surplus")
    lookback_days: int = 365
    min_history_days: int = 60
    min_regime_days: int = 14
    update_every_days: int = 7
    minimum_gain_eur_mwh: float = 0.05
    minimum_relative_gain: float = 0.005
    maximum_weight_step: float = 0.25
    maximum_adverse_day_increase_eur_mwh: float = 5.0
    require_oof_evidence: bool = True
    require_label_availability: bool = True

    def __post_init__(self) -> None:
        # YAML/JSON lists and Python tuples describe the same fixed recipe.
        object.__setattr__(self, "weights", tuple(self.weights))
        object.__setattr__(self, "regimes", tuple(self.regimes))
        if self.schema_version != 1:
            raise GovernanceError("Unsupported governance schema_version.")
        if (not self.weights or self.weights[0] != 0
                or tuple(sorted(set(self.weights))) != self.weights
                or any(not np.isfinite(w) or not 0 <= w <= 0.5 for w in self.weights)):
            raise GovernanceError("weights must be unique, sorted, start at 0, and stay in [0, 0.5].")
        if not self.regimes or len(set(self.regimes)) != len(self.regimes) or "neutral" in self.regimes:
            raise GovernanceError("regimes must be unique active names, excluding neutral.")
        for name in ("lookback_days", "min_history_days", "min_regime_days", "update_every_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise GovernanceError(f"{name} must be a positive integer.")
        if self.min_history_days > self.lookback_days:
            raise GovernanceError("min_history_days exceeds lookback_days.")
        for name in ("minimum_gain_eur_mwh", "minimum_relative_gain", "maximum_weight_step",
                     "maximum_adverse_day_increase_eur_mwh"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise GovernanceError(f"{name} must be finite and nonnegative.")


@dataclass(frozen=True)
class GuardPolicy:
    cutoff_utc: str
    weights_by_zone: dict[str, dict[str, float]]
    decisions_by_zone: dict[str, dict[str, str]]
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardResult:
    predictions: pd.DataFrame
    policies: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)


def _aware_utc(values: Any, label: str) -> pd.DatetimeIndex:
    parsed = [pd.Timestamp(v) for v in values]
    if any(not pd.isna(v) and v.tzinfo is None for v in parsed):
        raise GovernanceError(f"{label}: explicit timezone required.")
    return pd.DatetimeIndex(pd.to_datetime(parsed, utc=True))


def _normalise(frame: pd.DataFrame, config: GuardConfig) -> pd.DataFrame:
    required = {"timestamp", "zone", "base", "expert", "actual", "risk"}
    if missing := required.difference(frame.columns):
        raise GovernanceError(f"Missing governance columns: {sorted(missing)}")
    data = frame.copy(deep=True).reset_index(drop=True)
    data["timestamp"] = _aware_utc(data["timestamp"], "timestamp")
    if data["timestamp"].isna().any() or data["zone"].isna().any():
        raise GovernanceError("timestamp and zone must be present.")
    if not pd.DatetimeIndex(data["timestamp"]).equals(pd.DatetimeIndex(data["timestamp"]).floor("h")):
        raise GovernanceError("Governance expects physical hourly timestamps.")
    data["zone"] = data["zone"].astype(str)
    if data.duplicated(["zone", "timestamp"]).any():
        raise GovernanceError("Duplicate physical hour for a zone.")
    for name in ("base", "expert", "actual"):
        data[name] = pd.to_numeric(data[name], errors="raise").astype(float)
    data["_day"] = data["timestamp"].dt.tz_convert(config.timezone).dt.tz_localize(None).dt.normalize()
    names = {"neutral", *config.regimes}
    data["_risk"] = data["risk"].where(data["risk"].isin(names), "neutral")
    data["_known_risk"] = data["risk"].isin(names)
    data["_expert_ok"] = np.isfinite(data["expert"])
    for name in ("expert_available", "risk_available"):
        if name in data:
            data["_expert_ok"] &= data[name].eq(True).fillna(False)
    if "expert_oof" in data:
        data["_oof"] = data["expert_oof"].eq(True).fillna(False)
    else:
        data["_oof"] = not config.require_oof_evidence
    # Existing issued comparator archives may not certify full neural OOF.
    # Absence remains a caller provenance contract, not invented certification;
    # an explicit negative flag, however, must fail closed.
    data["_base_oof"] = (data["baseline_oof"].eq(True).fillna(False)
                         if "baseline_oof" in data else True)
    if "label_available_at_utc" in data:
        data["_label_available"] = _aware_utc(data["label_available_at_utc"], "label_available_at_utc")
        data["_label_evidence"] = True
    elif config.require_label_availability:
        data["_label_available"] = pd.NaT
        data["_label_evidence"] = False
    else:
        # Explicitly opt-in assumption, never represented as provider evidence.
        local = data["_day"] - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
        data["_label_available"] = local.dt.tz_localize(config.timezone).dt.tz_convert("UTC")
        data["_label_evidence"] = False
    return data.sort_values(["zone", "timestamp"], kind="stable").reset_index(drop=True)


def _cutoff(day: pd.Timestamp, config: GuardConfig) -> pd.Timestamp:
    civil = pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    return civil.tz_localize(config.timezone).tz_convert("UTC")


def _complete_days(data: pd.DataFrame, config: GuardConfig) -> tuple[pd.DataFrame, int]:
    accepted = []
    skipped = 0
    for day, group in data.groupby("_day", sort=True):
        expected = pd.date_range(day.tz_localize(config.timezone),
                                 (day + pd.Timedelta(days=1)).tz_localize(config.timezone),
                                 freq="h", inclusive="left").tz_convert("UTC")
        timestamps = pd.DatetimeIndex(group["timestamp"])
        if timestamps.equals(expected):
            accepted.append(group)
        else:
            skipped += 1
    return (pd.concat(accepted, ignore_index=True) if accepted else data.iloc[:0].copy()), skipped


def calibrate(history_frame: pd.DataFrame, *, cutoff: Any, config: GuardConfig | None = None,
              previous_policy: GuardPolicy | None = None) -> GuardPolicy:
    """Choose small regime weights using complete, labelled days before cutoff.

    The gain and adverse-day guards describe the past calibration sample only.
    They do not guarantee non-regression on unseen days or on the final year.
    """
    settings = config or GuardConfig()
    data = _normalise(history_frame, settings)
    origin = _aware_utc([cutoff], "cutoff")[0]
    if pd.isna(origin):
        raise GovernanceError("cutoff must be present.")
    if previous_policy is not None and pd.Timestamp(previous_policy.cutoff_utc) >= origin:
        raise GovernanceError("previous_policy must precede the current cutoff.")
    local = origin.tz_convert(settings.timezone)
    if (local.hour, local.minute, local.second, local.microsecond) != (8, 0, 0, 0):
        raise GovernanceError("Governance cutoff must be civil D-1 08:00.")
    delivery = local.tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = delivery - pd.Timedelta(days=settings.lookback_days)
    within = (data["_day"] >= start) & (data["_day"] < delivery)
    finite = np.isfinite(data["base"]) & np.isfinite(data["actual"])
    available = pd.to_datetime(data["_label_available"], utc=True) <= origin
    admissible = data.loc[within & finite & available & data["_base_oof"]].copy()
    weights: dict[str, dict[str, float]] = {}
    decisions: dict[str, dict[str, str]] = {}
    zones_audit: dict[str, Any] = {}
    for zone in sorted(data["zone"].unique()):
        subset, skipped = _complete_days(admissible.loc[admissible["zone"].eq(zone)], settings)
        base_days = int(subset["_day"].nunique())
        # A long base-only prefix must not satisfy the expert's warmup. Count
        # complete joint OOS days, never thousands of hourly rows or a neural
        # warmup whose expert flag is false.
        jointly_eligible = subset["_expert_ok"] & subset["_oof"] & subset["_known_risk"]
        complete_joint = jointly_eligible.groupby(subset["_day"]).all()
        joint_days = complete_joint.index[complete_joint]
        subset = subset.loc[subset["_day"].isin(joint_days)].copy()
        count = int(subset["_day"].nunique())
        weights[zone] = {name: 0.0 for name in settings.regimes}
        decisions[zone] = {name: "warmup" for name in settings.regimes}
        zone_audit: dict[str, Any] = {
            "complete_history_days": count, "hours": len(subset), "skipped_incomplete_days": skipped,
            "complete_base_days": base_days, "skipped_no_joint_oof_days": base_days - count,
            "history_start_day": None if not count else str(subset["_day"].min().date()),
            "history_end_day": None if not count else str(subset["_day"].max().date()),
            "full_365_calibration": count == 365,
            "regimes": {},
        }
        zones_audit[zone] = zone_audit
        if count < settings.min_history_days:
            continue
        error_base = (subset["base"] - subset["actual"]).abs()
        combined = subset["base"].copy()
        for regime in settings.regimes:
            eligible = subset["_risk"].eq(regime) & subset["_expert_ok"] & subset["_oof"]
            n_days = int(subset.loc[eligible, "_day"].nunique())
            entry: dict[str, Any] = {"days": n_days, "hours": int(eligible.sum()), "candidates": []}
            zone_audit["regimes"][regime] = entry
            if n_days < settings.min_regime_days:
                decisions[zone][regime] = "insufficient_regime_days"
                continue
            old_weight = (previous_policy.weights_by_zone.get(zone, {}).get(regime, 0.0)
                          if previous_policy is not None else 0.0)
            best_weight, best_mae = 0.0, float(error_base.loc[eligible].mean())
            for weight in settings.weights[1:]:
                if weight > old_weight + settings.maximum_weight_step + 1e-12:
                    continue
                candidate = subset["base"].copy()
                candidate.loc[eligible] += weight * (subset.loc[eligible, "expert"] - subset.loc[eligible, "base"])
                error = (candidate - subset["actual"]).abs()
                regime_mae = float(error.loc[eligible].mean())
                gain = float(error_base.loc[eligible].mean() - regime_mae)
                required_gain = max(settings.minimum_gain_eur_mwh,
                                    float(error_base.loc[eligible].mean()) * settings.minimum_relative_gain)
                daily_increase = (error - error_base).groupby(subset["_day"]).mean()
                worst_increase = float(daily_increase.max())
                annual_gain = float(error_base.mean() - error.mean())
                accepted = (gain >= required_gain and annual_gain >= -1e-12
                            and worst_increase <= settings.maximum_adverse_day_increase_eur_mwh + 1e-12)
                entry["candidates"].append({"weight": weight, "regime_mae": regime_mae,
                    "regime_gain": gain, "required_gain": required_gain,
                    "calibration_global_gain": annual_gain, "worst_day_mae_increase": worst_increase,
                    "accepted": bool(accepted)})
                if accepted and regime_mae < best_mae - 1e-12:
                    best_weight, best_mae = float(weight), regime_mae
            weights[zone][regime] = best_weight
            decisions[zone][regime] = "past_gain_validated" if best_weight else "no_admissible_past_gain"
            combined.loc[eligible] += best_weight * (subset.loc[eligible, "expert"] - subset.loc[eligible, "base"])
        combined_errors = (combined - subset["actual"]).abs()
        worst_combined = float((combined_errors - error_base).groupby(subset["_day"]).mean().max())
        # Distinct intraday regimes can coincide on a day. Enforce the same
        # downside guard after combining them, not merely per regime.
        if (combined_errors.mean() > error_base.mean() + 1e-12
                or worst_combined > settings.maximum_adverse_day_increase_eur_mwh + 1e-12):
            weights[zone] = {name: 0.0 for name in settings.regimes}
            decisions[zone] = {name: "combined_downside_guard" for name in settings.regimes}
            combined_errors = error_base
        zone_audit.update(base_mae=float(error_base.mean()), guarded_mae=float(combined_errors.mean()),
                          worst_combined_day_mae_increase=worst_combined)
    audit = {
        "config": asdict(settings), "zones": zones_audit,
        "selection": "complete_available_past_days_only; hour_weighted_MAE",
        "label_availability": ("explicit_caller_timestamps" if "label_available_at_utc" in history_frame
                               else "missing_fail_closed" if settings.require_label_availability
                               else "assumed_D_minus_1_noon_not_provider_evidence"),
        "oof_evidence": "caller_flags_not_independent_certification" if "expert_oof" in history_frame
                        else "missing_fail_closed" if settings.require_oof_evidence else "caller_assumption",
        "baseline_evidence": ("caller_oof_flags_not_neural_certification" if "baseline_oof" in history_frame
                              else "frozen_comparator_provenance_is_callers_responsibility"),
        "risk_evidence": "caller_supplied_physical_regimes; no_target_based_regime_fit_here",
        "training_window_semantics": "expanding_after_warmup_then_rolling_365",
        "future_non_regression_guarantee": False, "production_activation": False,
    }
    return GuardPolicy(origin.isoformat(), weights, decisions, audit)


def walkforward_guard(frame: pd.DataFrame, *, config: GuardConfig | None = None) -> GuardResult:
    """Replay weekly weight decisions without reading current/future outcomes.

    Neutral/invalid regimes, missing sources, insufficient evidence and cold
    starts preserve the base. The whole frame may contain future labels: only
    ``calibrate`` sees the admissible past subset. Weekly blocks are anchored to
    Monday in civil time, not to input row count or an event date.
    """
    settings = config or GuardConfig()
    data = _normalise(frame, settings)
    output = data.drop(columns=[name for name in data if name.startswith("_")]).copy()
    output["weight"] = 0.0
    output["guarded"] = output["base"]
    output["decision"] = "neutral"
    output["policy_cutoff_utc"] = None
    output["policy_history_days"] = 0
    output["policy_version"] = settings.policy_version
    policies: list[dict[str, Any]] = []
    for zone in sorted(data["zone"].unique()):
        zone_mask = data["zone"].eq(zone)
        days = sorted(data.loc[zone_mask, "_day"].unique())
        policy: GuardPolicy | None = None
        last_block: int | None = None
        # Monday 1970-01-05 is a calendar anchor, not a fitted event breakpoint.
        anchor = pd.Timestamp("1970-01-05")
        for raw_day in days:
            day = pd.Timestamp(raw_day)
            block = (day - anchor).days // settings.update_every_days
            if policy is None or block != last_block:
                policy = calibrate(frame.loc[frame["zone"].astype(str).eq(zone)],
                                   cutoff=_cutoff(day, settings), config=settings, previous_policy=policy)
                policies.append(policy.to_dict())
                last_block = block
            mask = zone_mask & data["_day"].eq(day)
            rows = data.loc[mask]
            learned = policy.weights_by_zone.get(zone, {})
            decisions = policy.decisions_by_zone.get(zone, {})
            for idx, row in rows.iterrows():
                regime = row["_risk"]
                decision = decisions.get(regime, "neutral")
                weight = float(learned.get(regime, 0.0))
                if not np.isfinite(row["base"]):
                    weight, decision = 0.0, "base_unavailable"
                elif not row["_known_risk"]:
                    weight, decision = 0.0, "unknown_risk"
                elif not row["_expert_ok"]:
                    weight, decision = 0.0, "expert_or_risk_unavailable"
                elif not row["_base_oof"]:
                    weight, decision = 0.0, "baseline_evidence_rejected"
                elif settings.require_oof_evidence and not row["_oof"]:
                    weight, decision = 0.0, "missing_oof_evidence"
                output.at[idx, "weight"] = weight
                output.at[idx, "guarded"] = (row["base"] if weight == 0 else
                    row["base"] + weight * (row["expert"] - row["base"]))
                output.at[idx, "decision"] = decision
                output.at[idx, "policy_cutoff_utc"] = policy.cutoff_utc
                output.at[idx, "policy_history_days"] = policy.audit["zones"].get(zone, {}).get("complete_history_days", 0)
    audit = {
        "schema_version": 1, "config": asdict(settings), "policies": len(policies),
        "rows": len(output), "nonzero_weight_rows": int(output["weight"].gt(0).sum()),
        "calibration_uses_current_day_labels": False,
        "warmup_is_base_identity": True, "selected_on_final_test": False,
        "evaluation_semantics": "predeclared_policy_learns_only_from_already_known_past_test_outcomes",
        "future_non_regression_guarantee": False, "production_activation": False,
    }
    return GuardResult(output, policies, audit)
