"""Live-only observed-target context validation.

This module deliberately lives outside ``hourly_contract.py`` because that
training contract is part of the immutable source-code checksum of the frozen
autonomous models.  Live orchestration may evolve without rewriting the
identity of an already trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import HourlyTargetContractError


def _assert_hour_alignment(index: pd.DatetimeIndex, name: str) -> None:
    aligned = (
        (index.minute == 0)
        & (index.second == 0)
        & (index.microsecond == 0)
        & (index.nanosecond == 0)
    )
    if bool(np.all(aligned)):
        return
    examples = [str(value) for value in index[~aligned][:3]]
    raise HourlyTargetContractError(
        f"{name}: timestamps non alignes sur l'heure: {examples}."
    )


@dataclass(frozen=True)
class LiveTargetContextAudit:
    """Audit the observed-price context immediately preceding a live day.

    This contract deliberately has no recovery or imputation mode.  A recent
    missing suffix is reported separately from an internal historical gap so
    the operational error is actionable, but both remain blocking: the
    existing 23/24/25-hour Chronos live adapter may only start one hour after
    the last *observed* target.
    """

    status: str
    target_start_utc: pd.Timestamp
    actual_target_end_utc: pd.Timestamp
    required_target_end_utc: pd.Timestamp
    delivery_start_utc: pd.Timestamp
    delivery_end_utc: pd.Timestamp
    delivery_hours: int
    missing_final_suffix_hours: int
    internal_gap_hours: int
    target_hours_after_required_end: int
    actual_values_imputed: bool = False

    @property
    def is_exact(self) -> bool:
        return self.status == "exact_observed_context"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe audit payload for run diagnostics."""

        return {
            "status": self.status,
            "target_start_utc": self.target_start_utc.isoformat(),
            "actual_target_end_utc": self.actual_target_end_utc.isoformat(),
            "required_target_end_utc": self.required_target_end_utc.isoformat(),
            "delivery_start_utc": self.delivery_start_utc.isoformat(),
            "delivery_end_utc": self.delivery_end_utc.isoformat(),
            "delivery_hours": int(self.delivery_hours),
            "missing_final_suffix_hours": int(
                self.missing_final_suffix_hours
            ),
            "internal_gap_hours": int(self.internal_gap_hours),
            "target_hours_after_required_end": int(
                self.target_hours_after_required_end
            ),
            "actual_values_imputed": False,
            "forecast_allowed": bool(self.is_exact),
        }

    def raise_if_not_exact(self, *, zone: str) -> None:
        """Fail closed with a precise, non-imputation diagnostic."""

        if self.is_exact:
            return
        prefix = f"{str(zone).upper()}: contexte cible live bloque"
        if self.internal_gap_hours:
            reason = (
                f"{self.internal_gap_hours} heure(s) manquante(s) a "
                "l'interieur de l'historique observe"
            )
        elif self.missing_final_suffix_hours:
            reason = (
                "suffixe final observe absent de "
                f"{self.missing_final_suffix_hours} heure(s)"
            )
        else:
            reason = (
                "la cible depasse la derniere heure autorisee de "
                f"{self.target_hours_after_required_end} heure(s)"
            )
        raise HourlyTargetContractError(
            f"{prefix}: {reason}; derniere observation reelle="
            f"{self.actual_target_end_utc.isoformat()}; derniere heure "
            f"requise={self.required_target_end_utc.isoformat()}. "
            "Aucune observation cible n'est interpolee ou fabriquee; "
            "le forecast J+1 n'est pas lance."
        )


def audit_live_target_context(
    target: pd.Series,
    delivery_index: pd.DatetimeIndex,
) -> LiveTargetContextAudit:
    """Describe whether ``target`` is the exact observed prefix of J+1.

    UTC is used for the continuity check, which makes the audit valid on both
    daylight-saving-time transition days.  A final missing suffix is never
    confused with an internal gap and no value is inserted in either case.
    """

    if not isinstance(target, pd.Series) or target.empty:
        raise TypeError("target doit etre une pandas.Series non vide.")
    if not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None:
        raise HourlyTargetContractError(
            "target doit avoir un DatetimeIndex timezone-aware."
        )
    if target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise HourlyTargetContractError(
            "target doit avoir un index unique et croissant."
        )
    numeric = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise HourlyTargetContractError(
            "target contient une observation manquante ou non finie."
        )
    if not isinstance(delivery_index, pd.DatetimeIndex) or delivery_index.empty:
        raise TypeError("delivery_index doit etre un DatetimeIndex non vide.")
    if delivery_index.tz is None:
        raise HourlyTargetContractError(
            "delivery_index doit etre timezone-aware."
        )
    target_utc = target.index.tz_convert("UTC")
    delivery_utc = delivery_index.tz_convert("UTC")
    if delivery_utc.has_duplicates or not delivery_utc.is_monotonic_increasing:
        raise HourlyTargetContractError(
            "delivery_index doit etre unique et croissant."
        )
    _assert_hour_alignment(target_utc, "target")
    _assert_hour_alignment(delivery_utc, "delivery_index")
    expected_delivery = pd.date_range(
        delivery_utc[0],
        periods=len(delivery_utc),
        freq="h",
        tz="UTC",
    )
    if not expected_delivery.equals(delivery_utc):
        raise HourlyTargetContractError(
            "delivery_index doit etre horaire et continu en UTC."
        )

    expected_target = pd.date_range(
        target_utc[0],
        target_utc[-1],
        freq="h",
        tz="UTC",
    )
    internal_gap_hours = int(len(expected_target.difference(target_utc)))
    required_end = delivery_utc[0] - pd.Timedelta(hours=1)
    delta_hours = int(
        abs((required_end - target_utc[-1]) / pd.Timedelta(hours=1))
    )
    missing_suffix = delta_hours if target_utc[-1] < required_end else 0
    hours_after = delta_hours if target_utc[-1] > required_end else 0
    if internal_gap_hours:
        status = "internal_gap_blocked"
    elif missing_suffix:
        status = "missing_final_suffix_blocked"
    elif hours_after:
        status = "target_after_required_end_blocked"
    else:
        status = "exact_observed_context"
    return LiveTargetContextAudit(
        status=status,
        target_start_utc=target_utc[0],
        actual_target_end_utc=target_utc[-1],
        required_target_end_utc=required_end,
        delivery_start_utc=delivery_utc[0],
        delivery_end_utc=delivery_utc[-1],
        delivery_hours=len(delivery_utc),
        missing_final_suffix_hours=missing_suffix,
        internal_gap_hours=internal_gap_hours,
        target_hours_after_required_end=hours_after,
    )
