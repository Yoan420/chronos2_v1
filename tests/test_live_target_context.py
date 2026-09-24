from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import (
    HourlyTargetContractError,
    local_delivery_day_index,
)
from chronos2_hourly.live_target_context import audit_live_target_context


def _target_until(
    end: pd.Timestamp,
    *,
    hours: int = 240,
) -> pd.Series:
    index = pd.date_range(
        end=end,
        periods=hours,
        freq="h",
        tz="UTC",
    )
    return pd.Series(np.arange(hours, dtype=float), index=index, name="target")


@pytest.mark.parametrize("delivery_day", ["2026-03-29", "2026-08-31", "2026-10-25"])
def test_live_target_context_accepts_only_the_exact_observed_prefix(
    delivery_day: str,
) -> None:
    delivery = local_delivery_day_index(delivery_day, timezone="Europe/Paris")
    target = _target_until(delivery[0] - pd.Timedelta(hours=1))

    audit = audit_live_target_context(target, delivery)

    assert audit.is_exact
    assert audit.status == "exact_observed_context"
    assert audit.delivery_hours == len(delivery)
    assert audit.missing_final_suffix_hours == 0
    assert audit.internal_gap_hours == 0
    assert audit.actual_values_imputed is False
    assert audit.as_dict()["forecast_allowed"] is True


def test_live_target_context_reports_a_final_suffix_without_filling_it() -> None:
    delivery = local_delivery_day_index("2026-09-02", timezone="Europe/Madrid")
    target = _target_until(delivery[0] - pd.Timedelta(hours=49))

    audit = audit_live_target_context(target, delivery)

    assert audit.status == "missing_final_suffix_blocked"
    assert audit.missing_final_suffix_hours == 48
    assert audit.internal_gap_hours == 0
    assert audit.as_dict()["actual_values_imputed"] is False
    assert audit.as_dict()["forecast_allowed"] is False
    with pytest.raises(
        HourlyTargetContractError,
        match=r"ES: contexte cible live bloque.*48 heure.*Aucune observation",
    ):
        audit.raise_if_not_exact(zone="ES")


def test_live_target_context_distinguishes_an_internal_gap_from_a_suffix() -> None:
    delivery = local_delivery_day_index("2026-09-02", timezone="Europe/Paris")
    target = _target_until(delivery[0] - pd.Timedelta(hours=1))
    target = target.drop(target.index[-12])

    audit = audit_live_target_context(target, delivery)

    assert audit.status == "internal_gap_blocked"
    assert audit.internal_gap_hours == 1
    assert audit.missing_final_suffix_hours == 0
    with pytest.raises(HourlyTargetContractError, match=r"interieur.*historique"):
        audit.raise_if_not_exact(zone="FR")


def test_live_target_context_rejects_a_target_extending_into_delivery() -> None:
    delivery = local_delivery_day_index("2026-09-02", timezone="Europe/Paris")
    target = _target_until(delivery[0] + pd.Timedelta(hours=2))

    audit = audit_live_target_context(target, delivery)

    assert audit.status == "target_after_required_end_blocked"
    assert audit.target_hours_after_required_end == 3
    with pytest.raises(HourlyTargetContractError, match=r"depasse.*3 heure"):
        audit.raise_if_not_exact(zone="FR")
