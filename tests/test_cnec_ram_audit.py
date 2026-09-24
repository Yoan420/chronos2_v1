from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.jao_flowbased import FLOWBASED_FEATURE_COLUMNS
from run_cnec_ram_audit import CnecRamAuditError, build_audit


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    timezone = "Europe/Paris"
    delivery_day = date(2026, 9, 2)
    start_day = delivery_day - timedelta(days=365)
    index = pd.date_range(
        pd.Timestamp(start_day, tz=timezone),
        pd.Timestamp(delivery_day, tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    actual = 50.0 + np.sin(np.arange(len(index)) / 24.0)
    statistics = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": actual,
            "residual_corrected__q50": actual + 2.0,
        }
    )
    flow = pd.DataFrame(index=index)
    flow["flowbased_pit_eligible"] = True
    for position, column in enumerate(FLOWBASED_FEATURE_COLUMNS, start=1):
        flow[column] = position + np.linspace(0.0, 1.0, len(index))
    flow["flowbased_cnec_mtu_availability"] = 1.0
    flow["flowbased_missing_mtu_share"] = 0.0
    flow["flowbased_hour_imputed"] = 0.0
    return statistics, flow


def test_cnec_ram_audit_is_exactly_365_days_and_dst_complete() -> None:
    statistics, flow = _frames()
    paired, daily, summary = build_audit(
        statistics=statistics,
        flow=flow,
        timezone="Europe/Paris",
        delivery_day=date(2026, 9, 2),
    )
    assert summary["days"] == 365
    assert len(daily) == 365
    assert summary["hours"] == len(paired) == 8760
    assert summary["mae_eur_mwh"] == pytest.approx(2.0)
    assert summary["postcoupling_labels_used_as_input"] is False


def test_cnec_ram_audit_refuses_one_missing_flow_hour() -> None:
    statistics, flow = _frames()
    flow = flow.iloc[1:]
    with pytest.raises(CnecRamAuditError, match="absentes"):
        build_audit(
            statistics=statistics,
            flow=flow,
            timezone="Europe/Paris",
            delivery_day=date(2026, 9, 2),
        )


def test_cnec_ram_audit_refuses_a_non_pit_hour() -> None:
    statistics, flow = _frames()
    flow.iloc[0, flow.columns.get_loc("flowbased_pit_eligible")] = False
    with pytest.raises(CnecRamAuditError, match="non admissible"):
        build_audit(
            statistics=statistics,
            flow=flow,
            timezone="Europe/Paris",
            delivery_day=date(2026, 9, 2),
        )


def test_cnec_ram_audit_accepts_constant_features() -> None:
    statistics, flow = _frames()
    flow.loc[:, list(FLOWBASED_FEATURE_COLUMNS)] = 1.0
    paired, daily, summary = build_audit(
        statistics=statistics,
        flow=flow,
        timezone="Europe/Paris",
        delivery_day=date(2026, 9, 2),
    )
    assert len(paired) == 8760
    assert len(daily) == 365
    assert all(
        value is None for value in summary["spearman_absolute_error"].values()
    )
