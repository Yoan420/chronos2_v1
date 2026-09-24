from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyx_quarterhour.data import MatchedInputs, calendar, day_index
from nyx_intrahour.data import HOURLY_ALIASES, ZONES


def fixture(day="2026-06-18"):
    start = (pd.Timestamp(day)-pd.Timedelta(days=2)).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start,end,freq="15min",inclusive="left").tz_convert("UTC")
    native = pd.concat([pd.DataFrame({"timestamp_utc":index,"zone":zone,
                         "actual_15m":np.arange(len(index),dtype=float)+i}) for i,zone in enumerate(ZONES)],ignore_index=True)
    hours = index[::4]
    base = pd.concat([pd.DataFrame({"timestamp_utc":hours,"zone":zone,"actual":10.,"nyx_q50":11.,
                                  **{f"feature_hourly_{a}":float(i+1) for i,a in enumerate(HOURLY_ALIASES)}})
                      for zone in ZONES],ignore_index=True)
    return native,base


@pytest.mark.parametrize("day,length",[("2026-06-18",96),("2026-03-29",92),("2025-10-26",100)])
def test_matched_context_duration_and_physical_delivery(day,length):
    native,base=fixture(day)
    data=MatchedInputs(native,base)
    qcontext,qfuture,qaudit=data.build_origin(day,"15min",context_hours=24)
    hcontext,hfuture,haudit=data.build_origin(day,"h",context_hours=24)
    for qc,qf,hc,hf in zip(qcontext,qfuture,hcontext,hfuture):
        assert len(qc)==96 and len(hc)==24
        assert len(qf)==length and len(hf)==length//4
        assert qc.timestamp.min()==hc.timestamp.min()
        assert qc.timestamp.max()+pd.Timedelta(minutes=15)==qf.timestamp.min()
        assert hc.timestamp.max()+pd.Timedelta(hours=1)==hf.timestamp.min()
        np.testing.assert_allclose(qc.target.to_numpy().reshape(-1,4).mean(axis=1),hc.target)
        assert "target" not in qf and "target" not in hf
        assert set(qf)-{"item_id","timestamp"}==set(hf)-{"item_id","timestamp"}
    assert {a["context_hours"] for a in qaudit+haudit}=={24}
    assert all(a["future_realized_target_used"] is False for a in qaudit+haudit)


def test_future_target_perturbation_does_not_change_any_model_input():
    native,base=fixture()
    before=MatchedInputs(native,base)
    changed=native.copy()
    changed.loc[changed.timestamp_utc>=day_index("2026-06-18","15min")[0],"actual_15m"]+=999999
    after=MatchedInputs(changed,base)
    for freq in ("h","15min"):
        old=before.build_origin("2026-06-18",freq,context_hours=24)
        new=after.build_origin("2026-06-18",freq,context_hours=24)
        for group in (0,1):
            for left,right in zip(old[group],new[group]):
                pd.testing.assert_frame_equal(left,right)


def test_calendar_contains_subhour_time_and_dst_offsets():
    index=day_index("2025-10-26","15min")
    values=calendar(index)
    assert values.known_hour_sin.iloc[0]!=values.known_hour_sin.iloc[1]
    assert set(values.known_utc_offset_hours)=={1.,2.}
    assert len(values)==100


@pytest.mark.parametrize("failure",["gap","nan","duplicate","naive","one_country_missing"])
def test_native_target_defects_are_rejected(failure):
    native,base=fixture()
    if failure=="gap": native=native.drop(index=0)
    if failure=="nan": native.loc[0,"actual_15m"]=np.nan
    if failure=="duplicate": native=pd.concat([native,native.iloc[[0]]])
    if failure=="naive": native.timestamp_utc=native.timestamp_utc.dt.tz_localize(None)
    if failure=="one_country_missing": native=native.loc[native.zone.ne("NL")]
    with pytest.raises(ValueError): MatchedInputs(native,base)


def test_short_history_and_missing_forecast_covariate_refuse_inference():
    native,base=fixture()
    data=MatchedInputs(native,base)
    with pytest.raises(ValueError,match="insufficient genuine history"):
        data.build_origin("2026-06-18","15min",context_hours=2048)
    limited=base.loc[base.timestamp_utc<day_index("2026-06-18","h")[0]]
    with pytest.raises(ValueError,match="fundamentals do not cover"):
        MatchedInputs(native,limited).build_origin("2026-06-18","h",context_hours=24)


def test_scoring_uses_four_quarter_mean_and_preserves_archived_labels():
    native,base=fixture()
    original=base.copy(deep=True)
    result,audit=MatchedInputs(native,base).evaluation_baseline(base,"2026-06-18","2026-06-18")
    assert len(result)==96
    assert result.archived_hourly_actual.eq(10).all()
    for row in result.itertuples():
        expected=native.loc[native.zone.eq(row.zone)&native.timestamp_utc.between(row.timestamp_utc,row.timestamp_utc+pd.Timedelta(minutes=45)),"actual_15m"].mean()
        assert row.actual==expected
    assert audit["scoring_target"]=="arithmetic_mean_of_four_native_quarter_hour_prices"
    pd.testing.assert_frame_equal(base,original)
