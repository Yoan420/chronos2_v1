"""Synthetic-only tests of the isolated full-quarter-hour residual port."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder, ResidualCorrector
from nyx_fullquarterhour.residual import build_meta_features, fit_predict_day


RECIPE = dict(enabled=True, base_model="chronos2", backend="catboost", iterations=700,
              depth=6, learning_rate=.03, l2_leaf_reg=15, min_samples_leaf=30,
              min_training_rows=720, random_state=42, thread_count=4, verbose=False,
              correction_scale=1., max_abs_correction=40.)


def fixture(start="2026-01-01", days=32, frequency="15min"):
    first = pd.Timestamp(start, tz="Europe/Paris")
    end = (pd.Timestamp(start)+pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    index = pd.date_range(first, end, freq=frequency, inclusive="left").tz_convert("UTC")
    elapsed = (index-index[0]).total_seconds().to_numpy()/3600
    local = index.tz_convert("Europe/Paris")
    hour = local.hour.to_numpy()+local.minute.to_numpy()/60
    X = pd.DataFrame(index=index)
    for number, zone in enumerate(("fr", "de", "be", "nl", "es")):
        X[f"known_{zone}_residual_load_fcst_oracle"] = 20+number+np.sin(elapsed/8)
    X["known_fr_nuclear_generation_fcst_gw_oracle"] = 40+np.sin(elapsed/72)
    X["known_hour_sin"] = np.sin(2*np.pi*hour/24)
    X["known_hour_cos"] = np.cos(2*np.pi*hour/24)
    X["known_dow_sin"] = np.sin(2*np.pi*local.dayofweek.to_numpy()/7)
    X["known_dow_cos"] = np.cos(2*np.pi*local.dayofweek.to_numpy()/7)
    X["known_is_weekend"] = (local.dayofweek.to_numpy()>=5).astype(float)
    median = 50+10*np.sin(elapsed/6)
    raw = pd.DataFrame({"q10":median-8,"q50":median,"q90":median+12},index=index)
    actual = pd.Series(median+3, index=index, name="synthetic_actual")
    return X, raw, actual


def prepared(start="2026-01-01", days=32, frequency="15min"):
    X,raw,actual=fixture(start,days,frequency)
    meta=build_meta_features(X,raw,frequency=frequency)
    last=meta.index.tz_convert("Europe/Paris").date[-1]
    before=meta.index.tz_convert("Europe/Paris").date<last
    return (meta.loc[before],actual.loc[before],raw.loc[before],meta.loc[~before],raw.loc[~before])


class CapturingModel:
    def __init__(self, correction=100.):
        self.correction=correction
    def fit(self,X,y):
        self.train=X.copy()
        self.y=np.asarray(y).copy()
    def predict(self,X):
        self.future=X.copy()
        return np.full(len(X),self.correction)
    def get_params(self):
        return {"fixture":True}


def test_hourly_port_matches_all_production_features_exactly():
    X,raw,_=fixture(days=3,frequency="h")
    experts=pd.concat([raw.add_prefix("base__"),raw.add_prefix("chronos2__")],axis=1)
    production=ResidualMetaFeatureBuilder(timezone="Europe/Paris",include_rich_calendar=True,
        rich_calendar_primary_country="FR").fit_transform(X,experts)
    experimental=build_meta_features(X,raw,frequency="h")
    pd.testing.assert_frame_equal(experimental,production)
    assert len(experimental.columns)==182


def test_quarter_calendar_and_physical_hour_ramps():
    X,raw,_=fixture(days=2)
    raw.loc[:,"q50"]=np.arange(len(raw),dtype=float)
    raw.loc[:,"q10"]=raw.q50-1
    raw.loc[:,"q90"]=raw.q50+1
    meta=build_meta_features(X,raw)
    assert meta.iloc[1].calendar_local_hour==.25
    assert meta.iloc[1].calendar_hour_sin==pytest.approx(np.sin(2*np.pi*.25/24))
    assert meta.iloc[3]["chronos2__q50__ramp_1h"]==0
    assert meta.iloc[4]["chronos2__q50__ramp_1h"]==4
    assert meta.iloc[8]["chronos2__q50__ramp_2h"]==8
    assert meta.iloc[96]["chronos2__q50__ramp_1h"]==0
    assert meta.iloc[96]["chronos2__q50__ramp_2h"]==0
    assert not any("doy" in c or "day_of_year" in c for c in meta)


@pytest.mark.parametrize("day,rows",[("2026-03-29",92),("2025-10-26",100)])
def test_complete_physical_dst_days(day,rows):
    X,raw,_=fixture(day,1)
    meta=build_meta_features(X,raw)
    assert len(meta)==rows
    assert meta.index.is_unique
    assert meta.known_cal_dst_transition_day_oracle.eq(1).all()
    if rows==100:
        assert meta.calendar_dst_fold.sum()==4
        assert set(meta.calendar_utc_offset_hours)=={1,2}


def test_global_precompute_equal_per_day_and_future_does_not_change_past():
    X,raw,_=fixture(days=3)
    original_X,original_raw=X.copy(),raw.copy()
    combined=build_meta_features(X,raw)
    pieces=[]
    for day in pd.Index(X.index.tz_convert("Europe/Paris").date).unique():
        mask=X.index.tz_convert("Europe/Paris").date==day
        pieces.append(build_meta_features(X.loc[mask],raw.loc[mask]))
    pd.testing.assert_frame_equal(combined,pd.concat(pieces))
    raw.iloc[-96:]+=1000
    perturbed=build_meta_features(X,raw)
    pd.testing.assert_frame_equal(combined.iloc[:-96],perturbed.iloc[:-96])
    pd.testing.assert_frame_equal(X,original_X)
    assert original_raw.iloc[0].q50==raw.iloc[0].q50


@pytest.mark.parametrize("case",["gap","partial","duplicate","naive","wrongzone","unsorted","nan","actual","crossed"])
def test_builder_rejects_incomplete_or_unsafe_inputs(case):
    X,raw,_=fixture(days=2)
    if case=="gap": X,raw=X.drop(X.index[6]),raw.drop(raw.index[6])
    elif case=="partial": X,raw=X.iloc[1:],raw.iloc[1:]
    elif case=="duplicate": X,raw=pd.concat([X,X.iloc[-1:]]),pd.concat([raw,raw.iloc[-1:]])
    elif case=="naive": X.index=X.index.tz_localize(None);raw.index=X.index
    elif case=="wrongzone": X.index=X.index.tz_convert("Europe/Paris");raw.index=X.index
    elif case=="unsorted": X,raw=X.iloc[::-1],raw.iloc[::-1]
    elif case=="nan": X.iloc[5,0]=np.nan
    elif case=="actual": X["actual"]=999.
    elif case=="crossed": raw.iloc[0,0]=raw.iloc[0].q90+1
    with pytest.raises(ValueError): build_meta_features(X,raw)


def test_production_historical_price_features_are_excluded():
    X,raw,_=fixture(days=2)
    X["price_lag_24"]=np.arange(len(X))
    X["calendar_day_of_year"]=1.
    meta=build_meta_features(X,raw)
    assert "price_lag_24" not in meta
    assert "calendar_day_of_year" not in meta


def test_daily_fit_uses_residual_label_and_caps_all_quantiles(monkeypatch):
    arrays=prepared()
    original=[x.copy(deep=True) for x in arrays]
    model=CapturingModel()
    monkeypatch.setattr(ResidualCorrector,"_new_model",lambda self:model)
    out,audit=fit_predict_day(*arrays,recipe=RECIPE)
    np.testing.assert_allclose(model.y,3.)
    np.testing.assert_allclose(out-arrays[-1],40.)
    np.testing.assert_allclose(out.q90-out.q10,arrays[-1].q90-arrays[-1].q10)
    assert model.train.index[-1]<model.future.index[0]
    assert audit["minimum_training_rows"]==2880
    assert audit["generation_source"]=="daily_prequential_refit"
    assert audit["clipped_count"]==96
    assert audit["target_observations_used"] is False
    assert len(audit["training_actual_sha256"])==64
    for a,b in zip(arrays,original):
        (pd.testing.assert_series_equal if isinstance(a,pd.Series) else pd.testing.assert_frame_equal)(a,b)


@pytest.mark.parametrize("frequency,rows",[("h",720),("15min",2880)])
def test_cold_start_preserves_duration_and_does_not_load_backend(monkeypatch,frequency,rows):
    monkeypatch.setattr(ResidualCorrector,"_new_model",lambda self:pytest.fail("No fit below threshold"))
    data=prepared(days=30,frequency=frequency)
    out,audit=fit_predict_day(*data,recipe=RECIPE,frequency=frequency)
    pd.testing.assert_frame_equal(out,data[-1])
    assert audit["minimum_training_rows"]==rows
    assert audit["training_days"]==29
    assert audit["generation_source"]=="identity_chronos_cold_start"


def test_spring_30_civil_days_retains_physical_720_hour_threshold(monkeypatch):
    data=prepared("2026-03-01",31)
    monkeypatch.setattr(ResidualCorrector,"_new_model",lambda self:pytest.fail("2876 < 2880"))
    _,audit=fit_predict_day(*data,recipe=RECIPE)
    assert audit["training_days"]==30
    assert audit["training_rows"]==2876
    assert audit["generation_source"]=="identity_chronos_cold_start"


@pytest.mark.parametrize("case",["target_day_labels","missing_day","gap","partial_target","multiple_targets","schema","lookback","label_alignment","nanlabel","crossed"])
def test_daily_fit_rejects_temporal_or_schema_violations(case,monkeypatch):
    monkeypatch.setattr(ResidualCorrector,"_new_model",lambda self:pytest.fail("Invalid input must not fit"))
    mt,y,rt,md,rd=prepared()
    kwargs={}
    if case=="target_day_labels": mt=pd.concat([mt,md]);y=pd.concat([y,rd.q50]);rt=pd.concat([rt,rd])
    elif case=="missing_day": mt,y,rt=mt.iloc[:-96],y.iloc[:-96],rt.iloc[:-96]
    elif case=="gap": mt,y,rt=mt.drop(mt.index[9]),y.drop(y.index[9]),rt.drop(rt.index[9])
    elif case=="partial_target": md,rd=md.iloc[1:],rd.iloc[1:]
    elif case=="multiple_targets": md=pd.concat([mt.iloc[-96:],md]);rd=pd.concat([rt.iloc[-96:],rd])
    elif case=="schema": md=md.iloc[:,::-1]
    elif case=="lookback": kwargs["max_lookback_days"]=10
    elif case=="label_alignment": y=y.iloc[::-1]
    elif case=="nanlabel": y.iloc[1]=np.nan
    elif case=="crossed": rd.iloc[1,0]=rd.iloc[1].q90+1
    with pytest.raises(ValueError): fit_predict_day(mt,y,rt,md,rd,recipe=RECIPE,**kwargs)


def test_backend_matches_exact_production_parameters_without_training():
    options=deepcopy(RECIPE)
    options.pop("enabled");options.pop("base_model")
    corrector=ResidualCorrector(**options)
    corrector.backend_=corrector._resolve_backend()
    params=corrector._new_model().get_params()
    assert params["iterations"]==700 and params["depth"]==6
    assert params["loss_function"]=="MAE" and params["has_time"] is True
    assert params["allow_writing_files"] is False and params["nan_mode"]=="Min"
    assert params["learning_rate"]==.03 and params["l2_leaf_reg"]==15
    # The production CatBoost backend does not pass the sklearn-only setting.
    assert "min_samples_leaf" not in params


def test_same_inputs_reproduce_hashes_and_backend_is_fresh_each_day(monkeypatch):
    data=prepared()
    made=[]
    def new(self):
        model=CapturingModel(correction=-3.)
        made.append(model)
        return model
    monkeypatch.setattr(ResidualCorrector,"_new_model",new)
    first,a1=fit_predict_day(*data,recipe=RECIPE)
    second,a2=fit_predict_day(*data,recipe=RECIPE)
    pd.testing.assert_frame_equal(first,second)
    assert a1==a2 and len(made)==2 and made[0] is not made[1]


def test_alternate_backend_is_rejected():
    with pytest.raises(ValueError,match="genuine CatBoost"):
        fit_predict_day(*prepared(),recipe={**RECIPE,"backend":"sklearn"})
