"""Inference contracts with injected pipelines only; no weights or real prices."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_quarterhour import inference


class FakePipeline:
    model_context_length = 8192
    model_prediction_length = 1024

    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def predict_df(self, context, **kwargs):
        self.calls.append((context.copy(), kwargs))
        result = kwargs["future_df"][["item_id", "timestamp"]].copy()
        result["target_name"] = "target"
        result["0.1"], result["0.5"], result["0.9"], result["predictions"] = 1., 2., 3., 2.
        return self.transform(result) if self.transform else result


def frames(freq="15min", context_length=8, prediction_length=4, item="fixture_fr", start="2026-01-10"):
    future_index = pd.date_range(start, periods=prediction_length, freq=freq)
    context_index = pd.date_range(end=future_index[0]-pd.tseries.frequencies.to_offset(freq), periods=context_length, freq=freq)
    context = pd.DataFrame({"item_id": item, "timestamp": context_index, "target": np.arange(context_length)+10.,
                            "load_gw": np.ones(context_length)})
    future = pd.DataFrame({"item_id": item, "timestamp": future_index, "load_gw": np.ones(prediction_length)})
    return context, future


def infer(context, future, pipeline=None, **kwargs):
    return inference.infer_batch([context], [future], freq=kwargs.pop("freq", "15min"),
                                 context_length=kwargs.pop("context_length", len(context)),
                                 prediction_length=kwargs.pop("prediction_length", len(future)),
                                 pipeline=pipeline or FakePipeline(), **kwargs)


def test_native_forecast_call_is_local_protocol_and_does_not_modify_inputs():
    context, future = frames()
    before_context, before_future = context.copy(deep=True), future.copy(deep=True)
    pipeline = FakePipeline()
    result = infer(context, future, pipeline=pipeline, model_batch_size=1)
    assert list(result)==["item_id", "timestamp", "q10", "q50", "q90", "point"]
    assert result.q50.eq(2).all() and result.point.eq(2).all()
    call = pipeline.calls[0][1]
    assert call["cross_learning"] is False and call["validate_inputs"] is True
    assert call["freq"]=="15min" and call["context_length"]==8
    assert call["prediction_length"]==4 and call["quantile_levels"]==[.1,.5,.9]
    assert call["batch_size"]==2  # target + one known covariate, never split below a task.
    pd.testing.assert_frame_equal(context,before_context)
    pd.testing.assert_frame_equal(future,before_future)


def test_mixed_country_tasks_are_batched_without_cross_learning():
    first, future_first = frames(item="fixture_nl")
    second, future_second = frames(item="fixture_be")
    pipeline = FakePipeline()
    result = inference.infer_batch([first,second],[future_first,future_second],freq="15min",
                                    context_length=8,prediction_length=4,pipeline=pipeline)
    assert len(result)==8 and result.groupby("item_id").size().eq(4).all()
    assert not pipeline.calls[0][1]["cross_learning"]


@pytest.mark.parametrize("day,horizon",[("2026-03-29",92),("2026-10-25",100)])
def test_both_dst_folds_and_spring_short_day_remain_physical_quarters(day,horizon):
    begin=pd.Timestamp(day).tz_localize("Europe/Paris").tz_convert("UTC")
    context,future=frames(start=begin,prediction_length=horizon)
    result=infer(context,future)
    assert len(result)==horizon and result.timestamp.is_unique
    assert result.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    local=pd.DatetimeIndex(result.timestamp).tz_localize("UTC").tz_convert("Europe/Paris")
    assert (local.date==pd.Timestamp(day).date()).all()


@pytest.mark.parametrize("freq,context_length,horizon",[("15min",8192,96),("h",2048,24)])
def test_equal_physical_context_duration_and_frequency_specific_horizons(freq,context_length,horizon):
    context,future=frames(freq=freq,context_length=context_length,prediction_length=horizon)
    result=infer(context,future,freq=freq)
    assert len(result)==horizon
    assert future.timestamp.iloc[0]-context.timestamp.iloc[0]==pd.Timedelta(hours=2048)


def test_unknown_day_cannot_be_skipped_between_context_and_delivery():
    context,future=frames()
    future.timestamp+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match="immediately"):
        infer(context,future)
    # A full intermediate horizon is accepted; caller later selects delivery D.
    _,future=frames(prediction_length=100)
    assert len(infer(context,future))==100


@pytest.mark.parametrize("issue",["gap", "duplicate", "reverse", "nan", "future_target", "covariates", "ids"])
def test_invalid_inputs_are_rejected_before_calling_model(issue):
    context,future=frames()
    if issue=="gap": context=context.drop(index=2)
    elif issue=="duplicate": context.loc[1,"timestamp"]=context.timestamp.iloc[0]
    elif issue=="reverse": context=context.iloc[::-1]
    elif issue=="nan": future.loc[0,"load_gw"]=np.nan
    elif issue=="future_target": future["target"]=999.
    elif issue=="covariates": future=future.drop(columns="load_gw")
    elif issue=="ids": future["item_id"]="different"
    pipeline=FakePipeline()
    with pytest.raises(ValueError): infer(context,future,pipeline=pipeline)
    assert not pipeline.calls


def test_context_cap_cannot_be_silently_trimmed_or_horizon_unrolled():
    context,future=frames()
    with pytest.raises(ValueError,match="exceeds context_length"):
        infer(context,future,context_length=4)
    with pytest.raises(ValueError,match="native limits"):
        infer(context,future,context_length=8193)
    context,future=frames(prediction_length=1025)
    with pytest.raises(ValueError,match="native limits"):
        infer(context,future)


@pytest.mark.parametrize("change",[
    lambda f:f.iloc[:-1],
    lambda f:pd.concat([f,f.iloc[[0]]],ignore_index=True),
    lambda f:f.assign(timestamp=f.timestamp+pd.Timedelta(minutes=15)),
    lambda f:f.assign(**{"0.5":np.nan}),
    lambda f:f.assign(**{"0.1":10.}),
    lambda f:f.assign(target_name="wrong"),
    lambda f:f.drop(columns="0.1")])
def test_incomplete_wrong_or_invalid_model_outputs_are_not_repaired(change):
    context,future=frames()
    with pytest.raises(ValueError):
        infer(context,future,pipeline=FakePipeline(change))


def test_loader_forces_local_only_cpu_and_reuses_exact_model(monkeypatch,tmp_path):
    calls, settings = [], []
    pipeline=FakePipeline()
    chronos=ModuleType("chronos")
    def from_pretrained(model_id,**kwargs):
        calls.append((model_id,kwargs))
        return pipeline
    chronos.Chronos2Pipeline=SimpleNamespace(from_pretrained=from_pretrained)
    torch=ModuleType("torch")
    torch.float32="fixture-float32"
    torch.set_num_threads=lambda value:settings.append(("threads",value))
    torch.manual_seed=lambda value:settings.append(("seed",value))
    monkeypatch.setitem(sys.modules,"chronos",chronos)
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setattr(inference,"_PIPELINE",None)
    monkeypatch.setattr(inference,"_MODEL_ID",None)
    (tmp_path/"config.json").write_text("{}",encoding="utf-8")
    (tmp_path/"model.safetensors").write_bytes(b"fixture-not-real-weights")
    monkeypatch.setattr(inference,"_local_checkpoint",lambda model_id:str(tmp_path))
    monkeypatch.setattr(inference,"_CHECKPOINT_IDENTITY",None)
    monkeypatch.setattr(inference,"_CHECKPOINT_SIGNATURES",{})
    assert inference.load_pipeline("local/fixture",torch_threads=8,seed=42) is pipeline
    assert inference.load_pipeline("local/fixture") is pipeline
    assert calls==[(str(tmp_path),{"device_map":"cpu","local_files_only":True,"dtype":"fixture-float32"})]
    assert ("threads",8) in settings and ("seed",42) in settings
    identity=inference.checkpoint_identity()
    assert identity["checkpoint_path"]==str(tmp_path)
    assert len(identity["config_sha256"])==len(identity["weights_sha256"])==64
    assert identity["weights_bytes"]==(tmp_path/"model.safetensors").stat().st_size
    identity["device"]="wrong-device"
    assert inference.checkpoint_identity()["device"]=="cpu"
    with pytest.raises(RuntimeError,match="different model"):
        inference.load_pipeline("another/fixture")
    monkeypatch.setattr(inference,"_local_checkpoint",lambda model_id:str(tmp_path/"another_snapshot"))
    with pytest.raises(RuntimeError,match="cached model path changed"):
        inference.load_pipeline("local/fixture")
    monkeypatch.setattr(inference,"_local_checkpoint",lambda model_id:str(tmp_path))
    (tmp_path/"model.safetensors").write_bytes(b"changed-fixture")
    with pytest.raises(RuntimeError,match="checkpoint files changed"):
        inference.checkpoint_identity()


def test_checkpoint_can_be_sealed_before_any_pipeline_or_torch_loading(tmp_path,monkeypatch):
    (tmp_path/"config.json").write_text("{}",encoding="utf-8")
    (tmp_path/"model.safetensors").write_bytes(b"fixture-not-real-weights")
    monkeypatch.setattr(inference,"_PIPELINE",None)
    monkeypatch.setattr(inference,"_MODEL_ID",None)
    monkeypatch.setattr(inference,"_CHECKPOINT_IDENTITY",None)
    monkeypatch.setattr(inference,"_CHECKPOINT_SIGNATURES",{})
    monkeypatch.setitem(sys.modules,"chronos",ModuleType("chronos"))
    monkeypatch.setitem(sys.modules,"torch",ModuleType("torch"))
    identity=inference.checkpoint_identity(str(tmp_path))
    assert len(identity["weights_sha256"])==64
    assert identity["snapshot"]==tmp_path.name
    assert inference._PIPELINE is None
    assert inference.checkpoint_identity()==identity
    # Public rechecks hash bytes even when size and timestamp are preserved.
    import os
    weights=tmp_path/"model.safetensors"
    stat=weights.stat()
    payload=weights.read_bytes()
    weights.write_bytes(b"X"+payload[1:])
    os.utime(weights,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(RuntimeError,match="checkpoint files changed"):
        inference.checkpoint_identity()


def test_cached_repository_id_resolves_to_directory_without_download(tmp_path,monkeypatch):
    snapshot=tmp_path/"snapshots/fixture"
    snapshot.mkdir(parents=True)
    (snapshot/"config.json").write_text("{}",encoding="utf-8")
    (snapshot/"model.safetensors").write_bytes(b"fixture-not-real-weights")
    calls=[]
    hub=ModuleType("huggingface_hub")
    def lookup(model_id,filename,**kwargs):
        calls.append((model_id,filename,kwargs))
        return str(snapshot/filename)
    hub.try_to_load_from_cache=lookup
    monkeypatch.setitem(sys.modules,"huggingface_hub",hub)
    assert inference._local_checkpoint("fixture/cached-model")==str(snapshot.resolve())
    assert calls==[("fixture/cached-model","config.json",{"revision":"main"})]
    hub.try_to_load_from_cache=lambda *a,**k:None
    with pytest.raises(FileNotFoundError,match="downloads are disabled"):
        inference._local_checkpoint("fixture/not-cached")


def test_inference_never_implicitly_loads_a_model(monkeypatch):
    context,future=frames()
    monkeypatch.setattr(inference,"_PIPELINE",None)
    with pytest.raises(RuntimeError,match="load_pipeline"):
        inference.infer_batch([context],[future],freq="15min",context_length=8,prediction_length=4)
