"""Future labels, physical DST hours and numerical residual/Kalman boundaries."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_baseline as producer


def test_chronos_identity_binds_exact_weights_and_refuses_unpinned_or_escaping_files():
    from copy import deepcopy
    identity = {"model_id": "amazon/chronos-2", "revision": producer.CHRONOS_REVISION,
                "device": "cpu", "dtype": "torch.float32",
                "files": {"config.json": "a" * 64, "model.safetensors": "b" * 64}}
    first = producer.chronos_identity_digest(identity)
    changed = deepcopy(identity)
    changed["files"]["model.safetensors"] = "c" * 64
    assert producer.chronos_identity_digest(changed) != first
    for patch in ({"device": "cuda"}, {"revision": "main"}, {"files": {}},
                  {"files": {**identity["files"], "../weights.bin": "d" * 64}},
                  {"files": {**identity["files"], "C:\\weights.bin": "d" * 64}}):
        with pytest.raises(ValueError, match="Chronos"):
            producer.chronos_identity_digest({**identity, **patch})


def test_code_and_configuration_hashes_survive_checkout_newlines(tmp_path, monkeypatch):
    code = tmp_path / "example.py"
    code.write_bytes(b"x = 1\ny = 2\n")
    expected = producer.sha256_text(code)
    binary = producer.sha256(code)
    code.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert producer.sha256_text(code) == expected
    assert producer.sha256(code) != binary
    code.write_bytes(b"x = 2\r\ny = 2\r\n")
    assert producer.sha256_text(code) != expected
    expected_configuration, expected_hashes = producer._configuration("FR", 1)
    for name in expected_hashes:
        copy = tmp_path / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((producer.ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    configuration, hashes = producer._configuration("FR", 1)
    assert configuration == expected_configuration
    assert hashes == expected_hashes


class Pipeline:
    def __init__(self):
        self.inputs = []

    def predict_quantiles(self, inputs, **kwargs):
        self.inputs.extend(inputs)
        assert kwargs["cross_learning"] is False
        assert kwargs["context_length"] == 2048
        center = inputs[0]["target"][-1]
        result = np.tile([center - 10., center, center + 10.], (1, kwargs["prediction_length"], 1))
        return [result], None


def inputs(day="2026-10-25", history_days=90):
    date = pd.Timestamp(day).date()
    index = producer._grid(date - timedelta(days=history_days), date + timedelta(days=1))
    t = np.arange(len(index), dtype=float)
    covariates = pd.DataFrame({name: 10 + i + 3 * np.sin(t / 24 + i)
                               for i, name in enumerate(producer.RAW_ALIASES)}, index=index)
    labels = pd.Series(40 + 10 * np.sin(t / 24), index=index, name="target")
    return labels, covariates


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_native_chronos_scores_missing_future_labels_and_preserves_dst(day, hours):
    target, covariates = inputs(day)
    plan = producer.build_delivery_plan(day, timezone="Europe/Paris")
    target.loc[plan.delivery_index_utc] = np.nan
    pipeline = Pipeline()
    forecast = producer.predict_chronos_day(target=target, covariates=covariates, zone="FR",
                                            delivery_day=day, pipeline=pipeline)
    assert len(forecast) == hours
    assert "actual" not in forecast
    assert forecast.index.equals(plan.delivery_index_utc)
    assert forecast.forecast_origin_utc.eq(plan.forecast_origin_utc).all()
    changed = target.copy()
    changed.loc[plan.delivery_index_utc] = 1e9
    again = producer.predict_chronos_day(target=changed, covariates=covariates, zone="FR",
                                         delivery_day=day, pipeline=Pipeline())
    pd.testing.assert_frame_equal(forecast, again)
    assert set(pipeline.inputs[0]["future_covariates"]) == {
        *[f"known_{name}_oracle" for name in producer.INPUT_ALIASES], *producer.CALENDAR_COLUMNS}


def test_chronos_refuses_incomplete_future_source_and_past_target():
    target, covariates = inputs()
    covariates.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="delivery forecast covariates"):
        producer.predict_chronos_day(target=target, covariates=covariates, zone="FR",
                                     delivery_day="2026-10-25", pipeline=Pipeline())
    target, covariates = inputs()
    target.iloc[-100] = np.nan
    with pytest.raises(ValueError, match="past target context"):
        producer.predict_chronos_day(target=target, covariates=covariates, zone="FR",
                                     delivery_day="2026-10-25", pipeline=Pipeline())


def test_bootstrap_keeps_calendar_known_when_old_physical_covariates_are_missing():
    target, covariates = inputs()
    plan = producer.build_delivery_plan("2026-10-25", timezone="Europe/Paris")
    pipeline = Pipeline()
    producer.predict_chronos_day(target=target, covariates=covariates.loc[plan.delivery_index_utc],
                                 zone="FR", delivery_day="2026-10-25", pipeline=pipeline)
    past = pipeline.inputs[0]["past_covariates"]
    assert np.isnan(past[producer.INPUT_ALIASES[0]]).all()
    assert all(np.isfinite(past[column]).all() for column in producer.CALENDAR_COLUMNS)


def test_real_residual_and_kalman_score_an_unlabelled_future():
    day = "2026-10-25"
    target, covariates = inputs(day, history_days=92)
    config, _ = producer._configuration("DE", 1)
    first = pd.Timestamp(day).date() - timedelta(days=31)
    index = producer._grid(first, pd.Timestamp(day).date())
    plans = {d: producer.build_delivery_plan(d, timezone="Europe/Berlin") for d in pd.Index(index.tz_convert("Europe/Berlin").date).unique()}
    point = target.loc[index] - 3. - np.sin(np.arange(len(index)) / 17)
    raw = pd.DataFrame({"q10": point - 12, "q50": point, "q90": point + 12,
                        "actual": target.loc[index],
                        "forecast_origin_utc": [plans[d].forecast_origin_utc for d in index.tz_convert("Europe/Berlin").date]}, index=index)
    current = producer.predict_chronos_day(target=target, covariates=covariates, zone="DE",
                                           delivery_day=day, pipeline=Pipeline())
    interaction, _ = producer.build_pair_interaction(covariates, "DE", "Europe/Berlin")
    forecast, audit = producer.predict_residual_day(raw_history=raw, raw_future=current,
        covariates=covariates, zone="DE", delivery_day=day, configuration=config, interaction=interaction)
    assert audit["generation_source"] == "daily_prequential_refit"
    assert audit["future_labels_used"] is False
    assert "actual" not in forecast
    history = raw.rename(columns={q: f"residual_corrected__{q}" for q in producer.QUANTILES})
    history["residual_correction"] = 0.
    result, kalman_audit = producer.predict_kalman_day(history=history, future=forecast,
        covariates=covariates, zone="DE", delivery_day=day, configuration=config)
    assert result.index.equals(current.index)
    assert np.isfinite(result.to_numpy(float)).all()
    assert kalman_audit["target_observations_assimilated"] == 0
    assert kalman_audit["training_window_days"] == 31
    changed = forecast.assign(actual=1e9)
    second, _ = producer.predict_kalman_day(history=history, future=changed,
        covariates=covariates, zone="DE", delivery_day=day, configuration=config)
    pd.testing.assert_frame_equal(result, second)
