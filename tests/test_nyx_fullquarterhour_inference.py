from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_fullquarterhour.inference import infer_batch
from nyx_fullquarterhour.raw import load_config, validate_raw
from nyx_fullquarterhour.storage import safe_path
from nyx_quarterhour.data import day_index

ROOT = Path(__file__).resolve().parents[1]


def inputs(freq="15min"):
    future_index = day_index("2026-03-29", freq).tz_localize(None)
    offset = pd.tseries.frequencies.to_offset(freq)
    context_index = pd.date_range(end=future_index[0] - offset, periods=8, freq=freq)
    c = pd.DataFrame({"item_id": "FR", "timestamp": context_index, "target": np.arange(8.),
                      "fr_nuclear_generation_fcst_gw": 40., "known_fr_nuclear_generation_fcst_gw_oracle": 40.})
    f = pd.DataFrame({"item_id": "FR", "timestamp": future_index, "known_fr_nuclear_generation_fcst_gw_oracle": 41.})
    return c, f


def model():
    def predict(context, *, future_df, **kwargs):
        assert "fr_nuclear_generation_fcst_gw" in context
        assert "fr_nuclear_generation_fcst_gw" not in future_df
        assert kwargs["cross_learning"] is False
        assert kwargs["context_length"] == 8
        return future_df[["item_id", "timestamp"]].assign(**{"0.1": 1., "0.5": 2., "0.9": 3.})
    return SimpleNamespace(predict_df=predict, model_context_length=8192, model_prediction_length=1024)


@pytest.mark.parametrize("freq,count", [("h", 23), ("15min", 92)])
def test_past_only_not_leaked_to_future_and_dst_preserved(freq, count):
    c, f = inputs(freq)
    result = infer_batch([c], [f], freq=freq, context_length=8, prediction_length=count, pipeline=model())
    assert len(result) == count
    assert result.q50.eq(2).all()


@pytest.mark.parametrize("violation", ["future_target", "unknown_future", "missing_known_history", "gap", "short_context", "duplicate_id"])
def test_invalid_input_rejected_before_model(violation):
    c, f = inputs()
    if violation == "future_target":
        f["target"] = 1.
    elif violation == "unknown_future":
        f["fr_nuclear_generation_fcst_gw"] = 40.
    elif violation == "missing_known_history":
        f["known_new"] = 1.
    elif violation == "gap":
        f.loc[1, "timestamp"] = f.timestamp.iloc[0]
    elif violation == "short_context":
        c = c.iloc[1:]
    contexts, futures = ([c, c], [f, f]) if violation == "duplicate_id" else ([c], [f])
    with pytest.raises(ValueError):
        infer_batch(contexts, futures, freq="15min", context_length=8, prediction_length=92, pipeline=model())


@pytest.mark.parametrize("violation", ["missing", "duplicate", "nan", "crossed", "extra"])
def test_invalid_outputs_rejected(violation):
    c, f = inputs()
    pipeline = model()
    original = pipeline.predict_df
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if violation == "missing":
            result = result.iloc[1:]
        elif violation == "duplicate":
            result = pd.concat([result, result.iloc[:1]])
        elif violation == "nan":
            result.loc[0, "0.5"] = np.nan
        elif violation == "crossed":
            result.loc[0, "0.5"] = 4.
        else:
            result.loc[0, "timestamp"] -= pd.Timedelta(days=1)
        return result
    pipeline.predict_df = corrupt
    with pytest.raises(ValueError):
        infer_batch([c], [f], freq="15min", context_length=8, prediction_length=92, pipeline=pipeline)


def test_full_config_no_production_activation(tmp_path):
    import yaml
    cfg = load_config(ROOT / "config/nyx_fullquarterhour.yaml")
    for key, value in [("activation_performed", True), ("context_hours", 4096), ("training_lookback_days", 90), ("output_root", "runs/live")]:
        changed = dict(cfg, **{key: value})
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            load_config(path)


def test_artifacts_confined(tmp_path):
    good = tmp_path / "runs/experiments/nyx_fullquarterhour_v1/test/status.json"
    assert safe_path(tmp_path, good) == good
    for bad in [tmp_path / "runs/live/status.json", good.parent / "../../../../live/file"]:
        with pytest.raises(ValueError):
            safe_path(tmp_path, bad)


def test_raw_day_all_four_countries_and_grid():
    expected = day_index("2026-10-25", "15min")
    frame = pd.concat([pd.DataFrame({"timestamp_utc": expected, "zone": zone, "q10": 1., "q50": 2., "q90": 3.})
                       for zone in ("BE", "DE", "FR", "NL")], ignore_index=True)
    validate_raw(frame, "2026-10-25", "15min")
    with pytest.raises(ValueError):
        validate_raw(frame.iloc[1:], "2026-10-25", "15min")
