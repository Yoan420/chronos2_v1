from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.nuclear_residual_cache as cache_module
from chronos2_hourly.nuclear_residual_cache import ResidualDayCache


TIMEZONE = "Europe/Paris"


def _inputs():
    index = pd.date_range("2024-03-28", "2024-04-03", inclusive="left", freq="h", tz=TIMEZONE).tz_convert("UTC")
    features = pd.DataFrame({"known_fr_nuclear_generation_forecast": np.arange(len(index), dtype=float)}, index=index)
    raw = pd.DataFrame({"q10": 30.0, "q50": 40.0, "q90": 50.0, "actual": 45.0}, index=index)
    days = index.tz_convert(TIMEZONE).strftime("%Y-%m-%d")
    training = index[(days >= "2024-03-29") & (days < "2024-03-31")]
    target = index[days == "2024-03-31"]
    base = raw.loc[target, ["q10", "q50", "q90"]]
    return features, raw, training, target, base


def _cache(tmp_path, features, raw, **contract):
    return ResidualDayCache(tmp_path, {"lookback_days": 365, **contract}, features=features, raw=raw, timezone=TIMEZONE)


def test_persists_future_prediction_when_it_becomes_observed_history(tmp_path):
    features, raw, training, target, base = _inputs()
    raw.loc[target, "actual"] = np.nan
    cache = _cache(tmp_path, features.loc[:target[-1]], raw.loc[:target[-1]])
    assert cache.load("2024-03-31", training, target, base) is None
    corrected = base + 0.75
    columns = tuple(features.columns)
    assert cache.store("2024-03-31", training, target, base, corrected, columns)
    assert (cache.hits, cache.misses, cache.writes) == (0, 1, 1)

    # A new delivery adds data and labels yesterday; neither changes yesterday's
    # causal fit. The target also covers a physical 23-hour spring DST day.
    raw.loc[target, "actual"] = 1000.0
    resumed = _cache(tmp_path, features, raw)
    loaded, loaded_columns = resumed.load("2024-03-31", training, target, base)
    pd.testing.assert_frame_equal(loaded, corrected, check_freq=False)
    assert len(loaded) == 23
    assert loaded_columns == columns
    assert (resumed.hits, resumed.misses, resumed.writes) == (1, 0, 0)


@pytest.mark.parametrize("changed", ["label", "training_feature", "prediction_feature", "training_base", "prediction_base", "config", "dtype", "schema", "index"])
def test_exact_causal_inputs_and_recipe_changes_invalidate(tmp_path, changed):
    features, raw, training, target, base = _inputs()
    cache = _cache(tmp_path, features, raw)
    assert cache.store("2024-03-31", training, target, base, base + 1.0, tuple(features.columns))
    contract = {}
    if changed == "label":
        raw.loc[training[0], "actual"] += 1
    elif changed == "training_feature":
        features.iloc[features.index.get_loc(training[0]), 0] += 1
    elif changed == "prediction_feature":
        features.iloc[features.index.get_loc(target[0]), 0] += 1
    elif changed == "training_base":
        raw.loc[training[0], "q50"] += 1
    elif changed == "prediction_base":
        base.loc[target[0], "q50"] += 1
    elif changed == "config":
        contract["iterations"] = 20
    elif changed == "dtype":
        features = features.astype("float32")
    elif changed == "schema":
        features = features.rename(columns={features.columns[0]: "different_feature"})
    elif changed == "index":
        training = training[1:]
    assert _cache(tmp_path, features, raw, **contract).load("2024-03-31", training, target, base) is None


def test_unrelated_rows_and_target_labels_do_not_invalidate(tmp_path):
    features, raw, training, target, base = _inputs()
    key = _cache(tmp_path, features, raw).key("2024-03-31", training, target, base)
    features.iloc[0, 0] += 999
    features.iloc[-1, 0] += 999
    raw.iloc[0, :] += 999
    raw.iloc[-1, :] += 999
    raw.loc[target, "actual"] += 999
    assert _cache(tmp_path, features, raw).key("2024-03-31", training, target, base) == key


def test_sparse_training_selection_is_exact(tmp_path):
    features, raw, training, target, base = _inputs()
    selected = training.delete(2)
    cache = _cache(tmp_path, features, raw)
    key = cache.key("2024-03-31", selected, target, base)
    features.loc[training[2], features.columns[0]] += 999
    assert _cache(tmp_path, features, raw).key("2024-03-31", selected, target, base) == key
    assert cache.key("2024-03-31", training, target, base) != key


def test_training_data_is_hashed_once_instead_of_for_each_window(tmp_path, monkeypatch):
    features, raw, training, target, base = _inputs()
    original = pd.util.hash_pandas_object
    sizes = []

    def tracked_hash(frame, *args, **kwargs):
        sizes.append(len(frame))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.util, "hash_pandas_object", tracked_hash)
    cache = _cache(tmp_path, features, raw)
    for _ in range(3):
        cache.key("2024-03-31", training, target, base)
    assert sizes == [len(features), len(raw), len(target), len(target), len(target)]


@pytest.mark.parametrize("corruption", ["json", "payload", "metadata", "missing_payload"])
def test_corrupt_cache_entries_are_disposable(tmp_path, corruption):
    features, raw, training, target, base = _inputs()
    cache = _cache(tmp_path, features, raw)
    assert cache.store("2024-03-31", training, target, base, base + 1.0, tuple(features.columns))
    record_path = next(tmp_path.glob("*.json"))
    payload_path = next(tmp_path.glob("*.parquet"))
    if corruption == "json":
        record_path.write_text("{")
    elif corruption == "payload":
        payload_path.write_bytes(b"truncated")
    elif corruption == "metadata":
        record = json.loads(record_path.read_text())
        record["feature_columns"] = ["incorrect"]
        record_path.write_text(json.dumps(record))
    else:
        payload_path.unlink()
    assert cache.load("2024-03-31", training, target, base) is None
    assert cache.store("2024-03-31", training, target, base, base + 1.0, tuple(features.columns))
    assert cache.load("2024-03-31", training, target, base) is not None


@pytest.mark.parametrize("invalid", ["unordered", "nonfinite", "index"])
def test_validates_predictions_even_with_matching_payload_checksum(tmp_path, invalid):
    features, raw, training, target, base = _inputs()
    cache = _cache(tmp_path, features, raw)
    assert cache.store("2024-03-31", training, target, base, base + 1, tuple(features.columns))
    invalid_prediction = base.copy()
    if invalid == "unordered":
        invalid_prediction.iloc[0, 0] = 500
    elif invalid == "nonfinite":
        invalid_prediction.iloc[0, 1] = np.nan
    else:
        invalid_prediction.index = invalid_prediction.index + pd.Timedelta(hours=1)
    assert not cache.store("2024-03-31", training, target, base, invalid_prediction, tuple(features.columns))
    buffer = BytesIO()
    invalid_prediction.to_parquet(buffer)
    payload = buffer.getvalue()
    record_path = next(tmp_path.glob("*.json"))
    record = json.loads(record_path.read_text())
    record.pop("record_sha256")
    record["payload_sha256"] = sha256(payload).hexdigest()
    (tmp_path / f'{record["payload_sha256"]}.parquet').write_bytes(payload)
    record["record_sha256"] = sha256(cache_module._json_bytes(record)).hexdigest()
    record_path.write_bytes(cache_module._json_bytes(record))
    assert cache.load("2024-03-31", training, target, base) is None


def test_unwritable_cache_does_not_break_prediction(tmp_path, monkeypatch):
    features, raw, training, target, base = _inputs()
    cache = _cache(tmp_path, features, raw)

    def unavailable(*args):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(cache, "_atomic_write", unavailable)
    assert not cache.store("2024-03-31", training, target, base, base, tuple(features.columns))
    assert cache.load("2024-03-31", training, target, base) is None


def test_real_corrector_diagnostic_attrs_do_not_prevent_persistence(tmp_path):
    from chronos2_hourly.models.residual_corrector import apply_residual_correction

    features, raw, training, target, base = _inputs()
    cache = _cache(tmp_path, features, raw)
    corrected = apply_residual_correction(base, np.ones(len(base)))
    assert isinstance(corrected.attrs["residual_correction"], pd.Series)
    assert cache.store("2024-03-31", training, target, base, corrected, tuple(features.columns))
    loaded, _ = cache.load("2024-03-31", training, target, base)
    assert loaded.attrs == {}
    assert "residual_correction" in corrected.attrs
    pd.testing.assert_frame_equal(loaded, corrected, check_freq=False)


def test_concurrent_writers_publish_complete_metadata_payload_pairs(tmp_path):
    features, raw, training, target, base = _inputs()
    caches = [_cache(tmp_path, features, raw), _cache(tmp_path, features, raw)]

    def write(which):
        return caches[which].store("2024-03-31", training, target, base, base + which, tuple(features.columns))

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Windows may reject one overlapping metadata replacement; the other
        # writer must still publish a complete, usable entry.
        assert any(list(pool.map(write, [0, 1])))
    loaded, _ = caches[0].load("2024-03-31", training, target, base)
    assert (loaded.q50 == 40).all() or (loaded.q50 == 41).all()
    assert not list(tmp_path.glob("*.tmp"))
