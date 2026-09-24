"""No-network checks of GRIB physical identity and safe native-handle release."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from auxiliary_lab import noaa_gfs
from auxiliary_lab.weather import ZONE_POINTS


RUN = pd.Timestamp("2023-09-07T00:00:00Z")
POINTS = ZONE_POINTS["FR"][:2]
RAW = b"GRIB\x00\x00\x00\x02" + (20).to_bytes(8, "big") + b"7777"


@pytest.fixture
def native_decoder(monkeypatch):
    state = SimpleNamespace(
        values=np.array([280.0, 285.0]), released=[], handle=object(),
        metadata={
            "edition": 2, "centre": "kwbc", "discipline": 0,
            "parameterCategory": 0, "parameterNumber": 0,
            "typeOfLevel": "heightAboveGround", "level": 2, "units": "K",
            "stepType": "instant", "stepUnits": 1, "startStep": 24,
            "endStep": 24, "dataDate": 20230907, "dataTime": 0,
            "validityDate": 20230908, "validityTime": 0,
            "gridType": "regular_ll", "Ni": 1440, "Nj": 721,
            "iDirectionIncrementInDegrees": 0.25,
            "jDirectionIncrementInDegrees": 0.25, "missingValue": 9999.0,
        },
    )

    def nearest(handle, is_lsm, latitudes, longitudes):
        assert handle is state.handle
        return [
            {"lat": latitude, "lon": longitude, "value": float(value), "distance": 0.0}
            for latitude, longitude, value in zip(latitudes, longitudes, state.values)
        ]

    fake = SimpleNamespace(
        codes_new_from_message=lambda raw: state.handle,
        codes_get=lambda handle, key: state.metadata[key],
        codes_grib_find_nearest_multiple=nearest,
        codes_get_api_version=lambda: "mock-2.48.0",
        codes_release=state.released.append,
    )
    monkeypatch.setitem(sys.modules, "eccodes", fake)
    return state


def test_valid_decoding_returns_physical_samples_and_releases_handle(native_decoder):
    decoded = noaa_gfs.decode_message(RAW, "temperature_2m", RUN, 24, POINTS)
    np.testing.assert_array_equal(decoded.values, [280.0, 285.0])
    assert decoded.metadata["units"] == "K"
    assert decoded.metadata["missingValue"] == 9999.0
    assert (decoded.start_step, decoded.end_step) == (24, 24)
    assert native_decoder.released == [native_decoder.handle]


@pytest.mark.parametrize("missing", [9999.0, np.nan])
def test_missing_sentinel_and_nonfinite_samples_are_rejected(native_decoder, missing):
    native_decoder.values[0] = missing
    with pytest.raises(noaa_gfs.GfsError, match="Missing|non-finite"):
        noaa_gfs.decode_message(RAW, "temperature_2m", RUN, 24, POINTS)
    assert native_decoder.released == [native_decoder.handle]


@pytest.mark.parametrize("name,value", [
    ("temperature_2m", 149.0), ("temperature_2m", 351.0),
    ("u100", 201.0), ("v100", -201.0),
    ("solar", 1601.0), ("solar", -0.6),
])
def test_finite_but_physically_impossible_values_are_rejected(native_decoder, name, value):
    spec = noaa_gfs.FIELDS[name]
    native_decoder.metadata.update(
        parameterCategory=spec.category, parameterNumber=spec.number,
        typeOfLevel=spec.level_type, level=spec.level, units=spec.units[0],
        stepType="avg" if spec.average else "instant",
        startStep=18 if spec.average else 24,
    )
    native_decoder.values[:] = value
    with pytest.raises(noaa_gfs.GfsError, match="physical bounds"):
        noaa_gfs.decode_message(RAW, name, RUN, 24, POINTS)
    assert native_decoder.released == [native_decoder.handle]


@pytest.mark.parametrize("key,value", [
    ("units", "degC"), ("centre", "ecmf"), ("parameterNumber", 1),
    ("dataDate", 20230908), ("validityTime", 100), ("Ni", 720),
])
def test_wrong_physical_identity_units_or_forecast_time_releases_handle(native_decoder, key, value):
    native_decoder.metadata[key] = value
    with pytest.raises(noaa_gfs.GfsError):
        noaa_gfs.decode_message(RAW, "temperature_2m", RUN, 24, POINTS)
    assert native_decoder.released == [native_decoder.handle]


@pytest.mark.parametrize("mismatch", ["u100", "solar_later"])
def test_materializer_rejects_grid_changes_between_variables_and_endpoints(tmp_path, monkeypatch, mismatch):
    class NoNetworkClient:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def fetch_endpoint(self, run, cutoff, hour, fields):
            messages = {
                name: {"start_step": ((hour - 1) // 6) * 6 if name == "solar" else hour,
                       "end_step": hour, "bytes": 1}
                for name in fields
            }
            return {name: b"fixture" for name in fields}, {
                "messages": messages, "from_cache": True,
                "publication_max_utc": "2023-09-07T04:00:00+00:00",
            }

    def decoded(raw, name, run, hour, points, radiation_tolerance=0.5):
        grid = [{"lat": point.latitude, "lon": point.longitude} for point in points]
        if name == mismatch or (mismatch == "solar_later" and name == "solar" and hour == 23):
            grid[0]["lat"] += 0.25
        start = ((hour - 1) // 6) * 6 if name == "solar" else hour
        value = {"temperature_2m": 280, "u100": 3, "v100": 4, "solar": 100}[name]
        return noaa_gfs.DecodedField(np.full(len(points), value, dtype=float), start, hour, {"grid_points": grid})

    monkeypatch.setattr(noaa_gfs, "GfsClient", NoNetworkClient)
    monkeypatch.setattr(noaa_gfs, "decode_message", decoded)
    output = tmp_path / "weather.parquet"
    with pytest.raises(noaa_gfs.GfsError, match="different ordered grid points"):
        noaa_gfs.materialize_noaa_gfs_weather(
            start_day="2023-09-08", end_day="2023-09-08",
            output_path=output, workers=2,
        )
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not list(tmp_path.glob("*.publish.lock"))
