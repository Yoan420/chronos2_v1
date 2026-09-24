from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_modular.forecast_explanation import (
    attach_forecast_components, build_forecast_components_html, attribution_scope_html,
)
from chronos2_hourly.report_attribution_cache import cached_attribution, FILES


def test_layers_reconstruct_issued_forecast_and_are_not_weights():
    stamps = pd.date_range("2026-09-08", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame({"delivery_start_utc": stamps, "chronos2__q50": 60.,
                          "residual_corrected__q50": 55., "residual_kalman__q50": 52.})
    result = SimpleNamespace(forecast_native=pd.DataFrame({"timestamp": stamps, "q50": 52.}))
    original = frame.copy(deep=True)
    attach_forecast_components(result, frame, model="residual_kalman")
    assert result.forecast_components.iloc[0].to_dict() == {
        "chronos_q50": 60., "residual_correction": -5., "kalman_correction": -3., "final_q50": 52.}
    rendered = build_forecast_components_html(result)
    assert "-5.00" in rendered and "-3.00" in rendered and "52.00" in rendered
    assert "pas des poids de variables" in rendered
    pd.testing.assert_frame_equal(frame, original)
    frame["residual_kalman__q50"] = 53.
    with pytest.raises(ValueError, match="composantes"):
        attach_forecast_components(result, frame, model="residual_kalman")


def test_missing_and_legacy_prices_are_not_zero_influence():
    assert "aucun poids n'est inventé" in attribution_scope_html(None)
    legacy = {"hourly": pd.DataFrame({"variable_key": ["wind"]})}
    assert "ne signifie pas une influence nulle" in attribution_scope_html(legacy)
    current = {"hourly": pd.DataFrame({"variable_key": ["historical_target_price"]})}
    assert "ni les prix futurs" in attribution_scope_html(current)


def test_attribution_cache_reuses_complete_payload_and_refuses_corruption(tmp_path):
    source = tmp_path / "forecast.csv"
    source.write_text("unchanged")
    calls = []

    def materialize(directory):
        calls.append(directory)
        (directory / FILES[0]).write_bytes(b"sealed hourly")
        (directory / FILES[1]).write_text(json.dumps({"groups": [{"key": "historical_target_price"}]}))

    options = dict(root=tmp_path / "cache", sources=[source], materialize=materialize)
    first = cached_attribution(**options)
    assert cached_attribution(**options) == first
    assert len(calls) == 1 and source.read_text() == "unchanged"
    (first / FILES[0]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="divergent"):
        cached_attribution(**options)
    source.write_text("new forecast")
    assert cached_attribution(**options) != first
    assert len(calls) == 2


def test_attribution_cache_does_not_publish_on_changed_source(tmp_path):
    source = tmp_path / "source"
    source.write_text("before")
    def changing(directory):
        (directory / FILES[0]).write_bytes(b"x")
        (directory / FILES[1]).write_text(json.dumps({"groups": [{"key": "historical_target_price"}]}))
        source.write_text("after")
    with pytest.raises(ValueError, match="sources ont changé"):
        cached_attribution(root=tmp_path / "cache", sources=[source], materialize=changing)
    assert not list((tmp_path / "cache").iterdir())
