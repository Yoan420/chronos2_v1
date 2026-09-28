"""Check the standalone final selector against the frozen historical rule."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_reference_pair import SIGNALS, build_confirmed_pair


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/experiments/nyx_improvement_to20260923"


def sample_signals():
    index = pd.date_range("2026-09-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame({
        "ensemble__q50": [100., 100., 100., 100.],
        "nyx__q50": [299., 299., 300., 300.],
        "test2__q50": [320., 320., 320., 320.],
        "prior90_active": [True, True, True, False],
        "spike_probability": [.95, .949, .95, .99],
        "partner_nyx__q50": [300., 300., 299., 300.],
    }, index=index).loc[:, SIGNALS]


def test_pair_formula_and_thresholds():
    output = build_confirmed_pair(sample_signals(), zone="FR", partner_zone="BE")
    assert output.scarcity_confirmed_pair.tolist() == [320., 100., 320., 100.]
    assert output.selected_confirmed_pair.tolist() == [True, False, True, False]


def test_missing_wrong_or_noncausal_inputs_fail_closed():
    frame = sample_signals()
    with pytest.raises(ValueError, match="partner"):
        build_confirmed_pair(frame, zone="FR", partner_zone="NL")
    with pytest.raises(ValueError, match="six declared"):
        build_confirmed_pair(frame.drop(columns="spike_probability"), zone="FR", partner_zone="BE")
    changed = frame.copy()
    changed["prior90_active"] = changed.prior90_active.astype(object)
    changed.loc[changed.index[0], "prior90_active"] = None
    with pytest.raises(ValueError, match="boolean"):
        build_confirmed_pair(changed, zone="FR", partner_zone="BE")
    changed = frame.copy()
    changed.loc[changed.index[0], "test2__q50"] = 99.
    with pytest.raises(ValueError, match="inconsistent"):
        build_confirmed_pair(changed, zone="FR", partner_zone="BE")
    changed = frame.copy()
    changed.index = changed.index.tz_convert("Europe/Paris")
    with pytest.raises(ValueError, match="UTC"):
        build_confirmed_pair(changed, zone="FR", partner_zone="BE")


@pytest.mark.parametrize("zone,partner", [("FR", "BE"), ("BE", "FR"), ("NL", "DE")])
def test_archived_annual_reference_is_bit_exact_when_available(zone, partner):
    path = ARCHIVE / "test2_confirmed_gate_v2/annual" / f"{zone}.parquet"
    if not path.is_file():
        pytest.skip("The ignored historical research archive is unavailable in a clean clone")
    archived = pd.read_parquet(path)
    current = archived.loc[:, ["ensemble__q50", "nyx__q50", "test2__q50",
                               "selected_scarcity_gate", "spike_probability", "partner_nyx__q50"]]
    current = current.rename(columns={"selected_scarcity_gate": "prior90_active"})
    predicted = build_confirmed_pair(current, zone=zone, partner_zone=partner)
    assert np.array_equal(predicted.scarcity_confirmed_pair.to_numpy(),
                          archived.scarcity_confirmed_pair.to_numpy())
    assert np.array_equal(predicted.selected_confirmed_pair.to_numpy(),
                          archived.selected_confirmed_pair.to_numpy())
    composition_path = ARCHIVE / "rmse_exchange_composition_v1/oof" / f"{zone}.parquet"
    if composition_path.is_file():
        composition = pd.read_parquet(composition_path)
        assert np.array_equal(predicted.scarcity_confirmed_pair.to_numpy(),
                              composition.loc[predicted.index, "reference"].to_numpy())
