"""DE replay must resume immutable checkpoints and keep the frozen composition."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_nyx_de_cpu_historical as de


def test_de_replay_keeps_four_country_points_and_checks_models(tmp_path, monkeypatch):
    grid = pd.date_range("2026-09-22 22:00", periods=24, freq="h", tz="UTC")
    monkeypatch.setattr(de.archived, "ANNUAL_GRID", grid)
    monkeypatch.setattr(de.archived, "annual_origins", lambda: [("2026-09-23", "2026-09-24")])
    monkeypatch.setattr(de.archived, "_plan", lambda *_: {})
    monkeypatch.setattr(de.archived, "_load_baselines", lambda *_: ({}, {}, {"input": "fixed"}))
    monkeypatch.setattr(de.archived, "_load_features", lambda *_: ({}, {"input": "fixed"}))
    reference = tmp_path / "reference.parquet"
    comparison = tmp_path / "comparison.parquet"
    pd.DataFrame({"reference": np.full(24, 100.)}, index=grid).to_parquet(reference)
    pd.DataFrame({"actual": np.full(24, 100.), "storm": np.full(24, 110.)}, index=grid).to_parquet(comparison)
    monkeypatch.setattr(de.archived, "source_path",
        lambda p: comparison if "comparison" in str(p) else reference)
    monkeypatch.setattr(de.archived, "score_selected", lambda point, actual: {"rmse": float(point.mean())})
    fits = []
    def fit(*args, config, model_path, **kwargs):
        fits.append(config.target_mode)
        model_path.write_bytes(b"test cpu CBM")
        points = {z: pd.DataFrame({"point": np.r_[np.full(12, 100.), np.full(12, 140.)]}, index=grid)
                  for z in de.archived.ZONES}
        return points, {"model": {"sha256": de.archived.sha256(model_path)}}
    monkeypatch.setattr(de, "fit_pooled_block", fit)
    # The production constant 53 also guards aggregate receipts; shortened test
    # intentionally exercises checkpoints without creating a final score.
    output = tmp_path / "replay"
    de.replay(output, roles=(de.ROLES[0],))
    de.replay(output, roles=(de.ROLES[0],))
    assert fits == ["residual"]
    saved = pd.read_parquet(output / de.ROLES[0] / "2026-09-23.parquet")
    assert list(saved) == list(de.archived.ZONES)
    (output / de.ROLES[0] / "2026-09-23.cbm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint or model was modified"):
        de.replay(output, roles=(de.ROLES[0],))
