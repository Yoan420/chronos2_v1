"""Read-only actual-fold smoke before sealing the new full replay."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from nyx_scarcity import policy as base
from nyx_stress_guard.features import make_stress_features
from nyx_stress_guard.policy import fit_model, predict_model

source = Path("runs/experiments/nyx_scarcity_v1/coherent_p50/snapshots/20260914T160104Z_d7366f80")
panel = pd.read_parquet(source/"panel.parquet")
settings = json.loads((source/"manifest.json").read_text(encoding="utf-8"))["settings"]
started = time.monotonic()
frame, names, required, _ = make_stress_features(panel)
p = base._parameters({**settings, "feature_columns": names, "required_feature_columns": required, "probability_gate": .5})
data = base._prepare(frame, p)
day = "2026-09-14"
current = data.loc[data._day.eq(day)]
state, record = fit_model(data, day, current.forecast_origin_utc.iloc[0], p, tuple(sorted(data.zone.unique())))
assert state is not None, record
probability, raw, _, detail = predict_model(state, current, p)
print(json.dumps({"status":record["status"], "features":len(names), "core_rows":record["cdf_core_rows"],
    "elapsed_seconds":time.monotonic()-started}, indent=2))
print("Fit/inference contract passed. This smoke does not select parameters or publish a forecast.")
