"""Read-only descriptive analysis of the frozen KPI input snapshots."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kpi_report.data import load_recent_models

frame, catalog, audit = load_recent_models(ROOT)
keys = ["zone", "timestamp_utc"]
wide = frame.pivot(index=keys, columns="model_id", values="forecast")
refs = frame.groupby(keys)[["actual", "storm"]].first()
wide = wide.join(refs).dropna().reset_index()
local = pd.to_datetime(wide.timestamp_utc, utc=True).dt.tz_convert("Europe/Paris")
wide["day"] = local.dt.strftime("%Y-%m-%d")
wide["hour"] = local.dt.hour
models = [c["id"] for c in catalog] + ["storm"]

def scores(f):
    return {m: {"n": len(f), "mae": float((f[m]-f.actual).abs().mean()),
                "rmse": float(np.sqrt(((f[m]-f.actual)**2).mean())),
                "bias": float((f[m]-f.actual).mean()),
                "ae_sum": float((f[m]-f.actual).abs().sum())} for m in models}

out = {"rows": len(wide), "tail": {}, "interventions": {}, "cases": [], "june": {}}
for zone in ["ALL"] + sorted(wide.zone.unique().tolist()):
    f = wide if zone == "ALL" else wide[wide.zone == zone]
    threshold = float(f.actual.quantile(.99))
    out["tail"][zone] = {
        "top1_threshold": threshold,
        "top1": scores(f[f.actual >= threshold]),
        "not_top1": scores(f[f.actual < threshold]),
        "over200": scores(f[f.actual > 200]),
        "hour19": scores(f[f.hour == 19]),
    }
    interventions = {}
    for m in models:
        if m in ["storm", "nuclear_kalman"]:
            continue
        shift = f[m] - f.nuclear_kalman
        active = shift.abs() > 1e-9
        gain = (f.nuclear_kalman-f.actual).abs() - (f[m]-f.actual).abs()
        severe = f.actual-f.nuclear_kalman > 50
        interventions[m] = {
            "active": int(active.sum()), "improved": int((active & (gain>1e-9)).sum()),
            "worse": int((active & (gain < -1e-9)).sum()),
            "total_gain": float(gain.sum()), "mean_gain_active": float(gain[active].mean()) if active.any() else None,
            "severe_underprediction_hours": int(severe.sum()),
            "corrected_severe_hours": int((severe & active).sum()),
            "max_correction": float(shift.max()),
        }
    out["interventions"][zone] = interventions
    out["june"][zone] = {d:scores(f[f.day == d]) for d in ["2026-06-24","2026-06-25","2026-06-26"]}

for _, row in wide[(wide.day == "2026-09-14") & (wide.hour == 19)].iterrows():
    out["cases"].append({"zone": row.zone, "day": row.day, "hour": int(row.hour),
                         "actual": row.actual, **{m: row[m] for m in models}})

families = audit["selected_families"]
stress = Path(families["stress_guard"]["snapshot"])
coherent = Path(families["coherent_p50"]["snapshot"])
out["case_probabilities"] = {}
for label, path in [("stress", stress/"physics_direct.parquet"),
                    ("forest", coherent/"forest/predictions.parquet"),
                    ("empirical", coherent/"empirical/predictions.parquet")]:
    p = pd.read_parquet(path)
    ts = pd.to_datetime(p.timestamp_utc, utc=True).dt.tz_convert("Europe/Paris")
    selected = p[(ts.dt.strftime("%Y-%m-%d") == "2026-09-14") & (ts.dt.hour == 19)]
    columns = [c for c in p if c in {"zone","forecast","actual","candidate_forecast","mixture_raw_p50_eur_mwh","mixture_error_q50","expert_ready","interval_status","intervention_active","correction_eur_mwh","governance_weight","scarcity_probability","event_probability","gate_probability","event_threshold_eur_mwh","policy_reason","decision_reason"} or ("probability" in c and "feature" not in c)]
    out["case_probabilities"][label] = selected[columns].replace({np.nan:None}).to_dict("records")
    out[label+"_columns"] = [c for c in p if not c.startswith("feature_")]

print(json.dumps(out, indent=2, default=str))
