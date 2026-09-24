"""Paired, sealed research report for the fixed 90-day calibration experiment."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kpi_report.economic import compute_economic_kpis
from kpi_report.metrics import compute_kpis
from nyx_congestion.report import assemble_panel as assemble_previous, _numeric
from nyx_physical_p50.report import _safe, _same, _sha, _typed

KEYS = ["zone", "timestamp_utc"]
MODELS = ("calibrated_control_direct", "calibrated_control_governed",
          "congestion_calibrated_direct", "nyx_congestion_calibrated")
PREVIOUS = ("congestion_control_direct", "congestion_control_governed", "congestion_direct", "nyx_congestion",
            "network_fuel_direct", "nyx_physical_p50")
CATALOG = [
    {"id": "nuclear_kalman", "label": "NYX opérationnel", "kind": "production"},
    {"id": "nyx_congestion_calibrated", "label": "Congestion · calibration 90 j · gouverné", "kind": "candidate"},
    {"id": "congestion_calibrated_direct", "label": "Congestion · calibration 90 j · direct", "kind": "candidate"},
    {"id": "calibrated_control_governed", "label": "Contrôle · calibration 90 j · gouverné", "kind": "control"},
    {"id": "calibrated_control_direct", "label": "Contrôle · calibration 90 j · direct", "kind": "control"},
    {"id": "nyx_congestion", "label": "Congestion précédent · gouverné", "kind": "previous"},
    {"id": "congestion_direct", "label": "Congestion précédent · direct", "kind": "previous"},
    {"id": "congestion_control_governed", "label": "Contrôle précédent · gouverné", "kind": "previous"},
    {"id": "congestion_control_direct", "label": "Contrôle précédent · direct", "kind": "previous"},
    {"id": "nyx_physical_p50", "label": "Réseau + CGC · gouverné", "kind": "previous"},
    {"id": "network_fuel_direct", "label": "Réseau + CGC · direct", "kind": "previous"},
    {"id": "__storm__", "label": "Storm", "kind": "benchmark"},
]


def _capture(path, seals, expected=None):
    value = _sha(path)
    if expected is not None and value != expected:
        raise ValueError(f"Source checksum mismatch: {path}")
    seals[str(Path(path).resolve())] = value
    return value


def assemble_panel(predictions, source_directory):
    source, seals = Path(source_directory).resolve(), {}
    manifest_sha = _capture(source/"manifest.json", seals)
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf8"))
    _capture(source/"results_manifest.json", seals)
    result = json.loads((source/"results_manifest.json").read_text(encoding="utf8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Old congestion source is not completed and sealed to its manifest.")
    for name, mapping in (("panel.parquet", "input_files"), ("predictions.parquet", "result_files")):
        owner = manifest if mapping == "input_files" else result
        expected = owner.get(mapping, {}).get(name)
        if not expected:
            raise ValueError(f"Old congestion source does not seal {name}.")
        _capture(source/name, seals, expected)
    physical = Path(manifest["source_dir"]).resolve()
    expected = manifest.get("source_files", {}).get("manifest.json")
    if not expected:
        raise ValueError("Old congestion source does not seal its physical ancestor.")
    _capture(physical/"manifest.json", seals, expected)
    _, previous, source_audit, previous_seals = assemble_previous(pd.read_parquet(source/"predictions.parquet"), physical)
    seals.update(previous_seals)
    base = _typed(pd.read_parquet(source/"panel.parquet"))
    wide, previous = _typed(predictions), _typed(previous)
    _same(base, previous)
    _same(base, wide)
    for name in base:
        if name not in wide:
            raise ValueError(f"Missing preserved source field {name}.")
        try:
            pd.testing.assert_series_equal(base[name], wide[name], check_dtype=False, check_exact=True, check_names=False)
        except AssertionError as exc:
            raise ValueError(f"Changed preserved source field {name}.") from exc
    if set(MODELS).difference(wide):
        raise ValueError("Missing new calibrated price candidates.")
    wide["nuclear_kalman"] = wide.forecast
    for model in PREVIOUS:
        if model in wide and not np.allclose(wide[model], previous[model], equal_nan=True, rtol=0, atol=1e-9):
            raise ValueError(f"Changed frozen comparator {model}.")
        wide[model] = previous[model]
    for model in (*MODELS, *PREVIOUS):
        _numeric(wide, model)
    for model in MODELS:
        lower, upper = model+"_q10", model+"_q90"
        if lower not in wide or upper not in wide:
            raise ValueError(f"Missing calibrated intervals: {model}.")
        _numeric(wide, lower)
        _numeric(wide, upper)
        complete = wide[[lower, model, upper]].notna().all(axis=1)
        if ((wide.loc[complete, lower] > wide.loc[complete, model]+1e-9)
                | (wide.loc[complete, upper] < wide.loc[complete, model]-1e-9)).any():
            raise ValueError(f"Unordered calibrated quantiles: {model}.")
    for name in wide:
        if name.endswith(("_raw_probability", "_spike_probability")):
            _numeric(wide, name, probability=True)
    columns = ["actual", "storm", "sample", "forecast_origin_utc"]
    columns += [name for name in ("forecast_eligible", "benchmark_eligible") if name in wide]
    parts = []
    for item in CATALOG:
        if item["id"] == "__storm__":
            continue
        part = wide[columns].copy()
        part["model_id"], part["forecast"] = item["id"], wide[item["id"]]
        parts.append(part.reset_index())
    return pd.concat(parts, ignore_index=True), wide.reset_index(), source_audit, seals


def common_rows(wide, *, end_day, days):
    end = date.fromisoformat(end_day)
    civil = wide.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    selected = wide.loc[civil.between(end-timedelta(days=days-1), end) & wide["sample"].eq("evaluation")].copy()
    for name in ("forecast_eligible", "benchmark_eligible"):
        if name in selected:
            selected = selected.loc[selected[name].fillna(False).astype(bool)]
    names = [row["id"] for row in CATALOG if row["id"] != "__storm__"]
    return selected.dropna(subset=[*names, "actual", "storm"])


def diagnostics(wide, *, end_day, days, zones):
    selected = common_rows(wide, end_day=end_day, days=days)
    selected["hour"] = selected.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    changes, hourly, readiness = [], [], []
    for zone in ["ALL", *zones]:
        part = selected if zone == "ALL" else selected.loc[selected.zone.eq(zone)]
        baseline = (part.nuclear_kalman-part.actual).abs()
        for strategy in ("control", "congestion"):
            flag = strategy+"_expert_ready"
            ready = part[flag].fillna(False).astype(bool) if flag in part else None
            readiness.append({"zone": zone, "strategy": strategy, "n_common_country_hours": len(part),
                "ready_country_hours": int(ready.sum()) if ready is not None else None})
        for item in CATALOG:
            model = item["id"]
            point = part.storm if model == "__storm__" else part[model]
            error = point-part.actual
            changed, gain = (point-part.nuclear_kalman).abs().gt(1e-9), baseline-error.abs()
            tail = part.actual.ge(200)
            changes.append({"zone": zone, "model_id": model, "n_hours": len(part),
                "changed_hours": int(changed.sum()), "better": int((changed & gain.gt(1e-9)).sum()),
                "worse": int((changed & gain.lt(-1e-9)).sum()), "tail_n": int(tail.sum()),
                "tail_mae": error.loc[tail].abs().mean(), "tail_rmse": np.sqrt((error.loc[tail]**2).mean()),
                "normal_mae": error.loc[~tail].abs().mean()})
            for hour, group in pd.DataFrame({"hour": part.hour, "ae": error.abs()}).groupby("hour"):
                hourly.append({"zone": zone, "model_id": model, "hour": int(hour), "mae": group.ae.mean()})
    return _safe({"interventions": changes, "hourly": hourly, "readiness": readiness})


def probability_diagnostics(wide, *, end_day, days, zones):
    """Score prequential outputs, never the held-out block used to fit Platt."""
    selected = common_rows(wide, end_day=end_day, days=days)
    strategies = ("control", "congestion")
    fields = ("expert_ready", "raw_probability", "spike_probability", "threshold_eur_mwh")
    missing = [f"{s}_{f}" for s in strategies for f in fields if f"{s}_{f}" not in selected]
    if missing:
        return {"available": False, "missing_fields": missing, "rows": [], "reliability": []}
    ready = np.logical_and.reduce([selected[s+"_expert_ready"].eq(True).to_numpy() for s in strategies])
    values = [s+"_"+f for s in strategies for f in fields if f != "expert_ready"]
    numeric = selected[values].apply(pd.to_numeric, errors="raise")
    finite = np.isfinite(numeric).all(axis=1)
    selected = selected.loc[ready & finite].copy()
    if not np.allclose(selected.control_threshold_eur_mwh, selected.congestion_threshold_eur_mwh,
                       rtol=0, atol=1e-9):
        raise ValueError("OOS probability comparison requires identical control/congestion event thresholds.")
    selected["event"] = (selected.actual-selected.nuclear_kalman).ge(selected.control_threshold_eur_mwh)
    rows, bins = [], []
    for zone in ["ALL", *zones]:
        part = selected if zone == "ALL" else selected.loc[selected.zone.eq(zone)]
        y = part.event.to_numpy(float)
        for strategy in strategies:
            row = {"zone": zone, "strategy": strategy, "n_country_hours": len(part),
                   "n_events": int(y.sum()), "event_rate": y.mean() if len(y) else None,
                   "sparse_calibration_hours": int(part.get(strategy+"_calibration_status",
                       pd.Series(index=part.index, dtype="object")).eq("regularized_sparse_support").sum())}
            for variant, field in (("raw", "raw_probability"), ("calibrated", "spike_probability")):
                p = part[strategy+"_"+field].to_numpy(float)
                if ((p < 0) | (p > 1)).any():
                    raise ValueError("OOS probability outside [0,1].")
                clipped = np.clip(p, 1e-15, 1-1e-15)
                row[variant+"_brier"] = np.mean((p-y)**2) if len(p) else None
                row[variant+"_log_loss"] = -np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped)) if len(p) else None
                assigned = np.minimum((p*10).astype(int), 9)
                for b in range(10):
                    mask = assigned == b
                    bins.append({"zone": zone, "strategy": strategy, "variant": variant,
                        "bin_lower": b/10, "bin_upper": (b+1)/10, "n_country_hours": int(mask.sum()),
                        "n_events": int(y[mask].sum()),
                        "mean_probability": p[mask].mean() if mask.any() else None,
                        "observed_frequency": y[mask].mean() if mask.any() else None})
            rows.append(row)
    return _safe({"available": True, "rows": rows, "reliability": bins,
        "support": "evaluation outputs only; joint expert-ready and finite probabilities for both new chains; identical thresholds",
        "target": "actual minus frozen NYX >= current CORE90 threshold; thresholds may vary by past-only fit",
        "log_loss_clip": 1e-15, "bin_edges": list(np.linspace(0, 1, 11))})


def read_folds(audit, *, root, seals):
    if not audit.get("snapshot"):
        return None
    source = Path(audit["snapshot"]).resolve()
    source.relative_to(Path(root).resolve()/"runs/experiments/nyx_congestion_calibration_v1")
    manifest_sha = _capture(source/"manifest.json", seals)
    if audit.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Calibration report snapshot identity mismatch.")
    _capture(source/"results_manifest.json", seals)
    result = json.loads((source/"results_manifest.json").read_text(encoding="utf8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Calibration folds require completed results of the same snapshot.")
    expected = result.get("result_files", {}).get("folds.parquet")
    if not expected:
        raise ValueError("Calibration result does not seal folds.parquet.")
    _capture(source/"folds.parquet", seals, expected)
    frame = pd.read_parquet(source/"folds.parquet")
    if {"strategy", "fit_day", "status", "reason"}.difference(frame):
        raise ValueError("Missing calibration fold diagnostics.")
    if frame.duplicated(["strategy", "fit_day"]).any() or not set(frame.strategy).issubset({"control", "congestion"}):
        raise ValueError("Invalid calibration fold identities.")
    frame["fit_day"] = pd.to_datetime(frame.fit_day, errors="raise").dt.strftime("%Y-%m-%d")
    return frame


def fold_summary(folds, *, end_day, days):
    if folds is None:
        return {"available": False, "rows": [], "case": [], "reasons": []}
    end = date.fromisoformat(end_day)
    selected = folds.loc[folds.fit_day.between((end-timedelta(days=days-1)).isoformat(), end_day)]
    rows, case, reasons = [], [], []
    for strategy in ("control", "congestion"):
        part = selected.loc[selected.strategy.eq(strategy)]
        trained = part.loc[part.status.eq("trained")]
        rows.append({"strategy": strategy, "n_attempts": len(part), "n_trained": len(trained),
                     "n_not_trained": len(part)-len(trained),
                     "first_trained_fit": trained.fit_day.min() if len(trained) else None,
                     "last_trained_fit": trained.fit_day.max() if len(trained) else None})
        for (status, reason), group in part.fillna({"reason": ""}).groupby(["status", "reason"], dropna=False):
            reasons.append({"strategy": strategy, "status": status, "reason": reason, "n_fits": len(group)})
        before = folds.loc[folds.strategy.eq(strategy) & folds.fit_day.le("2026-09-14")].sort_values("fit_day")
        if len(before):
            case.append(before.iloc[-1].to_dict())
    return _safe({"available": True, "rows": rows, "case": case, "reasons": reasons})


def build_report(predictions, source_directory, destination, *, root, audit):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.relative_to(root/"runs/experiments/nyx_congestion_calibration_v1")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Report destination is not empty.")
    long, wide, source_audit, seals = assemble_panel(predictions, source_directory)
    config = source_audit["source_config"]
    delivery = date.fromisoformat(config["delivery_day"])
    end_day = config.get("end_day") or (delivery-timedelta(days=1)).isoformat()
    if (config.get("evaluation_days") != 365 or date.fromisoformat(end_day) != delivery-timedelta(days=1)
            or config.get("timezone") != "Europe/Paris" or config.get("cutoff_time") != "08:00"):
        raise ValueError("Calibration report requires 365 days before delivery and strict 08:00 Paris origin.")
    zones = sorted(wide.zone.unique())
    if set(zones) != set(config["zones"]):
        raise ValueError("Calibration report countries differ from sealed source.")
    folds = read_folds(audit, root=root, seals=seals)
    economic_config = root/"config/economic_value.yaml"
    _capture(economic_config, seals)
    scored = long.loc[long["sample"].eq("evaluation")]
    periods = {}
    for days in (365, 7):
        result = compute_kpis(scored, end_day=end_day, days=days, zones=zones)
        result.pop("daily_rows", None)
        result["economic"] = compute_economic_kpis(scored, end_day=end_day, days=days, zones=zones, config_path=economic_config)
        result.update(diagnostics(wide, end_day=end_day, days=days, zones=zones))
        result["probability"] = probability_diagnostics(wide, end_day=end_day, days=days, zones=zones)
        result["fits"] = fold_summary(folds, end_day=end_day, days=days)
        periods[str(days)] = result
    june = compute_kpis(scored, end_day="2026-06-26", days=3, zones=zones)
    june.pop("daily_rows", None)
    june.update(diagnostics(wide, end_day="2026-06-26", days=3, zones=zones))
    civil = wide.timestamp_utc.dt.tz_convert("Europe/Paris")
    dates = ["2026-09-14", "2026-06-24", "2026-06-25", "2026-06-26"]
    names = [item["id"] for item in CATALOG if item["id"] != "__storm__"]
    columns = [*KEYS, "actual", "storm", *names]
    columns += [name for name in wide if name.startswith(("control_", "congestion_")) and name not in columns]
    cases = wide.loc[civil.dt.strftime("%Y-%m-%d").isin(dates) & wide["sample"].eq("evaluation"), columns].copy()
    cases["day"] = cases.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    cases["hour"] = cases.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    payload = _safe({"schema_version": 1, "catalog": CATALOG, "zones": zones, "periods": periods,
        "june": june, "case": cases.to_dict("records"), "case_days": dates,
        "delivery_day": delivery, "end_day": end_day, "audit": audit, "source_audit": source_audit,
        "source_sha256": seals, "generated_at_utc": datetime.now(timezone.utc),
        "method": {"calibration_calendar_days": 90, "training_window_calendar_days": 365,
            "calibrator": "regularized monotone hierarchical Platt; slope>=0 shrunk to1, global intercept/zone offsets shrunk to0",
            "thresholds": "CORE only, before fixed90 calendar calibration block",
            "classifier_exclusion_days": 90, "severity_exclusion_days": 28,
            "severity_overlap_with_probability_calibration": "first62 calendar days may enter severityCDF; never the classifier or probability calibrator inputs",
            "minimum_calendar_span_days": 118,
            "ridge": {"slope": 10., "intercept": 2., "zone": 20.},
            "stage1_retrained": False, "zero_positive_calibration_allowed": True,
            "detector_comparison": "new control vs new enriched only; old28-day CORE thresholds differ"},
        "diagnostic_only": True, "production_modified": False, "activation_performed": False,
        "independent_validation": False, "forecast_pit_certified": False, "economic_reference_executable": False})
    for path, expected in seals.items():
        if _sha(path) != expected:
            raise ValueError("Calibration report source changed during calculation.")
    destination.mkdir(parents=True, exist_ok=True)
    metrics, report = destination/"metrics.json", destination/"nyx_congestion_calibration_report.html"
    with metrics.open("x", encoding="utf8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    render_report(payload, report)
    for path, expected in seals.items():
        if _sha(path) != expected:
            raise ValueError("Calibration report source changed during publication.")
    return {"status": "completed", "report_path": str(report), "metrics_path": str(metrics),
        "report_sha256": _sha(report), "metrics_sha256": _sha(metrics), "diagnostic_only": True,
        "production_modified": False, "summary": {
            "annual_rows": [r for r in periods["365"]["rows"] if r["zone"] == "ALL"],
            "annual_economic_rows": [r for r in periods["365"]["economic"]["rows"] if r["zone"] == "ALL"],
            "coverage": periods["365"]["coverage"], "fits": periods["365"]["fits"]}}


def render_report(payload, destination):
    encoded = json.dumps(_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(old, new)
    with Path(destination).open("x", encoding="utf8") as handle:
        handle.write(TEMPLATE.replace("@@DATA@@", encoded))
    return Path(destination)


TEMPLATE = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX · Calibration 90 jours</title><style>
:root{--bg:#f1f5f9;--paper:#fff;--ink:#1b2c43;--muted:#52657b;--line:#d8e1ec;--nyx:#b87504;--storm:#037eaf;--new:#7943bd;--obs:#27384d;--good:#14724c;--bad:#b33b46;--accent:#255abb}
:root[data-theme=dark]{--bg:#111a26;--paper:#1a283a;--ink:#ebf2fc;--muted:#b1c1d6;--line:#354961;--nyx:#ffc268;--storm:#67d0fa;--new:#c4a0ff;--obs:#eef4ff;--good:#7bddaa;--bad:#ff9ea8;--accent:#91b8ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:1700px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:18px;align-items:start}h1{font-size:31px;margin:4px 0}h2{font-size:19px;margin:0 0 10px}h3{font-size:15px;margin:12px 0}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:2px;text-transform:uppercase}.muted,small{color:var(--muted)}button,select{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:var(--paper);color:var(--ink);max-width:100%}label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:3px}.filters{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}.card,.notice{background:var(--paper);padding:20px;border:1px solid var(--line);border-radius:10px;margin:18px 0}.notice{border-left:4px solid var(--accent)}.scroll{overflow-x:auto}table{width:100%;white-space:nowrap;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted)}th:first-child,td:first-child{text-align:left}tr.primary td:first-child{font-weight:700;color:var(--accent)}.good{color:var(--good)}.bad{color:var(--bad)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{min-width:0}.plot{width:100%}svg{width:100%;height:auto}svg text{fill:var(--muted);font:11px system-ui}.tip{min-height:25px;color:var(--muted);font-size:12px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px}.legend span:before{content:'━';margin-right:6px}.foot{font-size:12px;color:var(--muted)}pre{max-height:500px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:11px}code{font-size:12px}@media(max-width:850px){main{padding:14px}.grid{grid-template-columns:1fr}header{flex-direction:column}h1{font-size:24px}}
</style></head><body><main><header><div><div class="eyebrow">Laboratoire isolé · aucun changement opérationnel</div><h1>Calibration chronologique · 90 jours</h1><div id="subtitle" class="muted"></div></div><button id="theme">Mode nuit</button></header>
<div class="notice"><strong>Objectif : mieux calibrer sans forcer une correction.</strong> Le bloc de calibration est fixé à 90 jours calendaires, avec régularisation et partage d'information entre pays. Un bloc contenant peu ou aucun événement positif n'est plus rejeté pour cette seule raison. Les signaux CNEC et régionaux amont restent inchangés. Cette année déjà examinée est exploratoire ; aucun modèle n'est choisi ni promu à partir du seul 14 septembre.</div>
<div class="filters"><label>Pays<select id="zone"></select></label><label>Période des KPI<select id="period"><option value="365">365 derniers jours</option><option value="7">7 derniers jours</option></select></label><label>Courbe de prix<select id="model"></select></label><label>Épisode détaillé<select id="day"></select></label></div>
<section class="card"><h2>KPI · comparaison appariée des douze modèles</h2><p id="coverage" class="muted"></p><div class="scroll"><table id="kpi"><thead><tr><th>Modèle</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Win rate heure<br>vs Storm</th><th>Win rate MAE jour<br>vs Storm</th><th>Win rate prix moyen jour<br>vs Storm</th><th>MAE prix moyen jour<br>€/MWh</th><th>Prix moyen<br>€/MWh</th></tr></thead><tbody></tbody></table></div><p class="foot">Même intersection d'heures physiques pour tous les modèles, observé inclus. Replis sur NYX conservés. Les jours incomplets sont exclus seulement des scores journaliers ; DST : 23/24/25 heures. Ex æquo au dénominateur, sans victoire. Couleurs des erreurs : comparaison à NYX, sans test de significativité.</p></section>
<section class="card"><h2>EVA · politique économique inchangée</h2><p id="policy" class="muted"></p><div class="scroll"><table id="eva"><thead><tr><th>Modèle</th><th>P&amp;L net simulé €</th><th>Gain vs Storm €</th><th>Gain vs NYX €</th><th>Gain vs Storm<br>€/MWh potentiel</th><th>Heures-pays communes</th></tr></thead><tbody></tbody></table></div><p class="foot">Référence : prix observé day-ahead D−1, à la même heure civile. Proxy non exécutable à 08 h, pas P&amp;L négociable démontré. Seuil, allocation, coûts et référence conservés ; références manquantes ou ambiguës exclues symétriquement. Aucune annualisation et aucune optimisation de règle sur les résultats.</p></section>
<section class="card"><h2>Disponibilité réelle du correcteur</h2><div class="scroll"><table id="ready"><thead><tr><th>Chaîne</th><th>Heures-pays prêtes</th><th>Heures-pays communes</th><th>Fits entraînés / tentés</th><th>Premier fit entraîné</th><th>Dernier fit entraîné</th></tr></thead><tbody></tbody></table></div><details><summary>Raisons enregistrées des entraînements et replis</summary><pre id="fit-reasons"></pre></details><p class="foot">Un signal de risque amont disponible ne signifie pas que le correcteur est entraîné ou autorisé à intervenir. Les scores incluent les abstentions. Les diagnostics de fits sont lus dans les résultats scellés du nouveau snapshot ; un échec récent n'est pas masqué par le dernier fit réussi.</p></section>
<section class="card"><h2>Interventions et heures chères</h2><div class="scroll"><table id="changes"><thead><tr><th>Modèle</th><th>Heures changées</th><th>Améliorées</th><th>Dégradées</th><th>MAE prix ≥200</th><th>RMSE prix ≥200</th><th>MAE prix &lt;200</th></tr></thead><tbody></tbody></table></div><p class="foot">Découpage prix observé ≥200 €/MWh : diagnostic ex post, jamais une variable de déclenchement. Comparer les nouveaux modèles enrichis aux nouveaux contrôles appariés, avant d'attribuer un gain aux signaux congestion.</p><h3>Erreur moyenne par heure civile</h3><div id="hour-plot" class="plot"></div><div id="hour-tip" class="tip"></div></section>
<section class="card"><h2 id="case-title">Épisode détaillé</h2><p class="muted">Prix à 19 h Europe/Paris et profil complet. Ces épisodes servent au diagnostic, pas à choisir une règle spécifique par pays ou date.</p><div class="scroll"><table id="case"><thead><tr><th>Pays</th><th>Observé</th><th>Storm</th><th>NYX</th><th>Congestion précédente<br>gouvernée</th><th>Nouveau contrôle<br>direct</th><th>Nouveau contrôle<br>gouverné</th><th>Nouvelle congestion<br>directe</th><th>Nouvelle congestion<br>gouvernée</th></tr></thead><tbody></tbody></table></div><div class="legend" id="price-legend"></div><div class="grid" id="case-charts"></div><h3>À 19 h · du risque brut au risque calibré</h3><div class="scroll"><table id="probability"><thead><tr><th>Pays / chaîne</th><th>Probabilité brute</th><th>Probabilité calibrée</th><th>Seuil événement<br>€/MWh</th><th>Fenêtre CAL<br>jours</th><th>État calibration</th><th>Expert prêt</th><th>Fit de l'expert</th><th>Raison proposition</th></tr></thead><tbody></tbody></table></div><p class="foot"><strong>Comparer les probabilités uniquement entre les deux nouvelles chaînes.</strong> Les anciens détecteurs à calibration 28 jours utilisaient un CORE et des seuils d'événement différents. Comparer leurs P50 reste pertinent ; comparer directement leurs probabilités comme si la cible était identique ne l'est pas. Une probabilité recalibrée ne force pas une hausse : la médiane conditionnelle, les limites et la gouvernance restent déterminantes.</p><details><summary>Dernières tentatives de fit au plus tard le 14 septembre</summary><pre id="case-fits"></pre></details></section>
<section class="card"><h2>24–26 juin 2026 · résumé commun aux douze modèles</h2><p id="june-coverage" class="muted"></p><div class="scroll"><table id="june"><thead><tr><th>Modèle</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Win rate horaire<br>vs Storm</th><th>MAE prix moyen jour</th><th>Prix moyen</th><th>Heures communes</th></tr></thead><tbody></tbody></table></div><p class="foot">Période fixe de trois jours, indépendante du filtre 365/7 jours. Les graphiques de chaque journée sont accessibles dans « Épisode détaillé ». Ni les prix réalisés ni ces journées ne déterminent une règle spéciale.</p></section>
<section class="card"><h2>Méthode et limites</h2><div class="grid"><div><p><strong>Découpage fixe, pas sélection de fenêtre.</strong> Dans les 365 jours passés au maximum, le classifieur et les seuils d'événement apprennent avant D−90 ; ses scores des 90 derniers jours calibrent les probabilités. La distribution de sévérité conserve séparément son cutoff D−28 : elle peut utiliser les 62 premiers jours du bloc CAL, mais ses sorties n'entrent jamais dans le calibrateur. CAL est donc disjoint de l'entraînement du classifieur, <strong>pas de tous les composants de la chaîne</strong>. Toutes ces observations doivent être disponibles au cutoff courant.</p><p><strong>Platt hiérarchique monotone.</strong> Pente ≥0, ramenée vers 1 ; intercept global et décalages pays ramenés vers 0. Pénalités fixes : pente 10, intercept 2, pays 20, appliquées à la somme des log-loss. Une pente nulle peut créer des ex æquo ; la monotonie porte sur chaque pays, pas sur un classement global invariant. Ce calibrateur spécifique reste une extension empirique, pas une garantie de calibration, notamment en absence d'événements positifs. Le principe de scores de calibration séparés de l'apprentissage du classifieur est décrit dans la <a href="https://scikit-learn.org/stable/modules/calibration.html" target="_blank" rel="noopener">documentation scikit-learn</a>.</p><p><strong>Minima maintenus.</strong> Au moins 118 jours calendaires d'historique, 90 jours passés admissibles par pays, 28 jours CORE et 14 jours CAL par pays. CORE et sévérité requièrent 30 observations de chaque classe et cinq dates positives distinctes. Il s'agit de jours avec observations admissibles, pas forcément de journées complètes. Peu ou zéro positif dans CAL n'est plus à lui seul un blocage ; les autres contrôles et la convergence restent requis.</p></div><div><p><strong>P50, pas moyenne.</strong> Les probabilités conditionnent la distribution d'erreur à deux régimes. Son quantile 0,5, puis la gouvernance et la calibration des intervalles, donnent la proposition. On n'ajoute pas « probabilité × amplitude » manuellement à NYX. Une calibration plus disponible peut augmenter les fausses corrections : elles restent visibles.</p><p><strong>Signaux amont gelés.</strong> Ni les experts CNEC ni la pression régionale ne sont réentraînés. Leurs prévisions OOF, leurs limites de couverture et les délais de disponibilité des labels restent ceux de la source. Cutoff inchangé : 08 h Europe/Paris. La présence du label dans une archive ne suffit pas à le rendre disponible lors d'un fit ancien.</p><p><strong>Comparaison exploratoire, production intacte.</strong> Les douze modèles partagent une intersection exacte, mais cette année a déjà servi à formuler des hypothèses. Les anciens et nouveaux résultats comparent des chaînes complètes. Aucun choix sur le 14 septembre, aucune garantie de gain et aucune promotion automatique : une période future gelée reste nécessaire.</p></div></div><details><summary>Paramètres, splits, folds, provenance et SHA</summary><pre id="audit"></pre></details></section>
<noscript>JavaScript est nécessaire pour filtrer ce rapport. Les métriques sont aussi disponibles dans metrics.json.</noscript></main><script id="calibration-data" type="application/json">@@DATA@@</script><script>
'use strict';const D=JSON.parse(document.getElementById('calibration-data').textContent),$=id=>document.getElementById(id),labels=Object.fromEntries(D.catalog.map(m=>[m.id,m.label]));const fmt=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',pct=v=>Number.isFinite(v)?fmt(v*100)+' %':'—',color=k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim(),find=(rows,z,m)=>rows.find(r=>r.zone===z&&r.model_id===m)||{};
function opt(id,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;$(id).append(o)}opt('zone','ALL','ALL · pays agrégés');D.zones.forEach(z=>opt('zone',z,z));D.catalog.filter(m=>m.id!=='__storm__'&&m.id!=='nuclear_kalman').forEach(m=>opt('model',m.id,m.label));$('model').value='nyx_congestion_calibrated';D.case_days.forEach(d=>opt('day',d,d));
function td(tr,text,cl=''){const e=document.createElement('td');e.textContent=text;e.className=cl;tr.append(e)}function cls(v,b,lower=true){return Number.isFinite(v)&&Number.isFinite(b)&&Math.abs(v-b)>1e-9?((lower?v<b:v>b)?'good':'bad'):''}function table(id,rows,fn){const b=$(id).querySelector('tbody');b.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');if(r.model_id==='nyx_congestion_calibrated')tr.className='primary';td(tr,labels[r.model_id]||r.label||r.zone||r.model_id);fn(tr,r);b.append(tr)})}
function plot(id,tip,series){const host=$(id);host.replaceChildren();$(tip).textContent='';const vals=series.flatMap(s=>s.points.filter(p=>Number.isFinite(p.y)).map(p=>p.y));if(!vals.length){host.textContent='Aucune donnée commune.';return}let lo=Math.min(0,...vals),hi=Math.max(...vals);if(hi===lo)hi+=1;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 680 280');svg.setAttribute('role','img');svg.setAttribute('aria-label','Profil horaire en EUR par MWh');host.append(svg);const x=h=>60+h*595/23,y=v=>235-(v-lo)*210/(hi-lo);function el(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;el('line',{x1:60,x2:655,y1:y(v),y2:y(v),stroke:color('line')});el('text',{x:52,y:y(v)+4,'text-anchor':'end'},fmt(v,0))}for(const h of [0,6,12,18,23])el('text',{x:x(h),y:260,'text-anchor':'middle'},h+' h');series.forEach(s=>{let path='',open=false;s.points.forEach(p=>{if(!Number.isFinite(p.y)){open=false;return}path+=(open?'L':'M')+x(p.x)+','+y(p.y)+' ';open=true});el('path',{d:path,fill:'none',stroke:s.color,'stroke-width':2})});svg.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect(),h=Math.max(0,Math.min(23,Math.round(((e.clientX-r.left)*680/r.width-60)/595*23)));$(tip).textContent=h+' h · '+series.map(s=>s.label+': '+fmt(s.points.find(p=>p.x===h)?.y)).join(' · ')})}
function render(){const z=$('zone').value,m=$('model').value,p=D.periods[$('period').value],day=$('day').value,rows=p.rows.filter(r=>r.zone===z),base=find(rows,z,'nuclear_kalman');$('subtitle').textContent=p.period.start_day+' → '+p.period.end_day+' · livraison '+D.delivery_day+' non évaluée · cutoff 08 h Europe/Paris';const cov=p.coverage.filter(r=>z==='ALL'||r.zone===z),sum=k=>cov.reduce((a,r)=>a+(r[k]||0),0);$('coverage').textContent=fmt(sum('n_common_hours'),0)+' / '+fmt(sum('n_expected_hours'),0)+' heures-pays communes · '+fmt(sum('n_complete_days'),0)+' jours-pays complets · prix observé moyen '+fmt(base.observed_mean_price_eur_mwh)+' €/MWh';table('kpi',rows,(tr,r)=>{['mae_eur_mwh','rmse_eur_mwh'].forEach(k=>td(tr,fmt(r[k]),cls(r[k],base[k])));['win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct'].forEach(k=>td(tr,Number.isFinite(r[k])?fmt(r[k])+' %':'—'));td(tr,fmt(r.mae_day_mean_price_eur_mwh),cls(r.mae_day_mean_price_eur_mwh,base.mae_day_mean_price_eur_mwh));td(tr,fmt(r.mean_price_eur_mwh))});const ea=p.economic.audit,eb=find(p.economic.rows,z,'nuclear_kalman');$('policy').textContent='Portefeuille alternatif '+fmt(ea.portfolio_capacity_mw,0)+' MW ; '+Object.entries(ea.zone_capacity_mw).map(([a,b])=>a+' '+fmt(b,0)+' MW').join(', ')+' fixes. Seuil |edge| > '+fmt(ea.signal_hurdle_eur_mwh)+' €/MWh ; coûts '+fmt(ea.net_cost_eur_mwh)+' €/MWh.';table('eva',p.economic.rows.filter(r=>r.zone===z),(tr,r)=>{td(tr,fmt(r.pnl_net_eur,0));td(tr,fmt(r.gain_vs_storm_eur,0),cls(r.gain_vs_storm_eur,0,false));const d=Number.isFinite(r.pnl_net_eur)&&Number.isFinite(eb.pnl_net_eur)?r.pnl_net_eur-eb.pnl_net_eur:null;td(tr,fmt(d,0),cls(d,0,false));td(tr,fmt(r.gain_vs_storm_per_potential_mwh));td(tr,fmt(r.n_country_hours,0))});
table('ready',p.readiness.filter(r=>r.zone===z).map(r=>({...r,label:r.strategy==='control'?'Nouveau contrôle':'Nouvelle congestion'})),(tr,r)=>{td(tr,fmt(r.ready_country_hours,0));td(tr,fmt(r.n_common_country_hours,0));const f=p.fits.rows.find(v=>v.strategy===r.strategy)||{};td(tr,fmt(f.n_trained,0)+' / '+fmt(f.n_attempts,0));td(tr,f.first_trained_fit||'—');td(tr,f.last_trained_fit||'—')});$('fit-reasons').textContent=p.fits.available?JSON.stringify(p.fits.reasons,null,2):'Aucun folds scellé disponible.';$('case-fits').textContent=JSON.stringify(p.fits.case,null,2);table('changes',p.interventions.filter(r=>r.zone===z&&r.model_id!=='__storm__'),(tr,r)=>{['changed_hours','better','worse'].forEach(k=>td(tr,fmt(r[k],0)));['tail_mae','tail_rmse','normal_mae'].forEach(k=>td(tr,fmt(r[k])))});
const specs=[{id:'actual',label:'Observé',color:color('obs')},{id:'storm',label:'Storm',color:color('storm')},{id:'nuclear_kalman',label:'NYX',color:color('nyx')},{id:m,label:labels[m],color:color('new')}];plot('hour-plot','hour-tip',specs.slice(1).map(s=>({...s,points:p.hourly.filter(r=>r.zone===z&&r.model_id===(s.id==='storm'?'__storm__':s.id)).map(r=>({x:r.hour,y:r.mae}))})));const selected=D.case.filter(r=>r.day===day&&(z==='ALL'||r.zone===z));$('case-title').textContent=day+' · épisode détaillé';table('case',selected.filter(r=>r.hour===19),(tr,r)=>{['actual','storm','nuclear_kalman','nyx_congestion','calibrated_control_direct','calibrated_control_governed','congestion_calibrated_direct','nyx_congestion_calibrated'].forEach(k=>td(tr,fmt(r[k])))});$('price-legend').replaceChildren();specs.forEach(s=>{const e=document.createElement('span');e.textContent=s.label;e.style.color=s.color;$('price-legend').append(e)});$('case-charts').replaceChildren();D.zones.filter(a=>z==='ALL'||a===z).forEach(zone=>{const box=document.createElement('div');box.className='chart';const title=document.createElement('h3');title.textContent=zone;const host=document.createElement('div');host.id='case-'+zone;const tip=document.createElement('div');tip.id='tip-'+zone;tip.className='tip';box.append(title,host,tip);$('case-charts').append(box);const points=selected.filter(r=>r.zone===zone).sort((a,b)=>a.hour-b.hour);plot(host.id,tip.id,specs.map(s=>({...s,points:points.map(r=>({x:r.hour,y:r[s.id]}))}))) });
table('probability',selected.filter(r=>r.hour===19).flatMap(r=>['control','congestion'].map(s=>({...r,strategy:s,label:r.zone+' · '+(s==='control'?'contrôle':'congestion')}))),(tr,r)=>{const s=r.strategy+'_';td(tr,pct(r[s+'raw_probability']));td(tr,pct(r[s+'spike_probability']));td(tr,fmt(r[s+'threshold_eur_mwh']));td(tr,fmt(r[s+'calibration_window_days'],0));td(tr,r[s+'calibration_status']||'—');td(tr,r[s+'expert_ready']===true?'Oui':r[s+'expert_ready']===false?'Non':'—');td(tr,r[s+'expert_fit_day']||'—');td(tr,r[s+'proposal_reason']||'—')});const jr=D.june.rows.filter(r=>r.zone===z),jb=find(jr,z,'nuclear_kalman');$('june-coverage').textContent='24/06/2026 → 26/06/2026 · '+fmt(jb.n_hours,0)+' heures-pays communes · prix observé moyen '+fmt(jb.observed_mean_price_eur_mwh)+' €/MWh';table('june',jr,(tr,r)=>{td(tr,fmt(r.mae_eur_mwh),cls(r.mae_eur_mwh,jb.mae_eur_mwh));td(tr,fmt(r.rmse_eur_mwh),cls(r.rmse_eur_mwh,jb.rmse_eur_mwh));td(tr,Number.isFinite(r.win_rate_hour_pct)?fmt(r.win_rate_hour_pct)+' %':'—');td(tr,fmt(r.mae_day_mean_price_eur_mwh));td(tr,fmt(r.mean_price_eur_mwh));td(tr,fmt(r.n_hours,0))});$('audit').textContent=JSON.stringify({method:D.method,audit:D.audit,source_sha256:D.source_sha256,production_modified:false,independent_validation:false},null,2)}
['zone','period','model','day'].forEach(id=>$(id).addEventListener('change',render));$('theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'Mode jour':'Mode nuit';render()});render();
</script></body></html>'''

PROBABILITY_SECTION = r'''<section class="card"><h2>Probabilités · évaluation chronologique hors échantillon</h2><p class="muted">Brut et calibré, sur exactement les mêmes heures des deux nouvelles chaînes prêtes. Le filtre 365/7 jours et le pays s'appliquent ici aussi.</p><div class="scroll"><table id="probability-kpi"><thead><tr><th>Chaîne</th><th>Heures-pays</th><th>Événements</th><th>Fréquence observée</th><th>Brier brut</th><th>Brier calibré</th><th>Log-loss brute</th><th>Log-loss calibrée</th><th>CAL support faible<br>heures prédites</th></tr></thead><tbody></tbody></table></div><p class="foot">Événement : observé − NYX ≥ seuil CORE90 du fit courant ; seuils identiques vérifiés entre contrôle et congestion. Scores calculés uniquement sur les prévisions chronologiques évaluées, jamais sur le bloc CAL ayant ajusté ces probabilités. Moins est mieux pour Brier et log-loss, mais leur amélioration ne prouve pas à elle seule une meilleure calibration : ils reflètent aussi la discrimination. Les observations horaires sont dépendantes ; peu d'événements et cette année déjà examinée limitent la portée de la comparaison. L'intersection prête exclut les replis, qui restent inclus dans tous les KPI de prix.</p><details><summary>Fiabilité · classes fixes de probabilité, effectifs et événements</summary><p class="foot">Dix classes [0 ; 0,1[, …, [0,9 ; 1]. Comparer la probabilité moyenne à la fréquence observée ; classes vides laissées sans valeur. Classes peu peuplées : diagnostic fragile, sans garantie statistique.</p><div class="scroll"><table id="reliability"><thead><tr><th>Chaîne / score</th><th>Classe probabilité</th><th>Heures-pays</th><th>Événements</th><th>Probabilité moyenne</th><th>Fréquence observée</th></tr></thead><tbody></tbody></table></div></details></section>'''

PROBABILITY_SCRIPT = r'''function renderProbabilities(p,z){const a=p.probability||{rows:[],reliability:[]};table('probability-kpi',a.rows.filter(r=>r.zone===z).map(r=>({...r,label:r.strategy==='control'?'Nouveau contrôle':'Nouvelle congestion'})),(tr,r)=>{td(tr,fmt(r.n_country_hours,0));td(tr,fmt(r.n_events,0));td(tr,pct(r.event_rate));td(tr,fmt(r.raw_brier,5));td(tr,fmt(r.calibrated_brier,5),cls(r.calibrated_brier,r.raw_brier));td(tr,fmt(r.raw_log_loss,5));td(tr,fmt(r.calibrated_log_loss,5),cls(r.calibrated_log_loss,r.raw_log_loss));td(tr,fmt(r.sparse_calibration_hours,0))});table('reliability',a.reliability.filter(r=>r.zone===z).map(r=>({...r,label:(r.strategy==='control'?'Contrôle':'Congestion')+' · '+(r.variant==='raw'?'brut':'calibré')})),(tr,r)=>{td(tr,'['+fmt(r.bin_lower,1)+' ; '+fmt(r.bin_upper,1)+(r.bin_upper===1?']':'['));td(tr,fmt(r.n_country_hours,0));td(tr,fmt(r.n_events,0));td(tr,pct(r.mean_probability));td(tr,pct(r.observed_frequency))})}'''

_PROBABILITY_ANCHOR = '<section class="card"><h2>Interventions et heures chères</h2>'
TEMPLATE = (TEMPLATE.replace(_PROBABILITY_ANCHOR, PROBABILITY_SECTION+_PROBABILITY_ANCHOR)
            .replace('function render(){', PROBABILITY_SCRIPT+'\nfunction render(){')
            .replace('const specs=', 'renderProbabilities(p,z);const specs='))
