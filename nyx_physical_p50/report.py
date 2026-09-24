"""Paired, offline research report for physical P50 candidates; no promotion."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kpi_report.economic import compute_economic_kpis
from kpi_report.metrics import compute_kpis

KEYS = ["zone", "timestamp_utc"]
MODELS = ("fuel_transport_direct", "fuel_transport_governed", "network_fuel_direct", "nyx_physical_p50")
CATALOG = [
    {"id": "nuclear_kalman", "label": "NYX opérationnel", "kind": "production"},
    {"id": "nyx_physical_p50", "label": "Réseau + CGC · gouverné", "kind": "candidate"},
    {"id": "network_fuel_direct", "label": "Réseau + CGC · direct", "kind": "candidate"},
    {"id": "fuel_transport_governed", "label": "CGC · gouverné", "kind": "candidate"},
    {"id": "fuel_transport_direct", "label": "CGC · direct", "kind": "candidate"},
    {"id": "coherent_forest_direct", "label": "P50 précédent · direct", "kind": "previous"},
    {"id": "coherent_forest_governed", "label": "P50 précédent · gouverné", "kind": "previous"},
    {"id": "__storm__", "label": "Storm", "kind": "benchmark"},
]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (Path, date, datetime, pd.Timestamp)):
        return str(value)
    return value


def _typed(frame):
    frame = frame.copy(deep=True)
    if "storm" not in frame and "benchmark_forecast" in frame:
        frame["storm"] = frame.benchmark_forecast
    required = {*KEYS, "forecast", "actual", "storm", "sample", "forecast_origin_utc"}
    if required.difference(frame):
        raise ValueError(f"Report missing columns: {sorted(required.difference(frame))}")
    for name in ("timestamp_utc", "forecast_origin_utc", "label_available_at_utc"):
        if name not in frame:
            continue
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype):
            if any(pd.notna(v) and pd.Timestamp(v).tzinfo is None for v in frame[name]):
                raise ValueError(f"Report {name} must be timezone aware.")
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if frame.timestamp_utc.isna().any() or frame.forecast_origin_utc.isna().any():
        raise ValueError("Report identity timestamps missing.")
    if frame.duplicated(KEYS).any():
        raise ValueError("Report requires unique physical-hour identities.")
    for name in ("forecast", "actual", "storm", *MODELS):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
            if np.isinf(frame[name]).any():
                raise ValueError(f"Report {name} is infinite.")
    return frame.set_index(KEYS).sort_index()


def _same(base, current):
    if not base.index.equals(current.index):
        raise ValueError("Changed physical-hour support.")
    for name in ("forecast", "actual", "storm", "q10", "q90", "sample", "forecast_origin_utc", "label_available_at_utc"):
        if name not in base and name not in current:
            continue
        if name not in base or name not in current:
            raise ValueError(f"Missing shared {name}.")
        if name in ("forecast", "actual", "storm", "q10", "q90"):
            equal = np.allclose(base[name].to_numpy(float), current[name].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
        else:
            equal = base[name].equals(current[name])
        if not equal:
            raise ValueError(f"Changed shared {name}.")


def assemble_panel(predictions, source):
    source = Path(source).resolve()
    seals = {}
    def capture(path, expected=None):
        sha = _sha(path)
        if expected is not None and sha != expected:
            raise ValueError(f"Source checksum mismatch: {path}")
        seals[str(path)] = sha
        return sha
    manifest_sha = capture(source/"manifest.json")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf8"))
    for name in ("source_audit.json", "panel.parquet"):
        expected = manifest.get("input_files", {}).get(name)
        if not expected:
            raise ValueError(f"Source does not seal {name}.")
        capture(source/name, expected)
    source_audit = json.loads((source/"source_audit.json").read_text(encoding="utf8"))
    base, wide = _typed(pd.read_parquet(source/"panel.parquet")), _typed(predictions)
    _same(base, wide)
    if set(MODELS).difference(wide):
        raise ValueError("Missing physical P50 candidates.")
    wide["nuclear_kalman"] = wide.forecast
    results_path = source/"forest/results_manifest.json"
    capture(results_path)
    result = json.loads(results_path.read_text(encoding="utf8"))
    if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
        raise ValueError("Prior forest comparator is not sealed/completed.")
    for name, model in (("predictions.parquet", "coherent_forest_direct"),
                        ("governed_predictions.parquet", "coherent_forest_governed")):
        expected = result.get("result_files", {}).get(name)
        if not expected:
            raise ValueError(f"Prior forest does not seal {name}.")
        capture(source/"forest"/name, expected)
        previous = _typed(pd.read_parquet(source/"forest"/name))
        _same(base, previous)
        point = pd.to_numeric(previous.candidate_forecast, errors="raise")
        if np.isinf(point).any():
            raise ValueError("Infinite comparator forecast.")
        wide[model] = point
    long = []
    columns = ["actual", "storm", "sample", "forecast_origin_utc"]
    columns += [k for k in ("forecast_eligible", "benchmark_eligible") if k in wide]
    for item in CATALOG:
        if item["id"] == "__storm__":
            continue
        part = wide[columns].copy()
        part["model_id"], part["forecast"] = item["id"], wide[item["id"]]
        long.append(part.reset_index())
    return pd.concat(long, ignore_index=True), wide.reset_index(), source_audit, seals


def diagnostics(wide, *, end_day, days, zones):
    local = wide.timestamp_utc.dt.tz_convert("Europe/Paris")
    start = date.fromisoformat(end_day)-timedelta(days=days-1)
    selected = wide.loc[local.dt.date.between(start, date.fromisoformat(end_day)) & wide["sample"].eq("evaluation")].copy()
    names = [m["id"] for m in CATALOG if m["id"] != "__storm__"]
    selected = selected.dropna(subset=[*names, "actual", "storm"])
    selected["hour"] = selected.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    hourly, interventions = [], []
    for zone in ["ALL", *zones]:
        block = selected if zone == "ALL" else selected.loc[selected.zone.eq(zone)]
        baseline = (block.nuclear_kalman-block.actual).abs()
        for model in [*names, "__storm__"]:
            point = block.storm if model == "__storm__" else block[model]
            error = point-block.actual
            changed = (point-block.nuclear_kalman).abs().gt(1e-9)
            gain = baseline-error.abs()
            tail = block.actual.ge(200)
            interventions.append({"zone": zone, "model_id": model, "n_hours": len(block),
                "changed_hours": int(changed.sum()), "better": int((changed & gain.gt(1e-9)).sum()),
                "worse": int((changed & gain.lt(-1e-9)).sum()),
                "tail_n": int(tail.sum()), "tail_mae": error.loc[tail].abs().mean(),
                "tail_rmse": np.sqrt((error.loc[tail]**2).mean()),
                "normal_mae": error.loc[~tail].abs().mean()})
            for hour, group in pd.DataFrame({"hour": block.hour, "ae": error.abs(), "se": error**2}).groupby("hour"):
                hourly.append({"zone": zone, "model_id": model, "hour": int(hour),
                    "mae": group.ae.mean(), "rmse": np.sqrt(group.se.mean()), "n_hours": len(group)})
    return {"hourly": _safe(hourly), "interventions": _safe(interventions)}


def build_report(predictions, source: Path, destination: Path, *, root: Path, audit: dict):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.relative_to(root/"runs/experiments/nyx_physical_p50_v1")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Report destination is not empty.")
    long, wide, source_audit, seals = assemble_panel(predictions, source)
    config = source_audit["source_config"]
    delivery = date.fromisoformat(config["delivery_day"])
    end_day = config.get("end_day") or (delivery-timedelta(days=1)).isoformat()
    if date.fromisoformat(end_day) != delivery-timedelta(days=1) or config["evaluation_days"] != 365:
        raise ValueError("Report requires 365 days ending before sealed delivery.")
    if config.get("timezone") != "Europe/Paris" or config.get("cutoff_time") != "08:00":
        raise ValueError("Report requires strict 08:00 Europe/Paris source contract.")
    zones = sorted(wide.zone.unique().tolist())
    if set(zones) != set(config["zones"]):
        raise ValueError("Report source zones mismatch.")
    civil = wide.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    if (civil.le(date.fromisoformat(end_day)) & ~wide["sample"].eq("evaluation")).any():
        raise ValueError("Non-evaluation sample overlaps scored history.")
    scored = long.loc[long["sample"].eq("evaluation")].copy()
    econ_config = root/"config/economic_value.yaml"
    seals[str(econ_config)] = _sha(econ_config)
    periods = {}
    for days in (365, 7):
        result = compute_kpis(scored, end_day=end_day, days=days, zones=zones)
        result.pop("daily_rows", None)
        result["economic"] = compute_economic_kpis(scored, end_day=end_day, days=days, zones=zones, config_path=econ_config)
        result.update(diagnostics(wide, end_day=end_day, days=days, zones=zones))
        periods[str(days)] = result
    case_columns = [*KEYS, "actual", "storm", "nuclear_kalman", *MODELS,
                    "coherent_forest_direct", "coherent_forest_governed"]
    case_columns += [k for k in wide if k.endswith(("_fuel_ratio_to_core_max", "_fuel_outside_core_support"))]
    case = wide.loc[civil.eq(date(2026, 9, 14)), case_columns].copy()
    case["hour"] = case.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    forensic = None
    candidates = sorted((root/"runs/experiments/nyx_physical_p50_v1/forensics/2026-09-14").glob("*/spread_diagnostic.json"))
    if candidates:
        path = candidates[-1]
        seals[str(path)] = _sha(path)
        forensic = {"path": str(path), "sha256": seals[str(path)], "data": json.loads(path.read_text(encoding="utf8"))}
    initial_signal = None
    initial_path = root/"runs/experiments/nyx_physical_p50_v1/forensics/2026-09-14/initial_signal_audit.json"
    if initial_path.is_file():
        seals[str(initial_path)] = _sha(initial_path)
        initial_signal = json.loads(initial_path.read_text(encoding="utf8"))
        raw = Path(initial_signal["raw_source"]).resolve()
        raw.relative_to(root/"runs/experiments/nyx_physical_p50_v1")
        if _sha(raw) != initial_signal["source_sha256"]:
            raise ValueError("Initial forensic raw source changed.")
        seals[str(raw)] = initial_signal["source_sha256"]
    payload = _safe({"schema_version": 1, "catalog": CATALOG, "zones": zones, "periods": periods,
        "delivery_day": delivery, "end_day": end_day, "case": case.to_dict("records"), "forensic": forensic,
        "initial_signal": initial_signal,
        "audit": audit, "source_audit": source_audit, "source_sha256": seals,
        "generated_at_utc": datetime.now(timezone.utc), "production_modified": False,
        "activation_performed": False, "independent_validation": False, "forecast_pit_certified": False,
        "economic_reference_executable": False})
    for path, expected in seals.items():
        if _sha(path) != expected:
            raise ValueError("Report sources changed during calculation.")
    destination.mkdir(parents=True, exist_ok=True)
    metrics, report = destination/"metrics.json", destination/"nyx_physical_p50_report.html"
    with metrics.open("x", encoding="utf8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    render_report(payload, report)
    for path, expected in seals.items():
        if _sha(path) != expected:
            raise ValueError("Report sources changed during publication.")
    return {"status": "completed", "report_path": str(report), "metrics_path": str(metrics),
        "report_sha256": _sha(report), "metrics_sha256": _sha(metrics), "diagnostic_only": True,
        "production_modified": False, "summary": {"annual_rows": [r for r in periods["365"]["rows"] if r["zone"] == "ALL"],
            "annual_economic_rows": [r for r in periods["365"]["economic"]["rows"] if r["zone"] == "ALL"],
            "coverage": periods["365"]["coverage"]}}


def render_report(payload, destination):
    encoded = json.dumps(_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(old, new)
    with Path(destination).open("x", encoding="utf8") as handle:
        handle.write(TEMPLATE.replace("@@DATA@@", encoded))
    return Path(destination)


TEMPLATE = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYX · P50 physique</title><style>
:root{--bg:#f1f5f9;--paper:#fff;--ink:#1b2c43;--muted:#52657b;--line:#d8e1ec;--nyx:#bd7907;--storm:#037eaf;--new:#7743c5;--obs:#27384d;--good:#14724c;--bad:#b33b46;--accent:#255abb}
:root[data-theme=dark]{--bg:#111a26;--paper:#1a283a;--ink:#ebf2fc;--muted:#b1c1d6;--line:#354961;--nyx:#ffc268;--storm:#67d0fa;--new:#c4a0ff;--obs:#eef4ff;--good:#7bddaa;--bad:#ff9ea8;--accent:#91b8ff}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:14px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:1650px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:15px}h1{font-size:31px;margin:3px 0}h2{font-size:19px;margin:0 0 8px}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:2px;text-transform:uppercase}.muted,small{color:var(--muted)}button,select{color:var(--ink);background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:8px 12px;font:inherit}label{display:flex;flex-direction:column;gap:3px;color:var(--muted);font-size:12px}.filters{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}.card,.notice{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:20px;margin:18px 0}.notice{border-left:4px solid var(--accent)}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted)}th:first-child,td:first-child{text-align:left}tr.primary td:first-child{font-weight:bold;color:var(--accent)}.good{color:var(--good)}.bad{color:var(--bad)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart{min-width:0}.plot{width:100%}svg{width:100%;height:auto}svg text{fill:var(--muted);font:11px system-ui}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px}.legend span:before{content:'━';margin-right:5px}.tooltip{min-height:26px;font-size:12px;color:var(--muted)}.foot{font-size:12px;color:var(--muted)}pre{max-height:450px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:11px}a{color:var(--accent)}@media(max-width:850px){main{padding:14px}.grid{grid-template-columns:1fr}header{flex-direction:column}h1{font-size:25px}}
</style></head><body><main><header><div><div class="eyebrow">Laboratoire isolé · aucune activation</div><h1>Du stress physique au P50</h1><div class="muted" id="subtitle"></div></div><button id="theme">Mode nuit</button></header>
<div class="notice"><strong>Production inchangée.</strong> Hypothèse testée : adapter la sévérité des erreurs extrêmes au coût du gaz propre et à la vulnérabilité directionnelle du réseau. Le détecteur de pics reste figé. Les résultats de l'année déjà examinée ne sont pas une validation indépendante.</div>
<div class="filters"><label>Pays<select id="zone"></select></label><label>Période<select id="period"><option value="365">365 derniers jours</option><option value="7">7 derniers jours</option></select></label><label>P50 détaillé<select id="model"></select></label></div>
<section class="card"><h2>KPI · précision et prix moyens</h2><p class="muted" id="coverage"></p><div class="scroll"><table id="kpi"><thead><tr><th>Modèle</th><th>MAE €/MWh</th><th>RMSE €/MWh</th><th>Win rate heure<br>vs Storm</th><th>Win rate MAE jour<br>vs Storm</th><th>Win rate prix moyen jour<br>vs Storm</th><th>MAE prix moyen jour<br>€/MWh</th><th>Prix moyen<br>€/MWh</th></tr></thead><tbody></tbody></table></div><p class="foot">Même intersection d'heures physiques pour tous. Les ex æquo restent au dénominateur sans être des victoires. Les jours incomplets sont exclus des scores journaliers ; DST : 23/24/25 heures. Couleurs des erreurs : comparaison à NYX, pas une preuve de significativité.</p></section>
<section class="card"><h2>EVA · simulation à politique inchangée</h2><p class="muted" id="policy"></p><div class="scroll"><table id="eva"><thead><tr><th>Modèle</th><th>P&amp;L net simulé €</th><th>Gain vs Storm €</th><th>Gain vs NYX €</th><th>Gain vs Storm<br>€/MWh potentiel</th><th>Heures-pays communes</th></tr></thead><tbody></tbody></table></div><p class="foot">Référence = prix day-ahead observé D−1 à la même heure civile : proxy non exécutable à 08 h, pas un P&amp;L négociable. Références absentes/ambiguës exclues symétriquement. Aucune annualisation. Une hausse du même signal BUY ne modifie pas le volume.</p></section>
<section class="card"><h2>Interventions · faux positifs et heures chères</h2><div class="scroll"><table id="changes"><thead><tr><th>Modèle</th><th>Heures changées</th><th>Améliorées</th><th>Dégradées</th><th>MAE prix ≥200</th><th>RMSE prix ≥200</th><th>MAE prix &lt;200</th></tr></thead><tbody></tbody></table></div><p class="foot">Seuil observé ≥200 €/MWh : découpage descriptif après réalisation, jamais un input. Les replis sur NYX restent inclus dans les KPI annuels.</p><h2>Erreur moyenne par heure</h2><div id="hour-plot" class="plot"></div><div id="hour-tip" class="tooltip"></div></section>
<section class="card"><h2>14 septembre 2026 · 19 h Europe/Paris</h2><p class="muted">Comparaison à l'heure du pic, puis profil journalier complet. Ces exemples ne servent pas à choisir une règle par pays après coup.</p><div class="scroll"><table id="case"><thead><tr><th>Pays</th><th>Observé</th><th>Storm</th><th>NYX</th><th>P50 précédent direct</th><th>CGC direct</th><th>CGC gouverné</th><th>Réseau + CGC direct</th><th>Réseau + CGC gouverné</th><th>CGC / max entraînement</th></tr></thead><tbody></tbody></table></div><div class="legend" id="legend"></div><div class="grid" id="case-charts"></div></section>
<section class="card"><h2>Dossier réseau · preuve ex post, pas variable de forecast</h2><div id="forensic"></div><p class="foot">Les prix duaux et positions nettes sont publiés après le couplage. Leur rapprochement explique une composante des <strong>écarts entre pays</strong>, pas le niveau absolu de 697 €/MWh en Allemagne et pas l'identité de la dernière centrale appelée. Le résidu de rapprochement reste affiché : aucune fermeture exacte du bilan n'est revendiquée.</p></section>
<section class="card"><h2>Que contenait déjà le domaine initial ?</h2><p id="initial-status" class="muted"></p><div class="scroll"><table id="initial"><thead><tr><th>Heure Paris</th><th>RAM Gronau · MW</th><th>RAM Vigy inverse · MW</th><th>Sensibilité FR→DE Gronau</th><th>Sensibilité FR→DE Vigy inverse</th></tr></thead><tbody></tbody></table></div><p class="foot">Gronau : même transformateur, alias de contingence Zwart/Z et PTDF cohérents, sans correspondance EIC de contingence certifiée. Vigy : VIGY1 sous VIGY2 est la paire inverse de VIGY2 sous VIGY1 active après coupling, pas le même CNEC. Un RAM initial négatif est relatif au scénario RefProg : ce n'est ni une surcharge réalisée ni une capacité d'import négative. Le creux Gronau est à 18 h, le pic de prix à 19 h : le réseau seul ne suffit pas. Noms choisis après examen du résultat, réservés à ce diagnostic et non à une règle de trading.</p></section>
<section class="card"><h2>Méthode et limites</h2><div class="grid"><div>
<p><strong>P50 cohérent avec une distribution.</strong> Dans le régime extrême, l'excès est transporté selon <code>u + CGC × (erreur passée − u passé) / CGC passé</code>. Le régime normal n'est pas redimensionné par le CGC. La variante réseau réapprend cependant les poids conditionnels des deux régimes sur son sous-ensemble CORE qualifié : la comparaison porte sur les chaînes complètes, abstentions incluses, pas sur l'effet isolé d'une variable. Le quantile 0,5 du mélange est calculé ; ce n'est pas « probabilité × amplitude », ni une moyenne renommée P50.</p>
<p><strong>Intervention contrôlée.</strong> Filtres physiques, correction positive plafonnée et gouvernance historique peuvent annuler la proposition. L'interpolation monotone des quantiles puis la calibration chronologique des intervalles ne garantissent pas une couverture future. Le détecteur figé peut toujours manquer un pic ; ce test cible surtout sa sévérité.</p>
<p><strong>Information à 08 h.</strong> Les nouvelles expositions PTDF proviennent du domaine initial. Ce sont des indicateurs directionnels, <em>pas</em> des imports physiquement livrables ni une résolution complète de merit order/dispatch. La production, les commandes habituelles et leurs fichiers n'ont pas été remplacés.</p></div><div>
<p><strong>Extrapolation combustible.</strong> Le cas du 14/09 est à vérifier avec le ratio CGC/max entraînement ci-dessus : un ratio autour de 1,20 signale environ 20 % au-dessus du support historique. La transformation de queue est une hypothèse falsifiable, pas une garantie d'extrapolation correcte.</p>
<p><strong>Historique incomplet.</strong> L'année de backtest ne dispose pas de 365 jours antérieurs complets pour chaque apprentissage. L'échauffement et les replis restent visibles. Manquent encore des vintages de révision, une couverture météo complète, l'offre flexible horaire réelle et une capacité d'importation simultanée validée ; une incohérence de composantes NL reste à résoudre.</p>
<p><strong>Audit ≠ certification PIT.</strong> Un watermark antérieur au cutoff récupéré après l'événement ne vaut pas capture temps réel indépendante. Les vintages des courbes NYX/Storm de référence ne sont pas certifiés. Aucune garantie de non-régression annuelle, aucune promotion automatique : une période future gelée est nécessaire.</p></div></div><details><summary>Provenance, SHA, paramètres et gouvernance</summary><pre id="audit"></pre></details></section>
<noscript>JavaScript est nécessaire pour filtrer ce rapport. Les résultats complets sont dans metrics.json.</noscript></main><script id="physical-data" type="application/json">@@DATA@@</script><script>
'use strict';const D=JSON.parse(document.getElementById('physical-data').textContent),$=id=>document.getElementById(id),labels=Object.fromEntries(D.catalog.map(x=>[x.id,x.label]));
const fmt=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',find=(rows,z,m)=>rows.find(r=>r.zone===z&&r.model_id===m)||{};
function opt(id,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;$(id).append(o)}opt('zone','ALL','ALL · pays agrégés');D.zones.forEach(z=>opt('zone',z,z));D.catalog.filter(m=>m.kind==='candidate').forEach(m=>opt('model',m.id,m.label));$('model').value='nyx_physical_p50';
function td(tr,text,cl=''){const e=document.createElement('td');e.textContent=text;e.className=cl;tr.append(e)}function cls(v,b,lower=true){return Number.isFinite(v)&&Number.isFinite(b)&&Math.abs(v-b)>1e-9?((lower?v<b:v>b)?'good':'bad'):''}function table(id,rows,fn){const b=$(id).querySelector('tbody');b.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');if(r.model_id==='nyx_physical_p50')tr.className='primary';td(tr,labels[r.model_id]||r.zone||r.model_id);fn(tr,r);b.append(tr)})}const color=k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();
function plot(id,tip,axis,series){const host=$(id);host.replaceChildren();$(tip).textContent='';const vals=series.flatMap(s=>s.values.filter(Number.isFinite));if(!axis.length||!vals.length){host.textContent='Aucune donnée.';return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 680 275');svg.setAttribute('role','img');svg.setAttribute('aria-label','Profil horaire en EUR par MWh');host.append(svg);let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.07;lo-=pad;hi+=pad;const x=i=>55+i*605/Math.max(1,axis.length-1),y=v=>20+(hi-v)*215/(hi-lo);function el(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}for(let i=0;i<5;i++){const v=lo+i*(hi-lo)/4;el('line',{x1:55,x2:660,y1:y(v),y2:y(v),stroke:color('line')});el('text',{x:48,y:y(v)+4,'text-anchor':'end'},fmt(v,0))}axis.forEach((v,i)=>{if(i%4===0||i===axis.length-1)el('text',{x:x(i),y:260,'text-anchor':'middle'},v)});series.forEach(s=>{let d='',open=false;s.values.forEach((v,i)=>{if(!Number.isFinite(v)){open=false;return}d+=(open?'L':'M')+x(i)+','+y(v)+' ';open=true});el('path',{d,fill:'none',stroke:s.color,'stroke-width':2})});svg.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect(),i=Math.max(0,Math.min(axis.length-1,Math.round(((e.clientX-r.left)*680/r.width-55)/605*(axis.length-1))));$(tip).textContent=axis[i]+' · '+series.map(s=>s.label+': '+fmt(s.values[i])).join(' · ')})}
function render(){const z=$('zone').value,m=$('model').value,p=D.periods[$('period').value],rows=p.rows.filter(r=>r.zone===z),base=find(rows,z,'nuclear_kalman');$('subtitle').textContent=p.period.start_day+' → '+p.period.end_day+' · livraison '+D.delivery_day+' non évaluée · cutoff 08 h Europe/Paris';const c=p.coverage.filter(r=>z==='ALL'||r.zone===z),sum=k=>c.reduce((a,r)=>a+r[k],0);$('coverage').textContent=fmt(sum('n_common_hours'),0)+' / '+fmt(sum('n_expected_hours'),0)+' heures-pays communes · '+fmt(sum('n_complete_days'),0)+' jours-pays complets · prix observé moyen '+fmt(base.observed_mean_price_eur_mwh)+' €/MWh';table('kpi',rows,(tr,r)=>{['mae_eur_mwh','rmse_eur_mwh'].forEach(k=>td(tr,fmt(r[k]),cls(r[k],base[k])));['win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct'].forEach(k=>td(tr,Number.isFinite(r[k])?fmt(r[k])+' %':'—',cls(r[k],50,false)));td(tr,fmt(r.mae_day_mean_price_eur_mwh),cls(r.mae_day_mean_price_eur_mwh,base.mae_day_mean_price_eur_mwh));td(tr,fmt(r.mean_price_eur_mwh))});const ep=p.economic.audit,eb=find(p.economic.rows,z,'nuclear_kalman');$('policy').textContent='Portefeuille alternatif '+fmt(ep.portfolio_capacity_mw,0)+' MW ; '+Object.entries(ep.zone_capacity_mw).map(([a,b])=>a+' '+fmt(b,0)+' MW').join(', ')+' fixes, sans redistribution. Seuil effectif |edge| > '+fmt(ep.signal_hurdle_eur_mwh)+' €/MWh ; coûts '+fmt(ep.net_cost_eur_mwh)+' €/MWh.';table('eva',p.economic.rows.filter(r=>r.zone===z),(tr,r)=>{td(tr,fmt(r.pnl_net_eur,0));td(tr,fmt(r.gain_vs_storm_eur,0),cls(r.gain_vs_storm_eur,0,false));const d=Number.isFinite(r.pnl_net_eur)&&Number.isFinite(eb.pnl_net_eur)?r.pnl_net_eur-eb.pnl_net_eur:null;td(tr,fmt(d,0),cls(d,0,false));td(tr,fmt(r.gain_vs_storm_per_potential_mwh));td(tr,fmt(r.n_country_hours,0))});table('changes',p.interventions.filter(r=>r.zone===z&&r.model_id!=='__storm__'),(tr,r)=>{['changed_hours','better','worse'].forEach(k=>td(tr,fmt(r[k],0)));['tail_mae','tail_rmse','normal_mae'].forEach(k=>td(tr,fmt(r[k])))});
const lineSpecs=[{id:'actual',label:'Observé',color:color('obs')},{id:'storm',label:'Storm',color:color('storm')},{id:'nuclear_kalman',label:'NYX',color:color('nyx')},{id:m,label:labels[m],color:color('new')}];$('legend').replaceChildren();lineSpecs.forEach(s=>{const e=document.createElement('span');e.textContent=s.label;e.style.color=s.color;$('legend').append(e)});const hours=Array.from({length:24},(_,i)=>i);plot('hour-plot','hour-tip',hours.map(h=>h+' h'),lineSpecs.slice(1).map(s=>({...s,values:hours.map(h=>p.hourly.find(r=>r.zone===z&&r.model_id===(s.id==='storm'?'__storm__':s.id)&&r.hour===h)?.mae??null)})));
const selected=D.case.filter(r=>z==='ALL'||r.zone===z);table('case',selected.filter(r=>r.hour===19),(tr,r)=>{['actual','storm','nuclear_kalman','coherent_forest_direct','fuel_transport_direct','fuel_transport_governed','network_fuel_direct','nyx_physical_p50'].forEach(k=>td(tr,fmt(r[k])));td(tr,fmt(r.fuel_transport_fuel_ratio_to_core_max,3))});$('case-charts').replaceChildren();D.zones.filter(a=>z==='ALL'||a===z).forEach(zone=>{const box=document.createElement('div');box.className='chart';const title=document.createElement('h3');title.textContent=zone;box.append(title);const host=document.createElement('div');host.id='case-'+zone;box.append(host);const tip=document.createElement('div');tip.id='tip-'+zone;tip.className='tooltip';box.append(tip);$('case-charts').append(box);const data=selected.filter(r=>r.zone===zone).sort((a,b)=>a.hour-b.hour);plot(host.id,tip.id,data.map(r=>r.hour+' h'),lineSpecs.map(s=>({...s,values:data.map(r=>r[s.id])}))) });
$('forensic').replaceChildren();if(D.forensic){const data=D.forensic.data;const intro=document.createElement('p');intro.textContent='Données officielles JAO après couplage : principaux CNEC actifs Ensdorf–Vigy (contingence parallèle) et PST Gronau (contingence Doetinchem–Hengelo). Somme des prix duaux × différences PTDF, moyennée sur quatre quarts d’heure :';$('forensic').append(intro);const t=document.createElement('table');const b=document.createElement('tbody');t.append(b);(data.spread_comparison||[]).filter(r=>z==='ALL'||r.zone===z).forEach(r=>{const tr=document.createElement('tr');td(tr,r.zone+' − FR');td(tr,'Observé '+fmt(r.observed_spread_eur_mwh,3));td(tr,'Somme duale '+fmt(r.active_cnec_dual_sum_eur_mwh,3));td(tr,'Résidu '+fmt(r.unexplained_difference_eur_mwh,3)+' €/MWh');b.append(tr)});$('forensic').append(t);const a=document.createElement('a');a.href='https://publicationtool.jao.eu/core/';a.textContent='JAO · publication officielle';a.target='_blank';a.rel='noopener';$('forensic').append(a)}else $('forensic').textContent='Aucun dossier réseau post-couplage scellé joint à ce rapport.';$('audit').textContent=JSON.stringify({audit:D.audit,source_sha256:D.source_sha256,forensic:D.forensic?{path:D.forensic.path,sha256:D.forensic.sha256}:null,forecast_pit_certified:false,independent_validation:false,production_modified:false},null,2)}
if(D.initial_signal){const a=D.initial_signal;$('initial-status').textContent='Watermark initial '+a.initial_last_modified_utc+' ; récupération après événement '+a.retrieved_at_utc+'. Signal historique de réseau, pas certification temps réel.';const body=$('initial').querySelector('tbody');const hours=[...new Set(a.rows.map(r=>r.hour_paris))].sort((x,y)=>x-y);hours.forEach(h=>{const g=a.rows.find(r=>r.hour_paris===h&&r.cne_name.includes('Gronau'))||{},v=a.rows.find(r=>r.hour_paris===h&&r.cne_name.includes('Vigy'))||{},tr=document.createElement('tr');td(tr,h+' h');td(tr,fmt(g.ram_mw,0));td(tr,fmt(v.ram_mw,0));td(tr,fmt(g.transfer_fr_to_de,5));td(tr,fmt(v.transfer_fr_to_de,5));body.append(tr)})}else $('initial-status').textContent='Aucun rapprochement initial scellé disponible.';
['zone','period','model'].forEach(id=>$(id).addEventListener('change',render));$('theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'Mode jour':'Mode nuit';render()});render();
</script></body></html>'''
