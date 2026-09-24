"""Isolated, paired RMSE research report; no production or policy mutation."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from kpi_report.economic import compute_economic_kpis
from kpi_report.metrics import compute_kpis


KEYS = ["zone", "timestamp_utc"]
TIMEZONE = "Europe/Paris"
CANDIDATES = ("residual_mse_direct", "nyx_rmse", "mixture_mean_direct", "mixture_mean_governed")
CATALOG = [
    {"id": "nuclear_kalman", "label": "NYX opérationnel · nuclear_kalman", "kind": "production"},
    {"id": "nyx_rmse", "label": "NYX RMSE · MSE gouverné", "kind": "candidate"},
    {"id": "residual_mse_direct", "label": "Résiduel MSE · direct", "kind": "candidate"},
    {"id": "mixture_mean_governed", "label": "Moyenne du mélange · gouvernée", "kind": "candidate"},
    {"id": "mixture_mean_direct", "label": "Moyenne du mélange · directe", "kind": "candidate"},
    {"id": "coherent_forest_direct", "label": "Forêt · P50 antérieur", "kind": "previous"},
    {"id": "coherent_empirical_direct", "label": "Empirique · P50 antérieur", "kind": "previous"},
    {"id": "__storm__", "label": "Storm", "kind": "benchmark"},
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (Path, date, datetime, pd.Timestamp)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "storm" not in out and "benchmark_forecast" in out:
        out["storm"] = out.benchmark_forecast
    required = {*KEYS, "forecast", "actual", "storm", "forecast_origin_utc", "sample"}
    if required.difference(out):
        raise ValueError(f"Report panel missing columns: {sorted(required.difference(out))}")
    for name in ("timestamp_utc", "forecast_origin_utc", "label_available_at_utc"):
        if name not in out:
            continue
        if isinstance(out[name].dtype, pd.DatetimeTZDtype):
            out[name] = out[name].dt.tz_convert("UTC")
        else:
            values = [pd.Timestamp(value) for value in out[name]]
            if any(pd.notna(value) and value.tzinfo is None for value in values):
                raise ValueError(f"Report {name} requires aware timestamps.")
            out[name] = pd.to_datetime(values, utc=True)
    if out.timestamp_utc.isna().any() or out.forecast_origin_utc.isna().any():
        raise ValueError("Report timestamp/origin cannot be missing.")
    if out.duplicated(KEYS).any():
        raise ValueError("Report requires unique zone / physical-hour identities.")
    for name in ("forecast", "actual", "storm", *CANDIDATES):
        if name in out:
            out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
            if np.isinf(out[name]).any():
                raise ValueError(f"Report {name} contains infinity.")
    return out.set_index(KEYS).sort_index()


def _same_inputs(left: pd.DataFrame, right: pd.DataFrame, context: str) -> None:
    if not left.index.equals(right.index):
        raise ValueError(f"{context}: different physical-hour support.")
    for name in ("forecast", "actual", "storm", "q10", "q90", "forecast_origin_utc", "sample", "label_available_at_utc"):
        if name not in left or name not in right:
            if name in {"q10", "q90", "label_available_at_utc"}:
                continue
            raise ValueError(f"{context}: missing source identity {name}.")
        if name in ("forecast", "actual", "storm", "q10", "q90"):
            equal = np.allclose(left[name].to_numpy(float), right[name].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
        else:
            equal = left[name].equals(right[name])
        if not equal:
            raise ValueError(f"{context}: changed shared {name}.")


def assemble_panel(predictions: pd.DataFrame, source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Verify the frozen comparators and build a long, shared-reference panel."""
    source_dir = Path(source_dir).resolve()
    seals = {}

    def capture(path: Path, expected: str | None = None) -> str:
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Report source checksum mismatch: {path}")
        seals[str(path)] = digest
        return digest

    manifest_path = source_dir / "manifest.json"
    manifest_sha = capture(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("source_audit.json", "panel.parquet"):
        expected = manifest.get("input_files", {}).get(name)
        if not expected:
            raise ValueError(f"Source manifest missing sealed {name}.")
        capture(source_dir / name, expected)
    source_audit = json.loads((source_dir / "source_audit.json").read_text(encoding="utf-8"))
    base = _typed(pd.read_parquet(source_dir / "panel.parquet"))
    wide = _typed(predictions)
    _same_inputs(base, wide, "NYX RMSE")
    if set(CANDIDATES).difference(wide):
        raise ValueError(f"Missing RMSE candidate points: {sorted(set(CANDIDATES).difference(wide))}")
    wide["nuclear_kalman"] = wide.forecast
    for kind in ("forest", "empirical"):
        result_path = source_dir / kind / "results_manifest.json"
        capture(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("suite_manifest_sha256") != manifest_sha:
            raise ValueError(f"Unsealed/incomplete {kind} comparator.")
        expected = result.get("result_files", {}).get("predictions.parquet")
        if not expected:
            raise ValueError(f"Missing {kind} predictions checksum.")
        capture(source_dir / kind / "predictions.parquet", expected)
        previous = _typed(pd.read_parquet(source_dir / kind / "predictions.parquet"))
        _same_inputs(base, previous, kind)
        point = pd.to_numeric(previous.candidate_forecast, errors="raise")
        if np.isinf(point).any():
            raise ValueError(f"Infinite {kind} point forecast.")
        wide[f"coherent_{kind}_direct"] = point
    records = []
    columns = ["actual", "storm", "forecast_origin_utc", "sample"]
    columns += [name for name in ("forecast_eligible", "benchmark_eligible") if name in wide]
    for item in CATALOG:
        if item["id"] == "__storm__":
            continue
        current = wide[columns].copy()
        current["model_id"] = item["id"]
        current["forecast"] = wide[item["id"]]
        records.append(current.reset_index())
    return pd.concat(records, ignore_index=True), wide.reset_index(), source_audit, seals


def _extra_metrics(wide: pd.DataFrame, daily: list[dict], *, end_day: str, days: int, zones: list[str]) -> dict:
    """Descriptive diagnostics on the exact all-model/Storm paired hour support."""
    lower = pd.Timestamp(date.fromisoformat(end_day)-timedelta(days=days-1), tz=TIMEZONE).tz_convert("UTC")
    upper = pd.Timestamp(date.fromisoformat(end_day)+timedelta(days=1), tz=TIMEZONE).tz_convert("UTC")
    models = [item["id"] for item in CATALOG if item["id"] != "__storm__"]
    selected = wide.loc[wide.timestamp_utc.ge(lower) & wide.timestamp_utc.lt(upper)].copy()
    selected = selected.dropna(subset=[*models, "actual", "storm"])
    selected["hour"] = selected.timestamp_utc.dt.tz_convert(TIMEZONE).dt.hour
    hourly, interventions, tails, ready_metrics = [], [], [], []
    readiness = next((name for name in ("rmse_expert_ready", "expert_ready", "mse_ready") if name in selected), None)
    for zone in [*zones, "ALL"]:
        block = selected if zone == "ALL" else selected.loc[selected.zone.eq(zone)]
        base_error = (block.nuclear_kalman-block.actual).abs()
        for model in [*models, "__storm__"]:
            point = block.storm if model == "__storm__" else block[model]
            errors = point-block.actual
            temp = pd.DataFrame({"hour": block.hour, "error": errors, "abs_error": errors.abs(), "square": errors**2})
            for hour, group in temp.groupby("hour"):
                hourly.append({"zone": zone, "model_id": model, "hour": int(hour), "n_hours": len(group),
                               "mae_eur_mwh": float(group.abs_error.mean()), "rmse_eur_mwh": float(np.sqrt(group.square.mean()))})
            changed = (point-block.nuclear_kalman).abs().gt(1e-9)
            gain = base_error-errors.abs()
            interventions.append({"zone": zone, "model_id": model, "n_hours": len(block), "changed_hours": int(changed.sum()),
                "improved_hours": int((changed & gain.gt(1e-9)).sum()), "worsened_hours": int((changed & gain.lt(-1e-9)).sum()),
                "mean_abs_shift_eur_mwh": float((point-block.nuclear_kalman).abs().mean()) if len(block) else None,
                "mae_on_interventions_eur_mwh": float(errors.loc[changed].abs().mean()) if changed.any() else None,
                "nyx_mae_on_interventions_eur_mwh": float(base_error.loc[changed].mean()) if changed.any() else None})
            for regime, mask in (("price_ge_200", block.actual.ge(200)), ("price_lt_200", block.actual.lt(200))):
                part = errors.loc[mask]
                tails.append({"zone": zone, "model_id": model, "regime": regime, "n_hours": int(mask.sum()),
                              "mae_eur_mwh": float(part.abs().mean()) if len(part) else None,
                              "rmse_eur_mwh": float(np.sqrt((part**2).mean())) if len(part) else None})
            if readiness is not None:
                eligible = block[readiness].eq(True)
                part = errors.loc[eligible]
                ready_metrics.append({"zone": zone, "model_id": model, "n_hours": int(eligible.sum()),
                                      "mae_eur_mwh": float(part.abs().mean()) if len(part) else None,
                                      "rmse_eur_mwh": float(np.sqrt((part**2).mean())) if len(part) else None})
    # A compact chart series; ALL means the equally weighted available complete
    # country-days at each date, NOT the pooled-hour KPI (explicitly labelled).
    daily_rows = []
    for row in daily:
        daily_rows.append({name: row[name] for name in ("zone", "delivery_day", "model_id", "mean_price_eur_mwh", "observed_mean_price_eur_mwh")})
    return {"hourly": hourly, "interventions": interventions, "tails": tails,
            "expert_ready": ready_metrics, "readiness_column": readiness, "daily": daily_rows}


def build_report(predictions: pd.DataFrame, source_dir: Path, output_dir: Path, *, root: Path, audit: dict) -> dict:
    """Build one exclusive, self-contained research HTML and machine-readable KPI."""
    output_dir = Path(output_dir).resolve()
    root = Path(root).resolve()
    output_dir.relative_to(root / "runs" / "experiments" / "nyx_rmse_v1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Report directory is not empty: {output_dir}")
    long, wide, source_audit, seals = assemble_panel(predictions, source_dir)
    source = source_audit["source_config"]
    delivery = date.fromisoformat(source["delivery_day"])
    end_day = source.get("end_day") or (delivery-timedelta(days=1)).isoformat()
    if date.fromisoformat(end_day) != delivery-timedelta(days=1) or int(source["evaluation_days"]) != 365:
        raise ValueError("RMSE report requires the sealed delivery minus one / 365-day evaluation contract.")
    if source.get("timezone") != TIMEZONE or source.get("cutoff_time") != "08:00":
        raise ValueError("RMSE report requires the sealed 08:00 Paris forecast contract.")
    zones = sorted(wide.zone.unique().tolist())
    if set(zones) != set(source["zones"]):
        raise ValueError("RMSE report countries differ from the source experiment.")
    # Never score live rows, even when their auction labels have since arrived.
    civil = wide.timestamp_utc.dt.tz_convert(TIMEZONE).dt.date
    eval_mask = wide["sample"].eq("evaluation")
    if ((civil.le(date.fromisoformat(end_day))) & ~eval_mask).any():
        raise ValueError("Non-evaluation samples overlap the scored historical period.")
    long = long.loc[long["sample"].eq("evaluation")].copy()
    scored_wide = wide.loc[eval_mask].copy()
    config_path = root / "config" / "economic_value.yaml"
    seals[str(config_path)] = _sha(config_path)
    periods = {}
    for days in (365, 90, 30, 7):
        result = compute_kpis(long, end_day=end_day, days=days, zones=zones)
        extra = _extra_metrics(scored_wide, result.pop("daily_rows"), end_day=end_day, days=days, zones=zones)
        economic = compute_economic_kpis(long, end_day=end_day, days=days, zones=zones, config_path=config_path)
        result.update(extra)
        result["economic"] = economic
        periods[str(days)] = result
    payload = _json_safe({"schema_version": 1, "catalog": CATALOG, "zones": zones,
        "delivery_day": delivery.isoformat(), "end_day": end_day,
        "generated_at": datetime.now(timezone.utc).isoformat(), "periods": periods,
        "audit": audit, "source_directory": str(Path(source_dir).resolve()), "source_sha256": seals,
        "forecast_pit_certified": False, "benchmark_pit_certified": False,
        "forecast_provenance": "published_report_replay_and_latest_dashboard_not_certified_execution_vintages",
        "diagnostic_only": True, "production_modified": False, "activation_performed": False,
        "independent_validation": False, "mean_is_not_p50": True})
    for path, expected in seals.items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"Report source changed during evaluation: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "nyx_rmse_report.html"
    with metrics_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    render_report(payload, report_path)
    for path, expected in seals.items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"Report source changed during publication: {path}")
    annual = periods["365"]
    summary = {"annual_rows": [row for row in annual["rows"] if row["zone"] == "ALL"],
               "annual_economic_rows": [row for row in annual["economic"]["rows"] if row["zone"] == "ALL"],
               "coverage": annual["coverage"], "source_end_day": end_day}
    return {"status": "completed", "report_path": str(report_path), "metrics_path": str(metrics_path),
            "report_sha256": _sha(report_path), "metrics_sha256": _sha(metrics_path),
            "summary": _json_safe(summary), "diagnostic_only": True, "production_modified": False}


def render_report(payload: dict, output_path: Path) -> Path:
    """Render safe inline JSON and native SVG; no external scripts or services."""
    safe = _json_safe(payload)
    encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    encoded = encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    replacements = {"DATA": encoded, "SOURCE": html.escape(str(payload.get("source_directory", "")))}
    text = re.sub(r"@@(DATA|SOURCE)@@", lambda match: replacements[match[1]], TEMPLATE)
    with Path(output_path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return Path(output_path)


TEMPLATE = r'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYX RMSE · comparaison expérimentale</title>
<style>
:root{--bg:#f2f5fa;--paper:#fff;--ink:#16243a;--muted:#526079;--line:#dbe3ee;--accent:#255cc8;--good:#126c4b;--goodbg:#e4f5ec;--bad:#ab3b3b;--badbg:#fbeaea;--warn:#94621c;--nyx:#a76808;--storm:#0878a9;--point:#7443cc;--obs:#37485d;--grid:#dce3ec}
:root[data-theme=dark]{--bg:#111924;--paper:#1b2635;--ink:#edf2fb;--muted:#b1bfd3;--line:#35455b;--accent:#91b7ff;--good:#83dfb2;--goodbg:#1e3d35;--bad:#ffaaaa;--badbg:#482d37;--warn:#efd094;--nyx:#ffc268;--storm:#67cff5;--point:#c4a0ff;--obs:#e2ebf9;--grid:#36465d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1700px;margin:auto;padding:28px 32px 60px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}h1{font-size:32px;letter-spacing:-1px;margin:2px 0 5px}h2{font-size:19px;margin:0 0 5px}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:750}.muted,small{color:var(--muted)}.tag{display:inline-block;border:1px solid var(--line);padding:4px 9px;border-radius:6px;font-size:11px;font-weight:650}.notice{margin:18px 0;padding:14px 17px;border-left:4px solid var(--warn);background:var(--paper);border-radius:6px}.filters{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:20px 0}label{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:12px}select,button{border:1px solid var(--line);border-radius:7px;background:var(--paper);color:var(--ink);padding:9px 12px;font:inherit;cursor:pointer}button:hover{border-color:var(--accent)}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{padding:17px;background:var(--paper);border:1px solid var(--line);border-radius:10px}.stat .value{font-size:23px;font-weight:700;display:block;line-height:1.6}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap;margin-top:15px}th{text-align:right;font-weight:650;color:var(--muted);padding:11px 9px;border-bottom:2px solid var(--line)}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--paper);min-width:235px}td{text-align:right;padding:11px 9px;border-bottom:1px solid var(--line)}td.good{color:var(--good);background:var(--goodbg)}td.bad{color:var(--bad);background:var(--badbg)}tr.primary td:first-child{font-weight:750;color:var(--accent)}tr.base td:first-child{font-weight:700}tr:last-child td{border-bottom:0}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}.twocol .card{min-width:0;margin:0}.plot{width:100%;min-height:285px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:8px 0}.dot{width:18px;height:3px;display:inline-block;margin-right:6px;vertical-align:middle}.notes{display:grid;grid-template-columns:1fr 1fr;gap:16px}.notes p{margin:8px 0}.foot{font-size:12px;margin-top:10px}details summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;word-break:break-word;max-height:440px;overflow:auto;font-size:11px}svg text{fill:var(--muted);font:11px system-ui}svg .grid{stroke:var(--grid);stroke-width:1}svg .curve{fill:none;stroke-width:2}svg .hover{stroke:var(--muted);stroke-dasharray:4 3}.tooltip{min-height:25px;font-size:12px;color:var(--muted)}noscript{display:block;padding:20px;color:var(--bad)}@media(max-width:950px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}.twocol,.notes{grid-template-columns:1fr}header{flex-direction:column}h1{font-size:27px}.card{padding:14px}}
</style></head><body><main>
<header><div><div class="eyebrow">NYX · laboratoire indépendant</div><h1>RMSE : corriger les grandes erreurs</h1><div class="muted" id="subtitle"></div></div><div><button id="theme" type="button">Mode nuit</button> <span class="tag">PRODUCTION INCHANGÉE</span></div></header>
<div class="notice"><strong>Diagnostic rétrospectif, pas une promotion.</strong> Les prévisions ci-dessous sont des estimations ponctuelles : une moyenne conditionnelle n'est pas un P50 et peut sortir de l'intervalle P10–P90 de NYX. Aucun nouvel intervalle calibré n'est revendiqué. L'année a déjà été examinée ; une amélioration ici doit encore être validée sur de nouvelles journées.</div>
<div class="filters"><label>Pays<select id="zone"></select></label><label>Fenêtre d'évaluation<select id="period"><option value="365">365 derniers jours</option><option value="90">90 derniers jours</option><option value="30">30 derniers jours</option><option value="7">7 derniers jours</option></select></label><label>Candidat détaillé<select id="model"></select></label><label>Classement<select id="sort"><option value="rmse_eur_mwh">RMSE croissante</option><option value="mae_eur_mwh">MAE croissante</option><option value="win_rate_hour_pct">Win rate horaire décroissant</option></select></label></div>
<div class="cards"><div class="stat"><span class="muted">RMSE du candidat</span><span class="value" id="rmse"></span><small id="rmse-delta"></small></div><div class="stat"><span class="muted">MAE du candidat</span><span class="value" id="mae"></span><small id="mae-delta"></small></div><div class="stat"><span class="muted">Gain EVA vs NYX</span><span class="value" id="eva"></span><small>Simulation à politique fixe ; pas un P&amp;L exécutable</small></div><div class="stat"><span class="muted">Interventions du candidat</span><span class="value" id="changed"></span><small id="changed-detail"></small></div></div>
<section class="card"><h2>KPI · prix & précision</h2><div class="muted" id="coverage"></div><div class="scroll"><table id="kpi"><thead><tr><th>Modèle</th><th>MAE<br>€/MWh</th><th>RMSE<br>€/MWh</th><th>Gain RMSE<br>vs NYX</th><th>Win rate<br>heures vs Storm</th><th>Win rate<br>MAE jour vs Storm</th><th>Win rate<br>prix moyen jour vs Storm</th><th>MAE prix moyen<br>journalier €/MWh</th><th>Prix moyen<br>€/MWh</th></tr></thead><tbody></tbody></table></div><p class="foot">Vert / rouge : meilleure / moins bonne erreur que NYX ; pour les win rates, supérieur / inférieur à 50 %. Les ex æquo restent au dénominateur sans compter comme victoire. Les jours incomplets sont exclus des KPI journaliers (23/24/25 heures physiques selon le changement d'heure). Le prix moyen est pondéré par les heures communes.</p></section>
<section class="card"><h2>EVA · même décision, mêmes hypothèses</h2><p class="muted">Prix de référence : prix day-ahead observé de la veille à la même heure civile. Ce proxy n'est pas un prix auquel on aurait pu traiter à 08 h. <span id="portfolio"></span> Aucune redistribution lors du filtrage. Signal et coûts inchangés, aucun ajustement à ces résultats.</p><div class="scroll"><table id="economic"><thead><tr><th>Modèle</th><th>P&amp;L net simulé €</th><th>Gain vs Storm €</th><th>Gain vs NYX €</th><th>Gain vs Storm<br>€/MWh potentiel</th><th>Heures-pays communes</th></tr></thead><tbody></tbody></table></div><p class="foot" id="economic-support"></p></section>
<div class="twocol"><section class="card"><h2>Performance par heure</h2><label style="display:inline-flex">Erreur<select id="hour-metric"><option value="mae_eur_mwh">MAE</option><option value="rmse_eur_mwh">RMSE</option></select></label><div class="legend" id="hour-legend"></div><div id="hour-plot" class="plot"></div><div class="tooltip" id="hour-tooltip"></div><p class="foot">Heure civile Europe/Paris, sur les mêmes heures appariées pour chaque modèle. Les deux heures d'automne restent deux observations.</p></section><section class="card"><h2>Prix moyens journaliers</h2><div class="legend" id="daily-legend"></div><div id="daily-plot" class="plot"></div><div class="tooltip" id="daily-tooltip"></div><p class="foot">Jours communs complets. Vue ALL : moyenne des pays disponibles complets pour chaque date ; ce tracé n'est pas le KPI agrégé pondéré par les heures.</p></section></div>
<section class="card"><h2>Interventions & heures chères</h2><div class="muted">Une correction faussement déclenchée compte aussi. Le seuil observé ≥ 200 €/MWh est un découpage descriptif a posteriori, pas un signal disponible au forecast.</div><div class="scroll"><table id="interventions"><thead><tr><th>Modèle</th><th>Heures corrigées</th><th>Améliorées</th><th>Dégradées</th><th>MAE corrigées<br>modèle / NYX</th><th>MAE prix ≥ 200<br>€/MWh</th><th>RMSE prix ≥ 200<br>€/MWh</th><th>MAE prix &lt; 200<br>€/MWh</th></tr></thead><tbody></tbody></table></div></section>
<section class="card"><h2>Lecture méthodologique & limites</h2><div class="notes"><div><p><strong>Deux estimations, pas un changement en production.</strong> Le correcteur MSE apprend l'erreur signée de NYX. La moyenne du mélange normal / extrême agrège les espérances conditionnelles, sans inversion coûteuse de la CDF. Les versions gouvernées peuvent rester à poids nul ; les versions directes sont des diagnostics non gouvernés.</p><p><strong>La comparaison à l'ancien P50 n'isole pas uniquement une loss.</strong> Une correction signée et la suppression des anciens filtres positif, physique et de risque modifient aussi la règle d'intervention. La moyenne estimée d'une distribution n'est pas la moyenne de ses quantiles.</p><p><strong>Fenêtre et maturité.</strong> L'historique source couvre l'année évaluée, pas 365 jours supplémentaires avant elle. Une fenêtre plafonnée à 365 jours ne garantit donc pas 365 jours d'entraînement pour chaque fold. Les replis sur NYX sont inclus dans les résultats annuels ; les métriques sur les seules heures où l'expert est prêt sont affichées séparément ci-dessous si disponibles.</p></div><div><p><strong>Gouvernance chronologique, sans garantie future.</strong> L'entraînement et la sélection des poids ne doivent utiliser que les labels disponibles avant le cutoff. Le score MSE pénalisé par une erreur-type journalière est un garde-fou heuristique, pas une preuve statistique à 95 %. Les garde-fous MAE sont historiques et ne garantissent pas une non-régression future ; les comparaisons multiples ne constituent pas une validation indépendante.</p><p><strong>Provenance des forecasts.</strong> Les courbes de base NYX et Storm proviennent de replays de rapports publiés et du dernier cache dashboard. Leurs vintages d'exécution ne sont pas certifiés PIT. Un correcteur causal à 08 h ne certifie pas rétrospectivement les vintages de ses forecasts de base.</p><p><strong>Interprétation économique.</strong> Une amélioration RMSE ne se traduit en EVA que si elle modifie une position sous la règle fixe. Une hausse du même signal BUY ne change pas son volume. Les jours sans référence D−1 non ambiguë sont exclus symétriquement ; aucune annualisation ni nouvelle optimisation de confiance.</p><p><strong>Prochaine preuve.</strong> Ne pas retenir une variante comme « validée » après exploration de cette année. Une période future gelée est nécessaire avant toute décision opérationnelle. Les forecasts actuels, leur configuration et leurs commandes restent inchangés.</p></div></div><div class="scroll"><table id="ready"><thead><tr><th>Modèle · heures expert prêt</th><th>Heures communes</th><th>MAE €/MWh</th><th>RMSE €/MWh</th></tr></thead><tbody></tbody></table></div><p class="foot" id="ready-note"></p><details><summary>Audit, paramètres et provenance</summary><p class="foot">Source figée : @@SOURCE@@</p><pre id="audit"></pre></details></section>
<noscript>Activez JavaScript pour filtrer ce rapport autonome. Les résultats complets sont également disponibles dans metrics.json.</noscript>
</main><script id="rmse-data" type="application/json">@@DATA@@</script><script>
'use strict';
const D=JSON.parse(document.getElementById('rmse-data').textContent), $=id=>document.getElementById(id);
const labels=Object.fromEntries(D.catalog.map(x=>[x.id,x.label]));
const fmt=(x,n=2)=>x===null||x===undefined||!Number.isFinite(x)?'—':x.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n});
const sign=x=>(x>0?'+':'')+fmt(x,Math.abs(x)>0&&Math.abs(x)<.01?4:2), row=(rows,z,m)=>rows.find(r=>r.zone===z&&r.model_id===m)||{};
function option(parent,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;parent.appendChild(o)}
option($('zone'),'ALL','ALL · pays agrégés');D.zones.forEach(z=>option($('zone'),z,z));D.catalog.filter(m=>m.kind==='candidate').forEach(m=>option($('model'),m.id,m.label));$('model').value='nyx_rmse';
const portfolio=D.periods['365'].economic.audit;$('portfolio').textContent='Portefeuille alternatif de '+fmt(portfolio.portfolio_capacity_mw,0)+' MW au total ; allocations fixes : '+Object.entries(portfolio.zone_capacity_mw).map(([z,v])=>z+' '+fmt(v,0)+' MW').join(', ')+'.';
function cls(value,reference,smaller=true){if(!Number.isFinite(value)||!Number.isFinite(reference)||Math.abs(value-reference)<=1e-9)return '';return (smaller?value<reference:value>reference)?'good':'bad'}
function cell(tr,value,className=''){const td=document.createElement('td');td.textContent=value;td.className=className;tr.appendChild(td)}
function table(id,rows,build){const body=$(id).querySelector('tbody');body.replaceChildren();rows.forEach(r=>{const tr=document.createElement('tr');if(r.model_id==='nyx_rmse')tr.className='primary';if(r.model_id==='nuclear_kalman')tr.className='base';cell(tr,labels[r.model_id]||r.model_id);build(tr,r);body.appendChild(tr)})}
function order(rows){const key=$('sort').value,mult=key.startsWith('win')?-1:1;return [...rows].sort((a,b)=>{const av=a[key],bv=b[key];if(!Number.isFinite(av))return Number.isFinite(bv)?1:0;if(!Number.isFinite(bv))return -1;return mult*(av-bv)})}
function color(name){return getComputedStyle(document.documentElement).getPropertyValue('--'+name).trim()}
function legend(id,series){$(id).replaceChildren();series.forEach(s=>{const el=document.createElement('span'),dot=document.createElement('span');dot.className='dot';dot.style.background=s.color;el.appendChild(dot);el.appendChild(document.createTextNode(s.label));$(id).appendChild(el)})}
function plot(id,tooltip,axis,series){const host=$(id);host.replaceChildren();$(tooltip).textContent='';const vals=series.flatMap(s=>s.values.filter(Number.isFinite));if(!axis.length||!vals.length){host.textContent='Aucune donnée commune.';return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 720 290');svg.setAttribute('role','img');svg.setAttribute('aria-label',id==='hour-plot'?'Erreur par heure':'Prix moyens journaliers');host.appendChild(svg);const W=720,H=290,L=58,R=18,T=15,B=38;let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;const x=i=>L+i*(W-L-R)/Math.max(1,axis.length-1),y=v=>T+(hi-v)*(H-T-B)/(hi-lo);function el(tag,attrs,text){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.appendChild(n);return n}for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);el('line',{x1:L,x2:W-R,y1:yy,y2:yy,class:'grid'});el('text',{x:L-8,y:yy+4,'text-anchor':'end'},fmt(v,0))}const step=Math.max(1,Math.ceil(axis.length/6));axis.forEach((v,i)=>{if(i%step===0||i===axis.length-1)el('text',{x:x(i),y:H-12,'text-anchor':'middle'},String(v).length>5?String(v).slice(5):v)});series.forEach(s=>{let d='',open=false;s.values.forEach((v,i)=>{if(!Number.isFinite(v)){open=false;return}d+=(open?' L':' M')+x(i).toFixed(2)+','+y(v).toFixed(2);open=true});el('path',{d,class:'curve',stroke:s.color})});const hover=el('line',{x1:L,x2:L,y1:T,y2:H-B,class:'hover',visibility:'hidden'});svg.addEventListener('pointermove',event=>{const rect=svg.getBoundingClientRect(),px=(event.clientX-rect.left)*W/rect.width;const i=Math.max(0,Math.min(axis.length-1,Math.round((px-L)/(W-L-R)*Math.max(1,axis.length-1))));hover.setAttribute('x1',x(i));hover.setAttribute('x2',x(i));hover.setAttribute('visibility','visible');$(tooltip).textContent=axis[i]+' · '+series.map(s=>s.label+': '+fmt(s.values[i])+' €/MWh').join(' · ')});svg.addEventListener('pointerleave',()=>hover.setAttribute('visibility','hidden'))}
function render(){const p=D.periods[$('period').value],z=$('zone').value,m=$('model').value;const rows=order(p.rows.filter(r=>r.zone===z)),b=row(p.rows,z,'nuclear_kalman'),v=row(p.rows,z,m),eb=row(p.economic.rows,z,'nuclear_kalman'),ev=row(p.economic.rows,z,m),iv=row(p.interventions,z,m);$('subtitle').textContent=p.period.start_day+' → '+p.period.end_day+' · livraison '+D.delivery_day+' exclue de l’évaluation · cutoff 08 h Europe/Paris';$('rmse').textContent=fmt(v.rmse_eur_mwh)+' €/MWh';$('mae').textContent=fmt(v.mae_eur_mwh)+' €/MWh';$('rmse-delta').textContent=Number.isFinite(v.rmse_eur_mwh)&&Number.isFinite(b.rmse_eur_mwh)?'Gain vs NYX : '+sign(b.rmse_eur_mwh-v.rmse_eur_mwh)+' €/MWh':'Support indisponible';$('mae-delta').textContent=Number.isFinite(v.mae_eur_mwh)&&Number.isFinite(b.mae_eur_mwh)?'Gain vs NYX : '+sign(b.mae_eur_mwh-v.mae_eur_mwh)+' €/MWh':'Support indisponible';$('eva').textContent=Number.isFinite(ev.pnl_net_eur)&&Number.isFinite(eb.pnl_net_eur)?sign(ev.pnl_net_eur-eb.pnl_net_eur)+' €':'—';$('changed').textContent=fmt(iv.changed_hours,0)+' / '+fmt(iv.n_hours,0);$('changed-detail').textContent=fmt(iv.improved_hours,0)+' améliorées · '+fmt(iv.worsened_hours,0)+' dégradées';const covers=p.coverage.filter(r=>z==='ALL'||r.zone===z),sum=k=>covers.reduce((a,r)=>a+r[k],0);$('coverage').textContent=fmt(sum('n_common_hours'),0)+' heures-pays communes / '+fmt(sum('n_expected_hours'),0)+' attendues · '+fmt(sum('n_complete_days'),0)+' jours-pays complets · prix observé moyen : '+fmt(v.observed_mean_price_eur_mwh)+' €/MWh';table('kpi',rows,(tr,r)=>{cell(tr,fmt(r.mae_eur_mwh),cls(r.mae_eur_mwh,b.mae_eur_mwh));cell(tr,fmt(r.rmse_eur_mwh),cls(r.rmse_eur_mwh,b.rmse_eur_mwh));cell(tr,Number.isFinite(r.rmse_eur_mwh)&&Number.isFinite(b.rmse_eur_mwh)?sign(b.rmse_eur_mwh-r.rmse_eur_mwh):'—',cls(r.rmse_eur_mwh,b.rmse_eur_mwh));['win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct'].forEach(k=>cell(tr,Number.isFinite(r[k])?fmt(r[k])+' %':'—',cls(r[k],50,false)));cell(tr,fmt(r.mae_day_mean_price_eur_mwh),cls(r.mae_day_mean_price_eur_mwh,b.mae_day_mean_price_eur_mwh));cell(tr,fmt(r.mean_price_eur_mwh))});table('economic',rows.map(r=>row(p.economic.rows,z,r.model_id)),(tr,r)=>{cell(tr,fmt(r.pnl_net_eur,0));cell(tr,fmt(r.gain_vs_storm_eur,0),cls(r.gain_vs_storm_eur,0,false));const gain=Number.isFinite(r.pnl_net_eur)&&Number.isFinite(eb.pnl_net_eur)?r.pnl_net_eur-eb.pnl_net_eur:null;cell(tr,fmt(gain,0),cls(gain,0,false));cell(tr,fmt(r.gain_vs_storm_per_potential_mwh),cls(r.gain_vs_storm_per_potential_mwh,0,false));cell(tr,fmt(r.n_country_hours,0))});const ea=p.economic.audit;$('economic-support').textContent='Seuil effectif |edge| > '+fmt(ea.signal_hurdle_eur_mwh)+' €/MWh ; coût total '+fmt(ea.net_cost_eur_mwh)+' €/MWh traité. Dénominateur commun : '+fmt(ev.potential_energy_mwh,0)+' MWh potentiels. ALL exige tous les pays sélectionnés à chaque heure physique. Pas d’annualisation.';const metric=$('hour-metric').value,hours=Array.from({length:24},(_,i)=>i);const series=[{id:m,label:labels[m],color:color('point')},{id:'nuclear_kalman',label:'NYX actuel',color:color('nyx')},{id:'__storm__',label:'Storm',color:color('storm')}].map(s=>({...s,values:hours.map(h=>p.hourly.find(r=>r.zone===z&&r.model_id===s.id&&r.hour===h)?.[metric]??null)}));legend('hour-legend',series);plot('hour-plot','hour-tooltip',hours.map(h=>String(h).padStart(2,'0')+' h'),series);const d=p.daily.filter(r=>z==='ALL'||r.zone===z),dates=[...new Set(d.map(r=>r.delivery_day))].sort();const daySeries=[{id:m,label:labels[m],color:color('point')},{id:'nuclear_kalman',label:'NYX actuel',color:color('nyx')},{id:'__storm__',label:'Storm',color:color('storm')},{id:'actual',label:'Observé',color:color('obs')}].map(s=>({...s,values:dates.map(day=>{const parts=d.filter(r=>r.delivery_day===day&&r.model_id===(s.id==='actual'?'nuclear_kalman':s.id));return parts.length?parts.reduce((a,r)=>a+(s.id==='actual'?r.observed_mean_price_eur_mwh:r.mean_price_eur_mwh),0)/parts.length:null})}));legend('daily-legend',daySeries);plot('daily-plot','daily-tooltip',dates,daySeries);table('interventions',rows.filter(r=>r.model_id!=='__storm__').map(r=>row(p.interventions,z,r.model_id)),(tr,r)=>{cell(tr,fmt(r.changed_hours,0));cell(tr,fmt(r.improved_hours,0));cell(tr,fmt(r.worsened_hours,0));cell(tr,fmt(r.mae_on_interventions_eur_mwh)+' / '+fmt(r.nyx_mae_on_interventions_eur_mwh),cls(r.mae_on_interventions_eur_mwh,r.nyx_mae_on_interventions_eur_mwh));const tail=p.tails.find(t=>t.zone===z&&t.model_id===r.model_id&&t.regime==='price_ge_200')||{},normal=p.tails.find(t=>t.zone===z&&t.model_id===r.model_id&&t.regime==='price_lt_200')||{};cell(tr,fmt(tail.mae_eur_mwh));cell(tr,fmt(tail.rmse_eur_mwh));cell(tr,fmt(normal.mae_eur_mwh))});const ready=p.expert_ready.filter(r=>r.zone===z);table('ready',ready,(tr,r)=>{cell(tr,fmt(r.n_hours,0));cell(tr,fmt(r.mae_eur_mwh));cell(tr,fmt(r.rmse_eur_mwh))});$('ready-note').textContent=p.readiness_column?'Sous-ensemble commun identifié par '+p.readiness_column+' ; ce diagnostic ne remplace pas les KPI de l’année entière.':'Aucun indicateur de maturité commun disponible dans ces résultats ; ne pas confondre intervention non nulle et expert prêt.';$('audit').textContent=JSON.stringify({audit:D.audit,source_sha256:D.source_sha256,period:p.period,economic_policy:p.economic.audit,diagnostic_only:D.diagnostic_only,independent_validation:D.independent_validation},null,2)}
['zone','period','model','sort','hour-metric'].forEach(id=>$(id).addEventListener('change',render));$('theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'Mode jour':'Mode nuit';render()});render();
</script></body></html>'''
