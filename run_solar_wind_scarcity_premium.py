"""Isolated, immutable evaluation of a learned post-Kalman scarcity premium.

Only sealed outputs are read. No Chronos/CatBoost fitting, source refresh,
production update, or changes to concurrent experiments are performed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from chronos2_hourly.atomic_directory import AtomicDirectoryStaging
from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle


ROOT = Path(__file__).resolve().parent
ENGINE = "solar_wind_scarcity_premium_v1"
DAY = "2026-09-22"
BASELINES = {"DE": "45ec8314c37a2fe5", "NL": "b1da388bb4df6c63"}
TIMEZONES = {"DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}
OUTPUT = ROOT / "runs" / "experiments" / ENGINE / DAY
IMPLEMENTATION = (
    "run_solar_wind_scarcity_premium.py",
    "chronos2_hourly/solar_wind_scarcity_premium.py",
    "docs/solar_wind_scarcity_premium.md",
    "chronos2_hourly/nuclear_run_archive.py",
    "chronos2_hourly/atomic_directory.py",
)
RESULT_FILES = (
    "experiment.json", "features.parquet", "feature_audit.json",
    "backtest.parquet", "forecast.parquet", "daily_audit.json",
    "metrics.json", "report.html",
)
VARIANTS = {
    "baseline": "residual_kalman__",
    "scarcity_uncapped": "scarcity__",
    "scarcity_cap40": "capped40__",
}
LABELS = {
    "baseline": "NYX de référence",
    "scarcity_uncapped": "Prime apprise · amplitude libre",
    "scarcity_cap40": "Contrôle · prime limitée à +40",
}
PROTOCOL = {
    "engine": ENGINE,
    "delivery_day": DAY,
    "position": "additive_after_frozen_final_kalman",
    "training_target": "actual_minus_frozen_final_q50",
    "objective": "nonnegative_least_squares_with_ridge",
    "train_days": 365,
    "min_train_days": 90,
    "ridge": 0.05,
    "premium_caps_eur_mwh": [None, 40.0],
    "cap40_control_uses_identical_coefficients": True,
    "baseline_available_days": 365,
    "expected_evaluation_days": 275,
    "production_modified": False,
    "post_hoc_experiment": True,
    "pit_publication_evidence_verified": False,
    "forecast_labels_used": False,
    "low_renewable_slice": "joint_deficit >= 0.5",
    "high_residual_stress_slice": "residual_stress >= 1.0",
}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                      allow_nan=False, indent=2).encode("utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(frame):
    result = frame.copy(deep=True)
    key = next((c for c in ("delivery_start_utc", "timestamp") if c in result), None)
    values = result[key] if key else result.index
    if any(pd.Timestamp(value).tzinfo is None for value in values):
        raise ValueError("Explicit timezone-aware input timestamps required")
    result.index = pd.DatetimeIndex(pd.to_datetime(values, utc=True), name="delivery_start_utc_index")
    if result.index.hasnans or result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("Unique ordered UTC timestamps required")
    return result


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen source changed or disappeared: {path}")


def prepare(zone):
    if zone not in BASELINES:
        raise ValueError(f"Unsupported zone: {zone}")
    work = ROOT / "runs/experiments/solar_wind_v1" / DAY / zone.lower() / BASELINES[zone]
    frozen = work / "report_only/frozen_result"
    snapshot = json.loads((work / "input_snapshot.json").read_text(encoding="utf-8"))
    scientific = snapshot["identity"]["scientific_identity"]
    pins = {}
    for relative, expected in scientific["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"Original scientific implementation differs from sealed baseline: {relative}")
        pins[str(path)] = expected
    dependencies = {}
    for name, expected in scientific["dependencies"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Original dependency differs from sealed baseline: {name} {actual} != {expected}")
        dependencies[name] = actual
    for name in ("scipy", "pyarrow", "threadpoolctl"):
        dependencies[name] = importlib.metadata.version(name)
    for entry in snapshot["files"]:
        path = Path(entry["snapshot"]).resolve()
        if not path.is_relative_to((work / "snapshot").resolve()):
            raise ValueError("Frozen input snapshot escapes its run")
        pins[str(path)] = entry["sha256"]
    for path in [work / "input_snapshot.json", work / "resolved_config.yaml",
                 *sorted(frozen.iterdir()), *(ROOT / name for name in IMPLEMENTATION)]:
        if not path.is_file():
            raise ValueError(f"Expected pinned file missing: {path}")
        pins[str(path.resolve())] = sha(path)
    verify_pins(pins)
    bundle = load_nuclear_result_bundle(workdir=work)
    if bundle.audit["zone"] != zone or bundle.audit["delivery_day"] != DAY:
        raise ValueError("Sealed baseline zone or delivery day mismatch")
    verify_pins(pins)
    identity = {
        "protocol": PROTOCOL,
        "zone": zone,
        "timezone": TIMEZONES[zone],
        "baseline_id": BASELINES[zone],
        "baseline_directory": str(work),
        "source_sha256": pins,
        "dependencies": dependencies,
    }
    identifier = hashlib.sha256(json_bytes(identity)).hexdigest()[:16]
    return bundle, identity, identifier


def add_capped_control(frame):
    result = frame.copy(deep=True)
    premium = result["scarcity_premium"].to_numpy(float)
    if not np.isfinite(premium).all() or (premium < 0).any():
        raise ValueError("Finite nonnegative premium required")
    result["capped40_premium"] = np.minimum(premium, 40.0)
    for q in ("q10", "q50", "q90"):
        result["capped40__" + q] = result["residual_kalman__" + q] + result["capped40_premium"]
    return result


def scores(frame, prefix):
    if frame.empty:
        return {"hours": 0, "mae": None, "rmse": None, "bias": None,
                "coverage_q10_q90": None, "pinball_mean": None}
    actual = frame["actual"].to_numpy(float)
    predicted = frame[prefix + "q50"].to_numpy(float)
    error = predicted - actual
    pinball = []
    for q, level in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
        residual = actual - frame[prefix + q].to_numpy(float)
        pinball.append(float(np.maximum(level * residual, (level - 1) * residual).mean()))
    return {"hours": len(frame), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()),
            "coverage_q10_q90": float(((actual >= frame[prefix + "q10"].to_numpy(float)) &
                                       (actual <= frame[prefix + "q90"].to_numpy(float))).mean()),
            "pinball_mean": float(np.mean(pinball))}


def evaluate(frame, timezone_name):
    required = ["actual", "joint_deficit", "residual_stress", "scarcity_premium",
                *(prefix + q for prefix in VARIANTS.values() for q in ("q10", "q50", "q90"))]
    if frame.empty or not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("A finite paired evaluation frame is required")
    months = frame.index.tz_convert(timezone_name).strftime("%Y-%m")
    low = frame["joint_deficit"] >= 0.5
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        **{"month/" + month: months == month for month in sorted(set(months))},
        "actual_ge_200": frame["actual"] >= 200,
        "actual_ge_300": frame["actual"] >= 300,
        "low_renewables": low,
        "low_renewables_low_residual": low & (frame["residual_stress"] == 0),
        "low_renewables_high_residual": low & (frame["residual_stress"] >= 1),
        "other_hours": ~low,
        "zero_joint_deficit": frame["joint_deficit"] == 0,
    }
    slices = {}
    for name, mask in masks.items():
        selected = frame.loc[mask]
        result = {variant: scores(selected, prefix) for variant, prefix in VARIANTS.items()}
        base = result["baseline"]
        for variant, metrics in result.items():
            for metric in ("mae", "rmse", "bias"):
                val, ref = metrics[metric], base[metric]
                metrics["delta_" + metric] = None if val is None else val - ref
                if metric != "bias":
                    metrics["delta_" + metric + "_pct"] = None if val is None or not ref else 100 * (val / ref - 1)
        slices[name] = result
    spikes = {}
    for threshold in (200, 300):
        actual = frame["actual"] >= threshold
        comparison = {}
        for variant, prefix in VARIANTS.items():
            predicted = frame[prefix + "q50"] >= threshold
            tp, fp, fn = int((predicted & actual).sum()), int((predicted & ~actual).sum()), int((~predicted & actual).sum())
            comparison[variant] = {"actual_hours": int(actual.sum()), "tp": tp, "fp": fp, "fn": fn,
                                   "precision": tp / (tp + fp) if tp + fp else None,
                                   "recall": tp / (tp + fn) if tp + fn else None}
        spikes[str(threshold)] = comparison
    premium = frame["scarcity_premium"]
    local_days = frame.index.tz_convert(timezone_name).date
    return {"support": {"hours": len(frame), "days": len(set(local_days)),
                        "first_day": str(local_days[0]), "last_day": str(local_days[-1]),
                        "first_timestamp_utc": str(frame.index[0]), "last_timestamp_utc": str(frame.index[-1]),
                        "calibration_days_omitted": PROTOCOL["min_train_days"]},
            "slices": slices, "spikes": spikes,
            "premium": {"active_hours": int((premium > 1e-10).sum()),
                        "above_40_hours": int((premium > 40).sum()),
                        "mean": float(premium.mean()), "p95": float(premium.quantile(.95)),
                        "max": float(premium.max())},
            "bias_convention": "prediction_minus_actual",
            "significance_test_performed": False}


def _number(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}".replace(",", " ")


def _comparison_table(comparison):
    rows = []
    for variant in VARIANTS:
        metric = comparison[variant]
        delta = metric["delta_mae_pct"]
        css = "good" if delta is not None and delta < 0 else "bad" if delta else ""
        rows.append(f'<tr><td>{html.escape(LABELS[variant])}</td><td>{metric["hours"]}</td>'
                    f'<td>{_number(metric["mae"])}</td><td>{_number(metric["rmse"])}</td>'
                    f'<td>{_number(metric["bias"])}</td><td class="{css}">{_number(delta, 2)} %</td></tr>')
    return ('<div class="table-wrap"><table><thead><tr><th>Variante</th><th>Heures</th>'
            '<th>MAE</th><th>RMSE</th><th>Biais</th><th>Δ MAE / NYX</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table></div>')


def render_report(zone, identifier, metrics, forecast):
    support, premium = metrics["support"], metrics["premium"]
    slice_labels = {
        "actual_ge_200": "Prix réalisé ≥ 200 €/MWh", "actual_ge_300": "Prix réalisé ≥ 300 €/MWh",
        "low_renewables": "Prévisions renouvelables faibles · déficit conjoint ≥ 0,5",
        "low_renewables_low_residual": "Renouvelables faibles · charge résiduelle sous sa médiane",
        "low_renewables_high_residual": "Renouvelables faibles · charge résiduelle élevée",
        "other_hours": "Autres heures · déficit conjoint < 0,5", "zero_joint_deficit": "Déficit conjoint nul",
    }
    details = ''.join(f'<details><summary>{html.escape(slice_labels.get(name, name.replace("month/", "Mois ")))}</summary>'
                      f'{_comparison_table(value)}</details>'
                      for name, value in metrics["slices"].items() if name != "all")
    spike_rows = []
    for threshold, variants in metrics["spikes"].items():
        for variant, values in variants.items():
            spike_rows.append(f'<tr><td>{threshold}</td><td>{LABELS[variant]}</td><td>{values["actual_hours"]}</td>'
                              f'<td>{values["tp"]}</td><td>{values["fp"]}</td><td>{values["fn"]}</td></tr>')
    forecast_rows = []
    for index, row in forecast.iterrows():
        local = index.tz_convert(TIMEZONES[zone]).strftime("%H:%M %z")
        forecast_rows.append(f'<tr><td>{local}</td><td>{_number(row["residual_kalman__q50"])}</td>'
                             f'<td>{_number(row["scarcity__q50"])}</td><td>{_number(row["capped40__q50"])}</td>'
                             f'<td class="accent">+{_number(row["scarcity_premium"])}</td></tr>')
    return f'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYX · Prime de rareté · {zone}</title><style>
:root{{color-scheme:dark;--bg:#090c12;--panel:#101720;--line:#27333f;--text:#e4eaf0;--muted:#a1b0bd;--cyan:#61e4e7;--amber:#ffbb63}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 90% 0%,#15303a66,transparent 50%),var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:40px 26px 70px}}h1{{font-size:clamp(26px,4vw,43px);font-weight:600;line-height:1.2;margin:14px 0}}
h2{{font-size:23px;margin:34px 0 14px}}p{{max-width:1000px;color:var(--muted)}}.brand{{letter-spacing:.2em;color:var(--cyan);font-weight:700}}
.tag{{display:inline-block;margin:8px 8px 4px 0;border:1px solid var(--line);border-radius:999px;padding:5px 12px;color:var(--amber);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:25px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px}}.card strong{{display:block;font-size:27px;color:var(--cyan)}}.card span{{color:var(--muted)}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}table{{width:100%;border-collapse:collapse;white-space:nowrap;background:#0d131b}}th,td{{text-align:right;padding:13px 15px;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:12px;font-weight:500}}th:first-child,td:first-child{{text-align:left}}tr:last-child td{{border-bottom:0}}tbody tr:hover{{background:#15202b}}.good{{color:#74e4b7}}.bad{{color:#ff9d91}}.accent{{color:var(--amber)}}
details{{margin:12px 0;border:1px solid var(--line);border-radius:10px;padding:14px}}summary{{cursor:pointer;color:var(--text)}}details .table-wrap{{margin-top:14px}}.note{{padding:18px 22px;border-left:3px solid var(--amber);background:#ffbb6308;border-radius:0 10px 10px 0}}a{{color:var(--cyan)}}footer{{margin-top:34px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
</style></head><body><main>
<div class="brand">NYX / RECHERCHE</div><h1>Prévoir l’amplitude de la rareté</h1>
<span class="tag">{zone} · expérience isolée</span><span class="tag">Après Kalman</span><span class="tag">Amplitude apprise · sans plafond additionnel</span>
<p>Une prime positive est apprise sur les erreurs passées de NYX lorsque les prévisions de vent et de solaire sont faibles. Son amplitude dépend aussi de la charge résiduelle. La variante principale n’impose aucun plafond à cette prime ; le contrôle +40 utilise exactement les mêmes coefficients.</p>
<div class="cards"><div class="card"><strong>{support["days"]} jours</strong><span>{support["hours"]} heures communes évaluées</span></div>
<div class="card"><strong>{premium["active_hours"]}</strong><span>heures avec une prime positive</span></div>
<div class="card"><strong>{_number(premium["max"], 1)} €/MWh</strong><span>prime maximale du backtest</span></div>
<div class="card"><strong>{premium["above_40_hours"]}</strong><span>heures au-delà de +40 €/MWh</span></div></div>
<h2>Performance sur les mêmes heures</h2><p>{support["first_day"]} → {support["last_day"]}. Les 90 premiers jours servent à calibrer la prime et sont exclus de toutes les métriques. Erreurs et biais en €/MWh ; biais = prévision − réalisé. Un Δ MAE négatif indique une amélioration.</p>
{_comparison_table(metrics["slices"]["all"])}
<h2>Où la prime aide-t-elle ?</h2><p>Les découpes ont été fixées avant calcul. Les découpes de prix élevé utilisent le réalisé uniquement pour l’évaluation, jamais comme variable prédictive.</p>{details}
<h2>Détection des prix élevés</h2><div class="table-wrap"><table><thead><tr><th>Seuil €/MWh</th><th>Variante</th><th>Heures réalisées</th><th>Détections</th><th>Fausses alertes</th><th>Manqués</th></tr></thead><tbody>{''.join(spike_rows)}</tbody></table></div>
<h2>Prévision du {DAY}</h2><p>Prévision séparée, sans réalisé utilisé pour cette journée. Heures locales {TIMEZONES[zone]}. La prime décale les trois quantiles du même montant et conserve leur ordre.</p>
<div class="table-wrap"><table><thead><tr><th>Heure locale</th><th>NYX Q50</th><th>Prime libre · Q50</th><th>Contrôle +40 · Q50</th><th>Prime libre</th></tr></thead><tbody>{''.join(forecast_rows)}</tbody></table></div>
<h2>Protocole et portée</h2><div class="note"><p>L’apprentissage minimise l’erreur quadratique avec régularisation L2 (ridge = 0,05), coefficients non négatifs, sans constante. Ce choix donne plus de poids aux fortes sous-estimations et peut améliorer la RMSE en dégradant la MAE. Chaque journée utilise uniquement les erreurs de journées antérieures, sur au plus 365 jours ; la fenêtre grandit après 90 jours de calibration.</p>
<p>La référence NYX, ses seuils internes et ses prévisions scellées restent inchangés. Ce complément s’applique après le Kalman final et ne rejoue aucun modèle. Il n’est pas activé en production. Il garantit le sens de la prime à contexte fixé, pas une hausse universelle du prix total. Des coefficients nuls restent possibles si les erreurs passées ne justifient aucune prime.</p>
<p>Étude rétrospective après observation des difficultés du modèle : ces résultats ne constituent pas une validation indépendante. La publication historique des covariables au moment exact de chaque prévision n’est pas certifiée. Les substitutions de source déjà auditées du cas NL sont héritées. Aucun test de significativité n’a été effectué.</p></div>
<footer>Identité {identifier} · Sources, code et sorties vérifiés par SHA-256.<br>
<a href="metrics.json">Métriques complètes</a> · <a href="experiment.json">Protocole et sources</a> · <a href="daily_audit.json">Calibration quotidienne</a> · <a href="feature_audit.json">Normalisation causale</a> · <a href="completion.json">Empreintes des résultats</a></footer>
</main></body></html>'''


def safe_destination(zone, identifier):
    destination = OUTPUT / zone.lower() / identifier
    if destination.absolute() != destination.resolve() or not destination.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Redirected experiment output refused")
    return destination


def verify_completed(directory, identity, identifier):
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("identity") != identifier:
        raise ValueError("Existing experiment completion identity differs")
    if set(completion["files"]) != set(RESULT_FILES):
        raise ValueError("Existing experiment file inventory differs")
    if {path.name for path in directory.iterdir()} != set(RESULT_FILES) | {"completion.json"}:
        raise ValueError("Unexpected files in completed experiment")
    for name, expected in completion["files"].items():
        path = directory / name
        if path.is_symlink() or path.resolve().parent != directory.resolve() or sha(path) != expected:
            raise ValueError(f"Existing result checksum mismatch: {name}")
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if experiment != identity:
        raise ValueError("Refusing to reuse a divergent experiment")
    verify_pins(identity["source_sha256"])
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


def run_zone(zone, *, action):
    from chronos2_hourly.solar_wind_scarcity_premium import build_scarcity_basis, run_premium_backtest

    bundle, identity, identifier = prepare(zone)
    destination = safe_destination(zone, identifier)
    if action == "validate":
        result = {"zone": zone, "identity": identifier, "sources_verified": True,
                  "output": str(destination), "writes_performed": False}
        if destination.exists():
            result["existing_results_verified"] = True
            result["metrics"] = verify_completed(destination, identity, identifier)["slices"]["all"]
        return result
    if destination.exists():
        metrics = verify_completed(destination, identity, identifier)
        return {"zone": zone, "identity": identifier, "status": "reused", "report": str(destination / "report.html"),
                "metrics": metrics["slices"]["all"], "premium": metrics["premium"]}
    features, feature_audit = build_scarcity_basis(indexed(bundle.covariates), zone=zone, timezone=TIMEZONES[zone])
    baseline, forecast = indexed(bundle.kalman_view.backtest), indexed(bundle.kalman_view.forecast)
    if "actual" in forecast:
        if forecast["actual"].notna().any():
            raise ValueError("Forecast delivery labels must never enter calibration")
        forecast = forecast.drop(columns="actual")
    with threadpool_limits(limits=1):
        outputs = run_premium_backtest(baseline, forecast, features, timezone=TIMEZONES[zone],
            train_days=PROTOCOL["train_days"], min_train_days=PROTOCOL["min_train_days"],
            ridge=PROTOCOL["ridge"], cap=None)
    backtest = add_capped_control(outputs["backtest"])
    forecast = add_capped_control(outputs["forecast"])
    metrics = evaluate(backtest, TIMEZONES[zone])
    if metrics["support"]["days"] != PROTOCOL["expected_evaluation_days"]:
        raise ValueError("Unexpected paired evaluation support; fixed 90/275-day split required")
    if set(forecast.index.tz_convert(TIMEZONES[zone]).strftime("%Y-%m-%d")) != {DAY}:
        raise ValueError("Forecast must remain separate on the sealed delivery day")
    verify_pins(identity["source_sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with AtomicDirectoryStaging(destination.parent, prefix=".scarcity-") as publication:
        staging = publication.path
        for name, value in (("experiment.json", identity), ("feature_audit.json", feature_audit),
                            ("daily_audit.json", outputs["daily_audit"]), ("metrics.json", metrics)):
            (staging / name).write_bytes(json_bytes(value))
        features.to_parquet(staging / "features.parquet")
        backtest.to_parquet(staging / "backtest.parquet")
        forecast.to_parquet(staging / "forecast.parquet")
        (staging / "report.html").write_text(render_report(zone, identifier, metrics, forecast), encoding="utf-8")
        completion = {"schema_version": 1, "status": "complete", "engine": ENGINE,
                      "identity": identifier, "zone": zone,
                      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "sources_unchanged": True, "production_modified": False,
                      "files": {name: sha(staging / name) for name in RESULT_FILES}}
        (staging / "completion.json").write_bytes(json_bytes(completion))
        verify_pins(identity["source_sha256"])
        publication.publish(destination)
    verify_completed(destination, identity, identifier)
    return {"zone": zone, "identity": identifier, "status": "complete", "report": str(destination / "report.html"),
            "support": metrics["support"], "metrics": metrics["slices"]["all"], "premium": metrics["premium"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("validate", "run"), default="validate")
    parser.add_argument("--zones", nargs="+", choices=tuple(BASELINES), default=list(BASELINES))
    args = parser.parse_args(argv)
    if len(set(args.zones)) != len(args.zones):
        parser.error("Duplicate zones are not allowed")
    for zone in args.zones:
        print(json.dumps(json_value(run_zone(zone, action=args.action)), ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
