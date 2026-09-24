"""Three isolated, paired scarcity corrections; immutable v1 comparator."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

import run_solar_wind_scarcity_regime as parent
from chronos2_hourly.process_lock import exclusive_process_lock

ROOT = parent.ROOT
DAY = parent.DAY
ZONES = parent.ZONES
TZ = parent.TZ
ENGINE = "solar_wind_scarcity_ablation_v1"
PARENT_ID = "696eae68b4a16a10"
FIRST_DAY = "2026-03-22"
OUTPUT = ROOT / "runs/experiments" / ENGINE / DAY
VARIANTS = ("direct_quantile", "regime_hour_local", "regime_calibration_oof90")
LABELS = {
    "interaction_40": "Référence · interaction 40",
    "regime_v1": "Régimes v1 · test précédent",
    "direct_quantile": "Test 1 · quantiles directs",
    "regime_hour_local": "Test 2 · sans maxima journaliers",
    "regime_calibration_oof90": "Test 3 · recalibration OOF 90 jours",
}
PROTOCOL = {
    "engine": ENGINE, "parent_identity": PARENT_ID, "delivery_day": DAY,
    "evaluation_start_day": FIRST_DAY, "evaluation_end_day": "2026-09-21",
    "evaluation_days": 184, "evaluation_hours_per_zone": 4415,
    "forecast_diagnostic_hours_per_zone": 24, "variants": VARIANTS,
    "comparators": ("interaction_40", "regime_v1"),
    "direct_quantile": "single MultiQuantile expert all residuals, same v1 features and hyperparameters",
    "regime_hour_local": "remove ONLY the four own/other_daily_peak_* features from v1",
    "regime_calibration_oof90": "regularized logistic logit(p_v1)+country offset, past 90 complete days; unchanged reconstructed experts",
    "calibration_C": 0.1, "calibration_min_positive": 20,
    "calibration_min_positive_per_country": 5, "calibration_min_negative": 32,
    "calibration_nonpositive_slope": "identity fallback, audited",
    "parent_reconstruction_parity": {"atol": 1e-7, "rtol": 1e-8},
    "calibration_input": "archived prequential v1 probabilities, mix of original raw and Platt outputs",
    "fit_window_days": 365, "refit_schedule": "unchanged v1 weekly origins; fresh Sep22 diagnostic",
    "threads": 1, "workers": 1, "iterations": 120, "seed": 20260923,
    "process_priority": "BelowNormal", "min_available_memory_gib": 3.5,
    "memory_reserve_target_gib": 3.0,
    "production_modified": False, "new_source_sync": False, "chronos_kalman_refit": False,
    "post_hoc": True, "prospective_validation": False,
    "pit_publication_evidence_verified": False, "automatic_promotion": False,
    "event_labels": "read only after all predictions are committed; never used for fitting",
}
IMPLEMENTATION = (
    "run_solar_wind_scarcity_ablation.py",
    "chronos2_hourly/solar_wind_scarcity_ablation.py",
    "docs/solar_wind_scarcity_ablation.md",
)


def blocks():
    return [b for b in parent.blocks() if b[0] >= date.fromisoformat(FIRST_DAY)]


def prepare():
    ready = parent.prepare()
    identifier, identity, history, forecast, features, feature_audit, observations = ready
    if identifier != PARENT_ID:
        raise ValueError("Parent code or sources changed; refusing a different comparator")
    directory = parent.OUTPUT / PARENT_ID
    done = parent.verify_completed(directory, PARENT_ID, identity)
    pins = dict(identity["sources_and_code_sha256"])
    for name, expected in done["files"].items():
        pins[str(directory / name)] = expected
    pins[str(directory / "completion.json")] = parent.sha(directory / "completion.json")
    for name in IMPLEMENTATION:
        pins[str(ROOT / name)] = parent.sha(ROOT / name)
    # Select metadata/probabilities/quantiles only. Diagnostic labels are not model inputs.
    cols = ["zone", "fit_origin", "spike_probability", "residual_kalman__q50",
            "regime__q10", "regime__q50", "regime__q90"]
    old_history = pd.read_parquet(directory / "backtest.parquet", columns=cols + ["actual"])
    old_forecast = pd.read_parquet(directory / "forecast_diagnostic.parquet", columns=cols)
    source_predictions = pd.concat([old_history[cols], old_forecast[cols]])
    for zone in ZONES:
        sub = source_predictions.loc[source_predictions.zone == zone]
        expected = parent.expected_index("2025-12-21", "2026-09-23")
        if not sub.index.equals(expected):
            raise ValueError("Incomplete v1 prediction timeline")
        full_baseline = pd.concat([history[zone], forecast[zone]])
        if not np.array_equal(sub.residual_kalman__q50.to_numpy(), full_baseline.loc[sub.index, "residual_kalman__q50"].to_numpy()):
            raise ValueError("Parent prediction baseline mismatch")
        earlier = old_history.loc[old_history.zone == zone]
        if not np.array_equal(earlier.actual.to_numpy(), history[zone].loc[earlier.index, "actual"].to_numpy()):
            raise ValueError("Parent historical labels differ from frozen baseline")
    calibration = old_history[["zone", "fit_origin", "spike_probability"]].copy()
    calibration["timestamp_utc"] = calibration.index
    calibration["residual"] = old_history.actual - old_history.residual_kalman__q50
    calibration = calibration.reset_index(drop=True)
    new_identity = {"protocol": PROTOCOL, "sources_and_code_sha256": pins,
                    "dependencies": identity["dependencies"],
                    "feature_columns": identity["feature_columns"]}
    new_id = hashlib.sha256(parent.encoded(new_identity)).hexdigest()[:16]
    parent.verify_pins(pins)
    return new_id, new_identity, history, forecast, features, feature_audit, observations, source_predictions, calibration


def frame_keys(frame):
    return pd.MultiIndex.from_arrays([frame.index, frame.zone], names=["timestamp_utc", "zone"])


def inputs(origin, stop, is_forecast, ready):
    _, _, history, forecast, features, _, _, source_predictions, calibration = ready
    train_x, train_y, train_days, test_x, test_parts, references = [], [], [], [], [], []
    for zone in ZONES:
        hist = history[zone]
        days = hist.index.tz_convert(TZ).date
        selected = hist.loc[(days >= origin - timedelta(days=365)) & (days < origin)]
        train_x.append(features[zone].loc[selected.index])
        train_y.append((selected.actual - selected.residual_kalman__q50).to_numpy(float))
        train_days.extend(selected.index.tz_convert(TZ).date)
        target = forecast[zone] if is_forecast else hist
        target_days = target.index.tz_convert(TZ).date
        part = target.loc[(target_days >= origin) & (target_days < stop)].copy()
        part["zone"] = zone
        test_parts.append(part)
        test_x.append(features[zone].loc[part.index])
        reference = source_predictions.loc[source_predictions.zone == zone].loc[part.index]
        if not (reference.fit_origin == str(origin)).all():
            raise ValueError("Parent prediction origin differs from fixed refit schedule")
        references.append(reference)
    reference = pd.concat(references)
    keys = frame_keys(reference)
    probability = pd.Series(reference.spike_probability.to_numpy(float), index=keys)
    prediction_origin = pd.Series(reference.fit_origin.to_numpy(), index=keys)
    residual_quantiles = pd.DataFrame(
        reference[["regime__q10", "regime__q50", "regime__q90"]].to_numpy(float)
        - reference.residual_kalman__q50.to_numpy(float)[:, None],
        index=keys, columns=["q10", "q50", "q90"])
    cal_days = pd.DatetimeIndex(calibration.timestamp_utc).tz_convert(TZ).date
    calibration_window = calibration.loc[(cal_days >= origin - timedelta(days=90)) & (cal_days < origin)].copy()
    return (pd.concat(train_x), np.concatenate(train_y), pd.concat(test_x), train_days,
            pd.concat(test_parts), probability, prediction_origin, residual_quantiles, calibration_window)


def controls(source_predictions, history, forecast, *, is_forecast):
    frames = []
    for zone in ZONES:
        base = (forecast if is_forecast else history)[zone]
        start = DAY if is_forecast else FIRST_DAY
        base = base.loc[base.index >= pd.Timestamp(start, tz=TZ).tz_convert("UTC")].copy()
        base["zone"] = zone
        old = source_predictions.loc[source_predictions.zone == zone].loc[base.index]
        for variant in ("interaction_40", "regime_v1"):
            result = base.copy()
            for q in ("q10", "q50", "q90"):
                result["regime__" + q] = (base["residual_kalman__" + q].to_numpy()
                                          if variant == "interaction_40" else old["regime__" + q].to_numpy())
            result["spike_probability"] = np.nan if variant == "interaction_40" else old.spike_probability.to_numpy()
            result["variant"] = variant
            result["is_forecast"] = is_forecast
            frames.append(result)
    return pd.concat(frames)


def evaluate(frame):
    metrics = {}
    for zone in ZONES:
        metrics[zone] = {}
        for variant in LABELS:
            sub = frame.loc[(frame.zone == zone) & (frame.variant == variant)]
            masks = {"all": np.ones(len(sub), bool), "actual_ge_200": sub.actual >= 200,
                     "actual_ge_300": sub.actual >= 300,
                     "low_renewables_without_price_spike": (sub.own_joint_deficit >= .5) & (sub.actual < 200),
                     "low_renewables": sub.own_joint_deficit >= .5}
            months = sub.index.tz_convert(TZ).strftime("%Y-%m")
            masks.update({"month/" + month: months == month for month in sorted(set(months))})
            result = {name: parent.score(sub.loc[mask], "regime__") for name, mask in masks.items()}
            for threshold in (200, 300):
                actual, pred = sub.actual >= threshold, sub.regime__q50 >= threshold
                result["spikes_" + str(threshold)] = {"tp": int((actual & pred).sum()),
                    "fp": int((~actual & pred).sum()), "fn": int((actual & ~pred).sum())}
            p = sub.spike_probability
            result["gate_brier"] = (float(np.mean((p - (sub.actual - sub.residual_kalman__q50 > 50).astype(float))**2))
                                      if p.notna().all() else None)
            metrics[zone][variant] = result
    return metrics


def render(identifier, metrics, event):
    rows = []
    for zone in ZONES:
        base = metrics[zone]["interaction_40"]["all"]
        for variant in LABELS:
            result = metrics[zone][variant]["all"]
            rows.append(f"<tr><td>{zone}</td><td>{LABELS[variant]}</td><td>{result['mae']:.3f}</td><td>{100*(result['mae']/base['mae']-1):+.2f}%</td><td>{result['rmse']:.3f}</td><td>{100*(result['rmse']/base['rmse']-1):+.2f}%</td><td>{result['coverage_q10_q90']:.1%}</td></tr>")
    event_rows = []
    for zone in ZONES:
        for hour in (18, 19, 20):
            sub = event.loc[(event.zone == zone) & (event.index.tz_convert(TZ).hour == hour)]
            prices = {row.variant: row.regime__q50 for _, row in sub.iterrows()}
            actual = float(sub.iloc[0].actual)
            event_rows.append(f"<tr><td>{zone}</td><td>{hour} h</td><td>{actual:.2f}</td>" + ''.join(f"<td>{prices[v]:.2f}</td>" for v in LABELS) + "</tr>")
    return f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX · Trois tests de correction des spikes</title>
<style>body{{background:#090c12;color:#e4eaf0;font:15px/1.55 system-ui;max-width:1450px;margin:35px auto;padding:24px}}h1,h2,a{{color:#61e4e7}}table{{width:100%;border-collapse:collapse;background:#101720}}td,th{{padding:11px;border-bottom:1px solid #27333f;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.note{{border-left:3px solid #ffbb63;padding:15px;background:#ffbb6308}}pre{{white-space:pre-wrap;font-size:12px}}.scroll{{overflow-x:auto}}</style>
<h1>NYX · Trois tests isolés de correction des spikes</h1>
<p>Test 1 : quantiles directs, sans partition à +50. Test 2 : même modèle à régimes, uniquement sans les quatre maxima journaliers. Test 3 : mêmes experts et mêmes probabilités v1 avant recalibrage chronologique 90 jours, avec ajustement pays régularisé.</p>
<p class="note">Comparaison appariée du 22 mars au 21 septembre 2026 : <b>184 jours / 4 415 heures par pays</b>. Les 90 premiers jours de prédictions OOF v1 alimentent le recalibrage, donc toutes les variantes sont évaluées sur une période commune plus courte que le test précédent. Aucun score de périodes différentes n'est mélangé. Recherche post-hoc, archives PIT non certifiées ; aucune promotion ou modification de production.</p>
<h2>Performance historique commune · €/MWh</h2><div class="scroll"><table><tr><th>Pays</th><th>Variante</th><th>MAE</th><th>Δ MAE / référence</th><th>RMSE</th><th>Δ RMSE</th><th>Couverture P10–P90</th></tr>{''.join(rows)}</table></div>
<h2>22 septembre · diagnostic déjà vu, non validation indépendante</h2><div class="scroll"><table><tr><th>Pays</th><th>Heure locale</th><th>Observé</th>{''.join('<th>'+html.escape(LABELS[v])+' P50</th>' for v in LABELS)}</tr>{''.join(event_rows)}</table></div>
<p>Le quantile direct ne fournit pas de probabilité de régime : aucun Brier de classification n'est inventé pour ce test. Le test 3 recalibre des probabilités v1 archivées, certaines brutes et d'autres déjà calibrées selon les anciens blocs. Sa pente commune et son ajustement pays ne garantissent pas une calibration prospective. Chaque reconstruction des experts est vérifiée contre les quantiles v1 avant changement de probabilité.</p>
<details><summary>Spikes, fausses alertes, mois et calibration</summary><pre>{html.escape(parent.encoded(metrics).decode())}</pre></details>
<p><a href="metrics.json">Métriques complètes</a> · <a href="fit_audits.json">Audits</a> · <a href="experiment.json">Protocole figé</a> · <a href="completion.json">Inventaire SHA</a></p><footer>Identité {identifier} · résultats exploratoires, sans sélection ou déploiement automatique.</footer></html>'''


def inventory():
    files = {"experiment.json", "feature_audit.json", "backtest.parquet", "forecast_diagnostic.parquet",
             "metrics.json", "fit_audits.json", "report.html"}
    for origin, _, _ in blocks():
        for variant in VARIANTS:
            files.update(f"checkpoints/{origin}_{variant}{suffix}" for suffix in (".json", ".audit.json", ".parquet"))
    return files


def verify_completed(directory, identifier, identity):
    done = parent.read_json(directory / "completion.json")
    if (done.get("status"), done.get("identity"), done.get("evaluation_days"),
        done.get("hours_per_zone"), done.get("diagnostic_hours_per_zone")) != ("COMPLETE", identifier, 184, 4415, 24):
        raise ValueError("Invalid completion support or identity")
    if set(done["files"]) != inventory() or parent.read_json(directory / "experiment.json") != parent.value(identity):
        raise ValueError("Completion inventory or identity mismatch")
    for name, expected in done["files"].items():
        if parent.sha(directory / name) != expected:
            raise ValueError(f"Completion SHA mismatch: {name}")
    for name, grid in (("backtest.parquet", parent.expected_index(FIRST_DAY, DAY)),
                       ("forecast_diagnostic.parquet", parent.expected_index(DAY, "2026-09-23"))):
        frame = pd.read_parquet(directory / name)
        if set(frame.variant) != set(LABELS) or set(frame.zone) != set(ZONES):
            raise ValueError("Missing completed variants or countries")
        for zone in ZONES:
            for variant in LABELS:
                sub = frame.loc[(frame.zone == zone) & (frame.variant == variant)]
                q = sub[["regime__q10", "regime__q50", "regime__q90"]].to_numpy(float)
                if not sub.index.equals(grid) or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
                    raise ValueError("Invalid completed grids or quantiles")
    parent.verify_pins(identity["sources_and_code_sha256"])
    return done


def run(ready):
    from chronos2_hourly.solar_wind_scarcity_ablation import fit_variant
    identifier, identity, history, forecast, features, feature_audit, observations, predictions, _ = ready
    directory = OUTPUT / identifier
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(OUTPUT / "experiment.lock"):
        if (directory / "completion.json").exists():
            return verify_completed(directory, identifier, identity)
        if (directory / "experiment.json").exists() and parent.read_json(directory / "experiment.json") != parent.value(identity):
            raise ValueError("Changed existing workdir identity")
        parent.atomic_json(directory / "experiment.json", identity)
        parent.atomic_json(directory / "feature_audit.json", feature_audit)
        checkpoint = directory / "checkpoints"
        checkpoint.mkdir(exist_ok=True)
        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        started = time.monotonic()
        state = {"engine": ENGINE, "identity": identifier, "status": "RUNNING", "pid": os.getpid(),
                 "process_create_time": process.create_time(), "command": process.cmdline(),
                 "started_utc": datetime.now(timezone.utc).isoformat(), "threads": 1, "workers": 1,
                 "total_units": len(blocks()) * len(VARIANTS), "completed_units": 0,
                 "variants": {variant: {"completed_blocks": 0, "total_blocks": len(blocks())} for variant in VARIANTS}}
        def status(**updates):
            state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
            n = state["completed_units"]
            state["eta_seconds"] = (state["total_units"] - n) * ((time.monotonic() - started) / n if n else 20)
            state["eta_basis"] = "elapsed_per_unit_rough" if n else "initial_20sec_per_unit_rough"
            parent.atomic_json(directory / "status.json", state)
            parent.atomic_json(OUTPUT / "latest_DE_NL.json", state | {"workdir": str(directory)})
        frames, audits = [], []
        try:
            status(phase="ready")
            for origin, stop, is_forecast in blocks():
                train_x, train_y, test_x, train_days, template, pp, po, pq, cal = inputs(origin, stop, is_forecast, ready)
                for variant in VARIANTS:
                    stem = checkpoint / f"{origin}_{variant}"
                    data_path, audit_path, receipt = (stem.with_suffix(s) for s in (".parquet", ".audit.json", ".json"))
                    if receipt.exists():
                        saved = parent.read_json(receipt)
                        if saved.get("identity") != identifier or saved.get("variant") != variant or saved.get("origin") != str(origin):
                            raise ValueError("Checkpoint identity mismatch")
                        for path in (data_path, audit_path):
                            if parent.sha(path) != saved["files"][path.name]:
                                raise ValueError("Checkpoint SHA mismatch")
                        output, audit = pd.read_parquet(data_path), parent.read_json(audit_path)
                    else:
                        if data_path.exists() or audit_path.exists():
                            raise ValueError("Unsealed partial checkpoint retained; inspection required")
                        while psutil.virtual_memory().available < PROTOCOL["min_available_memory_gib"] * 2**30:
                            status(status="WAITING_MEMORY", phase="resource_guard", current_variant=variant, next_origin=str(origin))
                            time.sleep(10)
                        parent.verify_pins(identity["sources_and_code_sha256"])
                        status(status="RUNNING", phase="fit", current_variant=variant, next_origin=str(origin))
                        kwargs = dict(origin_day=origin, threads=1, iterations=120, seed=20260923)
                        if variant == "regime_calibration_oof90":
                            kwargs.update(parent_probability=pp, parent_prediction_origin=po,
                                          parent_residual_quantiles=pq, calibration_frame=cal)
                        with threadpool_limits(limits=1):
                            quantiles, probability, audit = fit_variant(variant, train_x, train_y, test_x, train_days, **kwargs)
                        if quantiles.shape != (len(template), 3) or not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
                            raise ValueError("Invalid candidate quantiles")
                        if probability is not None and (probability.shape != (len(template),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()):
                            raise ValueError("Invalid candidate probabilities")
                        output = template.copy()
                        for index, q in enumerate(("q10", "q50", "q90")):
                            output["regime__" + q] = output.residual_kalman__q50.to_numpy(float) + quantiles[:, index]
                        output["spike_probability"] = np.nan if probability is None else probability
                        output["variant"] = variant
                        output["is_forecast"] = is_forecast
                        output["fit_origin"] = str(origin)
                        for col in ("own_joint_deficit", "own_residual_stress", "other_residual_stress"):
                            output[col] = test_x[col].to_numpy(float)
                        audit.update(variant=variant, origin=str(origin), stop=str(stop), is_forecast=is_forecast)
                        parent.verify_pins(identity["sources_and_code_sha256"])
                        parent.atomic_parquet(data_path, output)
                        parent.atomic_json(audit_path, audit)
                        parent.atomic_json(receipt, {"identity": identifier, "variant": variant, "origin": str(origin),
                            "files": {p.name: parent.sha(p) for p in (data_path, audit_path)}})
                    frames.append(output)
                    audits.append(audit)
                    state["variants"][variant]["completed_blocks"] += 1
                    status(completed_units=state["completed_units"] + 1, phase="checkpoint_saved")
                    print(json.dumps({"identity": identifier, "variant": variant, "origin": str(origin),
                        "completed_units": state["completed_units"], "total_units": state["total_units"], "eta_seconds": state["eta_seconds"]}), flush=True)
            all_predictions = pd.concat(frames)
            result_frames = []
            for is_forecast in (False, True):
                baseline = controls(predictions, history, forecast, is_forecast=is_forecast)
                for zone in ZONES:
                    mask = baseline.zone == zone
                    baseline.loc[mask, "own_joint_deficit"] = features[zone].loc[baseline.loc[mask].index, "own_joint_deficit"].to_numpy()
                result = pd.concat([baseline, all_predictions.loc[all_predictions.is_forecast == is_forecast]])
                grid = parent.expected_index(DAY, "2026-09-23") if is_forecast else parent.expected_index(FIRST_DAY, DAY)
                for zone in ZONES:
                    for variant in LABELS:
                        if not result.loc[(result.zone == zone) & (result.variant == variant)].index.equals(grid):
                            raise ValueError("Unpaired final comparison grid")
                result_frames.append(result)
            backtest, diagnostic = result_frames
            metrics = evaluate(backtest)
            # Independent diagnostic labels read only after every candidate prediction is saved.
            diagnostic["actual"] = np.nan
            for zone in ZONES:
                observed = parent.indexed(pd.read_parquet(observations[zone]))
                mask = diagnostic.zone == zone
                actual = observed.actual.reindex(diagnostic.loc[mask].index).to_numpy(float)
                if not np.isfinite(actual).all():
                    raise ValueError("Missing diagnostic actuals; no silent omissions")
                diagnostic.loc[mask, "actual"] = actual
            parent.atomic_parquet(directory / "backtest.parquet", backtest)
            parent.atomic_parquet(directory / "forecast_diagnostic.parquet", diagnostic)
            parent.atomic_json(directory / "metrics.json", metrics)
            parent.atomic_json(directory / "fit_audits.json", audits)
            parent.atomic_bytes(directory / "report.html", render(identifier, metrics, diagnostic).encode("utf-8"))
            parent.verify_pins(identity["sources_and_code_sha256"])
            done = {"status": "COMPLETE", "identity": identifier, "production_modified": False,
                    "evaluation_days": 184, "hours_per_zone": 4415, "diagnostic_hours_per_zone": 24,
                    "variants": VARIANTS, "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "files": {name: parent.sha(directory / name) for name in sorted(inventory())}}
            parent.atomic_json(directory / "completion.json", done)
            verify_completed(directory, identifier, identity)
            status(status="COMPLETE", phase="complete")
            parent.atomic_bytes(OUTPUT / "index.html", f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={identifier}/report.html"><a href="{identifier}/report.html">NYX · Trois tests DE/NL</a>'.encode())
            return {"status": "COMPLETE", "identity": identifier, "sealed_files": len(done["files"]), "report": str(directory / "report.html")}
        except BaseException as exc:
            status(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED", phase="exception", error=f"{type(exc).__name__}: {exc}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("validate", "run"), default="validate")
    parser.add_argument("--expected-identity")
    args = parser.parse_args(argv)
    ready = prepare()
    identifier, identity = ready[:2]
    if args.expected_identity and args.expected_identity != identifier:
        raise ValueError("Launch identity changed; refusing another experiment")
    if args.action == "validate":
        result = {"identity": identifier, "workdir": str(OUTPUT / identifier), "writes_performed": False,
                  "sources_verified": True, "origins": len(blocks()), "units": len(blocks()) * len(VARIANTS), "protocol": PROTOCOL}
        if (OUTPUT / identifier / "completion.json").exists():
            result["completion_verified"] = verify_completed(OUTPUT / identifier, identifier, identity)["status"]
    else:
        if not args.expected_identity:
            raise ValueError("--expected-identity from read-only validation is required")
        result = run(ready)
    print(json.dumps(parent.value(result), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
