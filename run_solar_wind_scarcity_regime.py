"""Sealed, offline DE/NL scarcity-regime experiment after interaction40.

No live source, production model, configuration, or old experiment is written.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from chronos2_hourly.process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parent
ENGINE = "solar_wind_scarcity_regime_v1"
DAY = "2026-09-22"
ZONES = ("DE", "NL")
TZ = "Europe/Berlin"
BASE_IDS = {"DE": "45ec8314c37a2fe5", "NL": "b1da388bb4df6c63"}
CORRECTOR_IDS = {"DE": "48d7c14fd8bc2e4f", "NL": "8f246d6bbd54b511"}
OUTPUT = ROOT / "runs/experiments" / ENGINE / DAY
IMPLEMENTATION = (
    "run_solar_wind_scarcity_regime.py",
    "chronos2_hourly/solar_wind_scarcity_regime.py",
    "docs/solar_wind_scarcity_regime.md",
    "chronos2_hourly/process_lock.py",
    "chronos2_hourly/__init__.py",
    "chronos2_hourly/hourly_contract.py",
)
PROTOCOL = {
    "engine": ENGINE, "delivery_day": DAY, "zones": ZONES,
    "baseline": "sealed interaction_40 final residual_kalman__ quantiles",
    "target": "actual - baseline final q50",
    "position": "conditional residual distribution after frozen final baseline",
    "spike_regime_residual_gt_eur_mwh": 50.0,
    "warmup_days": 90, "train_window_days": 365, "refit_every_days": 7,
    "forecast_day_fresh_fit": True, "iterations": 120, "threads": 1,
    "seed": 20260923, "min_available_memory_gib": 3.5,
    "memory_reserve_gib": 3.0, "process_priority": "BelowNormal",
    "expected_backtest_days": 275, "expected_backtest_hours_per_zone": 6599,
    "post_hoc": True, "prospective_validation": False,
    "production_modified": False, "pit_publication_evidence_verified": False,
    "event_observations": "reporting only; never supplied to model or features",
    "sparse_regime": "explicit empirical fallback, no forced premium",
    "quantiles": "invert mixture CDF; add residual quantiles to baseline q50",
    "amplitude_cap": None, "automatic_promotion": False,
}


def value(obj):
    if isinstance(obj, dict):
        return {str(k): value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [value(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (date, datetime, pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def encoded(obj):
    return json.dumps(value(obj), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, indent=2).encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_bytes(path, encoded(obj))


def atomic_parquet(path, frame):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def verify_pins(pins):
    for name, expected in pins.items():
        if sha(name) != expected:
            raise ValueError(f"SHA mismatch: {name}")


def seal(directory, filename, pins):
    document = read_json(directory / filename)
    for name, expected in document["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"Invalid sealed artifact: {path}")
        pins[str(path)] = expected
    pins[str((directory / filename).resolve())] = sha(directory / filename)
    return document


def indexed(frame):
    result = frame.copy()
    key = next((k for k in ("delivery_start_utc", "timestamp") if k in result), None)
    stamps = pd.DatetimeIndex(result[key] if key else result.index)
    if stamps.tz is None:
        raise ValueError("Timezone-aware inputs required")
    result.index = stamps.tz_convert("UTC")
    result.index.name = "delivery_start_utc"
    if result.index.has_duplicates or result.index.hasnans or not result.index.is_monotonic_increasing:
        raise ValueError("Unordered or duplicated input timestamps")
    return result


def expected_index(first, last):
    return pd.date_range(pd.Timestamp(first, tz=TZ), pd.Timestamp(last, tz=TZ),
                         freq="h", inclusive="left").tz_convert("UTC")


def prepare():
    from chronos2_hourly.solar_wind_scarcity_regime import build_features
    pins, history, forecast, covariates, observations = {}, {}, {}, {}, {}
    for zone in ZONES:
        work = ROOT / "runs/experiments/solar_wind_corrector_reuse_v1" / DAY / zone.lower() / CORRECTOR_IDS[zone]
        done = seal(work, "completion.json", pins)
        if done["identity"] != CORRECTOR_IDS[zone] or done.get("annual_complete") is not True:
            raise ValueError("Incomplete or wrong interaction40 source")
        source = ROOT / "runs/experiments/solar_wind_v1" / DAY / zone.lower() / BASE_IDS[zone]
        frozen = source / "report_only/frozen_result"
        expected = read_json(work / "experiment.json")["original_identity"]["baseline_manifest_sha256"]
        if sha(frozen / "manifest.json") != expected:
            raise ValueError("Baseline manifest does not match interaction40")
        seal(frozen, "manifest.json", pins)
        covariates[zone] = indexed(pd.read_parquet(frozen / "covariates.parquet"))
        for name, target, grid in (
            ("backtest", history, expected_index("2025-09-22", "2026-09-22")),
            ("forecast", forecast, expected_index(DAY, "2026-09-23")),
        ):
            frame = indexed(pd.read_parquet(work / "interaction_40" / (name + ".parquet")))
            if not frame.index.equals(grid):
                raise ValueError(f"Incorrect {zone} {name} grid")
            columns = ["residual_kalman__" + q for q in ("q10", "q50", "q90")]
            if name == "backtest":
                columns.append("actual")
            elif "actual" in frame and frame.actual.notna().any():
                raise ValueError("Forecast labels cannot enter model inputs")
            frame = frame[columns].copy()
            if not np.isfinite(frame.to_numpy(float)).all():
                raise ValueError("Nonfinite baseline or labels")
            if (np.diff(frame.iloc[:, :3].to_numpy(float), axis=1) < 0).any():
                raise ValueError("Crossed baseline quantiles")
            target[zone] = frame
        observations[zone] = source / "reference/reporting/inputs/observed_latest.parquet"
        pins[str(observations[zone])] = sha(observations[zone])
    features, audit = build_features(covariates)
    for zone in ZONES:
        baseline = pd.concat([history[zone], forecast[zone]])
        features[zone] = features[zone].loc[baseline.index].copy()
        features[zone]["baseline_p50"] = baseline["residual_kalman__q50"]
        if not np.isfinite(features[zone].to_numpy(float)).all():
            raise ValueError("Incomplete feature coverage after historical normalization")
    for name in IMPLEMENTATION:
        pins[str(ROOT / name)] = sha(ROOT / name)
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "pyarrow", "psutil", "filelock", "threadpoolctl")}
    identity = {"protocol": PROTOCOL, "sources_and_code_sha256": pins,
                "source_ids": CORRECTOR_IDS, "dependencies": dependencies,
                "feature_columns": list(features["DE"].columns)}
    identifier = hashlib.sha256(encoded(identity)).hexdigest()[:16]
    verify_pins(pins)
    return identifier, identity, history, forecast, features, audit, observations


def blocks():
    start = date(2025, 9, 22) + timedelta(days=PROTOCOL["warmup_days"])
    end = date.fromisoformat(DAY)
    result = []
    while start < end:
        stop = min(start + timedelta(days=7), end)
        result.append((start, stop, False))
        start = stop
    result.append((end, end + timedelta(days=1), True))
    return result


def score(frame, prefix):
    if frame.empty:
        return {"hours": 0}
    y = frame.actual.to_numpy(float)
    error = frame[prefix + "q50"].to_numpy(float) - y
    answer = {"hours": len(frame), "mae": float(np.mean(abs(error))),
              "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}
    for q, alpha in (("q10", .1), ("q50", .5), ("q90", .9)):
        r = y - frame[prefix + q].to_numpy(float)
        answer["pinball_" + q] = float(np.maximum(alpha * r, (alpha - 1) * r).mean())
        answer["observed_fraction_below_" + q] = float((r <= 0).mean())
    answer["coverage_q10_q90"] = float(((y >= frame[prefix + "q10"]) & (y <= frame[prefix + "q90"])).mean())
    return answer


def evaluate(frame):
    result = {}
    for zone in ZONES:
        sub = frame.loc[frame.zone == zone]
        masks = {"all": np.ones(len(sub), bool), "actual_ge_200": sub.actual >= 200,
                 "actual_ge_300": sub.actual >= 300,
                 "large_positive_baseline_error": sub.actual - sub.residual_kalman__q50 > 50,
                 "low_renewables": sub.own_joint_deficit >= .5,
                 "low_renewables_without_price_spike": (sub.own_joint_deficit >= .5) & (sub.actual < 200),
                 "low_renewables_high_residual_stress": (sub.own_joint_deficit >= .5) & (sub.own_residual_stress >= 1)}
        months = sub.index.tz_convert(TZ).strftime("%Y-%m")
        masks.update({"month/" + month: months == month for month in sorted(set(months))})
        # All slice definitions fixed before viewing candidate outcomes.
        result[zone] = {name: {variant: score(sub.loc[mask], prefix) for variant, prefix in
                                (("interaction_40", "residual_kalman__"), ("scarcity_regime", "regime__"))}
                        for name, mask in masks.items()}
        for threshold in (200, 300):
            actual = sub.actual >= threshold
            result[zone]["spikes_" + str(threshold)] = {}
            for prefix in ("residual_kalman__", "regime__"):
                pred = sub[prefix + "q50"] >= threshold
                tp, fp, fn = int((pred & actual).sum()), int((pred & ~actual).sum()), int((~pred & actual).sum())
                result[zone]["spikes_" + str(threshold)][prefix] = {"tp": tp, "fp": fp, "fn": fn}
        event = sub.actual - sub.residual_kalman__q50 > 50
        result[zone]["gate_brier"] = float(np.mean((sub.spike_probability - event.astype(float)) ** 2))
        result[zone]["support"] = {"hours": len(sub), "days": len(set(sub.index.tz_convert(TZ).date))}
    return result


def report(identifier, metrics, event):
    rows = []
    for zone in ZONES:
        for variant, scores in metrics[zone]["all"].items():
            rows.append(f"<tr><td>{zone}</td><td>{variant}</td><td>{scores['mae']:.3f}</td><td>{scores['rmse']:.3f}</td><td>{scores['bias']:.3f}</td><td>{scores['coverage_q10_q90']:.1%}</td></tr>")
    event_rows = []
    for stamp, row in event.iterrows():
        if stamp.tz_convert(TZ).hour in (18, 19, 20):
            observed = "—" if pd.isna(row.get("actual", np.nan)) else f"{row.actual:.2f}"
            event_rows.append(f"<tr><td>{row.zone}</td><td>{stamp.tz_convert(TZ).strftime('%H:%M')}</td><td>{observed}</td><td>{row.residual_kalman__q50:.2f}</td><td>{row.regime__q50:.2f}</td><td>{row.regime__q90:.2f}</td><td>{row.spike_probability:.1%}</td></tr>")
    return f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX · Régime de rareté DE/NL</title>
<style>body{{background:#090c12;color:#e4eaf0;font:15px/1.55 system-ui;max-width:1250px;margin:40px auto;padding:24px}}h1,h2,a{{color:#61e4e7}}table{{width:100%;border-collapse:collapse;background:#101720}}td,th{{padding:12px;border-bottom:1px solid #27333f;text-align:right}}td:first-child{{text-align:left}}.note{{border-left:3px solid #ffbb63;padding:15px;background:#ffbb6308}}pre{{white-space:pre-wrap;font-size:12px}}</style>
<h1>NYX · Correcteur spécialisé rareté · DE / NL</h1><p>Interaction 40 + Kalman gelés. Détecteur commun et distributions résiduelles conditionnelles ; médiane de la CDF mélangée, sans plafond de correction ±40.</p>
<p class="note">Recherche rétrospective post-hoc, aucune promotion en production. Archives de publication PIT non certifiées et substitutions historiques NL héritées. 90 jours de calibration exclus ; 275 jours / 6 599 heures par pays évalués. Le 22 septembre a inspiré l’hypothèse : diagnostic distinct, jamais validation indépendante ni cible d’apprentissage.</p>
<h2>Comparaison sur les mêmes heures historiques</h2><table><tr><th>Pays</th><th>Variante</th><th>MAE</th><th>RMSE</th><th>Biais</th><th>Couverture 10–90</th></tr>{''.join(rows)}</table>
<h2>Diagnostic du 22 septembre · heures locales</h2><table><tr><th>Pays</th><th>Heure</th><th>Observé</th><th>Interaction 40 P50</th><th>Candidat P50</th><th>Candidat P90</th><th>P(erreur &gt;50)</th></tr>{''.join(event_rows)}</table>
<p>La probabilité concerne une sous-estimation de plus de 50 €/MWh de la référence, pas directement un prix supérieur à 200 ou 300. Les quantiles sont ordonnés par construction ; leur calibration doit être évaluée, pas présumée. Les amplitudes rares peuvent employer une distribution empirique faute d’exemples, tracée dans les audits.</p>
<details><summary>Spikes, fausses alertes et calibration</summary><pre>{html.escape(encoded(metrics).decode())}</pre></details>
<p><a href="metrics.json">Métriques</a> · <a href="experiment.json">Protocole et empreintes</a> · <a href="fit_audits.json">Audits des apprentissages</a> · <a href="completion.json">Inventaire scellé</a></p><footer>Identité {identifier} · production inchangée · pas de validation prospective</footer></html>'''


def verify_completed(directory, identifier, identity):
    done = read_json(directory / "completion.json")
    if done.get("status") != "COMPLETE" or done.get("identity") != identifier:
        raise ValueError("Invalid completion")
    if (done.get("evaluation_days"), done.get("hours_per_zone"),
            done.get("diagnostic_forecast_hours_per_zone")) != (275, 6599, 24):
        raise ValueError("Incomplete declared evaluation support")
    expected_files = {"experiment.json", "feature_audit.json", "backtest.parquet",
                      "forecast_diagnostic.parquet", "metrics.json", "fit_audits.json", "report.html"}
    for origin, _, _ in blocks():
        expected_files.update(f"checkpoints/{origin}{suffix}" for suffix in (".json", ".audit.json", ".parquet"))
    if set(done["files"]) != expected_files:
        raise ValueError("Completion must seal the exact expected result inventory")
    if read_json(directory / "experiment.json") != value(identity):
        raise ValueError("Changed identity")
    for name, expected in done["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"Invalid completion file: {name}")
    for name, grid in (("backtest.parquet", expected_index("2025-12-21", DAY)),
                       ("forecast_diagnostic.parquet", expected_index(DAY, "2026-09-23"))):
        frame = pd.read_parquet(directory / name)
        if set(frame.zone) != set(ZONES):
            raise ValueError("Missing or unexpected completed country")
        for zone in ZONES:
            selected = frame.loc[frame.zone == zone]
            if not selected.index.equals(grid):
                raise ValueError("Completed hourly grid mismatch")
            columns = [prefix + q for prefix in ("regime__", "residual_kalman__") for q in ("q10", "q50", "q90")]
            if not np.isfinite(selected[columns].to_numpy(float)).all():
                raise ValueError("Nonfinite completed quantiles")
            for prefix in ("regime__", "residual_kalman__"):
                if (np.diff(selected[[prefix + q for q in ("q10", "q50", "q90")]].to_numpy(float), axis=1) < 0).any():
                    raise ValueError("Crossed completed quantiles")
    verify_pins(identity["sources_and_code_sha256"])
    return done


def run(prepared):
    from chronos2_hourly.solar_wind_scarcity_regime import fit_predict
    identifier, identity, history, forecast, features, feature_audit, observations = prepared
    directory = OUTPUT / identifier
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(OUTPUT / "experiment.lock"):
        if (directory / "completion.json").exists():
            return verify_completed(directory, identifier, identity)
        if (directory / "experiment.json").exists() and read_json(directory / "experiment.json") != value(identity):
            raise ValueError("Existing workdir identity mismatch")
        atomic_json(directory / "experiment.json", identity)
        atomic_json(directory / "feature_audit.json", feature_audit)
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        started = time.monotonic()
        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        state = {"engine": ENGINE, "identity": identifier, "status": "RUNNING",
                 "pid": os.getpid(), "process_create_time": process.create_time(),
                 "started_utc": datetime.now(timezone.utc).isoformat(), "threads": 1,
                 "workers": 1, "total_blocks": len(blocks()), "completed_blocks": 0,
                 "report": str(directory / "report.html"), "command": process.cmdline()}
        def status(**updates):
            state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
            done = state["completed_blocks"]
            elapsed = time.monotonic() - started
            state["eta_seconds"] = (state["total_blocks"] - done) * (elapsed / done if done else 30)
            state["eta_estimate_basis"] = "elapsed_per_finished_block" if done else "initial_30_seconds_per_block_rough"
            atomic_json(directory / "status.json", state)
            atomic_json(OUTPUT / "latest_DE_NL.json", state | {"workdir": str(directory)})
        frames, audits = [], []
        try:
            status(phase="features_ready")
            for number, (origin, stop, is_forecast) in enumerate(blocks()):
                stem = checkpoints / origin.isoformat()
                receipt = stem.with_suffix(".json")
                data_path = stem.with_suffix(".parquet")
                audit_path = stem.with_suffix(".audit.json")
                if receipt.exists():
                    saved = read_json(receipt)
                    if saved.get("identity") != identifier or saved.get("origin") != str(origin):
                        raise ValueError("Checkpoint identity mismatch")
                    for path in (data_path, audit_path):
                        if sha(path) != saved["files"][path.name]:
                            raise ValueError("Checkpoint SHA mismatch")
                    output, audit = pd.read_parquet(data_path), read_json(audit_path)
                else:
                    # Partial files are preserved on interrupted publication; never silently overwritten.
                    if data_path.exists() or audit_path.exists():
                        raise ValueError(f"Unsealed partial checkpoint requires review: {origin}")
                    while psutil.virtual_memory().available < PROTOCOL["min_available_memory_gib"] * 2**30:
                        status(status="WAITING_MEMORY", phase="resource_guard", next_origin=str(origin))
                        time.sleep(10)
                    verify_pins(identity["sources_and_code_sha256"])
                    status(status="RUNNING", phase="fit", next_origin=str(origin))
                    train_x, train_y, train_days, test_x, test_parts = [], [], [], [], []
                    for zone in ZONES:
                        hist = history[zone]
                        days = hist.index.tz_convert(TZ).date
                        use = (days >= origin - timedelta(days=365)) & (days < origin)
                        selected = hist.loc[use]
                        train_x.append(features[zone].loc[selected.index])
                        train_y.append((selected.actual - selected.residual_kalman__q50).to_numpy(float))
                        train_days.extend(selected.index.tz_convert(TZ).date)
                        target = forecast[zone] if is_forecast else hist
                        target_days = target.index.tz_convert(TZ).date
                        part = target.loc[(target_days >= origin) & (target_days < stop)].copy()
                        part["zone"] = zone
                        test_parts.append(part)
                        test_x.append(features[zone].loc[part.index])
                    with threadpool_limits(limits=1):
                        residual_quantiles, probability, audit = fit_predict(
                            pd.concat(train_x), np.concatenate(train_y), pd.concat(test_x), train_days,
                            origin_day=origin, threads=1, iterations=PROTOCOL["iterations"], seed=PROTOCOL["seed"])
                    output = pd.concat(test_parts)
                    if residual_quantiles.shape != (len(output), 3) or probability.shape != (len(output),):
                        raise ValueError("Incorrect model output shape")
                    if not np.isfinite(residual_quantiles).all() or not np.isfinite(probability).all():
                        raise ValueError("Nonfinite model output")
                    if (np.diff(residual_quantiles, axis=1) < 0).any() or ((probability < 0) | (probability > 1)).any():
                        raise ValueError("Invalid quantiles/probabilities")
                    for column, q in enumerate(("q10", "q50", "q90")):
                        output["regime__" + q] = output.residual_kalman__q50.to_numpy(float) + residual_quantiles[:, column]
                    output["spike_probability"] = probability
                    diagnostic_features = pd.concat(test_x)
                    for feature in ("own_joint_deficit", "own_residual_stress", "other_residual_stress",
                                    "own_wind_gw", "own_solar_gw", "own_residual_load_gw"):
                        output[feature] = diagnostic_features[feature].to_numpy(float)
                    output["fit_origin"] = str(origin)
                    output["is_forecast"] = is_forecast
                    audit.update(origin=str(origin), stop=str(stop), is_forecast=is_forecast,
                                 train_hours_total=len(train_days), test_hours_total=len(output))
                    verify_pins(identity["sources_and_code_sha256"])
                    atomic_parquet(data_path, output)
                    atomic_json(audit_path, audit)
                    atomic_json(receipt, {"identity": identifier, "origin": str(origin),
                                         "files": {p.name: sha(p) for p in (data_path, audit_path)}})
                frames.append(output)
                audits.append(audit)
                status(completed_blocks=number + 1, phase="checkpoint_saved")
                print(json.dumps({"identity": identifier, "completed_blocks": number + 1,
                                  "total_blocks": len(blocks()), "origin": str(origin),
                                  "eta_seconds": state["eta_seconds"]}), flush=True)
            combined = pd.concat(frames)
            backtest = combined.loc[~combined.is_forecast].copy()
            event = combined.loc[combined.is_forecast].copy()
            for zone in ZONES:
                rows = backtest.loc[backtest.zone == zone]
                expected = expected_index("2025-12-21", DAY)
                if not rows.index.equals(expected) or len(rows) != 6599:
                    raise ValueError("Historical output grid mismatch")
                if not event.loc[event.zone == zone].index.equals(expected_index(DAY, "2026-09-23")):
                    raise ValueError("Diagnostic forecast grid mismatch")
            metrics = evaluate(backtest)
            # Only now read event labels, when all fitted predictions have been committed.
            for zone in ZONES:
                observed = indexed(pd.read_parquet(observations[zone]))
                key = next((k for k in ("actual", "price_eur_mwh", "value") if k in observed), None)
                if key is None:
                    raise ValueError("Unknown diagnostic observation schema")
                take = event.zone == zone
                event.loc[take, "actual"] = pd.to_numeric(observed[key], errors="raise").reindex(event.loc[take].index).to_numpy()
            atomic_parquet(directory / "backtest.parquet", backtest)
            atomic_parquet(directory / "forecast_diagnostic.parquet", event)
            atomic_json(directory / "metrics.json", metrics)
            atomic_json(directory / "fit_audits.json", audits)
            atomic_bytes(directory / "report.html", report(identifier, metrics, event).encode("utf-8"))
            verify_pins(identity["sources_and_code_sha256"])
            inventory = ["experiment.json", "feature_audit.json", "backtest.parquet", "forecast_diagnostic.parquet",
                         "metrics.json", "fit_audits.json", "report.html"]
            inventory += [p.relative_to(directory).as_posix() for p in sorted(checkpoints.iterdir()) if p.suffix in (".json", ".parquet")]
            done = {"status": "COMPLETE", "identity": identifier, "production_modified": False,
                    "evaluation_days": 275, "hours_per_zone": 6599, "diagnostic_forecast_hours_per_zone": 24,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "files": {name: sha(directory / name) for name in inventory}}
            atomic_json(directory / "completion.json", done)
            verify_completed(directory, identifier, identity)
            status(status="COMPLETE", phase="complete", eta_seconds=0)
            atomic_bytes(OUTPUT / "index.html", f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={identifier}/report.html"><a href="{identifier}/report.html">NYX · résultats DE/NL</a>'.encode())
            return done
        except BaseException as exc:
            status(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                   phase="exception", error=f"{type(exc).__name__}: {exc}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("validate", "run"), default="validate")
    parser.add_argument("--expected-identity")
    args = parser.parse_args(argv)
    prepared = prepare()
    identifier, identity = prepared[:2]
    if args.expected_identity and args.expected_identity != identifier:
        raise ValueError("Pinned launch identity changed; refusing another experiment")
    if args.action == "validate":
        result = {"identity": identifier, "workdir": str(OUTPUT / identifier), "writes_performed": False,
                  "blocks": len(blocks()), "sources_verified": True, "protocol": PROTOCOL}
        if (OUTPUT / identifier / "completion.json").exists():
            result["completion_verified"] = verify_completed(OUTPUT / identifier, identifier, identity)["status"]
    else:
        if not args.expected_identity:
            raise ValueError("--expected-identity from read-only validation is required to run")
        result = run(prepared)
    print(json.dumps(value(result), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
