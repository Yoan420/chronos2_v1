"""Offline causal hourly selection of sealed NYX interaction40 / scarcity Test 2."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import json
import os
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

import run_solar_wind_scarcity_ablation as ablation
import run_solar_wind_scarcity_regime as common
from chronos2_hourly.process_lock import exclusive_process_lock

ROOT, DAY, ZONES, TZ = common.ROOT, common.DAY, common.ZONES, common.TZ
ENGINE = "solar_wind_scarcity_hybrid_v1"
PARENT_ID = "397682c4250baa54"
FIRST_DAY = "2026-06-21"
OUTPUT = ROOT / "runs/experiments" / ENGINE / DAY
LABELS = {"nyx": "NYX · interaction ±40", "test2": "Test 2 · toutes les heures", "hybrid": "Hybride · sélection horaire"}
IMPLEMENTATION = ("run_solar_wind_scarcity_hybrid.py", "chronos2_hourly/solar_wind_scarcity_hybrid.py",
                  "docs/solar_wind_scarcity_hybrid.md")
PROTOCOL = {
    "engine": ENGINE, "parent_identity": PARENT_ID, "delivery_day": DAY,
    "selector": "12 NYX-only signal rules, chosen on prior complete 90-day DE/NL OOF panel",
    "nyx_p50_min": [150, 200, 250], "nyx_daily_peak_gap_max": [25, 50],
    "nyx_upside_min": [0, 50], "physical_gate": "own_joint_deficit>=0.5 OR own_residual_stress>=1.0",
    "objective": "minimum pooled MAE", "minimum_pooled_mae_gain": .02,
    "maximum_country_mae_increase": .05, "maximum_country_rmse_increase": 0,
    "minimum_selected_hours": 20, "minimum_selected_days": 5,
    "minimum_selected_hours_per_country": 5, "minimum_selected_days_per_country": 2,
    "calibration_window_days": 90, "fallback": "unaltered NYX if no eligible rule",
    "selection": "all three final quantiles from one source, exact copy, no forced increase",
    "evaluation_start_day": FIRST_DAY, "evaluation_end_day": "2026-09-21",
    "evaluation_days": 93, "hours_per_zone": 2232, "diagnostic_hours_per_zone": 24,
    "historical_origins": 14, "diagnostic_origins": 1, "refit_models": False,
    "threads": 1, "workers": 1, "priority": "BelowNormal", "min_available_memory_gib": 3.5,
    "production_modified": False, "new_source_sync": False, "automatic_promotion": False,
    "post_hoc": True, "independent_validation": False, "pit_publication_evidence_verified": False,
    "event_labels": "read for report only after all prediction checkpoints are sealed",
}


def blocks():
    return [b for b in ablation.blocks() if b[0] >= date.fromisoformat(FIRST_DAY)]


def make_panel(frame):
    panel = frame.loc[frame.variant == "regime_hour_local"].drop(columns="variant").copy()
    panel = panel.rename(columns={**{"residual_kalman__" + q: "nyx__" + q for q in ("q10", "q50", "q90")},
                                  **{"regime__" + q: "test2__" + q for q in ("q10", "q50", "q90")}})
    groups = [panel.zone, panel.index.tz_convert(TZ).date]
    panel["nyx_daily_peak_gap"] = panel.groupby(groups)["nyx__q50"].transform("max") - panel.nyx__q50
    return panel


def prepare():
    ready = ablation.prepare()
    identifier, identity = ready[:2]
    if identifier != PARENT_ID:
        raise ValueError("Frozen ablation identity changed")
    source = ablation.OUTPUT / PARENT_ID
    done = ablation.verify_completed(source, PARENT_ID, identity)
    pins = dict(identity["sources_and_code_sha256"])
    for name, expected in done["files"].items():
        pins[str(source / name)] = expected
    pins[str(source / "completion.json")] = common.sha(source / "completion.json")
    for name in IMPLEMENTATION:
        pins[str(ROOT / name)] = common.sha(ROOT / name)
    columns = ["zone", "variant", "fit_origin", "own_joint_deficit", "own_residual_stress"]
    columns += [prefix + q for prefix in ("residual_kalman__", "regime__") for q in ("q10", "q50", "q90")]
    history = make_panel(pd.read_parquet(source / "backtest.parquet", columns=columns + ["actual"]))
    forecast = make_panel(pd.read_parquet(source / "forecast_diagnostic.parquet", columns=columns))
    for panel, grid in ((history, common.expected_index(ablation.FIRST_DAY, DAY)),
                         (forecast, common.expected_index(DAY, "2026-09-23"))):
        if set(panel.zone) != set(ZONES):
            raise ValueError("Missing source countries")
        for zone in ZONES:
            sub = panel.loc[panel.zone == zone]
            if not sub.index.equals(grid):
                raise ValueError("Source Test 2 timeline changed")
            base = ready[3][zone] if panel is forecast else ready[2][zone]
            for q in ("q10", "q50", "q90"):
                if not np.array_equal(sub["nyx__" + q].to_numpy(), base.loc[sub.index, "residual_kalman__" + q].to_numpy()):
                    raise ValueError("Frozen NYX baseline mismatch")
            for column in ("own_joint_deficit", "own_residual_stress"):
                if not np.array_equal(sub[column].to_numpy(), ready[4][zone].loc[sub.index, column].to_numpy()):
                    raise ValueError("Physical forecast feature mismatch")
            if panel is history and not np.array_equal(sub.actual.to_numpy(), base.loc[sub.index, "actual"].to_numpy()):
                raise ValueError("Historical observations mismatch")
    new_identity = {"protocol": PROTOCOL, "sources_and_code_sha256": pins, "dependencies": identity["dependencies"]}
    new_id = hashlib.sha256(common.encoded(new_identity)).hexdigest()[:16]
    common.verify_pins(pins)
    return new_id, new_identity, history, forecast, ready[6], ready[5]


def block_inputs(origin, stop, is_forecast, history, forecast):
    days = np.asarray(history.index.tz_convert(TZ).date)
    past = history.loc[(days >= origin - timedelta(days=90)) & (days < origin)].copy()
    target = forecast if is_forecast else history
    target_days = np.asarray(target.index.tz_convert(TZ).date)
    current = target.loc[(target_days >= origin) & (target_days < stop)].drop(columns="actual", errors="ignore").copy()
    if not (current.fit_origin == str(origin)).all():
        raise ValueError("Test 2 weekly origin no longer matches inherited schedule")
    return past, current


def evaluate(frame):
    result = {}
    for zone in ZONES:
        sub = frame.loc[frame.zone == zone]
        masks = {"all": np.ones(len(sub), bool), "selected": sub.selected_test2,
                 "not_selected": ~sub.selected_test2, "actual_ge_200": sub.actual >= 200,
                 "actual_ge_300": sub.actual >= 300,
                 "large_positive_nyx_error": sub.actual - sub.nyx__q50 > 50,
                 "low_renewables_without_price_spike": (sub.own_joint_deficit >= .5) & (sub.actual < 200)}
        months = sub.index.tz_convert(TZ).strftime("%Y-%m")
        masks.update({"month/" + month: months == month for month in sorted(set(months))})
        metrics = {name: {variant: common.score(sub.loc[mask], variant + "__") for variant in LABELS}
                   for name, mask in masks.items()}
        for threshold in (200, 300):
            actual = sub.actual >= threshold
            metrics["spikes_" + str(threshold)] = {}
            for variant in LABELS:
                predicted = sub[variant + "__q50"] >= threshold
                metrics["spikes_" + str(threshold)][variant] = {
                    "tp": int((actual & predicted).sum()), "fp": int((~actual & predicted).sum()),
                    "fn": int((actual & ~predicted).sum())}
        gain = abs(sub.nyx__q50 - sub.actual) - abs(sub.hybrid__q50 - sub.actual)
        selected = sub.selected_test2
        metrics["routing"] = {"hours": len(sub), "selected_hours": int(selected.sum()),
            "selected_fraction": float(selected.mean()),
            "selected_days": len(set(sub.loc[selected].index.tz_convert(TZ).date)),
            "improved_hours": int((selected & (gain > 1e-9)).sum()),
            "worsened_hours": int((selected & (gain < -1e-9)).sum()),
            "unchanged_selected_hours": int((selected & (abs(gain) <= 1e-9)).sum()),
            "sum_absolute_error_reduction": float(gain.sum()), "gate_is_calibrated_probability": False}
        result[zone] = metrics
    return result


def render(identifier, metrics, event, policies):
    rows = []
    for zone in ZONES:
        baseline = metrics[zone]["all"]["nyx"]
        for variant, label in LABELS.items():
            m = metrics[zone]["all"][variant]
            rows.append(f"<tr><td>{zone}</td><td>{label}</td><td>{m['mae']:.3f}</td><td>{100*(m['mae']/baseline['mae']-1):+.2f}%</td><td>{m['rmse']:.3f}</td><td>{100*(m['rmse']/baseline['rmse']-1):+.2f}%</td><td>{m['coverage_q10_q90']:.1%}</td></tr>")
    event_rows = []
    for zone in ZONES:
        sub = event.loc[event.zone == zone]
        for stamp, row in sub.iterrows():
            event_rows.append(f"<tr><td>{zone}</td><td>{stamp.tz_convert(TZ).strftime('%H:%M')}</td><td>{row.actual:.2f}</td><td>{row.nyx__q50:.2f}</td><td>{row.test2__q50:.2f}</td><td>{row.hybrid__q50:.2f}</td><td>{'Test 2' if row.selected_test2 else 'NYX'}</td><td>{html.escape(str(row.reason))}</td></tr>")
    routing = ''.join(f"<p>{zone} : {metrics[zone]['routing']['selected_hours']} / 2 232 heures basculées ; {metrics[zone]['routing']['improved_hours']} améliorées, {metrics[zone]['routing']['worsened_hours']} dégradées contre NYX.</p>" for zone in ZONES)
    return f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX · Hybride interaction 40 / Test 2</title>
<style>body{{background:#090c12;color:#e4eaf0;font:15px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:24px}}h1,h2,a{{color:#61e4e7}}table{{width:100%;border-collapse:collapse;background:#101720}}td,th{{padding:10px;border-bottom:1px solid #27333f;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.note{{border-left:3px solid #ffbb63;padding:14px;background:#ffbb6308}}pre{{white-space:pre-wrap;font-size:12px}}.scroll{{overflow-x:auto}}</style>
<h1>NYX · Interaction ±40 par défaut, Test 2 sur alerte horaire</h1>
<p>Les trois quantiles proviennent ensemble de NYX ou de Test 2. Aucun nouveau modèle de prix entraîné, aucune addition de leurs prévisions, aucune hausse forcée.</p>
<p class="note">Comparaison commune : <b>21 juin–21 septembre 2026 · 93 jours / 2 232 heures par pays</b>. Les premiers 90 jours de Test 2 servent au réglage ; ils ne sont pas comptés dans ces scores. 12 règles définies avant cet essai, sélectionnées uniquement sur les 90 jours antérieurs, puis figées par bloc hebdomadaire. Retour NYX sans règle admissible. Recherche rétrospective exploratoire, non validation indépendante ; archives PIT non certifiées, aucune modification de production.</p>
<h2>Performances historiques · €/MWh</h2><div class="scroll"><table><tr><th>Pays</th><th>Méthode</th><th>MAE</th><th>Δ MAE / NYX</th><th>RMSE</th><th>Δ RMSE / NYX</th><th>Couverture P10–P90</th></tr>{''.join(rows)}</table></div>
<h2>Comportement de la bascule</h2>{routing}
<p>Le signal combine P50 NYX élevé, proximité du maximum de la courbe NYX prévue du même jour, risque haussier P90–P50 et tension physique prévue. Ce n'est pas une probabilité calibrée. Les gardes sur les erreurs passées ne garantissent pas les résultats futurs.</p>
<h2>22 septembre · diagnostic post-hoc, non test indépendant</h2><div class="scroll"><table><tr><th>Pays</th><th>Heure Paris</th><th>Observé</th><th>NYX P50</th><th>Test 2 P50</th><th>Hybride P50</th><th>Retenu</th><th>Motif</th></tr>{''.join(event_rows)}</table></div>
<details><summary>Règles choisies et rejets de chaque origine</summary><pre>{html.escape(common.encoded(policies).decode())}</pre></details>
<details><summary>Métriques complètes : heures sélectionnées, pics, fausses alertes, mois, quantiles</summary><pre>{html.escape(common.encoded(metrics).decode())}</pre></details>
<p><a href="metrics.json">Métriques</a> · <a href="gate_audits.json">Sélecteurs</a> · <a href="experiment.json">Protocole et sources</a> · <a href="completion.json">Scellement SHA</a></p><footer>Identité {identifier} · aucun déploiement automatique.</footer></html>'''


def inventory():
    files = {"experiment.json", "feature_audit.json", "backtest.parquet", "forecast_diagnostic.parquet",
             "metrics.json", "gate_audits.json", "report.html"}
    for origin, _, _ in blocks():
        files.update(f"checkpoints/{origin}{suffix}" for suffix in (".parquet", ".policy.json", ".json"))
    return files


def verify_frame(frame, grid, require_actual=True):
    if set(frame.zone) != set(ZONES) or frame.selected_test2.dtype != bool:
        raise ValueError("Invalid country or routing flags")
    for zone in ZONES:
        sub = frame.loc[frame.zone == zone]
        if not sub.index.equals(grid):
            raise ValueError("Invalid paired hourly grid")
        if require_actual and not np.isfinite(sub.actual.to_numpy(float)).all():
            raise ValueError("Nonfinite evaluation labels")
    for variant in LABELS:
        q = frame[[variant + "__" + q for q in ("q10", "q50", "q90")]].to_numpy(float)
        if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
            raise ValueError("Nonfinite or crossing quantiles")
    for q in ("q10", "q50", "q90"):
        expected = np.where(frame.selected_test2, frame["test2__" + q], frame["nyx__" + q])
        if not np.array_equal(frame["hybrid__" + q].to_numpy(), expected):
            raise ValueError("Hybrid must exactly select all three quantiles from one source")


def verify_completed(directory, identifier, identity):
    done = common.read_json(directory / "completion.json")
    if (done.get("status"), done.get("identity"), done.get("evaluation_days"), done.get("hours_per_zone"),
        done.get("diagnostic_hours_per_zone")) != ("COMPLETE", identifier, 93, 2232, 24):
        raise ValueError("Completion identity/support mismatch")
    if set(done["files"]) != inventory() or common.read_json(directory / "experiment.json") != common.value(identity):
        raise ValueError("Completion inventory/experiment mismatch")
    for name, expected in done["files"].items():
        if common.sha(directory / name) != expected:
            raise ValueError(f"Completion SHA mismatch: {name}")
    verify_frame(pd.read_parquet(directory / "backtest.parquet"), common.expected_index(FIRST_DAY, DAY))
    verify_frame(pd.read_parquet(directory / "forecast_diagnostic.parquet"), common.expected_index(DAY, "2026-09-23"))
    common.verify_pins(identity["sources_and_code_sha256"])
    return done


def run(ready):
    from chronos2_hourly.solar_wind_scarcity_hybrid import select_rule, apply_rule
    identifier, identity, history, forecast, observations, feature_audit = ready
    directory = OUTPUT / identifier
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(OUTPUT / "experiment.lock"):
        if (directory / "completion.json").exists():
            return verify_completed(directory, identifier, identity)
        if (directory / "experiment.json").exists() and common.read_json(directory / "experiment.json") != common.value(identity):
            raise ValueError("Existing workdir identity changed")
        common.atomic_json(directory / "experiment.json", identity)
        common.atomic_json(directory / "feature_audit.json", feature_audit)
        checkpoint = directory / "checkpoints"
        checkpoint.mkdir(exist_ok=True)
        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        started = time.monotonic()
        state = {"engine": ENGINE, "identity": identifier, "status": "RUNNING", "pid": os.getpid(),
                 "process_create_time": process.create_time(), "command": process.cmdline(),
                 "started_utc": datetime.now(timezone.utc).isoformat(), "threads": 1, "workers": 1,
                 "total_blocks": len(blocks()), "completed_blocks": 0}
        def status(**updates):
            state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
            n = state["completed_blocks"]
            state["eta_seconds"] = (len(blocks())-n) * ((time.monotonic()-started)/n if n else 5)
            state["eta_basis"] = "elapsed_per_block_rough" if n else "initial_5sec_per_block_rough"
            common.atomic_json(directory / "status.json", state)
            common.atomic_json(OUTPUT / "latest_DE_NL.json", state | {"workdir": str(directory)})
        frames, policies = [], []
        try:
            status(phase="ready")
            for origin, stop, is_forecast in blocks():
                stem = checkpoint / str(origin)
                data_path, policy_path, receipt = (stem.with_suffix(s) for s in (".parquet", ".policy.json", ".json"))
                past, current = block_inputs(origin, stop, is_forecast, history, forecast)
                if receipt.exists():
                    saved = common.read_json(receipt)
                    if saved.get("identity") != identifier or saved.get("origin") != str(origin):
                        raise ValueError("Checkpoint identity mismatch")
                    if set(saved["files"]) != {data_path.name, policy_path.name}:
                        raise ValueError("Checkpoint inventory mismatch")
                    for path in (data_path, policy_path):
                        if common.sha(path) != saved["files"][path.name]:
                            raise ValueError("Checkpoint SHA mismatch")
                    output, policy = pd.read_parquet(data_path), common.read_json(policy_path)
                    replay = apply_rule(current, policy)
                    pd.testing.assert_frame_equal(output.drop(columns=["selector_origin", "is_forecast"]), replay)
                else:
                    if data_path.exists() or policy_path.exists():
                        raise ValueError("Unsealed checkpoint retained for inspection")
                    while psutil.virtual_memory().available < 3.5 * 2**30:
                        status(status="WAITING_MEMORY", phase="resource_guard", next_origin=str(origin))
                        time.sleep(10)
                    common.verify_pins(identity["sources_and_code_sha256"])
                    status(status="RUNNING", phase="select_past_rule", next_origin=str(origin))
                    with threadpool_limits(limits=1):
                        policy = select_rule(past, origin_day=origin)
                        output = apply_rule(current, policy)
                    output["selector_origin"] = str(origin)
                    output["is_forecast"] = is_forecast
                    verify_frame(output, common.expected_index(str(origin), str(stop)), require_actual=False)
                    common.verify_pins(identity["sources_and_code_sha256"])
                    common.atomic_parquet(data_path, output)
                    common.atomic_json(policy_path, policy)
                    common.atomic_json(receipt, {"identity": identifier, "origin": str(origin),
                        "files": {p.name: common.sha(p) for p in (data_path, policy_path)}})
                frames.append(output)
                policies.append(policy)
                status(completed_blocks=state["completed_blocks"]+1, phase="checkpoint_saved")
                print(json.dumps({"identity": identifier, "origin": str(origin), "mode": policy["mode"],
                    "selected_hours": int(output.selected_test2.sum()), "completed_blocks": state["completed_blocks"],
                    "total_blocks": len(blocks()), "eta_seconds": state["eta_seconds"]}), flush=True)
            combined = pd.concat(frames)
            backtest = combined.loc[~combined.is_forecast].copy()
            diagnostic = combined.loc[combined.is_forecast].copy()
            # Labels are joined for evaluation only after all routed predictions are sealed.
            for frame in (backtest, diagnostic):
                frame["actual"] = np.nan
                for zone in ZONES:
                    observed = (common.indexed(pd.read_parquet(observations[zone])) if frame is diagnostic
                                else history.loc[history.zone == zone])
                    mask = frame.zone == zone
                    frame.loc[mask, "actual"] = observed.actual.reindex(frame.loc[mask].index).to_numpy(float)
            verify_frame(backtest, common.expected_index(FIRST_DAY, DAY))
            verify_frame(diagnostic, common.expected_index(DAY, "2026-09-23"))
            metrics = evaluate(backtest)
            common.atomic_parquet(directory / "backtest.parquet", backtest)
            common.atomic_parquet(directory / "forecast_diagnostic.parquet", diagnostic)
            common.atomic_json(directory / "metrics.json", metrics)
            common.atomic_json(directory / "gate_audits.json", policies)
            common.atomic_bytes(directory / "report.html", render(identifier, metrics, diagnostic, policies).encode("utf-8"))
            common.verify_pins(identity["sources_and_code_sha256"])
            done = {"status": "COMPLETE", "identity": identifier, "evaluation_days": 93, "hours_per_zone": 2232,
                    "diagnostic_hours_per_zone": 24, "production_modified": False,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "files": {name: common.sha(directory / name) for name in sorted(inventory())}}
            common.atomic_json(directory / "completion.json", done)
            verify_completed(directory, identifier, identity)
            common.atomic_bytes(OUTPUT / "index.html", f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={identifier}/report.html"><a href="{identifier}/report.html">NYX · Hybride DE/NL</a>'.encode())
            status(status="COMPLETE", phase="complete")
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
    if args.expected_identity and identifier != args.expected_identity:
        raise ValueError("Expected identity differs; refusing a changed experiment")
    if args.action == "validate":
        result = {"identity": identifier, "writes_performed": False, "sources_verified": True,
                  "workdir": str(OUTPUT / identifier), "blocks": len(blocks()), "protocol": PROTOCOL}
        if (OUTPUT / identifier / "completion.json").exists():
            result["completion_verified"] = verify_completed(OUTPUT / identifier, identifier, identity)["status"]
    else:
        if not args.expected_identity:
            raise ValueError("Read-only validation and explicit --expected-identity required")
        result = run(ready)
    print(json.dumps(common.value(result), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
