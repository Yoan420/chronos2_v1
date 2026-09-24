"""Provisional, read-only scoring of the completed native NYX workers.

Writes analysis beside this script, never into the sealed experiment.
No model is fitted, selected or modified here.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nyx_quarterhour.evaluation import aggregate_quarterhour_predictions, evaluate_predictions

RUN = ROOT / "runs/experiments/nyx_fullquarterhour_v1/20260916T111425Z_dff7f605"
ZONES = ["BE", "DE", "FR", "NL"]
NAMES = {
    "nyx": "NYX actuel (archivé)",
    "raw_hourly": "Chronos seul · horaire",
    "raw_quarterhour": "Chronos seul · 15 min",
    "residual_quarterhour": "Chronos + CatBoost · 15 min",
    "full_quarterhour": "NYX complet · 15 min",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, manifest):
    for name, expected in manifest.items():
        assert digest(directory / name) == expected, name


def main():
    lock = json.loads((RUN / "full_protocol.json").read_text())
    cfg = lock["protocol"]["config"]
    verify(ROOT, lock["protocol"]["code_sha256"])
    verify(RUN, json.loads((RUN / "inputs.manifest.json").read_text())["files"])
    raw_manifest = RUN / "raw_outputs.manifest.json"
    assert digest(raw_manifest) == lock["protocol"]["raw_manifest_sha256"]
    verify(RUN, json.loads(raw_manifest.read_text())["files"])
    for zone in ZONES:
        directory = RUN / "workers" / f"{zone}_15min"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["protocol_identity"] == lock["identity"]
        verify(directory, manifest["files"])

    forecasts = {}
    for name, frequency, stage in [
        ("raw_hourly", "h", "raw"), ("raw_quarterhour", "15min", "raw"),
        ("residual_quarterhour", "15min", "residual"), ("full_quarterhour", "15min", "full"),
    ]:
        parts = []
        raw = pd.read_parquet(RUN / f"raw_{frequency}.parquet") if stage == "raw" else None
        for zone in ZONES:
            if stage == "raw":
                frame = raw.loc[raw.zone.eq(zone)].set_index("timestamp_utc")
                column = "q50"
            else:
                filename = "kalman.parquet" if stage == "full" else "residual.parquet"
                frame = pd.read_parquet(RUN / "workers" / f"{zone}_{frequency}" / filename)
                column = "residual_kalman__q50" if stage == "full" else "q50"
            days = frame.index.tz_convert("Europe/Paris").date
            frame = frame.loc[(days >= pd.Timestamp(cfg["start_day"]).date()) &
                              (days <= pd.Timestamp(cfg["end_day"]).date())]
            parts.append(pd.DataFrame({"timestamp_utc": frame.index, "zone": zone,
                                       "prediction": frame[column].to_numpy(float)}))
        frame = pd.concat(parts, ignore_index=True)
        forecasts[name] = (aggregate_quarterhour_predictions(
            frame, start_day=cfg["start_day"], end_day=cfg["end_day"]
        ) if frequency == "15min" else frame)

    baseline = pd.read_parquet(RUN / "baseline.parquet")
    training = baseline.loc[baseline.timestamp_utc < pd.Timestamp("2026-03-14", tz="Europe/Paris").tz_convert("UTC")]
    q95 = {z: float(training.loc[training.zone.eq(z), "training_actual"].quantile(.95)) for z in ZONES}
    result = evaluate_predictions(
        pd.read_parquet(RUN / "scoring_baseline.parquet"), forecasts, q95_thresholds=q95,
        candidate_family="full_quarterhour", matched_control_family="matched_full_hourly",
        start_day=cfg["start_day"], end_day=cfg["end_day"],
        bootstrap_repetitions=cfg["bootstrap_repetitions"], seed=cfg["bootstrap_seed"],
        matched_control_verified=False,
    )
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(OUT / f"{name}.csv", index=False)
            value.to_parquet(OUT / f"{name}.parquet", index=False)
    meta = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
    meta.update(analysis_date="2026-09-17", provisional=True, source_run=str(RUN),
                source_protocol_identity=lock["identity"], fitted_models=False,
                missing_family="matched_full_hourly", production_modified=False)
    (OUT / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = result["metrics"]
    overall = metrics.loc[metrics.group.eq("overall")].set_index("family")
    country = metrics.loc[metrics.group.eq("country")].pivot(index="value", columns="family", values="mae")
    country["variation_pct"] = (country.full_quarterhour / country.nyx - 1) * 100
    paired = result["paired_deltas"].query("family == 'full_quarterhour' and baseline == 'nyx'").set_index("zone")
    pooled = paired.loc["all"]
    governance = []
    for zone in ZONES:
        audits = json.loads((RUN / "workers" / f"{zone}_15min" / "kalman_audit.json").read_text())
        counts = pd.Series([a["selected_filter"] for a in audits]).value_counts().to_dict()
        governance.append({"zone": zone, "days": len(audits), "selected_filters": counts,
                           "positive_weight_days": sum(a["selected_weight"] > 0 for a in audits)})
    (OUT / "kalman_governance.json").write_text(json.dumps(governance, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 4.2), layout="constrained")
    x = np.arange(4)
    for i, (family, color) in enumerate([("nyx", "#637887"), ("raw_quarterhour", "#ad78bf"),
                                         ("residual_quarterhour", "#e0ac54"), ("full_quarterhour", "#159ba7")]):
        ax.bar(x + (i - 1.5) * .19, country.loc[ZONES, family], width=.18, color=color, label=NAMES[family])
    ax.set(xticks=x, xticklabels=ZONES, ylabel="MAE horaire (€/MWh)", ylim=(0, 20),
           title="90 jours communs · une valeur plus basse est meilleure")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.savefig(OUT / "mae_by_country.svg")
    fig.savefig(OUT / "mae_by_country.png", dpi=180)
    stream = io.StringIO(); fig.savefig(stream, format="svg"); plt.close(fig)
    svg = stream.getvalue()[stream.getvalue().find("<svg"):]

    def table(frame):
        return frame.to_html(index=False, escape=True, float_format=lambda v: f"{v:.3f}", border=0)

    overall_table = overall.reset_index()[["family", "mae", "rmse", "bias"]].replace({"family": NAMES})
    overall_table.columns = ["Prévision", "MAE", "RMSE", "Biais"]
    country_table = country.reset_index()[["value", "nyx", "full_quarterhour", "variation_pct"]]
    country_table.columns = ["Pays", "MAE NYX actuel", "MAE NYX 15 min", "Variation MAE (%)"]
    interval_table = paired.reset_index()[["zone", "mae_delta", "mae_delta_ci_low", "mae_delta_ci_high"]]
    interval_table.columns = ["Périmètre", "Écart MAE", "IC 95 % bas", "IC 95 % haut"]
    regime = result["regime_checks"][["zone", "regime", "n", "candidate_mae", "baseline_mae", "mae_relative_degradation"]].copy()
    regime["mae_relative_degradation"] *= 100
    regime.columns = ["Pays", "Régime", "Heures", "MAE NYX 15 min", "MAE NYX actuel", "Variation (%)"]
    regime["Régime"] = regime["Régime"].replace({"negative": "Prix négatifs", "above_train_q95": "Au-dessus du q95 historique"})
    cb_gain = (1 - overall.loc["residual_quarterhour", "mae"] / overall.loc["raw_quarterhour", "mae"]) * 100
    kalman_delta = overall.loc["full_quarterhour", "mae"] - overall.loc["residual_quarterhour", "mae"]
    raw_gain = (1 - overall.loc["raw_quarterhour", "mae"] / overall.loc["raw_hourly", "mae"]) * 100
    report = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NYX — Première analyse du test complet à 15 minutes</title><style>
:root{{font-family:Segoe UI,Arial,sans-serif;color:#e6edf5;background:#090e17;color-scheme:dark}}body{{margin:0;background:radial-gradient(ellipse at top left,#163442,transparent 60%),#090e17}}main{{max-width:1100px;margin:auto;padding:36px 24px 60px}}h1{{font-size:34px}}h2{{font-size:21px}}p,li{{line-height:1.65;color:#c4d0df}}a{{color:#7fe3e8}}section{{margin-top:24px;padding:24px;background:#121e2ce8;border:1px solid #293c50;border-radius:12px}}.label{{color:#f2be73;font-weight:600}}.brand{{letter-spacing:.18em;color:#73d9df}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px;text-align:right;border-bottom:1px solid #2c3c50;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:#8edce3}}.figure{{background:white;border-radius:8px;padding:12px;margin:24px 0}}svg{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}footer{{color:#94a8bc;font-size:13px;margin-top:30px}}</style></head><body><main>
<div class="brand">NYX / RECHERCHE</div><h1>La chaîne complète à 15 minutes : premiers résultats</h1>
<p class="label">Analyse provisoire du 17 septembre 2026 · témoin horaire complet encore manquant</p>
<p>La chaîne Chronos-2 → CatBoost → Kalman à 15 minutes est terminée et vérifiée pour BE, DE, FR et NL. Le témoin complet horaire s’est interrompu le 16 septembre vers 21 h ; sa reprise est séparée de cette analyse. L’ancien <code>report.html</code> du dossier de calcul reflète une erreur de lancement antérieure et ne constitue pas le rapport de ces résultats.</p>
<section><h2>Conclusion disponible</h2><p>La MAE horaire est de <strong>{overall.loc['full_quarterhour','mae']:.3f} €/MWh</strong>, contre <strong>{overall.loc['nyx','mae']:.3f}</strong> pour NYX actuel, soit une dégradation observée de <strong>{-pooled.mae_relative_improvement*100:.2f} %</strong>. La RMSE change peu : {overall.loc['full_quarterhour','rmse']:.3f} contre {overall.loc['nyx','rmse']:.3f} €/MWh. Aucun gain global n’est démontré.</p>
<p>L’écart MAE de {pooled.mae_delta:+.3f} €/MWh a un intervalle à 95 % de [{pooled.mae_delta_ci_low:+.3f} ; {pooled.mae_delta_ci_high:+.3f}], qui contient zéro. Cet essai ne démontre donc ni une amélioration ni une dégradation globale statistiquement établie avec ce protocole.</p>
<p>Les critères de poursuite préétablis ne sont pas satisfaits face à NYX actuel. La comparaison permettant d’isoler l’effet du pas de temps attend toujours le témoin horaire complet à historique commun. Aucun changement de production n’est effectué.</p></section>
<section><h2>Scores sur une population identique</h2><p>Du 18/06/2026 au 15/09/2026 : 90 jours, 2 160 heures par pays, 8 640 points par famille. Les sorties natives sont moyennées par groupes de quatre quarts. MAE, RMSE et biais en €/MWh ; biais = prévision − observation.</p><div class="scroll">{table(overall_table)}</div></section>
<section><h2>Contribution des étapes</h2><p>CatBoost réduit la MAE du Chronos à 15 minutes de <strong>{cb_gain:.2f} %</strong> ({overall.loc['raw_quarterhour','mae']:.3f} → {overall.loc['residual_quarterhour','mae']:.3f}). Kalman ajoute ensuite {kalman_delta:+.3f} €/MWh de MAE sur cette période : il n’apporte pas de gain agrégé dans cette configuration. Sa gouvernance laisse la prévision inchangée dans 347 des 360 journées-pays ; elle active une correction dans 13 cas seulement. Ces ablations servent au diagnostic et ne justifient pas de retirer un composant sur la seule base de ce test.</p><p>Chronos seul à 15 minutes a une MAE observée inférieure de {raw_gain:.2f} % à son témoin brut horaire. Ce constat brut ne remplace pas la comparaison des deux chaînes complètes.</p></section>
<section><h2>Résultats par pays</h2><div class="scroll">{table(country_table)}</div><div class="figure">{svg}</div><p>La France présente la seule baisse de MAE observée (3,51 %), avec une RMSE légèrement supérieure. Les trois autres pays se dégradent en MAE. Les intervalles par pays contiennent tous zéro ; le signal français reste à confirmer.</p><div class="scroll">{table(interval_table)}</div><p>Écart = NYX 15 min − NYX actuel. Un écart négatif favorise le 15 min. Bootstrap apparié commun aux pays, blocs de sept jours, 1 000 répétitions, graine 20260916.</p></section>
<section><h2>Régimes de marché</h2><p>Les prix négatifs montrent une baisse observée de MAE dans les quatre pays : environ 27,1 % en BE, 15,8 % en DE, 10,8 % en FR et 17,7 % en NL. À l’inverse, sur les heures au-dessus du q95 historique en Allemagne, la MAE augmente de 5,74 %, dépassant le seuil de dégradation de 5 % prévu au protocole. Ces sous-groupes restent exploratoires, sans test multiple corrigé.</p><div class="scroll">{table(regime)}</div><p>Les seuils q95 sont fixés sur des observations antérieures au 14/03/2026, pas sur les 90 jours du test. Ils ne désignent donc pas les 5 % de prix les plus élevés de cette période. Chaque sous-groupe satisfait ici les minima prévus de 30 heures et cinq journées.</p></section>
<section><h2>Limites et suite</h2><p>Les correcteurs disposent de 174 à 263 journées passées, au lieu de 365 complètes, à cause de la disponibilité de l’archive native. NYX actuel conserve un historique plus long. Les fondamentaux restent horaires et sont répétés sur les quatre quarts ; seule la cible et la chaîne de modélisation passent au quart d’heure. Les observations sont rétrospectives et cette période a déjà été examinée lors d’essais précédents.</p><p>Il faut achever le témoin horaire, comparer les chaînes à historique égal, puis confirmer les éventuels gains sur une nouvelle période avant toute promotion. Les quatre moyennes de prévisions ponctuelles ne constituent pas des quantiles horaires calibrés.</p></section>
<section><h2>Fichiers et traçabilité</h2><p><a href="metrics.csv">Métriques détaillées</a> · <a href="paired_deltas.csv">Écarts et intervalles</a> · <a href="regime_checks.csv">Régimes</a> · <a href="predictions.csv">Prévisions horaires évaluées</a> · <a href="summary.json">Résumé machine</a> · <a href="kalman_governance.json">Choix Kalman</a> · <a href="analysis_manifest.json">Empreintes des fichiers</a></p><p>Les entrées, le brut, le code scellé et les 3 208 fichiers des quatre workers natifs ont été vérifiés avant calcul des scores. Cette analyse ne réentraîne aucun modèle.</p><p>Calcul source : <code>{html.escape(str(RUN))}</code><br>Protocole : <code>{lock['identity']}</code></p></section>
<footer>Instantané provisoire indépendant du rapport final · Données locales · Production inchangée</footer></main></body></html>'''
    (OUT / "report.html").write_text(report, encoding="utf-8")
    manifest = {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "analysis_manifest.json"}
    (OUT / "analysis_manifest.json").write_text(json.dumps({"provisional": True, "source_protocol_identity": lock["identity"], "files": manifest}, indent=2), encoding="utf-8")
    print(overall_table.to_string(index=False))
    print(json.dumps(governance, ensure_ascii=False))
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
