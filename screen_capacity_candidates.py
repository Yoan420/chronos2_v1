#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chronos2_modular.saturn import fetch_saturn_series


TARGET_SERIES = "power.price.everyday.fr.hourly.eurmwh"

CANDIDATE_SERIES = [
    "power.storm.flow..ch.fr.mw.h.obs.entsoe.capacity",
    "power.storm.flow..es.fr.mw.h.obs.entsoe.capacity",
    "power.storm.flow..fr.ch.mw.h.obs.entsoe.capacity",
    "power.storm.flow..fr.es.mw.h.obs.entsoe.capacity",
    "power.storm.flow..fr.it_north.mw.h.obs.entsoe.capacity",
    "power.storm.flow..fr.uk.mw.h.obs.entsoe.capacity",
    "power.storm.flow..it_north.fr.mw.h.obs.entsoe.capacity",
    "power.storm.flow..uk.fr.mw.h.obs.entsoe.capacity",
    "power.storm.flow.capacity.ch.fr.mw.h.entsoe.actual",
    "power.storm.flow.capacity.es.fr.mw.h.entsoe.actual",
    "power.storm.flow.capacity.fr.ch.mw.h.entsoe.actual",
    "power.storm.flow.capacity.fr.es.mw.h.entsoe.actual",
    "power.storm.flow.capacity.fr.it_north.mw.h.entsoe.actual",
    "power.storm.flow.capacity.it_north.fr.mw.h.entsoe.actual",
    "power.storm.flow.dayahead.capacity.ch.fr.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.es.fr.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.fr.ch.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.fr.es.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.fr.it_north.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.fr.uk.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.it_north.fr.mw.h.obs.entsoe",
    "power.storm.flow.dayahead.capacity.uk.fr.mw.h.obs.entsoe",
    "power.storm.prod.capacity.11wd7boll5bfrfbc.mw.h.entsoe.actual",
    "power.storm.prod.capacity.11wd7frim2b--p-g.mw.h.entsoe.actual",
    "power.storm.prod.capacity.11wd7frim2b--q-d.mw.h.entsoe.actual",
]


@dataclass(frozen=True)
class CandidateSpec:
    series: str
    family: str
    safe_variant: str
    causal_status: str
    semantic_priority: int


def classify_series(name: str) -> CandidateSpec:
    lower = name.lower()
    if ".flow.dayahead.capacity." in lower:
        return CandidateSpec(
            series=name,
            family="dayahead_capacity",
            safe_variant="level",
            causal_status="PIT_TO_VERIFY",
            semantic_priority=1,
        )
    if ".flow.capacity." in lower and lower.endswith(".actual"):
        return CandidateSpec(
            series=name,
            family="actual_capacity",
            safe_variant="lag24",
            causal_status="LAG_ONLY",
            semantic_priority=3,
        )
    if ".flow.." in lower and ".obs.entsoe.capacity" in lower:
        return CandidateSpec(
            series=name,
            family="observed_capacity",
            safe_variant="lag24",
            causal_status="LAG_ONLY",
            semantic_priority=2,
        )
    if ".prod.capacity." in lower and lower.endswith(".actual"):
        return CandidateSpec(
            series=name,
            family="production_capacity_actual",
            safe_variant="lag24",
            causal_status="IDENTITY_AND_PIT_TO_VERIFY",
            semantic_priority=4,
        )
    return CandidateSpec(
        series=name,
        family="unknown",
        safe_variant="lag24",
        causal_status="VERIFY",
        semantic_priority=5,
    )


def short_alias(name: str) -> str:
    value = name.lower()
    replacements = (
        ("power.storm.flow.dayahead.capacity.", "da_cap_"),
        ("power.storm.flow.capacity.", "actual_cap_"),
        ("power.storm.flow..", "obs_cap_"),
        ("power.storm.prod.capacity.", "prod_cap_"),
        (".mw.h.obs.entsoe", ""),
        (".mw.h.entsoe.actual", ""),
        (".mw.h.obs.entsoe.capacity", ""),
        (".capacity", ""),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value.replace(".", "_").replace("-", "_").strip("_")


def normalize_hourly(series: pd.Series, timezone: str) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").sort_index()
    index = pd.DatetimeIndex(result.index)
    if index.tz is None:
        index = index.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    else:
        index = index.tz_convert(timezone)
    result.index = index
    if result.index.duplicated().any():
        result = result.groupby(level=0).mean()
    return result.resample("h").mean()


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)
    doy = index.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_weekend": (dow >= 5).astype(float),
        },
        index=index,
    )


def baseline_frame(target: pd.Series) -> pd.DataFrame:
    frame = calendar_features(target.index)
    frame["price_lag24"] = target.shift(24)
    frame["price_lag48"] = target.shift(48)
    frame["price_lag168"] = target.shift(168)
    frame["price_roll24_mean"] = target.shift(24).rolling(24).mean()
    frame["price_roll168_mean"] = target.shift(24).rolling(168).mean()
    frame["price_roll168_std"] = target.shift(24).rolling(168).std()
    return frame


def new_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.05,
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    random_state=42,
                ),
            ),
        ]
    )


def fair_time_series_scores(
    frame: pd.DataFrame,
    feature: str,
    n_splits: int,
) -> list[dict[str, float]]:
    required = ["target", *BASELINE_COLUMNS, feature]
    work = frame[required].replace([np.inf, -np.inf], np.nan)
    work = work.loc[work["target"].notna() & work[feature].notna()].copy()

    min_rows = max(24 * 180, (n_splits + 1) * 24 * 30)
    if len(work) < min_rows:
        return []

    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(work), start=1):
        train = work.iloc[train_idx]
        test = work.iloc[test_idx]

        baseline_model = new_model()
        candidate_model = new_model()

        baseline_model.fit(train[BASELINE_COLUMNS], train["target"])
        candidate_model.fit(
            train[[*BASELINE_COLUMNS, feature]],
            train["target"],
        )

        pred_base = baseline_model.predict(test[BASELINE_COLUMNS])
        pred_candidate = candidate_model.predict(
            test[[*BASELINE_COLUMNS, feature]]
        )

        mae_base = mean_absolute_error(test["target"], pred_base)
        mae_candidate = mean_absolute_error(
            test["target"], pred_candidate
        )
        rmse_base = mean_squared_error(
            test["target"], pred_base
        ) ** 0.5
        rmse_candidate = mean_squared_error(
            test["target"], pred_candidate
        ) ** 0.5

        rows.append(
            {
                "fold": fold,
                "test_start": str(test.index.min()),
                "test_end": str(test.index.max()),
                "n_test": int(len(test)),
                "mae_baseline": float(mae_base),
                "mae_candidate": float(mae_candidate),
                "mae_gain_pct": float(
                    100 * (mae_base - mae_candidate) / mae_base
                ),
                "rmse_baseline": float(rmse_base),
                "rmse_candidate": float(rmse_candidate),
                "rmse_gain_pct": float(
                    100 * (rmse_base - rmse_candidate) / rmse_base
                ),
            }
        )
    return rows


def add_directional_aggregates(frame: pd.DataFrame) -> list[str]:
    created: list[str] = []
    borders = {
        "ch": ("da_cap_ch_fr", "da_cap_fr_ch"),
        "es": ("da_cap_es_fr", "da_cap_fr_es"),
        "it_north": ("da_cap_it_north_fr", "da_cap_fr_it_north"),
        "uk": ("da_cap_uk_fr", "da_cap_fr_uk"),
    }

    import_columns: list[str] = []
    export_columns: list[str] = []

    for border, (to_fr, from_fr) in borders.items():
        if to_fr in frame:
            import_columns.append(to_fr)
        if from_fr in frame:
            export_columns.append(from_fr)
        if to_fr in frame and from_fr in frame:
            frame[f"da_cap_{border}_total"] = frame[to_fr] + frame[from_fr]
            frame[f"da_cap_{border}_asymmetry"] = (
                frame[to_fr] - frame[from_fr]
            )
            created.extend(
                [
                    f"da_cap_{border}_total",
                    f"da_cap_{border}_asymmetry",
                ]
            )

    if import_columns:
        frame["da_cap_total_import_to_fr"] = frame[import_columns].sum(
            axis=1, min_count=1
        )
        frame["da_cap_min_import_to_fr"] = frame[import_columns].min(
            axis=1
        )
        created.extend(
            ["da_cap_total_import_to_fr", "da_cap_min_import_to_fr"]
        )
    if export_columns:
        frame["da_cap_total_export_from_fr"] = frame[
            export_columns
        ].sum(axis=1, min_count=1)
        created.append("da_cap_total_export_from_fr")
    if import_columns and export_columns:
        frame["da_cap_system_asymmetry"] = (
            frame["da_cap_total_import_to_fr"]
            - frame["da_cap_total_export_from_fr"]
        )
        created.append("da_cap_system_asymmetry")

    return created


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 100 or aligned.iloc[:, 0].nunique() < 2:
        return math.nan
    return float(aligned.corr(method="spearman").iloc[0, 1])


def make_html_report(
    ranking: pd.DataFrame,
    failures: pd.DataFrame,
    output_path: Path,
) -> None:
    if ranking.empty:
        ranked = pd.DataFrame(
            [{"message": "Aucun candidat n'a pu être évalué."}]
        )
        recommended = ranked.copy()
    else:
        ranked = ranking.sort_values(
            ["median_mae_gain_pct", "median_rmse_gain_pct"],
            ascending=False,
            na_position="last",
        )
        recommended = ranked.loc[
            ranked["causal_status"].isin(
                ["PIT_TO_VERIFY", "LAG_ONLY"]
            )
            & ranked["median_mae_gain_pct"].gt(0)
            & ranked["worst_mae_gain_pct"].gt(-1.0)
        ]

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Screening des capacités transfrontalières</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f2f2f2; position: sticky; top: 0; }}
.good {{ background: #eef8ee; padding: 16px; }}
.warn {{ background: #fff7e6; padding: 16px; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Screening rapide des capacités transfrontalières</h1>
<div class="warn">
<strong>Attention causale.</strong>
Les séries day-ahead ne sont utilisables en J+1 que si leur publication
est antérieure au cutoff opérationnel. Les séries actual/obs sont testées
avec un retard de 24 heures. Les résultats ne remplacent pas un backtest
point-in-time Chronos-2.
</div>
<h2>Candidats recommandés pour une ablation Chronos-2</h2>
<div class="good">{recommended.to_html(index=False, float_format=lambda x: f"{x:.3f}")}</div>
<h2>Classement complet</h2>
{ranked.to_html(index=False, float_format=lambda x: f"{x:.3f}")}
<h2>Échecs de téléchargement</h2>
{failures.to_html(index=False) if not failures.empty else "<p>Aucun.</p>"}
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screening causal rapide de séries Saturn de capacité "
            "transfrontalière pour le prix Day-Ahead France."
        )
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Début ISO. Par défaut : aujourd'hui moins deux ans.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Fin ISO. Par défaut : maintenant.",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Paris",
    )
    parser.add_argument(
        "--saturn-url",
        default="https://saturn-energyscan.gem.myengie.com//api",
    )
    parser.add_argument(
        "--author",
        default=os.getenv("SATURN_AUTHOR", "BQ6757"),
    )
    parser.add_argument(
        "--target",
        default=TARGET_SERIES,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-dir",
        default="runs/capacity_candidate_screening",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timezone = args.timezone
    now = pd.Timestamp.now(tz=timezone)
    start = (
        pd.Timestamp(args.start)
        if args.start
        else now - pd.DateOffset(years=2)
    )
    end = pd.Timestamp(args.end) if args.end else now
    if start.tzinfo is None:
        start = start.tz_localize(timezone)
    else:
        start = start.tz_convert(timezone)
    if end.tzinfo is None:
        end = end.tz_localize(timezone)
    else:
        end = end.tz_convert(timezone)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement cible : {args.target}")
    target = fetch_saturn_series(
        series_name=args.target,
        start=start,
        end=end,
        timezone=timezone,
        saturn_url=args.saturn_url,
        author=args.author,
    )
    target = normalize_hourly(target, timezone).rename("target")

    frame = baseline_frame(target)
    global BASELINE_COLUMNS
    BASELINE_COLUMNS = list(frame.columns)
    frame["target"] = target

    metadata_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    candidate_features: list[str] = []

    for number, name in enumerate(CANDIDATE_SERIES, start=1):
        spec = classify_series(name)
        alias = short_alias(name)
        print(f"[{number:02d}/{len(CANDIDATE_SERIES)}] {alias}")

        try:
            series = fetch_saturn_series(
                series_name=name,
                start=start - pd.Timedelta(days=8),
                end=end,
                timezone=timezone,
                saturn_url=args.saturn_url,
                author=args.author,
            )
            series = normalize_hourly(series, timezone).reindex(frame.index)
        except Exception as exc:
            failures.append(
                {
                    "series": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  ÉCHEC : {exc}")
            continue

        raw_alias = alias
        frame[raw_alias] = series

        if spec.safe_variant == "level":
            feature_alias = raw_alias
        elif spec.safe_variant == "lag24":
            feature_alias = f"{raw_alias}__lag24"
            frame[feature_alias] = series.shift(24)
        else:
            feature_alias = f"{raw_alias}__lag24"
            frame[feature_alias] = series.shift(24)

        candidate_features.append(feature_alias)
        metadata_rows.append(
            {
                "feature": feature_alias,
                "raw_alias": raw_alias,
                "series": name,
                "family": spec.family,
                "safe_variant": spec.safe_variant,
                "causal_status": spec.causal_status,
                "semantic_priority": spec.semantic_priority,
                "first_valid": str(series.first_valid_index()),
                "last_valid": str(series.last_valid_index()),
                "coverage_raw": float(series.notna().mean()),
                "coverage_feature": float(
                    frame[feature_alias].notna().mean()
                ),
                "n_unique": int(series.nunique(dropna=True)),
                "spearman_price_raw_upper_bound": safe_spearman(
                    series, target
                ),
                "spearman_abs_price_raw_upper_bound": safe_spearman(
                    series, target.abs()
                ),
                "spearman_price_safe_feature": safe_spearman(
                    frame[feature_alias], target
                ),
            }
        )

    derived = add_directional_aggregates(frame)
    for feature_alias in derived:
        candidate_features.append(feature_alias)
        metadata_rows.append(
            {
                "feature": feature_alias,
                "raw_alias": feature_alias,
                "series": "DERIVED_FROM_DAYAHEAD_CAPACITIES",
                "family": "dayahead_capacity_derived",
                "safe_variant": "level",
                "causal_status": "PIT_TO_VERIFY",
                "semantic_priority": 1,
                "first_valid": str(frame[feature_alias].first_valid_index()),
                "last_valid": str(frame[feature_alias].last_valid_index()),
                "coverage_raw": float(frame[feature_alias].notna().mean()),
                "coverage_feature": float(
                    frame[feature_alias].notna().mean()
                ),
                "n_unique": int(
                    frame[feature_alias].nunique(dropna=True)
                ),
                "spearman_price_raw_upper_bound": safe_spearman(
                    frame[feature_alias], target
                ),
                "spearman_abs_price_raw_upper_bound": safe_spearman(
                    frame[feature_alias], target.abs()
                ),
                "spearman_price_safe_feature": safe_spearman(
                    frame[feature_alias], target
                ),
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    fold_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for number, feature in enumerate(candidate_features, start=1):
        print(
            f"Backtest [{number:02d}/{len(candidate_features)}] {feature}"
        )
        scores = fair_time_series_scores(frame, feature, args.n_splits)
        if not scores:
            print("  Ignoré : historique ou couverture insuffisante.")
            continue

        for row in scores:
            fold_rows.append({"feature": feature, **row})

        score_frame = pd.DataFrame(scores)
        meta = metadata.loc[metadata["feature"].eq(feature)].iloc[0]
        ranking_rows.append(
            {
                **meta.to_dict(),
                "folds": int(len(score_frame)),
                "median_mae_gain_pct": float(
                    score_frame["mae_gain_pct"].median()
                ),
                "worst_mae_gain_pct": float(
                    score_frame["mae_gain_pct"].min()
                ),
                "mean_mae_gain_pct": float(
                    score_frame["mae_gain_pct"].mean()
                ),
                "std_mae_gain_pct": float(
                    score_frame["mae_gain_pct"].std()
                ),
                "positive_mae_folds": int(
                    score_frame["mae_gain_pct"].gt(0).sum()
                ),
                "median_rmse_gain_pct": float(
                    score_frame["rmse_gain_pct"].median()
                ),
                "worst_rmse_gain_pct": float(
                    score_frame["rmse_gain_pct"].min()
                ),
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    folds = pd.DataFrame(fold_rows)
    failures_frame = pd.DataFrame(failures)

    if not ranking.empty:
        ranking = ranking.sort_values(
            [
                "median_mae_gain_pct",
                "median_rmse_gain_pct",
                "coverage_feature",
            ],
            ascending=False,
        )
        ranking["screening_decision"] = np.select(
            [
                (
                    ranking["median_mae_gain_pct"].gt(0.5)
                    & ranking["worst_mae_gain_pct"].gt(-1.0)
                    & ranking["positive_mae_folds"].ge(
                        np.ceil(ranking["folds"] / 2)
                    )
                ),
                ranking["median_mae_gain_pct"].gt(0),
            ],
            ["PRIORITY_ABLATION", "SECONDARY_ABLATION"],
            default="DROP_FOR_NOW",
        )

    numeric_candidates = [
        column
        for column in candidate_features
        if column in frame and frame[column].notna().sum() >= 500
    ]
    correlation = (
        frame[numeric_candidates].corr(method="spearman")
        if numeric_candidates
        else pd.DataFrame()
    )

    metadata.to_csv(
        output_dir / "series_metadata_and_coverage.csv",
        index=False,
    )
    ranking.to_csv(output_dir / "screening_ranking.csv", index=False)
    folds.to_csv(output_dir / "screening_folds.csv", index=False)
    failures_frame.to_csv(
        output_dir / "download_failures.csv", index=False
    )
    correlation.to_csv(
        output_dir / "candidate_spearman_matrix.csv"
    )

    report_path = output_dir / "capacity_screening_report.html"
    make_html_report(ranking, failures_frame, report_path)

    manifest = {
        "target": args.target,
        "start": str(start),
        "end": str(end),
        "timezone": timezone,
        "n_splits": args.n_splits,
        "baseline_columns": BASELINE_COLUMNS,
        "candidate_count": len(candidate_features),
        "successful_ranked_count": int(len(ranking)),
        "report": str(report_path.resolve()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTop 15 :")
    if ranking.empty:
        print("Aucun candidat évalué.")
    else:
        columns = [
            "feature",
            "family",
            "causal_status",
            "coverage_feature",
            "median_mae_gain_pct",
            "worst_mae_gain_pct",
            "median_rmse_gain_pct",
            "screening_decision",
        ]
        print(ranking[columns].head(15).to_string(index=False))

    print(f"\nRapport : {report_path.resolve()}")
    return 0


BASELINE_COLUMNS: list[str] = []


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
