#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


LOWER_IS_BETTER = (
    "mae_q50",
    "rmse_q50",
    "crps_quantile_approx",
    "ramp_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="runs/feature_selection/selection_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/feature_selection/summary",
    )
    parser.add_argument(
        "--min-standalone-mae-gain",
        type=float,
        default=0.20,
        help="Gain médian minimal en pourcentage contre price-only.",
    )
    parser.add_argument(
        "--max-worst-fold-loss",
        type=float,
        default=0.50,
        help="Perte maximale tolérée sur le pire fold, en pourcentage.",
    )
    parser.add_argument(
        "--min-loo-contribution",
        type=float,
        default=0.10,
        help="Dégradation médiane minimale quand le groupe est retiré.",
    )
    return parser.parse_args()


def gain_percent(reference: pd.Series, candidate: pd.Series) -> pd.Series:
    reference = pd.to_numeric(reference, errors="coerce")
    candidate = pd.to_numeric(candidate, errors="coerce")
    return 100.0 * (reference - candidate) / reference


def add_gains(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for metric in LOWER_IS_BETTER:
        native = f"native_{metric}"
        baseline = f"baseline_{metric}"
        full = f"full_{metric}"
        if native in result and baseline in result:
            result[f"gain_vs_price_only_{metric}"] = gain_percent(
                result[baseline], result[native]
            )
        if native in result and full in result:
            # Positif : retirer le sujet dégrade le modèle complet.
            result[f"loo_contribution_{metric}"] = gain_percent(
                result[native], result[full]
            )
    if "native_bias_q50" in result:
        result["native_abs_bias_q50"] = pd.to_numeric(
            result["native_bias_q50"], errors="coerce"
        ).abs()
    return result


def aggregate_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (kind, subject), block in frame.groupby(
        ["selection_kind", "subject"], sort=True
    ):
        if kind == "full":
            continue
        row: dict[str, Any] = {
            "selection_kind": kind,
            "subject": subject,
            "folds": int(block["fold"].nunique()),
            "median_columns": float(
                pd.to_numeric(
                    block["retained_model_column_count"],
                    errors="coerce",
                ).median()
            ),
        }
        for metric in LOWER_IS_BETTER:
            gain_col = f"gain_vs_price_only_{metric}"
            loo_col = f"loo_contribution_{metric}"
            if gain_col in block:
                values = pd.to_numeric(block[gain_col], errors="coerce")
                row[f"median_{gain_col}"] = float(values.median())
                row[f"worst_{gain_col}"] = float(values.min())
                row[f"std_{gain_col}"] = float(values.std(ddof=0))
            if loo_col in block:
                values = pd.to_numeric(block[loo_col], errors="coerce")
                row[f"median_{loo_col}"] = float(values.median())
                row[f"worst_{loo_col}"] = float(values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def recommendation(
    summary: pd.DataFrame,
    min_standalone_gain: float,
    max_worst_loss: float,
    min_loo_contribution: float,
) -> pd.DataFrame:
    result = summary.copy()
    decisions: list[str] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        kind = row["selection_kind"]
        if kind == "only":
            median_gain = float(
                row.get("median_gain_vs_price_only_mae_q50", np.nan)
            )
            worst_gain = float(
                row.get("worst_gain_vs_price_only_mae_q50", np.nan)
            )
            crps_gain = float(
                row.get(
                    "median_gain_vs_price_only_crps_quantile_approx",
                    np.nan,
                )
            )
            ramp_gain = float(
                row.get("median_gain_vs_price_only_ramp_mae", np.nan)
            )
            keep = (
                np.isfinite(median_gain)
                and median_gain >= min_standalone_gain
                and (
                    not np.isfinite(worst_gain)
                    or worst_gain >= -max_worst_loss
                )
            ) or (
                max(
                    crps_gain if np.isfinite(crps_gain) else -np.inf,
                    ramp_gain if np.isfinite(ramp_gain) else -np.inf,
                )
                >= 1.0
                and (
                    not np.isfinite(median_gain) or median_gain >= -0.25
                )
            )
            decisions.append("retain_for_loo" if keep else "drop_candidate")
            reasons.append(
                f"gain MAE médian={median_gain:.3f}%, "
                f"pire fold={worst_gain:.3f}%"
            )
        elif kind == "loo":
            contribution = float(
                row.get("median_loo_contribution_mae_q50", np.nan)
            )
            crps = float(
                row.get(
                    "median_loo_contribution_crps_quantile_approx",
                    np.nan,
                )
            )
            ramp = float(
                row.get("median_loo_contribution_ramp_mae", np.nan)
            )
            keep = max(
                contribution if np.isfinite(contribution) else -np.inf,
                crps if np.isfinite(crps) else -np.inf,
                ramp if np.isfinite(ramp) else -np.inf,
            ) >= min_loo_contribution
            decisions.append("keep" if keep else "remove")
            reasons.append(
                f"contribution MAE={contribution:.3f}%, "
                f"CRPS={crps:.3f}%, ramp={ramp:.3f}%"
            )
        else:
            decisions.append("review")
            reasons.append("")
    result["decision"] = decisions
    result["reason"] = reasons
    return result


def main() -> int:
    args = parse_args()
    source = Path(args.results).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = add_gains(pd.read_csv(source))
    frame.to_csv(output_dir / "selection_results_with_gains.csv", index=False)

    summary = aggregate_subjects(frame)
    ranked = recommendation(
        summary,
        args.min_standalone_mae_gain,
        args.max_worst_fold_loss,
        args.min_loo_contribution,
    )
    ranked = ranked.sort_values(
        ["selection_kind", "decision", "subject"],
        kind="stable",
    )
    ranked.to_csv(output_dir / "feature_selection_ranking.csv", index=False)

    selected = ranked.loc[
        ranked["decision"].isin(["retain_for_loo", "keep"]),
        "subject",
    ].drop_duplicates().tolist()
    with (output_dir / "recommended_subjects.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            {"recommended_subjects": selected},
            handle,
            allow_unicode=True,
            sort_keys=False,
        )

    columns = [
        column
        for column in (
            "selection_kind",
            "subject",
            "median_gain_vs_price_only_mae_q50",
            "worst_gain_vs_price_only_mae_q50",
            "median_loo_contribution_mae_q50",
            "decision",
        )
        if column in ranked.columns
    ]
    print(ranked[columns].to_string(index=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
