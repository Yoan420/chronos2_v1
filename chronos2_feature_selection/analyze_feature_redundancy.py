#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="model_covariates_with_future.csv.gz ou model_covariates_selected.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/feature_selection/redundancy",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.97)
    parser.add_argument("--minimum-overlap", type=int, default=500)
    parser.add_argument("--round-decimals", type=int, default=8)
    return parser.parse_args()


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "timestamp" in frame:
        frame = frame.drop(columns=["timestamp"])
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.loc[:, numeric.notna().any(axis=0)]


def exact_duplicate_groups(
    frame: pd.DataFrame,
    decimals: int,
) -> list[list[str]]:
    signatures: dict[int, list[str]] = defaultdict(list)
    sentinel = np.float64(9.87654321012345e307)
    for column in frame.columns:
        values = frame[column].round(decimals).fillna(sentinel)
        signature = int(pd.util.hash_pandas_object(values, index=False).sum())
        signatures[signature].append(column)

    groups: list[list[str]] = []
    for candidates in signatures.values():
        if len(candidates) < 2:
            continue
        reference = frame[candidates[0]].round(decimals)
        equal = [candidates[0]]
        for column in candidates[1:]:
            if reference.equals(frame[column].round(decimals)):
                equal.append(column)
        if len(equal) > 1:
            groups.append(equal)
    return groups


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_frame(source)
    summary = pd.DataFrame(
        {
            "feature": frame.columns,
            "non_missing": [int(frame[c].notna().sum()) for c in frame],
            "missing_rate": [float(frame[c].isna().mean()) for c in frame],
            "n_unique": [int(frame[c].nunique(dropna=True)) for c in frame],
            "std": [float(frame[c].std()) for c in frame],
            "minimum": [float(frame[c].min()) for c in frame],
            "maximum": [float(frame[c].max()) for c in frame],
        }
    )
    summary["near_constant"] = (
        summary["n_unique"].le(1)
        | summary["std"].fillna(0.0).abs().lt(1e-10)
    )
    summary.to_csv(output_dir / "feature_summary.csv", index=False)

    duplicates = exact_duplicate_groups(frame, args.round_decimals)
    duplicate_rows = []
    for group_id, columns in enumerate(duplicates, start=1):
        for column in columns:
            duplicate_rows.append(
                {"duplicate_group": group_id, "feature": column}
            )
    pd.DataFrame(duplicate_rows).to_csv(
        output_dir / "exact_duplicate_groups.csv", index=False
    )

    pairs: list[dict[str, float | int | str]] = []
    columns = list(frame.columns)
    union_find = UnionFind(columns)
    for left_index, left in enumerate(columns):
        left_values = frame[left]
        for right in columns[left_index + 1 :]:
            valid = left_values.notna() & frame[right].notna()
            overlap = int(valid.sum())
            if overlap < args.minimum_overlap:
                continue
            correlation = float(
                left_values.loc[valid].corr(
                    frame.loc[valid, right], method="spearman"
                )
            )
            if np.isfinite(correlation) and abs(correlation) >= args.correlation_threshold:
                pairs.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "spearman": correlation,
                        "abs_spearman": abs(correlation),
                        "overlap": overlap,
                    }
                )
                union_find.union(left, right)

    pair_frame = pd.DataFrame(pairs)
    if not pair_frame.empty:
        pair_frame = pair_frame.sort_values(
            "abs_spearman", ascending=False
        )
    pair_frame.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    clusters: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        clusters[union_find.find(column)].append(column)
    cluster_rows = []
    cluster_id = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        cluster_id += 1
        for member in sorted(members):
            missing_rate = float(frame[member].isna().mean())
            cluster_rows.append(
                {
                    "cluster": cluster_id,
                    "feature": member,
                    "missing_rate": missing_rate,
                    "n_unique": int(frame[member].nunique(dropna=True)),
                }
            )
    pd.DataFrame(cluster_rows).to_csv(
        output_dir / "correlation_clusters.csv", index=False
    )

    print(
        f"{len(columns)} features | {len(duplicates)} groupes identiques | "
        f"{len(pairs)} paires avec |rho| >= {args.correlation_threshold}."
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
