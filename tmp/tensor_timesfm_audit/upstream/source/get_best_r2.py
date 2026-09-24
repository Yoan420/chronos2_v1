#!/usr/bin/env python3
"""
Scan up to max_depth levels under a root folder for CSVs named:
    results_epoch_10_*.csv

Verify each CSV has EXACTLY the expected columns (no more, no fewer),
then find the maximum Forecasted_R2 across all valid rows/files and
print:
    csv_path,Forecasted_R2,Forecasted_Kelly R2

Usage (defaults shown):
    python find_best_r2.py --root ./a --max-depth 3
"""

import os
import sys
import argparse
from pathlib import Path
from fnmatch import fnmatch
from typing import Optional

import pandas as pd


REQUIRED_COLS = [
    "Forecasted_RMSE",
    "Forecasted_R2",
    "Forecasted_Kelly R2",
    "Analyst_RMSE",
    "Analyst_R2",
    "Analyst_Kelly R2",
    "w_recon",
    "w_pred",
    "w_forecast_drivers",
    "w_regression",
]

PATTERN = "results_epoch_*.csv"


def iter_files_within_depth(root: Path, pattern: str, max_depth: int):
    """Yield files matching pattern, with depth <= max_depth (root is depth 0)."""
    root = root.resolve()
    root_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        cur_depth = len(Path(dirpath).resolve().parts) - root_depth
        if cur_depth > max_depth:
            # prune deeper traversal
            dirnames[:] = []
            continue
        for fname in filenames:
            if fnmatch(fname, pattern):
                yield Path(dirpath) / fname


def csv_has_exact_columns(path: Path) -> bool:
    try:
        df_head = pd.read_csv(path, nrows=0)
    except Exception as e:
        print(f"[skip] Failed to read header {path}: {e}", file=sys.stderr)
        return False
    cols = list(df_head.columns)
    # exact set match (no extras, no missing)
    return set(cols) == set(REQUIRED_COLS) and len(cols) == len(REQUIRED_COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./a", help="Root directory to search")
    ap.add_argument(
        "--max-depth", type=int, default=3, help="Max directory depth (root is depth 0)"
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(
            f"Error: root folder not found or not a directory: {root}", file=sys.stderr
        )
        sys.exit(2)

    best_val = None
    best_row = None
    best_csv_path: Optional[Path] = None  # which CSV produced the best row

    any_files = False

    for csv_path in iter_files_within_depth(root, PATTERN, args.max_depth):
        any_files = True
        if not csv_has_exact_columns(csv_path):
            print(f"[skip] Column mismatch in {csv_path}", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[skip] Failed to read {csv_path}: {e}", file=sys.stderr)
            continue

        # ensure numeric & drop rows without the necessary values
        for col in ["Forecasted_R2", "Forecasted_Kelly R2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Forecasted_R2", "Forecasted_Kelly R2"])
        if df.empty:
            print(f"[skip] No valid rows in {csv_path}", file=sys.stderr)
            continue

        # best row within this file
        local_idx = df["Forecasted_R2"].idxmax()
        local_val = df.at[local_idx, "Forecasted_R2"]

        if best_val is None or local_val > best_val:
            best_val = local_val
            best_row = df.loc[local_idx].copy()
            best_csv_path = csv_path  # save the winning CSV file

    if not any_files:
        print(
            "Error: no CSVs found matching pattern results_epoch_10_*.csv",
            file=sys.stderr,
        )
        sys.exit(3)

    if best_row is None:
        print(
            "Error: no valid CSV rows with the expected columns and numeric Forecasted_R2",
            file=sys.stderr,
        )
        sys.exit(4)

    # print csv path + the requested pair
    print(
        f"{best_csv_path},{best_row['Forecasted_R2']},{best_row['Forecasted_Kelly R2']}"
    )


if __name__ == "__main__":
    main()
