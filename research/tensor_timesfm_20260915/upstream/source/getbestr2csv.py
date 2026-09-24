# #!/usr/bin/env python3

# """

# Scan up to max_depth levels under a root folder for CSVs named:

#     results_epoch_10_*.csv


# Verify each CSV has EXACTLY the expected columns (no more, no fewer),

# then find the maximum Forecasted_R2 across all valid rows/files and

# print the triple:

#     csv_path,Forecasted_R2,Forecasted_Kelly R2

# """


# import os

# import sys

# import argparse

# from pathlib import Path

# from fnmatch import fnmatch


# import pandas as pd


# REQUIRED_COLS = [
#     "Forecasted_RMSE",
#     "Forecasted_R2",
#     "Forecasted_Kelly R2",
#     "Analyst_RMSE",
#     "Analyst_R2",
#     "Analyst_Kelly R2",
#     "w_recon",
#     "w_pred",
#     "w_forecast_drivers",
#     "w_regression",
# ]


# PATTERN = "results_epoch_10_*.csv"


# def iter_files_within_depth(root: Path, pattern: str, max_depth: int):
#     """Yield files matching pattern, with depth <= max_depth (root is depth 0)."""

#     root = root.resolve()

#     root_depth = len(root.parts)

#     for dirpath, dirnames, filenames in os.walk(root):
#         cur_depth = len(Path(dirpath).resolve().parts) - root_depth

#         if cur_depth > max_depth:
#             dirnames[:] = []

#             continue

#         for fname in filenames:
#             if fnmatch(fname, pattern):
#                 yield Path(dirpath) / fname


# def csv_has_exact_columns(path: Path) -> bool:
#     try:
#         df_head = pd.read_csv(path, nrows=0)

#     except Exception as e:
#         print(f"[skip] Failed to read header {path}: {e}", file=sys.stderr)

#         return False

#     cols = list(df_head.columns)

#     return set(cols) == set(REQUIRED_COLS) and len(cols) == len(REQUIRED_COLS)


# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--root", type=str, default="./a", help="Root directory to search")

#     ap.add_argument(
#         "--max-depth", type=int, default=3, help="Max directory depth (root is depth 0)"
#     )

#     args = ap.parse_args()

#     root = Path(args.root)

#     if not root.exists() or not root.is_dir():
#         print(
#             f"Error: root folder not found or not a directory: {root}", file=sys.stderr
#         )

#         sys.exit(2)

#     best_val = None

#     best_row = None

#     best_file = None

#     any_files = False

#     for csv_path in iter_files_within_depth(root, PATTERN, args.max_depth):
#         any_files = True

#         if not csv_has_exact_columns(csv_path):
#             print(f"[skip] Column mismatch in {csv_path}", file=sys.stderr)

#             continue

#         try:
#             df = pd.read_csv(csv_path)

#         except Exception as e:
#             print(f"[skip] Failed to read {csv_path}: {e}", file=sys.stderr)

#             continue

#         for col in ["Forecasted_R2", "Forecasted_Kelly R2"]:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#         df = df.dropna(subset=["Forecasted_R2", "Forecasted_Kelly R2"])

#         if df.empty:
#             print(f"[skip] No valid rows in {csv_path}", file=sys.stderr)

#             continue

#         local_idx = df["Forecasted_R2"].idxmax()

#         local_val = df.at[local_idx, "Forecasted_R2"]

#         if best_val is None or local_val > best_val:
#             best_val = local_val

#             best_row = df.loc[local_idx].copy()

#             best_file = csv_path

#     if not any_files:
#         print(
#             "Error: no CSVs found matching pattern results_epoch_10_*.csv",
#             file=sys.stderr,
#         )

#         sys.exit(3)

#     if best_row is None:
#         print(
#             "Error: no valid CSV rows with the expected columns and numeric Forecasted_R2",
#             file=sys.stderr,
#         )

#         sys.exit(4)

#     # print file path + values

#     # print file path + values + best weights
#     # (format: csv_path,Forecasted_R2,Forecasted_Kelly R2,w_recon,w_pred,w_forecast_drivers,w_regression)

#     best_w_recon: float = float(
#         best_row["w_recon"]
#     )  # float scalar; reconstruction-loss weight used in that row
#     best_w_pred: float = float(
#         best_row["w_pred"]
#     )  # float scalar; prediction-loss (driver) weight
#     best_w_forecast_drivers: float = float(
#         best_row["w_forecast_drivers"]
#     )  # float scalar; weight for TimesFM/driver loss
#     best_w_regression: float = float(
#         best_row["w_regression"]
#     )  # float scalar; regression head/regularizer weight

#     print(  # prints a single CSV line (plain string)
#         f"{best_file},"  # Path object auto-cast to string; CSV field 1: file path
#         f"{best_row['Forecasted_R2']},"  # float; CSV field 2: best Forecasted_R2
#         f"{best_row['Forecasted_Kelly R2']},"  # float; CSV field 3: best Forecasted_Kelly R2
#         f"{best_w_recon},"  # float; CSV field 4: w_recon
#         f"{best_w_pred},"  # float; CSV field 5: w_pred
#         f"{best_w_forecast_drivers},"  # float; CSV field 6: w_forecast_drivers
#         f"{best_w_regression}"  # float; CSV field 7: w_regression
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Scan up to --max-depth levels under a root folder for CSVs named:
    single_split_metrics_*.csv

For each CSV:
  - verify it has the column "MASE"
  - coerce "MASE" to numeric and drop NaNs
  - track the global MINIMUM MASE across all files/rows

Finally print ONE line:
    csv_path,MASE

Default root:
  ./examples/rideshare_3d_tensor/rideshare_timesfm_3d_multistep
Default max-depth:
  2   (root is depth 0; subfolder depth 1; file depth 2)
"""

import os  # stdlib: directory walking
import sys  # stdlib: stderr + exit codes
import argparse  # stdlib: CLI args
from pathlib import Path  # stdlib: path objects
from fnmatch import fnmatch  # stdlib: pattern match for filenames

import pandas as pd  # third-party: CSV reading

# ---- Required single column for metrics CSVs ----
REQUIRED_SINGLE_COL: str = "MASE"  # str: the only column we need to exist

# ---- File pattern inside each subfolder ----
PATTERN: str = "single_split_metrics_*.csv"  # str: glob-like filename filter


def iter_files_within_depth(root: Path, pattern: str, max_depth: int):
    """
    Yield files matching `pattern` under `root`, limited by directory depth.

    Params
    ------
    root: Path
        Path object, the starting directory (depth 0).
    pattern: str
        Filename wildcard, e.g., "single_split_metrics_*.csv".
    max_depth: int
        Maximum directory depth allowed from root (inclusive of files inside).

    Yields
    ------
    Path
        Full path to each matching file.
    """
    root = root.resolve()  # Path: absolute normalized root
    root_depth: int = len(root.parts)  # int: number of path segments

    for dirpath, dirnames, filenames in os.walk(root):
        # dirpath: str, current directory path
        # dirnames: list[str], subdirectories (we can prune)
        # filenames: list[str], files in current dir
        cur_depth: int = len(Path(dirpath).resolve().parts) - root_depth  # int

        if cur_depth > max_depth:
            dirnames[:] = []  # mutate in-place to stop deeper walk
            continue

        for fname in filenames:
            # fname: str, file name only
            if fnmatch(fname, pattern):  # bool: wildcard match
                yield Path(dirpath) / fname  # Path: full path to match


def csv_has_required_col(path: Path) -> bool:
    """
    Return True if CSV at `path` contains the REQUIRED_SINGLE_COL ("MASE").

    Uses header-only read for speed and safety.
    """
    try:
        # df_head: DataFrame with 0 rows, only header columns
        df_head: pd.DataFrame = pd.read_csv(path, nrows=0)
    except Exception as e:
        print(f"[skip] Failed to read header {path}: {e}", file=sys.stderr)
        return False

    cols = list(df_head.columns)  # list[str]: column names
    return REQUIRED_SINGLE_COL in cols


def main():
    # ---- CLI: pick root + depth ----
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="./examples/rideshare_3d_tensor/rideshare_timesfm_3d_multistep",
        help="Root directory to search (default matches your rideshare layout)",
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Max directory depth (root=0, its subfolder=1, files=2)",
    )
    args = ap.parse_args()

    # root_dir: Path = validated start folder
    root_dir: Path = Path(args.root)
    if not root_dir.exists() or not root_dir.is_dir():
        print(
            f"Error: root folder not found or not a directory: {root_dir}",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- Track global best (minimum) MASE ----
    best_mase_value: float | None = None  # float or None: current min MASE
    best_csv_file: Path | None = None  # Path or None: file where min found
    any_files_seen: bool = False  # bool: did we see any matching CSV?

    # ---- Walk matching files within depth ----
    for csv_path in iter_files_within_depth(root_dir, PATTERN, args.max_depth):
        any_files_seen = True  # we did encounter at least one candidate

        if not csv_has_required_col(csv_path):
            print(
                f"[skip] Missing column '{REQUIRED_SINGLE_COL}' in {csv_path}",
                file=sys.stderr,
            )
            continue

        try:
            # df: DataFrame with full content
            df: pd.DataFrame = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[skip] Failed to read {csv_path}: {e}", file=sys.stderr)
            continue

        # Coerce 'MASE' to numeric; invalid -> NaN
        # pd.to_numeric: uncommon param `errors='coerce'` converts bad strings to NaN
        df[REQUIRED_SINGLE_COL] = pd.to_numeric(
            df[REQUIRED_SINGLE_COL], errors="coerce"
        )

        # Drop rows where MASE is NaN
        df = df.dropna(subset=[REQUIRED_SINGLE_COL])

        if df.empty:
            print(
                f"[skip] No valid '{REQUIRED_SINGLE_COL}' rows in {csv_path}",
                file=sys.stderr,
            )
            continue

        # local_min_index: int label of row with minimum MASE
        local_min_index = df[REQUIRED_SINGLE_COL].idxmin()

        # local_min_value: float (from DataFrame cell)
        local_min_value: float = float(df.at[local_min_index, REQUIRED_SINGLE_COL])

        # Update global best if needed (we want MINIMUM)
        if (best_mase_value is None) or (local_min_value < best_mase_value):
            best_mase_value = local_min_value
            best_csv_file = csv_path

    # ---- Final checks and print result ----
    if not any_files_seen:
        print(f"Error: no CSVs found matching pattern {PATTERN}", file=sys.stderr)
        sys.exit(3)

    if best_mase_value is None or best_csv_file is None:
        print("Error: no valid MASE values found in matched CSVs", file=sys.stderr)
        sys.exit(4)

    # Print a single CSV line: file_path,MASE
    # best_csv_file: auto-casts to string; best_mase_value: float
    print(f"{best_csv_file},{best_mase_value}")


if __name__ == "__main__":
    main()
