#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find the smallest RMSE across metric CSV files located INSIDE:
  <base_dir> with max depth = 2, filenames matching 'single_split_metrics_*.csv'.

Each CSV is expected to have a header row with at least:
  RMSE, R2, R2_Kelly

Example row:
  H,C,t_test,t_end,RMSE,MAE,MASE,R2,R2_Kelly
  168,210,373,540,2.319619,1.082735,0.730005,0.936217,0.960399

Usage:
  python examples/find_lowest_rmse.py --base_dir examples/rideshare_3d_tensor/rideshare_timesfm_3d_168steps_Nov_9_original
  # Optional flags:
  #   --max-depth 2
  #   --pattern single_split_metrics_*.csv
"""

from __future__ import annotations

import argparse  # type: ignore  # CLI arguments
import csv  # type: ignore  # CSV parsing
import fnmatch  # type: ignore  # filename pattern globbing
import sys  # type: ignore  # exit status
from dataclasses import dataclass  # type: ignore  # simple typed container
from pathlib import Path  # type: ignore  # filesystem paths
from typing import Iterator, List, Optional, Tuple  # type: ignore  # type hints


@dataclass
class MetricRow:
    """One metrics row extracted from a CSV file."""

    source_file_path: Path  # Path: file path of the CSV
    row_index_in_file: int  # int: 0-based row index (excluding header)
    rmse_value: float  # float: RMSE value to minimize
    r2_value: float  # float: R2 from the same row
    r2_kelly_value: float  # float: R2_Kelly from the same row


def safe_float(text: str) -> Optional[float]:
    """Convert string to float safely. Returns None for invalid values."""
    try:
        return float(text)
    except Exception:
        return None


def within_max_depth(root_path: Path, current_path: Path, maximum_depth: int) -> bool:
    """
    Check if current_path is within maximum_depth levels below root_path.
    - root_path: Path -> the base folder provided by user
    - current_path: Path -> folder we are visiting
    - maximum_depth: int -> depth budget (2 means root, root/*, root/*/*)
    """
    try:
        relative_parts: Tuple[str, ...] = current_path.relative_to(root_path).parts
    except Exception:
        # current_path is not inside root_path
        return False
    return len(relative_parts) <= maximum_depth


def find_candidate_csv_files(
    base_directory_path: Path, file_name_pattern: str, maximum_depth: int
) -> List[Path]:
    """
    Find CSV files starting from base_directory_path (NOT its parent),
    limited by maximum_depth, and matching file_name_pattern.
    """
    # IMPORTANT: search ONLY inside the given base directory
    search_root_path: Path = base_directory_path

    candidate_file_paths: List[Path] = []
    directories_to_visit: List[Path] = [search_root_path]
    visited_index: int = 0

    # Manual breadth-first walk with depth limit (simple and explicit)
    while visited_index < len(directories_to_visit):
        current_dir_path: Path = directories_to_visit[visited_index]
        visited_index += 1

        if not within_max_depth(search_root_path, current_dir_path, maximum_depth):
            continue

        for entry in current_dir_path.iterdir():
            if entry.is_dir():
                if within_max_depth(search_root_path, entry, maximum_depth):
                    directories_to_visit.append(entry)
            else:
                if fnmatch.fnmatch(entry.name, file_name_pattern):
                    # Extra safety: ignore files that are not actually under base_directory_path
                    try:
                        entry.relative_to(search_root_path)
                    except Exception:
                        continue
                    candidate_file_paths.append(entry)

    return candidate_file_paths


def iter_metric_rows(csv_file_path: Path) -> Iterator[MetricRow]:
    """
    Yield MetricRow objects from one CSV.
    Expects header to contain 'RMSE', 'R2', 'R2_Kelly'.
    """
    with csv_file_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        csv_reader = csv.reader(file_handle)
        try:
            header_row: List[str] = next(csv_reader)
        except StopIteration:
            return  # empty file

        normalized_header: List[str] = [h.strip() for h in header_row]

        # Column indices (simple, explicit)
        try:
            rmse_index: int = normalized_header.index("Forecasted_RMSE")
            r2_index: int = normalized_header.index("Forecasted_R2")
            r2_kelly_index: int = normalized_header.index("Forecasted_R2 Kelly")
        except ValueError:
            # Required columns not found -> skip file
            return

        for row_index_in_file, row_values in enumerate(csv_reader):
            # Guard against short rows
            if max(rmse_index, r2_index, r2_kelly_index) >= len(row_values):
                continue

            rmse_str: str = row_values[rmse_index].strip()
            r2_str: str = row_values[r2_index].strip()
            r2_kelly_str: str = row_values[r2_kelly_index].strip()

            rmse_val: Optional[float] = safe_float(rmse_str)
            r2_val: Optional[float] = safe_float(r2_str)
            r2_kelly_val: Optional[float] = safe_float(r2_kelly_str)

            if rmse_val is None or r2_val is None or r2_kelly_val is None:
                continue

            yield MetricRow(
                source_file_path=csv_file_path,
                row_index_in_file=row_index_in_file,
                rmse_value=rmse_val,
                r2_value=r2_val,
                r2_kelly_value=r2_kelly_val,
            )


def find_min_rmse(
    base_directory_path: Path, file_name_pattern: str, maximum_depth: int
) -> Optional[MetricRow]:
    """Search CSVs and return the row with the smallest RMSE (or None if none)."""
    candidate_files: List[Path] = find_candidate_csv_files(
        base_directory_path=base_directory_path,
        file_name_pattern=file_name_pattern,
        maximum_depth=maximum_depth,
    )

    best_row: Optional[MetricRow] = None

    for csv_path in candidate_files:
        for metric_row in iter_metric_rows(csv_path):
            if (best_row is None) or (metric_row.rmse_value < best_row.rmse_value):
                best_row = metric_row

    return best_row


def main() -> None:
    """CLI entry point."""
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Find the smallest RMSE across metric CSV files inside a given folder."
    )
    argument_parser.add_argument(
        "--base_dir",
        type=str,
        required=True,  # make it mandatory
        help="Base folder (e.g., examples/rideshare_3d_tensor/rideshare_timesfm_3d_168steps_Nov_9_original)",
    )
    argument_parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Max directory depth from base_dir to search (default: 2).",
    )
    argument_parser.add_argument(
        "--pattern",
        type=str,
        default="single_split_metrics_*.csv",
        help="Filename pattern to match (default: single_split_metrics_*.csv).",
    )

    parsed_args = argument_parser.parse_args()

    base_directory_path: Path = Path(
        parsed_args.base_dir
    ).resolve()  # Path: absolute base dir
    maximum_depth: int = int(parsed_args.max_depth)  # int: depth limit
    file_name_pattern: str = str(parsed_args.pattern)  # str: glob pattern

    best_row: Optional[MetricRow] = find_min_rmse(
        base_directory_path=base_directory_path,
        file_name_pattern=file_name_pattern,
        maximum_depth=maximum_depth,
    )

    if best_row is None:
        print("No valid metrics found (check folder, depth, and pattern).")
        sys.exit(1)

    print("Smallest RMSE found:")
    print(f"  RMSE     = {best_row.rmse_value:.6f}")
    print(f"  R2       = {best_row.r2_value:.6f}")
    print(f"  R2_Kelly = {best_row.r2_kelly_value:.6f}")
    print("")
    print("Context:")
    print(f"  File     = {best_row.source_file_path}")
    print(f"  Row idx  = {best_row.row_index_in_file}  (0-based, excluding header)")


if __name__ == "__main__":
    main()
