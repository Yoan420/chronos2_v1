"""Verify annual CWE lineage or project 123 from supplied live 503 and 449.

Historical parity requires ignored local research archives.  The prospective
projection requires externally produced 503- and 449-column matrices in the
named bundle; it never collects sources or certifies publication vintages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_annual_feature_projection import FULL, POOLED, COMPACT, project_country
from chronos2_hourly.nyx_annual_live_preflight import (
    ROOT, ZONES, delivery_grid, load_schema, validate_feature_frame,
)


HISTORICAL_ROOT = ROOT / "runs/experiments/nyx_improvement_to20260923/feature_sets"
HISTORICAL_DIRS = {
    FULL: "pooled_jao_refresh_v1",
    POOLED: "pooled_fundamentals_v1",
    COMPACT: "pooled_jao_refresh_v1/compact",
}
MANIFEST = ROOT / "config/nyx_annual_cwe_historical.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=True)
    # pandas treats +0.0 and -0.0 as equal; preserve the archived IEEE bits.
    for name in right:
        if pd.api.types.is_float_dtype(right[name].dtype):
            original = right[name].to_numpy()
            observed = left[name].to_numpy()
            zeros = original == 0
            if not np.array_equal(np.signbit(observed[zeros]), np.signbit(original[zeros])):
                raise ValueError(f"{label}/{name}: signed zero changed")


def historical_parity(root: Path = HISTORICAL_ROOT) -> dict:
    """Compare all twelve selected archives and quantify the known JAO split."""
    schema = load_schema()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["source_feature_matrices"]
    checks = {}
    for zone in ZONES:
        base = root / HISTORICAL_DIRS[FULL] / f"features_{zone}.parquet"
        if not base.is_file():
            raise FileNotFoundError(f"Historical 503 matrix absent: {base}")
        if _sha256(base) != manifest[f"full_{zone}"]["sha256"]:
            raise ValueError(f"Historical 503 source SHA differs: {zone}")
        full = pd.read_parquet(base)
        pooled_path = root / HISTORICAL_DIRS[POOLED] / f"features_{zone}.parquet"
        if not pooled_path.is_file():
            raise FileNotFoundError(f"Historical 449 matrix absent: {pooled_path}")
        if _sha256(pooled_path) != manifest[f"pooled_{zone}"]["sha256"]:
            raise ValueError(f"Historical 449 source SHA differs: {zone}")
        pooled = pd.read_parquet(pooled_path)
        compact = project_country(full, pooled, zone=zone,
                                  expected_index=full.index, schema=schema)
        validate_feature_frame(full, schema["families"][FULL]["columns"][zone],
                               full.index, f"{zone}/503")
        checks[f"{zone}/{FULL}"] = {"rows": len(full), "columns": len(full.columns),
                                    "source_sha256": _sha256(base)}
        jao = [name for name in pooled if name.startswith("extra_jao_")]
        differs = pd.DataFrame({name: ~(full[name].eq(pooled[name]) |
                                        (full[name].isna() & pooled[name].isna()))
                                for name in jao}, index=full.index)
        checks[f"{zone}/{POOLED}"] = {
            "rows": len(pooled), "columns": len(pooled.columns),
            "source_sha256": _sha256(pooled_path),
            "non_jao_columns_exact": True, "jao_difference_hours": int(differs.any(axis=1).sum()),
            "jao_difference_cells": int(differs.to_numpy().sum()),
        }
        compact_path = root / HISTORICAL_DIRS[COMPACT] / f"features_{zone}.parquet"
        if not compact_path.is_file():
            raise FileNotFoundError(f"Historical 123 matrix absent: {compact_path}")
        if _sha256(compact_path) != manifest[f"compact_{zone}"]["sha256"]:
            raise ValueError(f"Historical 123 source SHA differs: {zone}")
        expected = pd.read_parquet(compact_path)
        _exact(compact, expected, f"{zone}/{COMPACT}")
        checks[f"{zone}/{COMPACT}"] = {
            "rows": len(expected), "columns": len(expected.columns),
            "source_sha256": _sha256(compact_path), "bit_exact_projection": True,
        }
    return {"protocol": "nyx_annual_feature_projection_parity_v1",
            "historical_archive_only": True, "passed": True, "checks": checks}


def project_live_bundle(bundle: Path, delivery_day: str) -> dict:
    """Write compact projections after validating four 503/449 source pairs.

    This writes feature files only.  The source receipts, reference forecast,
    baseline and CPU qualification gate are checked elsewhere.
    """
    bundle = bundle.resolve()
    full_index, _, _ = delivery_grid(delivery_day)
    schema = load_schema()
    # Validate every source before changing the bundle.
    pending = {}
    sources = {}
    for zone in ZONES:
        source = bundle / "features" / FULL / f"{zone}.parquet"
        if not source.is_file():
            raise FileNotFoundError(f"Prospective 503 matrix absent: {source}")
        pooled_path = bundle / "features" / POOLED / f"{zone}.parquet"
        if not pooled_path.is_file():
            raise FileNotFoundError(f"Prospective 449 matrix absent: {pooled_path}")
        initial = {"full503": _sha256(source), "pooled449": _sha256(pooled_path)}
        pending[zone] = project_country(pd.read_parquet(source), pd.read_parquet(pooled_path),
                                        zone=zone, expected_index=full_index, schema=schema)
        if initial != {"full503": _sha256(source), "pooled449": _sha256(pooled_path)}:
            raise ValueError(f"{zone}: input matrix changed while projecting")
        sources[zone] = initial
    for zone, hashes in sources.items():
        full_path = bundle / "features" / FULL / f"{zone}.parquet"
        pooled_path = bundle / "features" / POOLED / f"{zone}.parquet"
        if hashes != {"full503": _sha256(full_path), "pooled449": _sha256(pooled_path)}:
            raise ValueError(f"{zone}: input matrix changed before projection write")
    hashes = {}
    for zone in ZONES:
        target = bundle / "features" / COMPACT / f"{zone}.parquet"
        if target.exists():
            _exact(pd.read_parquet(target), pending[zone], f"{zone}/{COMPACT}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".nyx-projection-", suffix=".parquet",
                                             dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                pending[zone].to_parquet(temporary)
                _exact(pd.read_parquet(temporary), pending[zone], f"{zone}/{COMPACT}")
                try:
                    temporary.rename(target)
                except FileExistsError:
                    _exact(pd.read_parquet(target), pending[zone], f"{zone}/{COMPACT}")
            finally:
                temporary.unlink(missing_ok=True)
        hashes[f"{COMPACT}/{zone}"] = _sha256(target)
    for zone, hashes_by_family in sources.items():
        full_path = bundle / "features" / FULL / f"{zone}.parquet"
        pooled_path = bundle / "features" / POOLED / f"{zone}.parquet"
        if hashes_by_family != {"full503": _sha256(full_path),
                                "pooled449": _sha256(pooled_path)}:
            raise ValueError(f"{zone}: input matrix changed during projection write")
    return {"protocol": "nyx_annual_feature_projection_live_v1",
            "delivery_day": delivery_day, "bundle": str(bundle),
            "projection_only": True, "source_503_449_sha256": sources,
            "projection_sha256": hashes,
            "source_publication_and_503_449_production_certified": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("historical-parity", "project-live"), required=True)
    parser.add_argument("--historical-root", type=Path, default=HISTORICAL_ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--delivery-day")
    args = parser.parse_args(argv)
    if args.action == "historical-parity":
        result = historical_parity(args.historical_root)
    else:
        if args.bundle is None or args.delivery_day is None:
            parser.error("--bundle and --delivery-day are required for project-live")
        result = project_live_bundle(args.bundle, args.delivery_day)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
