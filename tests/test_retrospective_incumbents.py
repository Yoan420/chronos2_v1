from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.retrospective_incumbents import (
    IncumbentComparisonError,
    load_incumbent_comparison,
)


def _hours(day):
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = pd.Timestamp(date.fromisoformat(day) + timedelta(days=1)).tz_localize("Europe/Paris")
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _fixture(root: Path, day="2026-09-08", variants=("autonomous", "kalman")):
    batch = root / "runs" / "exports" / day
    manifest = {"schema_version": 1, "delivery_day": day, "mode": "both", "exports": []}
    for variant in variants:
        path = batch / "fr" / variant / f"forecast_fr_{day}_{variant}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        source = "residual_corrected" if variant == "autonomous" else "residual_kalman"
        frame = pd.DataFrame({"zone": "FR", "forecast_variant": variant, "source_model": source,
            "delivery_start_utc": _hours(day), "q10": 90., "q50": 100., "q90": 110.,
            "price_eur_mwh": -9999., "actual": -9999.})
        frame.to_csv(path, index=False)
        manifest["exports"].append({"zone": "FR", "variant": variant, "source_model": source,
            "csv": {"path": path.relative_to(batch).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "current_batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return batch, manifest


def _save_manifest(batch, manifest):
    (batch / "current_batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _mutate_csv(batch, manifest, change):
    entry = manifest["exports"][0]["csv"]
    path = batch / entry["path"]
    frame = change(pd.read_csv(path))
    frame.to_csv(path, index=False)
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _save_manifest(batch, manifest)


@pytest.mark.parametrize("day,count", [("2026-09-08", 24), ("2026-03-29", 23), ("2025-10-26", 25)])
def test_physical_day_verified_without_importing_observed_price(tmp_path, day, count):
    _fixture(tmp_path, day)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    frame, audit = load_incumbent_comparison(tmp_path, delivery_day=day, zones=["FR"])
    assert len(frame) == count
    assert set(frame.columns) == {"zone", "delivery_start_utc", *(
        f"incumbent_{variant}__q{q}" for variant in ("autonomous", "kalman") for q in (10, 50, 90))}
    assert frame["incumbent_autonomous__q50"].eq(100).all()
    assert frame["incumbent_kalman__q50"].eq(100).all()
    assert not audit["observations_imported"] and not audit["calibration_input"]
    assert len(audit["batch_manifest"]["sha256"]) == 64
    assert all(row["status"] == "available" and not row["upstream_checkpoint_verified"] for row in audit["comparators"])
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_missing_manifest_leaves_empty_comparators_and_no_files(tmp_path):
    frame, audit = load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR", "DE"])
    assert len(frame) == 48
    assert frame.filter(like="incumbent_").isna().all().all()
    assert audit["batch_manifest"] is None
    assert {row["reason"] for row in audit["comparators"]} == {"batch_manifest_unavailable"}
    assert not list(tmp_path.iterdir())


def test_missing_zone_and_variant_leave_nan(tmp_path):
    _fixture(tmp_path, variants=("autonomous",))
    frame, audit = load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR", "DE"])
    assert frame.loc[frame.zone.eq("FR"), "incumbent_autonomous__q50"].eq(100).all()
    assert frame["incumbent_kalman__q50"].isna().all()
    assert frame.loc[frame.zone.eq("DE")].filter(like="incumbent_").isna().all().all()
    assert sum(row["status"] == "unavailable" for row in audit["comparators"]) == 3


def test_csv_tampering_fails_instead_of_omitting_comparator(tmp_path):
    batch, manifest = _fixture(tmp_path)
    path = batch / manifest["exports"][0]["csv"]["path"]
    path.write_bytes(path.read_bytes().replace(b"100.0", b"105.0"))
    with pytest.raises(IncumbentComparisonError, match="SHA256"):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])


@pytest.mark.parametrize("path", ["../outside.csv", "C:/outside.csv", "/outside.csv", "fr/kalman/forecast_fr_2026-09-08_kalman.csv", "de/autonomous/forecast_de_2026-09-08_autonomous.csv"])
def test_manifest_path_escape_or_wrong_identity_fails(tmp_path, path):
    batch, manifest = _fixture(tmp_path)
    manifest["exports"][0]["csv"]["path"] = path
    _save_manifest(batch, manifest)
    with pytest.raises(IncumbentComparisonError, match="chemin CSV"):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])


@pytest.mark.parametrize("kind", ["missing_csv", "duplicate", "source_model", "delivery_day", "schema"])
def test_invalid_existing_manifest_is_not_silently_unavailable(tmp_path, kind):
    batch, manifest = _fixture(tmp_path)
    if kind == "missing_csv":
        (batch / manifest["exports"][0]["csv"]["path"]).unlink()
    elif kind == "duplicate":
        manifest["exports"].append(manifest["exports"][0])
    elif kind == "source_model":
        manifest["exports"][0]["source_model"] = "mkonline_blend"
    elif kind == "delivery_day":
        manifest["delivery_day"] = "2026-09-07"
    else:
        manifest["schema_version"] = True
    _save_manifest(batch, manifest)
    with pytest.raises(IncumbentComparisonError):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])


@pytest.mark.parametrize("column,value", [("zone", "DE"), ("forecast_variant", "blend"), ("source_model", "chronos2"), ("q50", np.inf), ("q50", np.nan), ("q50", 120.), ("delivery_start_utc", "2026-09-08 00:00:00")])
def test_csv_semantic_invalidity_fails_even_with_matching_sha(tmp_path, column, value):
    batch, manifest = _fixture(tmp_path)
    def mutate(frame):
        frame.loc[0, column] = value
        return frame
    _mutate_csv(batch, manifest, mutate)
    with pytest.raises(IncumbentComparisonError):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])


@pytest.mark.parametrize("change", [lambda frame: frame.iloc[:-1], lambda frame: pd.concat([frame, frame.iloc[:1]]), lambda frame: frame.drop(columns=["source_model"])])
def test_incomplete_duplicate_or_missing_columns_fail(tmp_path, change):
    batch, manifest = _fixture(tmp_path)
    _mutate_csv(batch, manifest, change)
    with pytest.raises(IncumbentComparisonError):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])


def test_unsorted_but_complete_export_aligns_by_timestamp(tmp_path):
    batch, manifest = _fixture(tmp_path)
    def mutate(frame):
        frame["q50"] = 95 + np.arange(len(frame)) / 10
        return frame.iloc[::-1]
    _mutate_csv(batch, manifest, mutate)
    result, _ = load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])
    np.testing.assert_allclose(result["incumbent_autonomous__q50"], 95 + np.arange(24) / 10)


def test_duplicate_json_key_is_rejected(tmp_path):
    batch, _ = _fixture(tmp_path)
    path = batch / "current_batch_manifest.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(IncumbentComparisonError, match="ambigu"):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])


@pytest.mark.parametrize("day,zones", [("../2026-09-08", ["FR"]), ("2026-09-08", ["fr"]), ("2026-09-08", []), ("2026-09-08", ["FR", "FR"])])
def test_invalid_request_is_rejected(tmp_path, day, zones):
    with pytest.raises(IncumbentComparisonError):
        load_incumbent_comparison(tmp_path, delivery_day=day, zones=zones)


def test_symlink_refused_when_platform_supports_it(tmp_path):
    batch, manifest = _fixture(tmp_path)
    path = batch / manifest["exports"][0]["csv"]["path"]
    original = path.with_suffix(".original.csv")
    path.rename(original)
    try:
        path.symlink_to(original)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this Windows account.")
    with pytest.raises(IncumbentComparisonError, match="Lien ou reparse"):
        load_incumbent_comparison(tmp_path, delivery_day="2026-09-08", zones=["FR"])
