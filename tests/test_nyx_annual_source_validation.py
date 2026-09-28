from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_source_validation as validator
from chronos2_hourly import nyx_annual_jao_source as jao_source
from test_nyx_annual_jao_source import FakeClient, _fetch
from test_run_nyx_annual_hydro_source import payload, Session


def test_jao_raw_packet_reconstructs_without_original_cache(tmp_path):
    day = date(2026, 9, 30)
    bundle = tmp_path / "bundle"
    prefix = jao_source.SOURCE_SUBDIR
    captures = bundle / prefix / "captures"
    cutoff = jao_source.jao.expected_cutoff_utc(day)
    jao_source.capture_day(day=day, cache_root=captures, client=FakeClient(_fetch(day)),
                           now_utc=cutoff - pd.Timedelta(minutes=9))
    verified = jao_source.verify_daily_capture(captures, day)
    _, current, _ = validator.gate.delivery_grid(day.isoformat())
    frame, _ = jao_source.build_strict_history_features(captures, current)
    feature_path = bundle / prefix / jao_source.FEATURE_NAME
    frame.to_parquet(feature_path)
    ledger_path = bundle / prefix / jao_source.LEDGER_NAME
    ledger_path.write_text(json.dumps({"protocol": jao_source.LEDGER_PROTOCOL,
        "delivery_day": day.isoformat(), "captures": [verified], "captured_days": 1,
        "missing_days": []}), encoding="utf-8")
    paths = [*jao_source._paths(captures, day), feature_path, ledger_path]
    receipt = {"artifact_sha256": {p.relative_to(bundle).as_posix(): validator.gate.sha256(p) for p in paths}}
    result = validator._jao(bundle, day.isoformat(), receipt, current)
    assert result["daily_raw_captures_recomputed"] == 1
    frame.iloc[0, 0] += 1
    frame.to_parquet(feature_path)
    receipt["artifact_sha256"][feature_path.relative_to(bundle).as_posix()] = validator.gate.sha256(feature_path)
    with pytest.raises(ValueError, match="raw reconstruction"):
        validator._jao(bundle, day.isoformat(), receipt, current)


def test_hydro_zip_raw_packet_recomputed_including_dst(tmp_path):
    import run_nyx_annual_hydro_source as hydro
    day = "2026-10-25"
    archive = tmp_path / "archive"
    hydro.capture(day, archive, session=Session(payload(day)),
                  now_utc=hydro.source_window(day)[-1] - pd.Timedelta(minutes=5))
    frame, _ = hydro.verify_capture(archive / day, day)
    bundle = tmp_path / "bundle"
    root = bundle / "source_artifacts/public_hydro"
    root.mkdir(parents=True)
    frame.to_parquet(root / "features.parquet")
    zip_path = root / "pit_captures.zip"
    with zipfile.ZipFile(zip_path, "w") as output:
        for name in hydro.NAMES:
            output.write(archive / day / name, f"{day}/{name}")
    receipt = {"artifact_sha256": {p.relative_to(bundle).as_posix(): validator.gate.sha256(p)
        for p in (root / "features.parquet", zip_path)},
        "capture_receipt_sha256": {day: validator.gate.sha256(archive / day / "capture.json")}}
    result = validator._captured_features(bundle, day, receipt, frame.index, group="public_hydro")
    assert result["actual_capture_before_own_cutoff_verified"] is True
    assert len(frame) == 25
    frame.iloc[0, 0] += 1
    frame.to_parquet(root / "features.parquet")
    receipt["artifact_sha256"]["source_artifacts/public_hydro/features.parquet"] = validator.gate.sha256(root / "features.parquet")
    with pytest.raises(ValueError, match="raw reconstruction"):
        validator._captured_features(bundle, day, receipt, frame.index, group="public_hydro")


def test_zip_member_escape_rejected_before_extraction(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="ZIP inventory differs"):
        validator._extract_capture_zip(path, tmp_path / "extract", ["2026-09-30"], ("FR.json",))
    assert not (tmp_path / "outside.txt").exists()


def test_thermal_validates_each_own_cutoff_even_if_bundle_hash_is_updated(tmp_path):
    import run_nyx_annual_thermal_source as source
    from test_run_nyx_annual_thermal_source import Saturn
    day = "2026-09-29"
    plan = source.plan()
    _, _, cutoff = validator.gate.delivery_grid(day)
    daily, evidence = source.collect(Saturn(plan["specs"]), day, now_utc=cutoff)
    path = source.publish(tmp_path, day, daily, evidence, plan, now_utc=cutoff)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert validator._thermal(tmp_path, day, receipt)["source_daily_states_verified"] == 366 * 13
    relative = f"source_artifacts/thermal_capacity/{source.SOURCES[0]}.parquet"
    source_path = tmp_path / relative
    frame = pd.read_parquet(source_path)
    frame.loc[frame.index[0], "asof_query_utc"] += pd.Timedelta(hours=1)
    frame.to_parquet(source_path)
    receipt["artifact_sha256"][relative] = validator.gate.sha256(source_path)
    with pytest.raises(ValueError, match="own daily cutoff"):
        validator._thermal(tmp_path, day, receipt)


def test_missing_real_source_packet_cannot_be_qualified(tmp_path):
    with pytest.raises(ValueError, match="source receipt missing"):
        validator.validate_source_packet(tmp_path, "2026-09-30")
