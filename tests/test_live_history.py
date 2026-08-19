from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import live_history
from chronos2_hourly.hourly_contract import local_delivery_day_index


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed_run(run_dir: Path, day: date) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    index = local_delivery_day_index(day, timezone="Europe/Paris")
    origin = live_history._cutoff_for_day(
        day, timezone="Europe/Paris"
    ).tz_convert("UTC")
    base = np.arange(len(index), dtype=float) + 50.0
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index.astype(str),
            "residual_corrected__q10": base - 5.0,
            "residual_corrected__q50": base,
            "residual_corrected__q90": base + 5.0,
            "mkonline_blend__q10": base - 4.0,
            "mkonline_blend__q50": base + 1.0,
            "mkonline_blend__q90": base + 6.0,
            "mkonline_primary__q50": base + 2.0,
            "storm_evaluation_only__q50": base + 3.0,
            "actual": base + 0.5,
            "fold_id": 1.0,
            "forecast_origin_utc": str(origin),
            "mkonline_blend_forecast_origin_utc": str(origin),
        }
    )
    backtest = run_dir / live_history.BACKTEST_NAME
    frame.to_csv(backtest, index=False, compression="gzip")
    (run_dir / live_history.METRICS_NAME).write_text(
        json.dumps(
            {
                "training_diagnostics": {
                    "evaluation_start_local_date": day.isoformat(),
                    "evaluation_end_local_date": day.isoformat(),
                }
            }
        ),
        encoding="utf-8",
    )
    return backtest


def _write_forecast_run(
    run_dir: Path,
    day: date,
    *,
    run_type: str,
    offset: float = 0.0,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    index = local_delivery_day_index(day, timezone="Europe/Paris")
    cutoff = live_history._cutoff_for_day(
        day, timezone="Europe/Paris"
    )
    values = np.arange(len(index), dtype=float) + 50.0 + offset
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index.astype(str),
            "residual_corrected__q10": values - 5.0,
            "residual_corrected__q50": values,
            "residual_corrected__q90": values + 5.0,
            "mkonline_blend__q10": values - 4.0,
            "mkonline_blend__q50": values + 1.0,
            "mkonline_blend__q90": values + 6.0,
            "mkonline_primary__q50": values + 2.0,
            "mkonline_blend_forecast_origin_utc": str(cutoff.tz_convert("UTC")),
        }
    )
    if run_type == "live_day_ahead":
        frame["forecast_origin_utc"] = str(cutoff.tz_convert("UTC"))
    forecast = run_dir / live_history.FORECAST_NAME
    frame.to_csv(forecast, index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": run_type,
                "zone": "FR",
                "timezone": "Europe/Paris",
                "target_series": (
                    "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
                ),
                "candidate_model": "mkonline_blend",
                "prediction_mode": "mkonline_blend",
                "prediction_inputs": ["autonomous", "mkonline_primary"],
                "storm_used_as_feature": False,
                "delivery_day_local": day.isoformat(),
                "forecast_cutoff_local": str(cutoff),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "artifacts": [
                    {
                        "path": live_history.FORECAST_NAME,
                        "role": "run_artifact",
                        "sha256": _sha256(forecast),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return forecast


def _fake_storm(
    path: Path,
    *,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    values = pd.Series(55.0, index=expected_index)
    selected = pd.DataFrame(
        {
            "value_time_utc": expected_index,
            "snapshot_time_utc": expected_index - pd.Timedelta(days=1),
            "revision_time_utc": expected_index - pd.Timedelta(days=1),
            "value": values.to_numpy(),
        }
    )
    return values, selected, {
        "role": "evaluation_only_comparator",
        "hours": len(expected_index),
        "used_for_prediction": False,
    }


def test_writes_separate_strict_history_with_scopes_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = tmp_path / "sealed"
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    staging = tmp_path / "staging"
    staging.mkdir()
    # These staging benchmark artifacts must remain byte-for-byte unchanged.
    staging_backtest = staging / live_history.BACKTEST_NAME
    staging_metrics = staging / live_history.METRICS_NAME
    staging_backtest.write_bytes(b"copied-backtest-marker")
    staging_metrics.write_bytes(b"copied-metrics-marker")
    immutable_paths = [
        _write_sealed_run(sealed, date(2026, 1, 1)),
        _write_forecast_run(
            replay_root / "fr_day_ahead_2026-01-02",
            date(2026, 1, 2),
            run_type="pit_replay",
        ),
        _write_forecast_run(
            live_root / "fr_day_ahead_2026-01-03",
            date(2026, 1, 3),
            run_type="live_day_ahead",
            offset=10.0,
        ),
        # The current delivery exists but must never be scored yet.
        _write_forecast_run(
            live_root / "fr_day_ahead_2026-01-04",
            date(2026, 1, 4),
            run_type="live_day_ahead",
            offset=20.0,
        ),
    ]
    immutable_hashes = {path: _sha256(path) for path in immutable_paths}
    staging_hashes = {
        staging_backtest: _sha256(staging_backtest),
        staging_metrics: _sha256(staging_metrics),
    }
    monkeypatch.setattr(live_history, "_load_storm_evaluation_only", _fake_storm)
    realized_index = local_delivery_day_index(
        date(2026, 1, 2), timezone="Europe/Paris"
    ).append(
        local_delivery_day_index(date(2026, 1, 3), timezone="Europe/Paris")
    )
    actual = pd.Series(
        np.arange(len(realized_index), dtype=float) + 45.0,
        index=realized_index,
    )

    audit = live_history.update_live_statistics_history(
        staging_run_dir=staging,
        sealed_benchmark_run=sealed,
        live_output_root=live_root,
        replay_output_root=replay_root,
        current_delivery_day=date(2026, 1, 4),
        canonical_target=actual,
        storm_pit_path=tmp_path / "storm.parquet",
    )

    assert audit["evaluated_realized_days"] == ["2026-01-02", "2026-01-03"]
    assert audit["n_evaluated_realized_hours"] == 48
    assert audit["n_total_statistics_hours"] == 72
    assert audit["staging_backtest_rewritten"] is False
    assert audit["staging_metrics_rewritten"] is False
    assert all(_sha256(path) == digest for path, digest in immutable_hashes.items())
    assert all(_sha256(path) == digest for path, digest in staging_hashes.items())

    history_path = staging / live_history.STATISTICS_HISTORY_NAME
    audit_path = staging / live_history.STATISTICS_AUDIT_NAME
    assert history_path.is_file() and audit_path.is_file()
    history = pd.read_csv(history_path)
    delivery = pd.to_datetime(history["delivery_start_utc"], utc=True)
    days = delivery.dt.tz_convert("Europe/Paris").dt.date
    assert set(days) == {
        date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)
    }
    assert set(history["statistics_scope"]) == {
        "sealed_benchmark", "realized_pit_replay", "realized_live"
    }
    replay = history.loc[days == date(2026, 1, 2)]
    live = history.loc[days == date(2026, 1, 3)]
    assert replay["statistics_run_type"].eq("pit_replay").all()
    assert live["statistics_run_type"].eq("live_day_ahead").all()
    assert replay["actual"].notna().all() and live["actual"].notna().all()
    assert replay["storm_evaluation_only__q50"].eq(55.0).all()
    assert live["storm_evaluation_only__q50"].eq(55.0).all()


def test_strict_contiguity_requires_every_post_benchmark_day(
    tmp_path: Path,
) -> None:
    sealed = tmp_path / "sealed"
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    _write_sealed_run(sealed, date(2026, 1, 1))
    _write_forecast_run(
        replay_root / "fr_day_ahead_2026-01-02",
        date(2026, 1, 2),
        run_type="pit_replay",
    )
    live_root.mkdir()
    actual = pd.Series(
        50.0,
        index=local_delivery_day_index(date(2026, 1, 2), timezone="Europe/Paris"),
    )
    with pytest.raises(ValueError, match="2026-01-03"):
        live_history.update_live_statistics_history(
            staging_run_dir=tmp_path / "staging",
            sealed_benchmark_run=sealed,
            live_output_root=live_root,
            replay_output_root=replay_root,
            current_delivery_day=date(2026, 1, 4),
            canonical_target=actual,
            storm_pit_path=tmp_path / "storm.parquet",
        )


def test_sealed_benchmark_forecast_is_never_used_as_operational_seed(
    tmp_path: Path,
) -> None:
    sealed = tmp_path / "sealed"
    _write_sealed_run(sealed, date(2026, 1, 1))
    # A development forecast exists inside the benchmark directory.  It must
    # not satisfy the explicit replay/live archive contract for 2 January.
    _write_forecast_run(
        sealed,
        date(2026, 1, 2),
        run_type="pit_replay",
    )
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    live_root.mkdir()
    replay_root.mkdir()
    actual = pd.Series(
        50.0,
        index=local_delivery_day_index(date(2026, 1, 2), timezone="Europe/Paris"),
    )
    with pytest.raises(ValueError, match="2026-01-02"):
        live_history.update_live_statistics_history(
            staging_run_dir=tmp_path / "staging",
            sealed_benchmark_run=sealed,
            live_output_root=live_root,
            replay_output_root=replay_root,
            current_delivery_day=date(2026, 1, 3),
            canonical_target=actual,
            storm_pit_path=tmp_path / "storm.parquet",
        )


def test_discovery_rejects_checksum_tampering(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    live_root.mkdir()
    forecast = _write_forecast_run(
        replay_root / "fr_day_ahead_2026-01-02",
        date(2026, 1, 2),
        run_type="pit_replay",
    )
    forecast.write_text(forecast.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        live_history.discover_archived_forecasts(
            live_output_root=live_root,
            replay_output_root=replay_root,
            current_delivery_day=date(2026, 1, 3),
            first_history_day=date(2026, 1, 2),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("zone", "DE", "zone"),
        ("target_series", "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh", "target_series"),
        ("candidate_model", "residual_corrected", "candidate_model"),
        ("prediction_mode", "autonomous_only", "prediction_mode"),
        ("storm_used_as_feature", True, "storm_used_as_feature"),
    ],
)
def test_discovery_rejects_cross_zone_or_model_archive(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    live_root.mkdir()
    run = replay_root / "fr_day_ahead_2026-01-02"
    _write_forecast_run(
        run,
        date(2026, 1, 2),
        run_type="pit_replay",
    )
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        live_history.discover_archived_forecasts(
            live_output_root=live_root,
            replay_output_root=replay_root,
            current_delivery_day=date(2026, 1, 3),
            first_history_day=date(2026, 1, 2),
        )


def test_discovery_accepts_only_the_exact_legacy_fr_identity(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    live_root.mkdir()
    run = replay_root / "fr_day_ahead_2026-01-02"
    _write_forecast_run(run, date(2026, 1, 2), run_type="pit_replay")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("target_series", "candidate_model", "prediction_mode"):
        manifest.pop(key)
    manifest["prediction_inputs"] = [
        "autonomous_extended_residual",
        "41551_native",
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    forecasts, _audit = live_history.discover_archived_forecasts(
        live_output_root=live_root,
        replay_output_root=replay_root,
        current_delivery_day=date(2026, 1, 3),
        first_history_day=date(2026, 1, 2),
    )
    assert set(forecasts) == {date(2026, 1, 2)}

    manifest["prediction_inputs"] = ["autonomous_extended_residual", "41550_native"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy FR prediction_inputs"):
        live_history.discover_archived_forecasts(
            live_output_root=live_root,
            replay_output_root=replay_root,
            current_delivery_day=date(2026, 1, 3),
            first_history_day=date(2026, 1, 2),
        )


def test_non_fr_archive_cannot_use_legacy_missing_identity(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    replay_root = tmp_path / "replays"
    live_root.mkdir()
    run = replay_root / "de_day_ahead_2026-01-02"
    _write_forecast_run(run, date(2026, 1, 2), run_type="pit_replay")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["zone"] = "DE"
    manifest["timezone"] = "Europe/Berlin"
    for key in ("target_series", "candidate_model", "prediction_mode"):
        manifest.pop(key)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete modern archive identity"):
        live_history.discover_archived_forecasts(
            live_output_root=live_root,
            replay_output_root=replay_root,
            current_delivery_day=date(2026, 1, 3),
            first_history_day=date(2026, 1, 2),
            zone="DE",
            timezone="Europe/Berlin",
            target_series="power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh",
            candidate_model="residual_corrected",
            prediction_mode="autonomous_only",
        )


@pytest.mark.parametrize(
    ("delivery_day", "expected_cutoff"),
    [
        (date(2026, 3, 30), "2026-03-29 08:00:00+02:00"),
        (date(2026, 10, 26), "2026-10-25 08:00:00+01:00"),
    ],
)
def test_cutoff_is_civil_0800_across_dst(
    delivery_day: date,
    expected_cutoff: str,
) -> None:
    assert str(
        live_history._cutoff_for_day(
            delivery_day, timezone="Europe/Paris"
        )
    ) == expected_cutoff
