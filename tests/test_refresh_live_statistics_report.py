from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

import refresh_live_statistics_report as refresher


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_run(run_dir: Path) -> Path:
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    forecast = run_dir / "forecast_hourly_fr.csv"
    forecast.write_text(
        "delivery_start_utc,q50\n2026-08-13T22:00:00Z,42.0\n",
        encoding="utf-8",
    )
    manifest = run_dir / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_type": "live_day_ahead",
                "delivery_day_local": "2026-08-14",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "timestamp": [
                "2026-08-12T22:00:00+00:00",
                "2026-08-12T23:00:00+00:00",
            ],
            "target": [40.0, 41.0],
        }
    ).to_csv(inputs / "aligned_inputs.csv.gz", index=False, compression="gzip")
    checksum = run_dir / "artifact_checksums.json"
    checksum.write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "artifacts": [
                    {
                        "path": "forecast_hourly_fr.csv",
                        "role": "run_artifact",
                        "sha256": _sha256(forecast),
                    },
                    {
                        "path": "run_manifest.json",
                        "role": "run_artifact",
                        "sha256": _sha256(manifest),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return forecast


def test_verify_published_run_checks_every_declared_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    forecast = _write_source_run(source)

    verified = refresher._verify_published_run(source)
    assert verified["forecast_hourly_fr.csv"] == _sha256(forecast)

    (source / "run_manifest.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Archive publiee modifiee"):
        refresher._verify_published_run(source)


def test_canonical_target_is_unique_and_utc(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_run(source)

    target = refresher._canonical_target(source)
    assert str(target.index.tz) == "UTC"
    assert not target.index.has_duplicates
    assert target.tolist() == [40.0, 41.0]

    duplicate = pd.DataFrame(
        {
            "timestamp": ["2026-08-12T22:00:00Z"] * 2,
            "target": [40.0, 41.0],
        }
    )
    duplicate.to_csv(
        source / "inputs" / "aligned_inputs.csv.gz",
        index=False,
        compression="gzip",
    )
    with pytest.raises(ValueError, match="dupliquee"):
        refresher._canonical_target(source)


def test_publish_refuses_an_existing_immutable_snapshot(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()

    with pytest.raises(FileExistsError, match="reste immuable"):
        refresher._publish(staging, output)
    assert staging.is_dir()
    assert output.is_dir()


def test_main_creates_derived_snapshot_without_changing_source_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    forecast = _write_source_run(source)
    source_forecast_bytes = forecast.read_bytes()
    live_root = tmp_path / "live"
    output = live_root / "_reports" / "snapshot"
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    config = tmp_path / "live.yaml"
    config.write_text(
        "\n".join(
            [
                "live:",
                f"  output_root: '{live_root.as_posix()}'",
                f"  sealed_benchmark_run: '{benchmark.as_posix()}'",
                "report:",
                "  forecast_history_hours: 168",
            ]
        ),
        encoding="utf-8",
    )
    events: list[str] = []

    def fake_history(**kwargs):
        events.append("history")
        staging = Path(kwargs["staging_run_dir"])
        (staging / "statistics_history_hourly.csv.gz").write_bytes(b"history")
        (staging / "statistics_history_audit.json").write_text(
            "{}", encoding="utf-8"
        )
        return {"status": "complete"}

    def fake_report(run_dir, *, output_path, **_kwargs):
        events.append("html")
        assert (Path(run_dir) / "statistics_history_hourly.csv.gz").is_file()
        Path(output_path).write_text("<html>snapshot</html>", encoding="utf-8")

    monkeypatch.setattr(refresher, "update_live_statistics_history", fake_history)
    monkeypatch.setattr(
        refresher,
        "_load_native_storm_snapshot",
        lambda **_kwargs: (
            pd.Series(42.0, index=pd.date_range("2025-08-12", periods=24, freq="h")),
            {
                "requested_series": "power.price.fr.euromwh.h.fcst.3mv.storm",
                "used_for_prediction": False,
            },
        ),
    )
    monkeypatch.setattr(refresher, "write_hourly_html_report", fake_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_live_statistics_report.py",
            "--config",
            str(config),
            "--source-run",
            str(source),
            "--output-dir",
            str(output),
        ],
    )

    assert refresher.main() == 0
    assert events == ["history", "html"]
    assert forecast.read_bytes() == source_forecast_bytes
    assert (output / "forecast_hourly_fr.csv").read_bytes() == source_forecast_bytes
    derived = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert derived["run_type"] == "live_statistics_report_snapshot"
    assert derived["forecast_recomputed"] is False
    assert derived["forecast_rewritten"] is False
    assert derived["source_forecast_sha256"] == hashlib.sha256(
        source_forecast_bytes
    ).hexdigest()
