from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.chronos_residual_load import (
    EXPECTED_ALIASES,
    RESIDUAL_LOAD_TREATMENT_COLUMNS,
)
from chronos2_hourly.historical_residual_comparison import (
    CHALLENGER_MODEL,
    CONTROL_MODEL,
    FINAL_HOURS,
    HistoricalResidualComparisonError,
    publish_historical_residual_comparison,
)


TIMEZONE = "Europe/Paris"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = []
    for delivery_day in index.tz_convert(TIMEZONE).date:
        local = pd.Timestamp(
            datetime.combine(delivery_day - timedelta(days=1), time(8, 0))
        ).tz_localize(TIMEZONE)
        values.append(local.tz_convert("UTC"))
    return pd.DatetimeIndex(values)


def _seal(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_checksums.json":
            continue
        relative = path.relative_to(run_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "role": (
                    "materialized_input"
                    if relative.startswith("inputs/")
                    else "run_artifact"
                ),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (run_dir / "artifact_checksums.json").write_text(
        json.dumps({"algorithm": "sha256", "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    *,
    name: str,
    residual_load_source: str,
    prediction_offset: float,
) -> Path:
    run = root / name
    inputs = run / "inputs"
    inputs.mkdir(parents=True)

    start = pd.Timestamp("2025-08-12").tz_localize(TIMEZONE)
    end = pd.Timestamp("2026-08-12").tz_localize(TIMEZONE)
    index = pd.date_range(start, end, inclusive="left", freq="h").tz_convert(
        "UTC"
    )
    assert len(index) == FINAL_HOURS
    actual = 55.0 + 8.0 * np.sin(np.arange(len(index)) / 37.0)
    q50 = actual + prediction_offset
    origin = _origins(index)
    backtest = pd.DataFrame(
        {
            "delivery_start_utc": index.astype(str),
            "residual_corrected__q10": q50 - 10.0,
            "residual_corrected__q50": q50,
            "residual_corrected__q90": q50 + 10.0,
            "actual": actual,
            "fold_id": 5,
            "forecast_origin_utc": origin.astype(str),
        }
    )
    backtest.to_csv(
        run / "backtest_hourly_oof.csv.gz", index=False, compression="gzip"
    )

    future_start = end
    future_end = end + pd.Timedelta(days=1)
    future_index = pd.date_range(
        future_start, future_end, inclusive="left", freq="h"
    ).tz_convert("UTC")
    future_q50 = 60.0 + prediction_offset + np.arange(len(future_index)) / 10.0
    future_origin = _origins(future_index)
    pd.DataFrame(
        {
            "delivery_start_utc": future_index.astype(str),
            "forecast_origin_utc": future_origin.astype(str),
            "residual_corrected__q10": future_q50 - 10.0,
            "residual_corrected__q50": future_q50,
            "residual_corrected__q90": future_q50 + 10.0,
            "q10": future_q50 - 10.0,
            "q50": future_q50,
            "q90": future_q50 + 10.0,
            "price_eur_mwh": future_q50,
        }
    ).to_csv(run / "forecast_hourly_fr.csv", index=False)

    treatment_value = 10.0 if residual_load_source == "saturn" else 20.0
    aligned = pd.DataFrame(
        {
            "timestamp": index.astype(str),
            "target": actual,
            "same_non_treatment": np.cos(np.arange(len(index)) / 11.0),
        }
    )
    for alias in EXPECTED_ALIASES:
        aligned[alias] = treatment_value
    aligned.to_csv(inputs / "aligned_inputs.csv.gz", index=False, compression="gzip")

    context_index = index.append(future_index)
    context = pd.DataFrame(
        {
            "timestamp": context_index.astype(str),
            "known_hour_sin": np.sin(np.arange(len(context_index)) / 24.0),
            "same_context_feature": 7.0,
        }
    )
    for column in RESIDUAL_LOAD_TREATMENT_COLUMNS:
        context[column] = treatment_value
    context.to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {
            "zone": "FR",
            "alias": list(EXPECTED_ALIASES),
            "series": list(EXPECTED_ALIASES),
            "coverage_exact": 1.0,
            "coverage_after_fill": 1.0,
            "missing_after_fill": 0,
        }
    ).to_csv(inputs / "input_coverage.csv", index=False)
    pd.DataFrame(
        {
            "alias": ["target", *EXPECTED_ALIASES],
            "role": ["target", *(["known_future_covariate"] * 5)],
            "series": ["price", *EXPECTED_ALIASES],
            "description": ["price", *("" for _ in EXPECTED_ALIASES)],
            "future_strategies": ["", *(["oracle"] * 5)],
            "known_future": [False, *([True] * 5)],
            "source": ["cache", *(["pit_parquet"] * 5)],
        }
    ).to_csv(inputs / "input_manifest.csv", index=False)
    (inputs / "residual_replay_manifest.json").write_text(
        json.dumps(
            {
                "residual_load_source": residual_load_source,
                "model_revision": "test-revision",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature": ["known_hour_sin", "same_context_feature"],
            "dtype": ["float64", "float64"],
            "historical_non_missing": [len(index), len(index)],
        }
    ).to_csv(run / "feature_manifest.csv", index=False)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "zone": "FR",
                "timezone": TIMEZONE,
                "native_model": "residual_corrected",
                "residual_load_source": residual_load_source,
            }
        ),
        encoding="utf-8",
    )
    _seal(run)
    return run


def test_publishes_standard_report_from_two_recalculated_runs(tmp_path: Path) -> None:
    control = _write_run(
        tmp_path,
        name="control",
        residual_load_source="saturn",
        prediction_offset=2.0,
    )
    challenger = _write_run(
        tmp_path,
        name="challenger",
        residual_load_source="chronos2_historical_replay",
        prediction_offset=1.0,
    )
    output = tmp_path / "comparison"

    result = publish_historical_residual_comparison(
        control,
        challenger,
        output,
        zone="FR",
    )

    assert result.output_dir == output.resolve()
    assert result.report_path.is_file()
    assert result.report_path.stat().st_size > 100_000
    rendered = result.report_path.read_text(encoding="utf-8")
    assert "Prix aval — residual_load Chronos-2" in rendered
    assert "Prix aval — residual_load Saturn" in rendered
    assert "Comparaison graphique au benchmark actif" in rendered
    assert result.control_mae == pytest.approx(2.0)
    assert result.challenger_mae == pytest.approx(1.0)
    assert not (output / "statistics_history_hourly.csv.gz").exists()
    assert not (output / "statistics_history_audit.json").exists()

    backtest = pd.read_csv(output / "backtest_hourly_oof.csv.gz")
    for model in (CONTROL_MODEL, CHALLENGER_MODEL):
        for quantile in ("q10", "q50", "q90"):
            assert f"{model}__{quantile}" in backtest
    metrics = pd.read_csv(output / "metrics_hourly.csv").set_index("model")
    assert metrics.loc[CONTROL_MODEL, "n_scored"] == FINAL_HOURS
    assert metrics.loc[CHALLENGER_MODEL, "mae"] == pytest.approx(1.0)

    copied = pd.read_csv(output / "inputs/model_covariates_with_future.csv.gz")
    original = pd.read_csv(
        challenger / "inputs/model_covariates_with_future.csv.gz"
    )
    pd.testing.assert_frame_equal(copied, original)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["residual_load_source"] == "chronos2_historical_replay"
    assert manifest["statistics_history_reused"] is False
    assert manifest["statistics_comparison"]["source"] == (
        "recalculated_paired_backtest"
    )
    audit = json.loads((output / "comparison_audit.json").read_text(encoding="utf-8"))
    assert audit["final_hours"] == FINAL_HOURS
    assert audit["non_treatment_inputs"][
        "model_covariates_with_future.csv.gz"
    ]["non_treatment_values_identical"] is True

    checksum_payload = json.loads(
        (output / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    paths = {entry["path"]: entry for entry in checksum_payload["artifacts"]}
    report_relative = result.report_path.relative_to(output).as_posix()
    assert paths[report_relative]["sha256"] == _sha256(result.report_path)
    assert paths["inputs/residual_replay_manifest.json"]["role"] == (
        "materialized_input"
    )


def test_rejects_a_non_treatment_input_difference(tmp_path: Path) -> None:
    control = _write_run(
        tmp_path,
        name="control",
        residual_load_source="saturn",
        prediction_offset=2.0,
    )
    challenger = _write_run(
        tmp_path,
        name="challenger",
        residual_load_source="chronos2_historical_replay",
        prediction_offset=1.0,
    )
    path = challenger / "inputs/model_covariates_with_future.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "same_context_feature"] = 99.0
    frame.to_csv(path, index=False, compression="gzip")
    _seal(challenger)

    with pytest.raises(
        HistoricalResidualComparisonError,
        match="same_context_feature",
    ):
        publish_historical_residual_comparison(
            control,
            challenger,
            tmp_path / "comparison",
            zone="FR",
        )


def test_rejects_wrong_challenger_source_and_existing_destination(
    tmp_path: Path,
) -> None:
    control = _write_run(
        tmp_path,
        name="control",
        residual_load_source="saturn",
        prediction_offset=2.0,
    )
    wrong = _write_run(
        tmp_path,
        name="wrong",
        residual_load_source="saturn",
        prediction_offset=1.0,
    )
    with pytest.raises(
        HistoricalResidualComparisonError,
        match="chronos2_historical_replay",
    ):
        publish_historical_residual_comparison(
            control,
            wrong,
            tmp_path / "comparison",
            zone="FR",
        )

    challenger = _write_run(
        tmp_path,
        name="challenger",
        residual_load_source="chronos2_historical_replay",
        prediction_offset=1.0,
    )
    destination = tmp_path / "already-there"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        publish_historical_residual_comparison(
            control,
            challenger,
            destination,
            zone="FR",
        )
