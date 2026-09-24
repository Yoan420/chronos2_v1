from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_residual_load_historical_comparison as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _protocol(tmp_path: Path) -> runner.Protocol:
    source = runner._load_protocol(
        PROJECT_ROOT / "config" / "residual_load_historical_comparison.yaml",
        PROJECT_ROOT,
    )
    return replace(source, experiment_root=tmp_path / "experiment")


def test_protocol_is_exact_223_730_365() -> None:
    protocol = runner._load_protocol(
        PROJECT_ROOT / "config" / "residual_load_historical_comparison.yaml",
        PROJECT_ROOT,
    )
    assert (protocol.extended_end - protocol.extended_start).days + 1 == 223
    assert (protocol.oof_end - protocol.oof_start).days + 1 == 730
    assert (protocol.final_end - protocol.final_start).days + 1 == 365
    assert protocol.residual_end.isoformat() == "2026-08-12"
    assert protocol.model_revision == runner.MODEL_REVISION


def test_price_execution_identity_pins_the_executor_protocol(
    tmp_path: Path,
) -> None:
    signature = runner._execution_signature(
        _protocol(tmp_path), "cpu", scope="price"
    )
    assert signature["executor_protocol_version"] == (
        runner.PRICE_EXECUTION_PROTOCOL_VERSION
    )
    assert signature["executor_variant"] == (
        "hourly_oof_residual_source_comparison"
    )
    assert signature["with_covariates"] is True


def test_generated_extended_config_references_plain_live_csv(
    tmp_path: Path,
) -> None:
    protocol = _protocol(tmp_path)
    zone = next(iter(protocol.zones.values()))
    config_path = runner._generated_extended_config(
        protocol,
        zone,
        "saturn",
        threads=1,
    )
    payload = runner.load_yaml(config_path)
    assert payload["extended_residual"]["chronos_live_file"].endswith(
        "chronos_price_live.csv"
    )


def test_pit_rows_use_civil_d_minus_one_cutoff_across_dst() -> None:
    index = runner.local_delivery_day_index(
        "2024-03-31", timezone="Europe/Paris"
    )
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    frame = runner._pit_rows(values, timezone="Europe/Paris")
    assert len(frame) == 23
    assert frame["snapshot_time_utc"].nunique() == 1
    assert pd.Timestamp(frame["snapshot_time_utc"].iloc[0]) == pd.Timestamp(
        "2024-03-30T07:00:00Z"
    )


def test_cutoff_remains_civil_0800_after_both_dst_switches() -> None:
    assert runner._cutoff_for_day(date(2024, 4, 1)) == pd.Timestamp(
        "2024-03-31T06:00:00Z"
    )
    assert runner._cutoff_for_day(date(2024, 10, 28)) == pd.Timestamp(
        "2024-10-27T07:00:00Z"
    )


def _data(value_shift: float) -> SimpleNamespace:
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    raw = pd.DataFrame(index=index)
    for offset, alias in enumerate(runner.EXPECTED_ALIASES):
        values = np.arange(4, dtype=float) + offset + value_shift
        values[0] = np.nan
        raw[alias] = values
        raw[f"known_{alias}_oracle"] = values
    raw["known_hour_sin"] = 0.5
    return SimpleNamespace(
        target=pd.Series(np.arange(4, dtype=float), index=index),
        covariates=raw.copy(),
        model_context_covariates=raw,
        known_future_columns=[
            f"known_{alias}_oracle" for alias in runner.EXPECTED_ALIASES
        ],
    )


def test_branch_input_parity_allows_only_treatment_value_changes() -> None:
    runner._assert_branch_input_parity(
        _data(0.0), _data(2.0), zone="FR"
    )


def test_branch_input_parity_rejects_mask_change() -> None:
    control = _data(0.0)
    challenger = _data(2.0)
    challenger.model_context_covariates.loc[
        challenger.model_context_covariates.index[1],
        "fr_residual_load_fcst",
    ] = np.nan
    with pytest.raises(runner.HistoricalComparisonRunnerError, match="mirror|masque"):
        runner._assert_branch_input_parity(control, challenger, zone="FR")


def test_export_has_the_same_two_file_shape_and_canonical_csv(
    tmp_path: Path,
) -> None:
    protocol = replace(
        _protocol(tmp_path),
        export_root=tmp_path / "exports" / "residual_load_chronos2",
    )
    zone = protocol.zones["FR"]
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    (comparison / "comparison.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    delivery = runner.local_delivery_day_index(
        protocol.residual_end, timezone=zone.timezone
    )
    pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "delivery_start_local": delivery.tz_convert(zone.timezone),
            "local_hour": delivery.tz_convert(zone.timezone).hour,
            "internal_model__q50": 999.0,
            "q10": 10.0,
            "q50": 20.0,
            "q90": 30.0,
            "price_eur_mwh": 20.0,
        }
    ).to_csv(comparison / "forecast_hourly_fr.csv", index=False)

    destination = runner._export_comparison(
        comparison,
        protocol,
        zone,
        "autonomous",
        export_day=protocol.residual_end,
        overwrite=False,
    )

    stem = f"forecast_fr_{protocol.residual_end.isoformat()}_autonomous"
    assert {path.name for path in destination.iterdir()} == {
        f"{stem}.html",
        f"{stem}.csv",
    }
    exported = pd.read_csv(destination / f"{stem}.csv")
    assert list(exported.columns) == [
        "zone",
        "forecast_variant",
        "source_model",
        "residual_load_source",
        "uses_mkonline",
        "delivery_start_utc",
        "delivery_start_local",
        "local_hour",
        "q10",
        "q50",
        "q90",
        "price_eur_mwh",
    ]
    assert "internal_model__q50" not in exported

    assert (
        runner._export_comparison(
            comparison,
            protocol,
            zone,
            "autonomous",
            export_day=protocol.residual_end,
            overwrite=False,
        )
        == destination
    )
    (comparison / "comparison.html").write_text(
        "<html>nouveau rapport</html>", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="obsolete"):
        runner._export_comparison(
            comparison,
            protocol,
            zone,
            "autonomous",
            export_day=protocol.residual_end,
            overwrite=False,
        )


def test_completed_run_requires_the_exact_contract_and_valid_seal(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    contract = {
        "run_type": "historical_residual_load_full_recalculation",
        "residual_load_source": "saturn",
        "price_replay_manifest_sha256": "price-a",
        "statistics_history_reused": False,
    }
    manifest = run / "run_manifest.json"
    manifest.write_text(json.dumps(contract), encoding="utf-8")
    source_code = tmp_path / "code.py"
    source_code.write_text("VALUE = 1\n", encoding="utf-8")
    (run / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "output_directory": str(run),
                "artifacts": [
                    {
                        "path": "run_manifest.json",
                        "size_bytes": manifest.stat().st_size,
                        "sha256": runner._sha256(manifest),
                    },
                    {
                        "path": "code.py",
                        "role": "source_code",
                        "size_bytes": source_code.stat().st_size,
                        "sha256": runner._sha256(source_code),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert runner._completed_run(
        run, expected_contract=contract, project_root=tmp_path
    )
    assert not runner._completed_run(
        run,
        expected_contract={**contract, "price_replay_manifest_sha256": "price-b"},
        project_root=tmp_path,
    )
    manifest.write_text(json.dumps({**contract, "tampered": True}), encoding="utf-8")
    assert not runner._completed_run(
        run, expected_contract=contract, project_root=tmp_path
    )
