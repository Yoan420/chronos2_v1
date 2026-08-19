from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import screen_mkonline_zone_crossfit as screen


def _dependency_manifest(
    path: Path,
    *,
    zone: str = "DE",
    primary: str = "41550_native",
    provider: str = "MKONLINE",
    storm_token_found: bool = False,
) -> Path:
    countries = {
        "DE": "GERMANY",
        "BE": "BELGIUM",
        "NL": "NETHERLANDS",
        "ES": "SPAIN",
    }
    path.write_text(
        json.dumps(
            {
                "series": [
                    {
                        "root": f"power.price.{zone.lower()}.euromwh.h.fcst.mkonline.ecop",
                        "terminal": primary,
                        "terminal_type": "primary",
                        "terminal_formula": None,
                        "provider": provider,
                        "source": "WATTSIGHT",
                        "country": countries[zone],
                        "label": f"{zone}_EC00_OP_HOURLY_INSTANTPRICE_FORECAST",
                        "storm_token_found": storm_token_found,
                        "dependency_gate_passed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _primary_frame(
    index: pd.DatetimeIndex,
    *,
    cutoff_timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    cutoff = screen._expected_cutoff(index, cutoff_timezone=cutoff_timezone)
    return pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": np.arange(len(index), dtype=float),
        }
    )


def test_exact_l1_weight_is_not_a_grid_approximation() -> None:
    actual = np.array([0.337, 0.337, 0.337])
    autonomous = np.zeros(3)
    expert = np.ones(3)

    assert screen._exact_l1_weight(actual, autonomous, expert) == pytest.approx(
        0.337
    )


def test_score_gate_requires_threshold_and_each_30_day_half() -> None:
    index = pd.date_range(
        "2025-06-12 22:00:00Z", periods=60 * 24, freq="h", tz="UTC"
    )
    actual = np.zeros(len(index))
    baseline = np.full(len(index), 2.0)
    passing = np.full(len(index), 1.0)

    score = screen._score_block(
        index, actual, baseline, passing, timezone="Europe/Berlin"
    )
    assert score["all"]["gain"] == pytest.approx(1.0)
    assert score["first30"]["gain"] > 0
    assert score["last30"]["gain"] > 0
    assert score["passes"] is True

    failing = passing.copy()
    local_days = pd.Index(index.tz_convert("Europe/Berlin").date)
    last_half = local_days.isin(local_days.unique()[30:])
    failing[last_half] = 2.25
    score = screen._score_block(
        index, actual, baseline, failing, timezone="Europe/Berlin"
    )
    assert score["last30"]["gain"] < 0
    assert score["passes"] is False


def test_cutoff_timezone_is_independent_from_delivery_timezone() -> None:
    delivery = pd.DatetimeIndex(
        [pd.Timestamp("2025-03-30 22:00:00Z")]
    )

    paris = screen._expected_cutoff(
        delivery, cutoff_timezone="Europe/Paris"
    )
    utc = screen._expected_cutoff(delivery, cutoff_timezone="UTC")

    assert paris[0] == pd.Timestamp("2025-03-30 06:00:00Z")
    assert utc[0] == pd.Timestamp("2025-03-29 08:00:00Z")
    assert paris[0] != utc[0]


def test_primary_loader_records_paris_cutoff_without_claiming_zone_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = pd.date_range(
        "2025-06-12 22:00:00Z", periods=60 * 24, freq="h", tz="UTC"
    )
    frame = _primary_frame(expected, cutoff_timezone="Europe/Paris")
    monkeypatch.setattr(screen, "_read_primary_window", lambda *_: frame)

    values, audit = screen._load_primary(
        tmp_path / "de_b2.parquet",
        expected,
        block="B2",
        delivery_timezone="Europe/Berlin",
        cutoff_timezone="Europe/Paris",
        expected_local_days=60,
    )

    assert values.index.equals(expected)
    assert audit["delivery_timezone"] == "Europe/Berlin"
    assert audit["cutoff_timezone"] == "Europe/Paris"
    assert audit["cutoff_contract"] == "civil D-1 08:00 Europe/Paris"
    assert audit["final_rows_loaded"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"provider": "STORM"}, "unexpected provider/source"),
        ({"storm_token_found": True}, "negative Storm proof"),
    ],
)
def test_dependency_audit_rejects_provider_or_storm_contamination(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    path = _dependency_manifest(tmp_path / "dependency.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["series"][0].update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        screen._audit_primary_dependency(
            path, primary_series="41550_native", zone="DE"
        )


def test_dependency_audit_checks_primary_identity_and_country(tmp_path: Path) -> None:
    path = _dependency_manifest(tmp_path / "dependency.json")
    audit = screen._audit_primary_dependency(
        path, primary_series="41550_native", zone="DE"
    )
    assert audit["terminal_series"] == "41550_native"
    assert audit["terminal_type"] == "primary"
    assert audit["country"] == "GERMANY"

    with pytest.raises(ValueError, match="expected one dependency entry"):
        screen._audit_primary_dependency(
            path, primary_series="41555_native", zone="DE"
        )


def test_cli_has_no_final_phase_and_b2_requires_frozen_inputs(tmp_path: Path) -> None:
    common = [
        "--zone",
        "DE",
        "--timezone",
        "Europe/Berlin",
        "--autonomous-run",
        str(tmp_path / "run"),
        "--extended-oof-file",
        str(tmp_path / "ext.csv.gz"),
        "--primary-series",
        "41550_native",
        "--primary-file",
        str(tmp_path / "primary.parquet"),
        "--dependency-manifest",
        str(tmp_path / "dependency.json"),
        "--output-json",
        str(tmp_path / "result.json"),
    ]
    with pytest.raises(SystemExit):
        screen.parse_args([*common, "--phase", "final"])
    with pytest.raises(SystemExit):
        screen.parse_args([*common, "--phase", "b2"])

    parsed = screen.parse_args(
        [
            *common,
            "--phase",
            "b2",
            "--b2-primary-file",
            str(tmp_path / "b2.parquet"),
            "--frozen-weight",
            "0.337",
        ]
    )
    assert parsed.frozen_weight == pytest.approx(0.337)
    assert parsed.cutoff_timezone == "Europe/Paris"


def test_autonomous_run_audit_rejects_storm_feature_and_wrong_timezone(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "feature_manifest.csv").write_text(
        "feature\nknown_storm_price\n", encoding="utf-8"
    )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "zone": "DE",
                "timezone": "Europe/Berlin",
                "target_contract": "hourly_utc_no_interpolation",
                "delivery_horizon": "dynamic_23_24_25",
                "active_features": ["known_storm_price"],
                "input_diagnostics": {
                    "covariates": {
                        "load": {"forecast_origin_local_time": "08:00"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden Storm token"):
        screen._audit_autonomous_run(
            run, zone="DE", timezone="Europe/Berlin"
        )
    with pytest.raises(ValueError, match="timezone"):
        screen._audit_autonomous_run(run, zone="DE", timezone="Europe/Paris")


def test_autonomous_run_audit_requires_explicit_origin_contract(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "feature_manifest.csv").write_text(
        "feature\nknown_load\n", encoding="utf-8"
    )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "zone": "BE",
                "timezone": "Europe/Brussels",
                "target_contract": "hourly_utc_no_interpolation",
                "delivery_horizon": "dynamic_23_24_25",
                "active_features": ["known_load"],
                "input_diagnostics": {"covariates": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forecast origins"):
        screen._audit_autonomous_run(
            run, zone="BE", timezone="Europe/Brussels"
        )


def test_calibration_contract_starts_at_first_real_oof_fold(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    timestamps = pd.date_range(
        "2024-08-10 22:00:00Z", periods=72, freq="h", tz="UTC"
    )
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps.astype(str),
            "fold_id": [np.nan] * 24 + [1.0] * 48,
        }
    ).to_csv(run / "backtest_hourly_oof.csv.gz", index=False)

    contract = screen._calibration_contract(
        run,
        zone="DE",
        timezone="Europe/Berlin",
    )

    assert contract.a_start_day == pd.Timestamp("2024-08-12")
    assert contract.a_start_utc == pd.Timestamp("2024-08-11 22:00:00Z")
    assert contract.ext_start_day == pd.Timestamp("2024-01-02")
