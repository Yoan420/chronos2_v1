from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_multicountry_forecast as launcher
from chronos2_hourly.reporting import (
    VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT,
    VARIABLE_ATTRIBUTION_COLUMNS,
    VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT,
    _attach_variable_attribution,
)
from chronos2_modular.report import build_variable_attribution_html


TIMEZONE = "Europe/Paris"
DELIVERY_DAY = "2026-08-27"
GROUPS = (
    {
        "key": "residual_load",
        "label": "Charge résiduelle",
        "context_columns": ["fr_residual_load"],
        "future_columns": ["fr_residual_load_fcst"],
        "baseline_reference": "profil climatologique PIT",
    },
    {
        "key": "calendar",
        "label": "Calendrier",
        "context_columns": ["hour_sin", "hour_cos"],
        "future_columns": [],
        "baseline_reference": "calendrier neutre",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_attribution(
    directory: Path,
    *,
    selected_model: str,
) -> SimpleNamespace:
    directory.mkdir(parents=True, exist_ok=True)
    delivery = pd.date_range(
        "2026-08-26T22:00:00Z",
        periods=24,
        freq="h",
        tz="UTC",
    )
    autonomous = 50.0 + np.arange(len(delivery), dtype=float)
    blend = 60.0 + np.arange(len(delivery), dtype=float)
    forecast = pd.DataFrame(
        {
            "delivery_start_utc": delivery.astype(str),
            "residual_corrected__q50": autonomous,
            "mkonline_blend__q50": blend,
        }
    )
    forecast_path = directory / "forecast_hourly_fr.csv"
    forecast.to_csv(forecast_path, index=False)

    rows: list[dict[str, object]] = []
    for variant, values in (
        ("autonomous", autonomous),
        ("mkonline_blend", blend),
    ):
        for timestamp, value in zip(delivery, values):
            baseline = float(value - 5.0)
            for group, contribution, weight in (
                (GROUPS[0], 2.0, 40.0),
                (GROUPS[1], 3.0, 60.0),
            ):
                rows.append(
                    {
                        "delivery_start_utc": timestamp.isoformat(),
                        "delivery_start_local": timestamp.tz_convert(
                            TIMEZONE
                        ).isoformat(),
                        "variant": variant,
                        "variable_key": group["key"],
                        "variable_label": group["label"],
                        "baseline_reference": group["baseline_reference"],
                        "forecast_q50": float(value),
                        "counterfactual_q50": baseline,
                        "contribution_eur_mwh": contribution,
                        "absolute_contribution_eur_mwh": abs(contribution),
                        "weight_pct": weight,
                    }
                )
    hourly = pd.DataFrame(rows, columns=VARIABLE_ATTRIBUTION_COLUMNS)
    hourly.to_csv(
        directory / VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT,
        index=False,
        compression="gzip",
    )
    audit = {
        "schema_version": 1,
        "status": "complete",
        "zone": "FR",
        "timezone": TIMEZONE,
        "delivery_day": DELIVERY_DAY,
        "quantile": "q50",
        "method": "exact_grouped_shapley_end_to_end",
        "scope": "local_delivery_day",
        "expected_hours": 24,
        "variants": ["autonomous", "mkonline_blend"],
        "architecture_weights": {
            "autonomous": {
                "autonomous": 1.0,
                "mkonline_primary": 0.0,
            },
            "mkonline_blend": {
                "autonomous": 0.4,
                "mkonline_primary": 0.6,
            },
        },
        "groups": list(GROUPS),
        "forecast_sha256": _sha256(forecast_path),
        "used_for_prediction": False,
        "storm_used": False,
        "forecast_modified": False,
        "reproduction_tolerance_eur_mwh": 1e-8,
        "max_base_prediction_error_eur_mwh": 0.0,
        "max_reconstruction_error_eur_mwh": 0.0,
    }
    (directory / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_values = autonomous if selected_model == "residual_corrected" else blend
    return SimpleNamespace(
        zone="FR",
        forecast_native=pd.DataFrame(
            {
                "timestamp": delivery.tz_convert(TIMEZONE),
                "q50": selected_values,
            }
        ),
    )


@pytest.mark.parametrize(
    ("native_model", "expected_variant", "expected_mkonline_weight"),
    (
        ("residual_corrected", "autonomous", "0.0%"),
        ("mkonline_blend", "mkonline_blend", "60.0%"),
    ),
)
def test_audited_attribution_is_selected_and_rendered_for_each_export_variant(
    tmp_path: Path,
    native_model: str,
    expected_variant: str,
    expected_mkonline_weight: str,
) -> None:
    result = _write_attribution(tmp_path, selected_model=native_model)

    _attach_variable_attribution(
        result,
        directory=tmp_path,
        native_model=native_model,
        timezone=TIMEZONE,
    )

    attribution = result.variable_attribution
    assert attribution["variant"] == expected_variant
    assert len(attribution["hourly"]) == 48
    assert attribution["hourly"]["weight_pct"].drop_duplicates().sum() == 100.0
    rendered = build_variable_attribution_html(result)
    assert rendered.count('data-report-section="variable-attribution"') == 1
    assert f'data-attribution-variant="{expected_variant}"' in rendered
    assert "Influence des variables dans la prévision finale" in rendered
    assert "Décomposition additive Shapley groupée" in rendered
    assert "sensibilité locale du modèle" in rendered
    assert "pas d'un coefficient causal" in rendered
    assert "Poids d&#x27;influence Shapley (%)" in rendered
    assert "Contributions additives par heure locale".lower() in rendered.lower()
    assert "Modèle autonome" in rendered
    assert "MKOnline primaire" in rendered
    assert expected_mkonline_weight in rendered
    assert rendered.count("plotly-graph-div") == 2


def test_attribution_is_optional_as_a_pair_and_partial_pair_fails_closed(
    tmp_path: Path,
) -> None:
    delivery = pd.date_range(
        "2026-08-26T22:00:00Z", periods=24, freq="h", tz="UTC"
    )
    result = SimpleNamespace(
        zone="FR",
        forecast_native=pd.DataFrame(
            {"timestamp": delivery.tz_convert(TIMEZONE), "q50": 50.0}
        ),
    )
    _attach_variable_attribution(
        result,
        directory=tmp_path,
        native_model="residual_corrected",
        timezone=TIMEZONE,
    )
    assert not hasattr(result, "variable_attribution")

    pd.DataFrame(columns=VARIABLE_ATTRIBUTION_COLUMNS).to_csv(
        tmp_path / VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT,
        index=False,
        compression="gzip",
    )
    with pytest.raises(FileNotFoundError, match="paire.*incomplete"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_corrected",
            timezone=TIMEZONE,
        )


@pytest.mark.parametrize(
    "native_model",
    (
        "residual_kalman",
        "residual_kalman_weather",
        "residual_kalman_hybrid",
    ),
)
def test_kalman_reports_render_copied_attribution_as_upstream_only(
    tmp_path: Path,
    native_model: str,
) -> None:
    result = _write_attribution(tmp_path, selected_model="residual_corrected")
    forecast_path = tmp_path / "forecast_hourly_fr.csv"
    original_sha = _sha256(forecast_path)
    forecast = pd.read_csv(forecast_path)
    forecast[f"{native_model}__q50"] = forecast["residual_corrected__q50"] + 7.0
    forecast.to_csv(forecast_path, index=False)
    result.forecast_native["q50"] = forecast[f"{native_model}__q50"].to_numpy()
    (tmp_path / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "artifacts": [
                    {
                        "path": "forecast_hourly_fr.csv",
                        "role": "run_artifact",
                        "sha256": original_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _sha256(forecast_path) != original_sha

    _attach_variable_attribution(
        result,
        directory=tmp_path,
        native_model=native_model,
        timezone=TIMEZONE,
    )

    attribution = result.variable_attribution
    assert attribution["variant"] == "autonomous"
    assert attribution["scope"] == "upstream_model"
    assert attribution["is_upstream_attribution"] is True
    assert attribution["explained_model"] == "residual_corrected"
    assert attribution["reported_model"] == native_model
    rendered = build_variable_attribution_html(result)
    assert 'data-attribution-scope="upstream-model"' in rendered
    assert "Influence des variables dans la prévision amont" in rendered
    assert "avant application du filtre Kalman" in rendered
    assert "diagnostics Kalman" in rendered


def test_kalman_upstream_attribution_fails_without_original_checksum_manifest(
    tmp_path: Path,
) -> None:
    result = _write_attribution(tmp_path, selected_model="residual_corrected")

    with pytest.raises(FileNotFoundError, match="artifact_checksums"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_kalman_hybrid",
            timezone=TIMEZONE,
        )


def test_kalman_upstream_attribution_fails_on_hash_or_upstream_value_divergence(
    tmp_path: Path,
) -> None:
    result = _write_attribution(tmp_path, selected_model="residual_corrected")
    forecast_path = tmp_path / "forecast_hourly_fr.csv"
    original_sha = _sha256(forecast_path)
    forecast = pd.read_csv(forecast_path)
    forecast["residual_kalman_hybrid__q50"] = (
        forecast["residual_corrected__q50"] + 7.0
    )
    forecast.to_csv(forecast_path, index=False)
    result.forecast_native["q50"] = forecast[
        "residual_kalman_hybrid__q50"
    ].to_numpy()
    manifest_path = tmp_path / "artifact_checksums.json"
    manifest = {
        "algorithm": "sha256",
        "artifacts": [
            {
                "path": "forecast_hourly_fr.csv",
                "role": "run_artifact",
                "sha256": "0" * 64,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="forecast original"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_kalman_hybrid",
            timezone=TIMEZONE,
        )

    manifest["artifacts"][0]["sha256"] = original_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    forecast = pd.read_csv(forecast_path)
    forecast["residual_corrected__q50"] += 1.0
    forecast.to_csv(forecast_path, index=False)
    with pytest.raises(ValueError, match="amont residual_corrected"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_kalman_hybrid",
            timezone=TIMEZONE,
        )


def test_non_additive_or_storm_tainted_attribution_fails_closed(
    tmp_path: Path,
) -> None:
    result = _write_attribution(tmp_path, selected_model="residual_corrected")
    hourly_path = tmp_path / VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT
    hourly = pd.read_csv(hourly_path)
    hourly.loc[0, "contribution_eur_mwh"] = 100.0
    hourly.loc[0, "absolute_contribution_eur_mwh"] = 100.0
    hourly.to_csv(hourly_path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="non additive|weight_pct"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_corrected",
            timezone=TIMEZONE,
        )


def test_declared_attribution_errors_are_bounded_by_audit_tolerance(
    tmp_path: Path,
) -> None:
    result = _write_attribution(tmp_path, selected_model="residual_corrected")
    audit_path = tmp_path / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["reproduction_tolerance_eur_mwh"] = 1e-9
    audit["max_base_prediction_error_eur_mwh"] = 2e-9
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="tolerance de reproduction"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_corrected",
            timezone=TIMEZONE,
        )

    _write_attribution(tmp_path, selected_model="residual_corrected")
    audit_path = tmp_path / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["storm_used"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="storm_used"):
        _attach_variable_attribution(
            result,
            directory=tmp_path,
            native_model="residual_corrected",
            timezone=TIMEZONE,
        )


def _minimal_export_spec(tmp_path: Path) -> tuple[object, Path, Path]:
    archive = tmp_path / "forecast_archive"
    history = tmp_path / "saturn_history"
    archive.mkdir()
    history.mkdir()
    delivery = pd.date_range("2026-08-26T22:00:00Z", periods=2, freq="h")
    pd.DataFrame(
        {
            "delivery_start_utc": delivery.astype(str),
            "residual_corrected__q10": 40.0,
            "residual_corrected__q50": 50.0,
            "residual_corrected__q90": 60.0,
            "chronos2__q10": 39.0,
            "chronos2__q50": 49.0,
            "chronos2__q90": 59.0,
        }
    ).to_csv(archive / "forecast_hourly_fr.csv", index=False)
    pd.DataFrame({"delivery_start_utc": delivery.astype(str)}).to_csv(
        history / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    inputs = archive / "inputs"
    inputs.mkdir()
    for filename in launcher._REPORT_INPUT_FILES:
        (inputs / filename).write_text("source\n", encoding="utf-8")
    (archive / VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT).write_bytes(b"hourly-source")
    (archive / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT).write_bytes(b"audit-source")
    (history / VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT).write_bytes(b"wrong-history")
    (history / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT).write_bytes(b"wrong-history")
    spec = launcher._ExportSpec(
        zone="FR",
        timezone=TIMEZONE,
        delivery_day=DELIVERY_DAY,
        variant="autonomous",
        source_model="residual_corrected",
        baseline_model="chronos2",
        archive=archive,
        history_archive=history,
        csv_path=tmp_path / "forecast.csv",
        report_path=tmp_path / "forecast.html",
        history_contract=None,
    )
    return spec, archive, history


def test_export_view_copies_attribution_pair_from_forecast_archive(
    tmp_path: Path,
) -> None:
    spec, archive, _history = _minimal_export_spec(tmp_path)
    destination = tmp_path / "report_view"

    launcher._copy_reporting_view(spec, destination)

    for filename in launcher._REPORT_ATTRIBUTION_FILES:
        assert (destination / filename).read_bytes() == (archive / filename).read_bytes()


def test_export_view_rejects_partial_attribution_pair(tmp_path: Path) -> None:
    spec, archive, _history = _minimal_export_spec(tmp_path)
    (archive / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT).unlink()

    with pytest.raises(FileNotFoundError, match="paire d'attribution incomplete"):
        launcher._copy_reporting_view(spec, tmp_path / "report_view")
