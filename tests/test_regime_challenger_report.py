from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.regime_challenger_report import (
    REPORT_SCHEMA,
    RegimeChallengerReportError,
    write_regime_challenger_report,
)


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.external_references: list[str] = []
        self.svg_count = 0
        self.scripts: list[str] = []
        self._script: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier:
            self.ids.add(str(identifier))
        if tag == "svg":
            self.svg_count += 1
        if tag == "script" and attributes.get("src") is None:
            self._script = []
        for name in ("src", "href"):
            value = attributes.get(name)
            if value and not str(value).startswith("#"):
                self.external_references.append(str(value))

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "zone": "FR",
        "timezone": "Europe/Paris",
        "forecast_status": "shadow_challenger",
        "production_eligible": False,
        "delivery_day_local": "2026-08-25",
        "model": 'regime</td><script id="manifest-injection">x</script>',
        "pit_audit": {
            "cutoff": "J-1 08:00 Europe/Paris",
            "revision_violations": 0,
        },
        "provenance": {
            "config_sha256": "a" * 64,
            "source": "sealed D-1 forecasts",
        },
    }


def _hourly(*, actual: bool = True) -> pd.DataFrame:
    timeline = pd.date_range(
        "2026-08-24T22:00:00Z",
        periods=24,
        freq="h",
        tz="UTC",
    )
    rows: list[pd.DataFrame] = []
    for offset, variant in enumerate(("level_only", "regime_gate")):
        baseline = np.linspace(50.0, 95.0, len(timeline)) + offset
        premium = np.where(np.arange(len(timeline)) >= 10, 18.0 + offset, 2.0)
        challenger = baseline + premium
        probability = np.linspace(0.05, 0.92, len(timeline))
        observed = challenger - 3.0 if actual else np.full(len(timeline), np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "delivery_start_utc": timeline,
                    "local_hour": timeline.tz_convert("Europe/Paris").hour,
                    "variant": variant,
                    "actual": observed,
                    "baseline_q10": baseline - 8.0,
                    "baseline_q50": baseline,
                    "baseline_q90": baseline + 8.0,
                    "challenger_q10": challenger - 10.0,
                    "challenger_q50": challenger,
                    "challenger_q90": challenger + 10.0,
                    "shock_probability": probability,
                    "shock_premium": premium,
                    "regime_label": np.where(
                        probability >= 0.7,
                        "spike",
                        "normal",
                    ),
                    "extra_statistic": "<b>tolérée</b>",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in ("level_only", "regime_gate"):
        for scope, hours in (
            ("global", 24),
            ("solar_10_16", 7),
            ("tail_p90", 3),
            ("regime_spike", 6),
        ):
            rows.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "n_hours": hours,
                    "baseline_mae": 12.5,
                    "challenger_mae": 8.25,
                    "ci95_lower": -6.0,
                    "note": '<img src="https://invalid.example/x">',
                }
            )
    return pd.DataFrame(rows)


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "local_date": day,
                "variant": variant,
                "n_hours": 24,
                "baseline_mae": 12.0 + day_index,
                "challenger_mae": 8.0 + day_index,
                "tail_recall": 0.75,
            }
            for variant in ("level_only", "regime_gate")
            for day_index, day in enumerate(("2026-08-24", "2026-08-25"))
        ]
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [
                "residual_load_d",
                "wind_drop_d_vs_d1",
                '<svg id="feature-injection"></svg>',
            ],
            "importance": [0.62, 0.29, -0.09],
            "method": ["gain", "gain", "gain"],
        }
    )


def _diagnostics() -> dict[str, object]:
    return {
        "day_diagnostic": {
            "summary": "Vent D en forte baisse; régime tendu.",
            "solar_window": "10:00-16:59",
        },
        "calibration_gate": {
            "passes": False,
            "minimum_tail_recall": 0.65,
            "observed_tail_recall": 0.61,
        },
        "pit": {
            "forecast_origin": "2026-08-24T06:00:00Z",
            "cutoff_violations": 0,
        },
        "provenance": {"recipe_sha256": "b" * 64},
        "limitations": [
            'limite</li><script id="limit-injection">x</script>',
        ],
    }


def _write(
    path: Path,
    *,
    manifest: dict[str, object] | None = None,
    hourly: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    daily: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> Path:
    return write_regime_challenger_report(
        path,
        manifest=manifest if manifest is not None else _manifest(),
        hourly=hourly if hourly is not None else _hourly(),
        metrics=metrics if metrics is not None else _metrics(),
        daily=daily if daily is not None else _daily(),
        feature_importance=features if features is not None else _features(),
        diagnostics=_diagnostics(),
    )


def test_writes_complete_standalone_escaped_report_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    hourly = _hourly()
    metrics = _metrics()
    daily = _daily()
    features = _features()
    snapshots = tuple(
        frame.copy(deep=True) for frame in (hourly, metrics, daily, features)
    )

    output = _write(
        tmp_path / "report.html",
        hourly=hourly,
        metrics=metrics,
        daily=daily,
        features=features,
    )

    assert output == (tmp_path / "report.html").resolve()
    source = output.read_text(encoding="utf-8")
    document = _Document()
    document.feed(source)
    assert source.startswith("<!doctype html>")
    assert REPORT_SCHEMA in source
    assert document.external_references == []
    assert document.svg_count >= 5
    assert len(document.scripts) == 1
    assert {
        "status",
        "day-diagnostic",
        "variant-curves",
        "metrics",
        "calibration-gate",
        "daily-performance",
        "feature-importance",
        "audit",
        "limitations",
    }.issubset(document.ids)
    assert "SHADOW_CHALLENGER" in source
    assert "NON PRODUCTION" in source
    assert "Heures solaires" in source
    assert "Queue de distribution / spikes" in source
    assert "Importance des features" in source
    assert "Audit PIT et provenance" in source
    assert "https://invalid.example" in source
    assert '<img src="https://invalid.example/x">' not in source
    assert "manifest-injection" not in document.ids
    assert "feature-injection" not in document.ids
    assert "limit-injection" not in document.ids
    assert "&lt;script id=&quot;manifest-injection&quot;&gt;" in source
    assert "connect-src 'none'" in source
    assert "object-src 'none'" in source
    for observed, expected in zip(
        (hourly, metrics, daily, features),
        snapshots,
        strict=True,
    ):
        pd.testing.assert_frame_equal(observed, expected)


def test_shadow_without_actuals_and_empty_optional_tables_is_explicit(
    tmp_path: Path,
) -> None:
    hourly = _hourly(actual=False).drop(columns="actual")
    metrics = pd.DataFrame(columns=sorted(_metrics().columns))
    daily = pd.DataFrame(columns=sorted(_daily().columns))
    features = pd.DataFrame(columns=sorted(_features().columns))

    output = _write(
        tmp_path / "shadow.html",
        hourly=hourly,
        metrics=metrics,
        daily=daily,
        features=features,
    )
    source = output.read_text(encoding="utf-8")

    assert "Non observé à cet instant" in source
    assert "Aucune métrique globale fournie" in source
    assert "Aucune performance quotidienne fournie" in source
    assert "Aucune importance de feature fournie" in source
    assert "0 observations réalisées" in source
    assert "NaN" not in source
    assert "Infinity" not in source


def test_missing_essential_column_fails_before_writing(tmp_path: Path) -> None:
    hourly = _hourly().drop(columns="shock_probability")
    output = tmp_path / "invalid.html"

    with pytest.raises(
        RegimeChallengerReportError,
        match="hourly incomplet.*shock_probability",
    ):
        _write(output, hourly=hourly)

    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.assign(shock_probability=1.2),
            "shock_probability",
        ),
        (
            lambda frame: frame.assign(
                challenger_q10=frame["challenger_q90"] + 1.0
            ),
            "croisement de quantiles challenger",
        ),
        (
            lambda frame: frame.assign(
                delivery_start_utc=frame["delivery_start_utc"].dt.tz_localize(
                    None
                )
            ),
            "timezone-aware",
        ),
    ],
)
def test_hourly_contract_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    with pytest.raises(RegimeChallengerReportError, match=message):
        _write(tmp_path / "invalid.html", hourly=mutation(_hourly()))


def test_refuses_production_eligible_manifest_and_non_html_output(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["production_eligible"] = True
    with pytest.raises(RegimeChallengerReportError, match="production_eligible=false"):
        _write(tmp_path / "report.html", manifest=manifest)

    with pytest.raises(RegimeChallengerReportError, match="extension .html"):
        _write(tmp_path / "report.txt")
