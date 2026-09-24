from __future__ import annotations

import builtins
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.residual_load_comparison_report import (
    REPORT_SCHEMA,
    ResidualLoadComparisonReportError,
    render_residual_load_comparison_report,
)


_PAIRED_COLUMNS = [
    "zone",
    "timezone",
    "delivery_day_local",
    "delivery_start_utc",
    "hours_in_local_day",
    "production_q50",
    "challenger_q50",
    "actual",
    "production_error",
    "challenger_error",
    "production_abs_error",
    "challenger_abs_error",
    "challenger_wins",
    "tie",
]

_METRIC_VALUE_COLUMNS = [
    "production_mae",
    "challenger_mae",
    "mae_delta",
    "production_rmse",
    "challenger_rmse",
    "rmse_delta",
    "production_bias",
    "challenger_bias",
    "absolute_bias_delta",
    "production_smape_pct",
    "challenger_smape_pct",
    "smape_delta_pct",
    "challenger_win_rate",
    "tie_rate",
    "challenger_day_win_rate",
]


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.external_references: list[str] = []
        self.svg_count = 0
        self.scripts: list[str] = []
        self._script: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(str(attributes["id"]))
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


def _assert_javascript_delimiters_are_balanced(script: str) -> None:
    """Catch truncated inline JS without requiring a Node runtime in CI."""

    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for character in script:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character in opening:
            stack.append(character)
        elif character in closing:
            assert stack and stack.pop() == closing[character]
    assert quote is None
    assert stack == []


def _paired(day: str = "2026-08-20") -> pd.DataFrame:
    timeline = local_delivery_day_index(day, timezone="Europe/Paris")
    actual = np.linspace(45.0, 70.0, len(timeline))
    frame = pd.DataFrame(
        {
            "zone": "FR",
            "timezone": "Europe/Paris",
            "delivery_day_local": day,
            "delivery_start_utc": timeline,
            "hours_in_local_day": len(timeline),
            "production_q50": actual + 2.0,
            "challenger_q50": actual + 1.0,
            "actual": actual,
        }
    )
    frame["production_error"] = frame["production_q50"] - frame["actual"]
    frame["challenger_error"] = frame["challenger_q50"] - frame["actual"]
    frame["production_abs_error"] = frame["production_error"].abs()
    frame["challenger_abs_error"] = frame["challenger_error"].abs()
    frame["challenger_wins"] = (
        frame["challenger_abs_error"] < frame["production_abs_error"]
    )
    frame["tie"] = (
        frame["challenger_abs_error"] == frame["production_abs_error"]
    )
    return frame.loc[:, _PAIRED_COLUMNS]


def _metric_row(frame: pd.DataFrame, *, zone: str, scope: str) -> dict[str, object]:
    if frame.empty:
        return {
            "zone": zone,
            "scope": scope,
            "n_days": 0,
            "n_hours": 0,
            **{column: np.nan for column in _METRIC_VALUE_COLUMNS},
        }
    actual = frame["actual"].to_numpy(float)
    production = frame["production_q50"].to_numpy(float)
    challenger = frame["challenger_q50"].to_numpy(float)
    production_error = production - actual
    challenger_error = challenger - actual
    production_abs = np.abs(production_error)
    challenger_abs = np.abs(challenger_error)

    def smape(forecast: np.ndarray) -> float:
        denominator = np.abs(forecast) + np.abs(actual)
        ratio = np.divide(
            2.0 * np.abs(forecast - actual),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        return 100.0 * float(np.mean(ratio))

    daily = frame.assign(
        production_abs=production_abs,
        challenger_abs=challenger_abs,
    ).groupby(["zone", "delivery_day_local"])[
        ["production_abs", "challenger_abs"]
    ].mean()
    production_mae = float(production_abs.mean())
    challenger_mae = float(challenger_abs.mean())
    production_rmse = float(np.sqrt(np.mean(np.square(production_error))))
    challenger_rmse = float(np.sqrt(np.mean(np.square(challenger_error))))
    production_bias = float(production_error.mean())
    challenger_bias = float(challenger_error.mean())
    production_smape = smape(production)
    challenger_smape = smape(challenger)
    return {
        "zone": zone,
        "scope": scope,
        "n_days": int(frame["delivery_day_local"].nunique()),
        "n_hours": len(frame),
        "production_mae": production_mae,
        "challenger_mae": challenger_mae,
        "mae_delta": challenger_mae - production_mae,
        "production_rmse": production_rmse,
        "challenger_rmse": challenger_rmse,
        "rmse_delta": challenger_rmse - production_rmse,
        "production_bias": production_bias,
        "challenger_bias": challenger_bias,
        "absolute_bias_delta": abs(challenger_bias) - abs(production_bias),
        "production_smape_pct": production_smape,
        "challenger_smape_pct": challenger_smape,
        "smape_delta_pct": challenger_smape - production_smape,
        "challenger_win_rate": float(np.mean(challenger_abs < production_abs)),
        "tie_rate": float(np.mean(challenger_abs == production_abs)),
        "challenger_day_win_rate": float(
            np.mean(daily["challenger_abs"] < daily["production_abs"])
        ),
    }


def _metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _metric_row(frame, zone="FR", scope="zone"),
            _metric_row(frame, zone="ALL", scope="aggregate"),
        ]
    )


def _manifest(
    frame: pd.DataFrame,
    *,
    day: str = "2026-08-20",
    scored: bool = True,
    actual_source: str = "audited_statistics_history",
) -> dict[str, object]:
    hours = len(local_delivery_day_index(day, timezone="Europe/Paris"))
    return {
        "schema_version": 1,
        "comparison_type": "prospective_paired_ab",
        "treatment": "residual_load_source_chronos2_vs_saturn",
        "downstream_policy": "frozen_current_production_downstream",
        "interpretation_warning": "Prospective paired comparison under distribution shift.",
        "zones": ["FR"],
        "start_delivery_day_local": day,
        "end_delivery_day_local": day,
        "n_discovered_pairs": 1,
        "n_scored_pairs": int(scored),
        "n_scored_hours": len(frame),
        "pairs": [
            {
                "zone": "FR",
                "delivery_day_local": day,
                "hours_in_local_day": hours,
                "production_manifest_sha256": "1" * 64,
                "challenger_manifest_sha256": "2" * 64,
                "downstream_source_identity_sha256": "3" * 64,
                "protected_input_identity_sha256": "4" * 64,
                "realized_complete": scored,
                "scored": scored,
            }
        ],
        "unpaired_archives": [],
        "actuals": {
            "FR": {
                "source": actual_source,
                "path": r"C:\private\statistics_history_hourly.csv.gz",
                "sha256": "5" * 64,
                "n_available_hours": hours,
            }
        },
    }


def _render(
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    return render_residual_load_comparison_report(
        paired,
        metrics,
        manifest,
        paired_sha256="a" * 64,
        metrics_sha256="b" * 64,
    )


def test_report_is_standalone_prospective_and_does_not_touch_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    metrics = _metrics(paired)
    manifest = _manifest(paired)
    paired_before = paired.copy(deep=True)
    metrics_before = metrics.copy(deep=True)

    def refuse_file_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Le renderer ne doit lire aucun fichier.")

    monkeypatch.setattr(builtins, "open", refuse_file_access)
    monkeypatch.setattr(Path, "read_text", refuse_file_access)
    rendered = _render(paired, metrics, manifest)

    pd.testing.assert_frame_equal(paired, paired_before)
    pd.testing.assert_frame_equal(metrics, metrics_before)
    document = _Document()
    document.feed(rendered)
    assert rendered.startswith("<!doctype html>")
    assert REPORT_SCHEMA in rendered
    assert "Comparaison prospective appariée réalisée" in rendered
    assert "Production · charge résiduelle Saturn" in rendered
    assert "Challenger · charge résiduelle Chronos-2" in rendered
    assert "1,00" in rendered
    assert "2,00" in rendered
    assert document.svg_count == 2
    assert document.external_references == []
    assert len(document.scripts) == 2
    for script in document.scripts:
        _assert_javascript_delimiters_are_balanced(script)
    assert "b.setAttribute('aria-label',t==='dark'?'Activer le mode clair':'Activer le mode nuit');" in rendered
    assert {"overview", "zone-fr", "hourly-values", "audit"}.issubset(document.ids)
    assert "connect-src 'none'" in rendered
    assert "object-src 'none'" in rendered
    assert "C:\\private" not in rendered
    assert "backtest" not in rendered.lower()
    assert "storm" not in rendered.lower()
    assert "statistics_history_hourly" not in rendered.lower()


def test_empty_report_uses_na_and_audits_unscored_pair_with_safe_escaping() -> None:
    paired = pd.DataFrame(columns=_PAIRED_COLUMNS)
    metrics = _metrics(paired)
    source = 'réel</td><script id="injected">alert(1)</script>'
    manifest = _manifest(paired, scored=False, actual_source=source)
    manifest["unpaired_archives"] = [
        {
            "zone": "FR",
            "delivery_day_local": "2026-08-20",
            "missing": "challenger",
        }
    ]

    rendered = _render(paired, metrics, manifest)

    document = _Document()
    document.feed(rendered)
    assert "Données insuffisantes" in rendered
    assert "N/A" in rendered
    assert "Non scorée · prix réels incomplets" in rendered
    assert "Aucune paire scorée" in rendered
    assert "challenger" in rendered
    assert document.svg_count == 0
    assert "injected" not in document.ids
    assert "&lt;script id=&quot;injected&quot;&gt;" in rendered
    assert "NaN" not in rendered
    assert "Infinity" not in rendered


@pytest.mark.parametrize(
    ("day", "expected_hours"),
    [("2026-03-29", 23), ("2026-10-25", 25)],
)
def test_report_accepts_complete_dst_days(day: str, expected_hours: int) -> None:
    paired = _paired(day)

    rendered = _render(paired, _metrics(paired), _manifest(paired, day=day))

    assert len(paired) == expected_hours
    assert f">{expected_hours}<" in rendered
    assert day in rendered


def test_report_rejects_metric_not_computed_from_paired_rows() -> None:
    paired = _paired()
    metrics = _metrics(paired)
    metrics.loc[metrics["zone"] == "ALL", "challenger_mae"] = 999.0

    with pytest.raises(
        ResidualLoadComparisonReportError,
        match="challenger_mae.*incohérent",
    ):
        _render(paired, metrics, _manifest(paired))


def test_report_rejects_naive_or_incomplete_timeline() -> None:
    paired = _paired()
    paired["delivery_start_utc"] = paired["delivery_start_utc"].dt.tz_localize(None)

    with pytest.raises(
        ResidualLoadComparisonReportError,
        match="timezone-aware",
    ):
        _render(paired, _metrics(_paired()), _manifest(paired))


def test_report_rejects_non_prospective_manifest() -> None:
    paired = _paired()
    manifest = _manifest(paired)
    manifest["comparison_type"] = "historical_relabelled"

    with pytest.raises(
        ResidualLoadComparisonReportError,
        match="comparison_type",
    ):
        _render(paired, _metrics(paired), manifest)
