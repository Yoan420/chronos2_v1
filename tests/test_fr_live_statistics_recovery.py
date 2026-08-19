from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import run_mkonline_live_hourly as live_runner


MISSING_DAYS = [
    "2026-08-16",
    "2026-08-17",
    "2026-08-18",
    "2026-08-19",
]
GAP_ERROR = (
    "Forecast archive is incomplete for Statistics; create causal PIT replay "
    "run(s): " + ", ".join(MISSING_DAYS)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Harness:
    output_root: Path
    output: Path
    forecast_sha256: dict[str, str]
    statistics_calls: list[dict[str, Any]]
    report_calls: list[Path]


def _install_live_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    statistics_gap: bool,
) -> _Harness:
    delivery_day = "2026-08-20"
    as_of = "2026-08-19T09:00:00+02:00"
    schedule = live_runner._resolve_schedule(as_of, delivery_day)
    frozen = tmp_path / "frozen"
    benchmark = tmp_path / "benchmark"
    output_root = tmp_path / "runs"
    output = output_root / f"fr_day_ahead_{delivery_day}"
    frozen.mkdir()
    benchmark.mkdir()
    (frozen / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    (benchmark / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    (benchmark / "run_manifest.json").write_text(
        json.dumps(
            {
                "target_series": (
                    "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
                ),
                "target_availability": {"stale": True},
            }
        ),
        encoding="utf-8",
    )
    base_config = tmp_path / "base.yaml"
    recipe = tmp_path / "recipe.json"
    dependency = tmp_path / "dependency.json"
    base_config.write_text("model: {}", encoding="utf-8")
    recipe.write_text("{}", encoding="utf-8")
    dependency.write_text("{}", encoding="utf-8")
    config = tmp_path / "live.yaml"
    config.write_text(
        "\n".join(
            [
                "live:",
                f"  base_config: '{base_config.as_posix()}'",
                f"  frozen_autonomous_run: '{frozen.as_posix()}'",
                f"  sealed_benchmark_run: '{benchmark.as_posix()}'",
                f"  recipe_manifest: '{recipe.as_posix()}'",
                f"  dependency_manifest: '{dependency.as_posix()}'",
                f"  output_root: '{output_root.as_posix()}'",
                "report:",
                "  filename: 'fr_detailed_{delivery_day}.html'",
                "  title: 'Forecast FR {delivery_day}'",
            ]
        ),
        encoding="utf-8",
    )

    index = schedule.delivery_index
    chronos = pd.DataFrame(
        {
            "forecast_origin_utc": schedule.cutoff_local.tz_convert("UTC"),
            "q10": 30.0,
            "q50": 40.0,
            "q90": 50.0,
        },
        index=index,
    )
    chronos.index.name = "delivery_start_utc"
    extended = chronos.loc[:, ["q10", "q50", "q90"]].copy()
    fresh = pd.DataFrame({"feature": 1.0}, index=index)
    availability = {"forecast_delivery_day_local": delivery_day}
    forecast_sha256: dict[str, str] = {}
    statistics_calls: list[dict[str, Any]] = []
    report_calls: list[Path] = []

    real_timestamp = pd.Timestamp

    class _FrozenTimestamp(real_timestamp):
        @classmethod
        def now(cls, tz=None):
            value = real_timestamp(as_of)
            return value.tz_convert(tz) if tz is not None else value.tz_localize(None)

    monkeypatch.setattr(live_runner.pd, "Timestamp", _FrozenTimestamp)
    monkeypatch.setattr(
        live_runner,
        "_load_recipe",
        lambda *_args, **_kwargs: {
            "external_expert": {"commercial_entitlement_status": "test"}
        },
    )
    monkeypatch.setattr(live_runner, "_verify_source_run", lambda *_args: None)
    monkeypatch.setattr(
        live_runner, "_verify_frozen_training_source", lambda *_args: {}
    )
    monkeypatch.setattr(live_runner, "_verify_benchmark", lambda *_args: {})
    monkeypatch.setattr(
        live_runner,
        "_build_dynamic_data",
        lambda **_kwargs: (
            {"model": {}},
            SimpleNamespace(
                target=pd.Series([50.0], index=[index[0]]), diagnostics={}
            ),
            fresh,
        ),
    )
    monkeypatch.setattr(
        live_runner, "_validate_live_input_audit", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        live_runner,
        "_audit_future_pit_freshness",
        lambda **_kwargs: {"status": "fresh"},
    )
    monkeypatch.setattr(
        live_runner, "load_model", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        live_runner,
        "run_existing_live_forecast",
        lambda *_args, **_kwargs: chronos,
    )
    monkeypatch.setattr(
        live_runner,
        "_train_and_predict_extended",
        lambda **_kwargs: (
            extended,
            {"target_availability": availability},
        ),
    )

    def fake_materialize(**kwargs: Any) -> list[str]:
        kwargs["output"].write_bytes(b"parquet placeholder")
        return ["materialize", delivery_day]

    monkeypatch.setattr(live_runner, "_materialize", fake_materialize)
    monkeypatch.setattr(
        live_runner,
        "_load_primary",
        lambda *_args, **_kwargs: (
            pd.Series(45.0, index=index),
            pd.Series(schedule.cutoff_local.tz_convert("UTC"), index=index),
            {"coverage": 1.0},
        ),
    )

    def fake_copy(_benchmark: Path, staging: Path) -> None:
        (staging / "metrics_hourly.json").write_text(
            json.dumps(
                {
                    "metrics": [{"model": "mkonline_blend", "mae": 11.23}],
                    "training_diagnostics": {"mkonline_blend": {}},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(live_runner, "_copy_benchmark_for_report", fake_copy)
    monkeypatch.setattr(
        live_runner,
        "_load_storm_dashboard_statistics_snapshot",
        lambda **_kwargs: (
            pd.Series(42.0, index=index),
            {
                "requested_series": (
                    "power.price.fr.euromwh.h.fcst.3mv.storm"
                ),
                "used_for_prediction": False,
            },
        ),
    )
    monkeypatch.setattr(
        live_runner,
        "missing_statistics_archive_days",
        lambda **_kwargs: (
            [date.fromisoformat(value) for value in MISSING_DAYS]
            if statistics_gap
            else []
        ),
    )

    def write_statistics_files(staging: Path, audit: dict[str, Any]) -> None:
        pd.DataFrame(
            {
                "delivery_start_utc": index,
                "actual": 40.0,
                "mkonline_blend__q10": 35.0,
                "mkonline_blend__q50": 40.0,
                "mkonline_blend__q90": 45.0,
                "storm_dashboard_official__q50": 42.0,
            }
        ).to_csv(
            staging / "statistics_history_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        (staging / "statistics_history_audit.json").write_text(
            json.dumps(
                {
                    **audit,
                    "report_scope_note": (
                        "Statistics causales partielles; prévision courante valide."
                    ),
                }
            ),
            encoding="utf-8",
        )

    def fake_history(**kwargs: Any) -> dict[str, Any]:
        staging = Path(kwargs["staging_run_dir"])
        forecast_path = staging / "forecast_hourly_fr.csv"
        assert forecast_path.is_file(), "Statistics must run after candidate freeze"
        frozen_sha = _sha256(forecast_path)
        forecast_sha256.setdefault("frozen", frozen_sha)
        assert forecast_sha256["frozen"] == frozen_sha
        statistics_calls.append(dict(kwargs))
        if statistics_gap and not kwargs.get("allow_partial_prefix", False):
            raise ValueError(GAP_ERROR)
        if statistics_gap:
            blocker = kwargs.get("statistics_blocker")
            assert blocker is not None
            assert blocker["status"] == "blocked_missing_causal_archives"
            assert blocker["missing_realized_days"] == MISSING_DAYS
            assert blocker["candidate_forecast_publication"] == "continued"
            assert blocker["statistics_scope"] == "contiguous_prefix"
            assert blocker["storm_used_for_prediction"] is False
            audit = {
                "status": "partial_contiguous_prefix",
                "statistics_complete": False,
                "statistics_prefix_end_local": "2026-08-15",
                "evaluated_realized_days": [
                    "2026-08-12",
                    "2026-08-13",
                    "2026-08-14",
                    "2026-08-15",
                ],
                "missing_realized_days": MISSING_DAYS,
                "excluded_after_first_gap_days": MISSING_DAYS[1:],
                "statistics_blocker": dict(blocker),
                "storm_used_for_prediction": False,
            }
        else:
            audit = {
                "status": "complete",
                "statistics_complete": True,
                "missing_realized_days": [],
                "storm_used_for_prediction": False,
            }
        write_statistics_files(staging, audit)
        assert _sha256(forecast_path) == frozen_sha
        return audit

    monkeypatch.setattr(
        live_runner, "update_live_statistics_history", fake_history
    )

    def fake_report(
        run_dir: str | Path,
        *,
        output_path: str | Path,
        **_kwargs: Any,
    ) -> Path:
        staging = Path(run_dir)
        report_path = Path(output_path)
        forecast_path = staging / "forecast_hourly_fr.csv"
        assert _sha256(forecast_path) == forecast_sha256["frozen"]
        assert (staging / "statistics_history_hourly.csv.gz").is_file()
        assert (staging / "statistics_history_audit.json").is_file()
        if statistics_gap:
            diagnostic_path = staging / "statistics_update_blocked.json"
            assert diagnostic_path.is_file()
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            assert diagnostic["status"] == "partial_contiguous_prefix"
            assert diagnostic["missing_realized_days"] == MISSING_DAYS
            assert diagnostic["candidate_forecast_publication"] == "continued"
        report_path.write_text(
            "<!doctype html><html><body>"
            "<h1>Forecast FR détaillé</h1>"
            "<div id='forecast-chart'>Graphique forecast</div>"
            "<section id='statistics'>Statistics partielles</section>"
            "</body></html>",
            encoding="utf-8",
        )
        report_calls.append(report_path)
        return report_path

    monkeypatch.setattr(live_runner, "write_hourly_html_report", fake_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mkonline_live_hourly.py",
            "--config",
            str(config),
            "--data-as-of",
            as_of,
            "--delivery-day",
            delivery_day,
        ],
    )
    return _Harness(
        output_root=output_root,
        output=output,
        forecast_sha256=forecast_sha256,
        statistics_calls=statistics_calls,
        report_calls=report_calls,
    )


def test_fr_statistics_gap_after_candidate_freeze_publishes_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_live_harness(
        tmp_path, monkeypatch, statistics_gap=True
    )

    assert live_runner.main() == 0

    output = harness.output
    assert output.is_dir()
    assert not list(harness.output_root.glob(".*.tmp-*"))
    forecast = output / "forecast_hourly_fr.csv"
    assert forecast.is_file()
    assert _sha256(forecast) == harness.forecast_sha256["frozen"]
    assert [
        bool(call.get("allow_partial_prefix", False))
        for call in harness.statistics_calls
    ] == [False, True]

    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    statistics = manifest["statistics_history"]
    assert statistics["status"] == "partial_contiguous_prefix"
    assert statistics["statistics_complete"] is False
    assert statistics["missing_realized_days"] == MISSING_DAYS
    assert statistics["diagnostic_path"] == "statistics_update_blocked.json"
    assert manifest["reporting_status"] == "complete"

    summary = json.loads(
        (output / "live_run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "forecast_complete_statistics_partial"
    assert summary["statistics_history"]["missing_realized_days"] == MISSING_DAYS

    diagnostic = json.loads(
        (output / "statistics_update_blocked.json").read_text(encoding="utf-8")
    )
    assert diagnostic["missing_realized_days"] == MISSING_DAYS
    assert diagnostic["candidate_forecast_publication"] == "continued"
    assert diagnostic["storm_used_for_prediction"] is False

    report = output / "fr_detailed_2026-08-20.html"
    assert len(harness.report_calls) == 1
    assert harness.report_calls[0].name == report.name
    rendered = report.read_text(encoding="utf-8")
    assert "forecast-chart" in rendered
    assert "Statistics partielles" in rendered

    checksums = json.loads(
        (output / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    forecast_entry = next(
        item
        for item in checksums["artifacts"]
        if item["path"] == "forecast_hourly_fr.csv"
    )
    assert forecast_entry["sha256"] == harness.forecast_sha256["frozen"]


@pytest.mark.parametrize(
    ("fatal_stage", "message"),
    [
        ("candidate", "candidate construction failed"),
        ("checksums", "checksum generation failed"),
        ("publication", "atomic publication failed"),
    ],
)
def test_fr_candidate_checksum_and_publication_failures_remain_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fatal_stage: str,
    message: str,
) -> None:
    harness = _install_live_harness(
        tmp_path, monkeypatch, statistics_gap=False
    )

    if fatal_stage == "candidate":
        monkeypatch.setattr(
            live_runner,
            "_forecast_frame",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )
    elif fatal_stage == "checksums":
        monkeypatch.setattr(
            live_runner,
            "_write_checksums",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )
    else:
        monkeypatch.setattr(
            live_runner,
            "_publish",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
        )

    with pytest.raises(RuntimeError, match=message):
        live_runner.main()

    assert not harness.output.exists()
    assert not list(harness.output_root.glob(".*.tmp-*"))
