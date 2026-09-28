"""The app must not bypass the annual CPU qualification or bundle gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_app_service as service
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid


DAY = "2026-09-29"


def test_fixed_bundle_matches_source_adapters_and_blocked_launch_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = service.paths_for_day(tmp_path, DAY)
    assert paths.bundle == tmp_path / "runs/live/nyx_annual_cpu" / DAY
    assert paths.output == tmp_path / "runs/nyx_annual_cpu_live" / DAY
    (tmp_path / "run_nyx_annual_cpu_live.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(service, "preflight", lambda *_: {
        "ready": False, "blockers": ["forecast_enabled:false"]})
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_args, **_kwargs:
                        pytest.fail("Popen must not run while blocked"))
    with pytest.raises(ValueError, match="forecast_enabled:false"):
        service.start_annual_cpu_process(tmp_path, DAY)
    assert not paths.logs.exists()
    assert not paths.output.exists()


def test_launch_uses_fixed_consumer_arguments_after_fresh_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "run_nyx_annual_cpu_live.py"
    runner.write_text("pass\n", encoding="utf-8")
    calls: list[tuple[Path, str, Path]] = []

    def fake_preflight(bundle: Path, day: str, output: Path) -> dict:
        calls.append((bundle, day, output))
        return {"ready": True, "blockers": []}

    command_seen: list[tuple[str, ...]] = []

    def fake_popen(command, **options):
        command_seen.append(tuple(command))
        assert options["shell"] is False
        assert options["cwd"] == tmp_path
        return SimpleNamespace(pid=1234, poll=lambda: None)

    monkeypatch.setattr(service, "preflight", fake_preflight)
    monkeypatch.setattr(service.subprocess, "Popen", fake_popen)
    handle = service.start_annual_cpu_process(
        tmp_path, DAY, python_executable=sys.executable)
    assert len(calls) == 1
    assert command_seen == [(
        str(Path(sys.executable).resolve()), str(runner),
        "--bundle", str(tmp_path / "runs/live/nyx_annual_cpu" / DAY),
        "--delivery-day", DAY,
        "--output", str(tmp_path / "runs/nyx_annual_cpu_live" / DAY),
    )]
    assert handle.return_code is None
    assert handle.log_path.is_file()
    assert not handle.output.exists()


def test_completed_forecast_display_requires_matching_csv_digest(
    tmp_path: Path,
) -> None:
    zone = "FR"
    output = service.paths_for_day(tmp_path, DAY).output
    directory = output / "zones" / zone
    directory.mkdir(parents=True)
    _, current, _ = delivery_grid(DAY)
    frame = pd.DataFrame({
        "timestamp_utc": current,
        "timestamp_local": [value.isoformat() for value in
                            current.tz_convert("Europe/Paris")],
        "price_eur_mwh": np.full(len(current), 50.),
        "p_negative": np.full(len(current), .25),
        "is_negative_predicted": np.zeros(len(current), dtype=bool),
    })
    csv = directory / "forecast_fr.csv"
    frame.to_csv(csv, index=False)
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    receipt = {"protocol": service.PROTOCOL, "status": "COMPLETE",
               "delivery_day": DAY, "countries": {
                   country: {"hours": len(current), "csv": str(csv),
                             "csv_sha256": digest}
                   for country in ("FR", "BE", "NL")}}
    (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    displayed = service.load_annual_cpu_forecast(tmp_path, DAY, zone)
    assert len(displayed) == len(current)
    assert displayed["price_eur_mwh"].eq(50.).all()
    csv.write_text(csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modifié"):
        service.load_annual_cpu_forecast(tmp_path, DAY, zone)


def test_streamlit_control_shows_disabled_annual_cpu_button(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    script = "\n".join([
        "from pathlib import Path",
        "import streamlit as st",
        "from app_nyx_annual_cpu import render_annual_cpu_controls",
        "st.session_state.setdefault('annual_cpu_process', None)",
        "render_annual_cpu_controls(st, project_root=Path(" + repr(str(tmp_path)) +
        "), delivery_day='2026-09-29', conventional_busy=False, "
        "status_provider=lambda root, day: {'ready': False, "
        "'blockers': ['forecast_enabled:false'], 'bundle_inspection': None})",
    ])
    at = AppTest.from_string(script, default_timeout=10).run()
    assert not at.exception
    button = at.button(key="launch_nyx_annual_cpu")
    assert button.disabled
    assert any("Lancement indisponible" in item.value for item in at.warning)
