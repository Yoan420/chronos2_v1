from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run_mkonline_live_zone as dispatcher


class _Audit:
    def as_dict(self):
        return {"status": "ready"}

    def require_ready(self) -> None:
        return None


@pytest.mark.parametrize(
    "runner_name",
    ["run_mkonline_live_hourly.py", "run_mkonline_live_model.py"],
)
def test_dispatcher_enables_zone_scoped_capture_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_name: str,
) -> None:
    runner = tmp_path / runner_name
    runner.write_text("pass\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text("zones: {}\n", encoding="utf-8")
    observed: list[list[str]] = []
    monkeypatch.setattr(dispatcher, "load_zone_registry", lambda _path: ({}, tmp_path))
    monkeypatch.setattr(dispatcher, "audit_zone_live_bundle", lambda *_args, **_kwargs: _Audit())
    monkeypatch.setattr(dispatcher, "strict_contract_preflight", lambda audit, **_kwargs: audit)
    monkeypatch.setattr(
        dispatcher,
        "build_zone_runner_command",
        lambda *_args, **_kwargs: ["python", str(runner), "--config", "live.yaml"],
    )
    monkeypatch.setattr(
        dispatcher.subprocess,
        "run",
        lambda command, **_kwargs: observed.append(list(command)),
    )
    monkeypatch.setattr(
        dispatcher,
        "parse_args",
        lambda: SimpleNamespace(
            registry=str(registry),
            zone="BE",
            preflight_only=False,
            data_as_of=None,
            delivery_day=None,
            output_dir=None,
            device=None,
            threads=None,
            workers=None,
            local_files_only=False,
            pit_replay=False,
            no_rolling365_capture=False,
        ),
    )
    assert dispatcher.main() == 0
    command = observed[0]
    flag = command.index("--rolling365-capture-root")
    assert Path(command[flag + 1]) == (
        tmp_path / "runs" / "rolling365_shadow"
    ).resolve()


def test_dispatcher_explicit_opt_out_and_replay_do_not_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "run_mkonline_live_hourly.py"
    runner.write_text("pass\n", encoding="utf-8")
    observed: list[list[str]] = []
    monkeypatch.setattr(dispatcher, "load_zone_registry", lambda _path: ({}, tmp_path))
    monkeypatch.setattr(dispatcher, "audit_zone_live_bundle", lambda *_args, **_kwargs: _Audit())
    monkeypatch.setattr(dispatcher, "strict_contract_preflight", lambda audit, **_kwargs: audit)
    monkeypatch.setattr(
        dispatcher,
        "build_zone_runner_command",
        lambda *_args, **_kwargs: ["python", str(runner)],
    )
    monkeypatch.setattr(
        dispatcher.subprocess,
        "run",
        lambda command, **_kwargs: observed.append(list(command)),
    )
    for no_capture, pit_replay in ((True, False), (False, True)):
        monkeypatch.setattr(
            dispatcher,
            "parse_args",
            lambda no_capture=no_capture, pit_replay=pit_replay: SimpleNamespace(
                registry=str(tmp_path / "registry.yaml"),
                zone="FR",
                preflight_only=False,
                data_as_of=None,
                delivery_day=None,
                output_dir=None,
                device=None,
                threads=None,
                workers=None,
                local_files_only=False,
                pit_replay=pit_replay,
                no_rolling365_capture=no_capture,
            ),
        )
        assert dispatcher.main() == 0
        assert "--rolling365-capture-root" not in observed[-1]

