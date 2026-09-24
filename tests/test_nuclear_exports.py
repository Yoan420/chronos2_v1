import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from chronos2_hourly import nuclear_exports as exports
from chronos2_hourly.hourly_contract import local_delivery_day_index


def fixture(tmp_path, day="2026-09-09"):
    frame = pd.DataFrame({"delivery_start_utc": local_delivery_day_index(day)})
    for model in ("residual_corrected", "residual_kalman"):
        for quantile, value in (("q10", 40.), ("q50", 50.), ("q90", 60.)):
            frame[f"{model}__{quantile}"] = value
    result = SimpleNamespace(source_forecast=frame, kalman_view=SimpleNamespace(forecast=frame))
    source = tmp_path / "reports"
    source.mkdir()
    reports = {}
    for kind in ("autonomous", "kalman"):
        reports[kind] = source / f"{kind}.html"
        reports[kind].write_text("<html>Statistics Storm</html>", encoding="utf-8")
    reports["audit"] = source / "audit.json"
    reports["audit"].write_text(json.dumps({"zone": "FR", "delivery_day": day}), encoding="utf-8")
    return result, reports


@pytest.mark.parametrize("day,hours", [("2026-09-09", 24), ("2025-03-30", 23), ("2025-10-26", 25)])
def test_same_export_format_preserves_incumbents_and_dst(tmp_path, day, hours):
    result, reports = fixture(tmp_path, day)
    incumbent = tmp_path / "runs/exports" / day / "fr/autonomous/forecast.html"
    incumbent.parent.mkdir(parents=True)
    incumbent.write_text("incumbent", encoding="utf-8")
    current = incumbent.parents[1] / "current_batch_manifest.json"
    current.write_text("incumbent manifest", encoding="utf-8")
    outputs = exports.publish_nuclear_exports(result, reports, project_root=tmp_path,
        zone="FR", delivery_day=day, timezone="Europe/Paris")
    assert incumbent.read_text() == "incumbent"
    assert current.read_text() == "incumbent manifest"
    assert Path(outputs["autonomous"]).is_file()
    values = pd.read_csv(Path(outputs["kalman"]).with_suffix(".csv"))
    assert len(values) == hours
    assert values.forecast_variant.eq("nuclear_kalman").all()
    assert values.price_eur_mwh.eq(50).all()
    assert not values.uses_mkonline.any()
    assert values.delivery_start_utc.is_unique
    if hours == 25:
        assert set(values.loc[values.local_hour.eq(2), "fold"]) == {0, 1}
    assert json.loads(Path(outputs["manifest"]).read_text())["incumbent_exports_modified"] is False


def test_invalid_second_variant_publishes_nothing(tmp_path):
    result, reports = fixture(tmp_path)
    result.kalman_view.forecast = result.kalman_view.forecast.iloc[:-1]
    with pytest.raises(ValueError, match="exact physical"):
        exports.publish_nuclear_exports(result, reports, project_root=tmp_path,
            zone="FR", delivery_day="2026-09-09", timezone="Europe/Paris")
    assert not list((tmp_path / "runs/exports").rglob("*.html"))


def test_publication_failure_restores_both_previous_reports(tmp_path, monkeypatch):
    result, reports = fixture(tmp_path)
    kwargs = dict(project_root=tmp_path, zone="FR", delivery_day="2026-09-09", timezone="Europe/Paris")
    outputs = exports.publish_nuclear_exports(result, reports, **kwargs)
    before = {str(path): path.read_bytes() for path in (tmp_path / "runs/exports").rglob("*") if path.is_file()}
    reports["autonomous"].write_text("<html>new report</html>", encoding="utf-8")
    replace = exports.os.replace
    def failing_once(source, target):
        if Path(source).name == "current_nuclear_batch_manifest.json":
            raise OSError("publication interrupted")
        return replace(source, target)

    monkeypatch.setattr(exports.os, "replace", failing_once)
    with pytest.raises(OSError, match="interrupted"):
        exports.publish_nuclear_exports(result, reports, **kwargs)
    assert {str(path): path.read_bytes() for path in (tmp_path / "runs/exports").rglob("*") if path.is_file()} == before
    assert Path(outputs["manifest"]).is_file()


def test_late_directory_collision_never_changes_earlier_exports(tmp_path):
    result, reports = fixture(tmp_path)
    kwargs = dict(project_root=tmp_path, zone="FR", delivery_day="2026-09-09", timezone="Europe/Paris")
    destination = tmp_path / "runs/exports/2026-09-09/fr"
    older = destination / "nuclear_autonomous/forecast_fr_2026-09-09_nuclear_autonomous.html"
    older.parent.mkdir(parents=True)
    older.write_text("previous autonomous export", encoding="utf-8")
    collision = destination / "nuclear_kalman/forecast_fr_2026-09-09_nuclear_kalman.html"
    collision.mkdir(parents=True)
    (collision / "keep.txt").write_text("unrelated directory", encoding="utf-8")
    before = {str(p): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="not a file"):
        exports.publish_nuclear_exports(result, reports, **kwargs)
    assert {str(p): p.read_bytes() for p in destination.rglob("*") if p.is_file()} == before
    assert collision.is_dir()


def test_kalman_only_publication_preserves_unselected_variant(tmp_path):
    result, reports = fixture(tmp_path)
    kwargs = dict(project_root=tmp_path, zone="FR", delivery_day="2026-09-09", timezone="Europe/Paris")
    first = exports.publish_nuclear_exports(result, reports, **kwargs)
    autonomous = Path(first["autonomous"])
    before = {p.name: p.read_bytes() for p in autonomous.parent.iterdir()}
    reports["kalman"].write_text("<html>new Kalman only</html>", encoding="utf-8")
    reports.pop("autonomous")
    second = exports.publish_nuclear_exports(result, reports, **kwargs, report_variants=("kalman",))
    assert "autonomous" not in second
    assert {p.name: p.read_bytes() for p in autonomous.parent.iterdir()} == before
    manifest = json.loads(Path(second["manifest"]).read_text())
    assert manifest["mode"] == "nuclear_kalman"
    assert [r["variant"] for r in manifest["exports"]] == ["nuclear_kalman"]
    assert "nuclear_autonomous/" not in Path(second["index"]).read_text(encoding="utf-8")
