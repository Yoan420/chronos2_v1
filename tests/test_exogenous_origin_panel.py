from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.feature_bank import ExogenousBank, delivery_utc_index
from chronos2_exogenous.panel import (
    OriginPanelError,
    build_origin_panel,
    load_target_cache,
    write_origin_panel,
)
from run_chronos2_exogenous_panel import (
    _canonical_target_path,
    _resolve_target,
)


ZONES = ("FR", "DE", "BE", "NL")
DAYS = ("2026-03-28", "2026-03-29", "2026-10-25")
CONTEXT = 4


def _required_index() -> pd.DatetimeIndex:
    pieces: list[pd.DatetimeIndex] = []
    for day in DAYS:
        horizon = delivery_utc_index(day, day)
        context = pd.date_range(
            end=horizon[0] - pd.Timedelta(hours=1),
            periods=CONTEXT,
            freq="h",
        )
        pieces.extend([context, horizon])
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.union(piece)
    return result.sort_values()


def _bank(zone: str, index: pd.DatetimeIndex) -> ExogenousBank:
    lower = zone.casefold()
    frame = pd.DataFrame(
        {
            "shared_residual_load": np.linspace(10.0, 20.0, len(index)),
            f"{lower}_temperature_fcst": np.linspace(0.0, 15.0, len(index)),
        },
        index=index,
    )
    catalogue = pd.DataFrame(
        [
            {
                "column": column,
                "source": "synthetic",
                "family": "test",
                "role": "value",
                "known_future": True,
                "chronos": True,
                "residual": False,
                "kalman": False,
            }
            for column in frame.columns
        ]
    )
    return ExogenousBank(
        frame=frame,
        catalogue=catalogue,
        audit={
            "source_hashes": {"synthetic": f"sha-{zone}"},
            "production_pit_evidence": {"synthetic": True},
            "production_ready": True,
            "production_blockers": [],
        },
    )


def _targets(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {
        zone: pd.Series(
            np.linspace(20.0 + number, 100.0 + number, len(index)),
            index=index,
        )
        for number, zone in enumerate(ZONES)
    }


def test_per_zone_panel_keeps_dst_days_for_evaluation_and_filters_fit() -> None:
    index = _required_index()
    panel = build_origin_panel(
        {zone: _bank(zone, index) for zone in ZONES},
        _targets(index),
        delivery_days=DAYS,
        context_length=CONTEXT,
        layout="per_zone",
    )

    assert panel.audit["horizon_day_counts"] == {"24": 1, "23": 1, "25": 1}
    assert panel.audit["fit_excluded_dst_days"] == ["2026-03-29", "2026-10-25"]
    assert panel.audit["evaluation_origins_reserved"] == 3
    assert set(panel.frame["item_id"]) == set(ZONES)
    assert "local_temperature_fcst" in panel.frame
    assert not any(f"{zone.casefold()}_temperature_fcst" in panel.frame for zone in ZONES)
    assert len(panel.for_fit()) == (CONTEXT + 24) * len(ZONES)
    assert len(panel.for_evaluation()) == (
        (CONTEXT + 23) + (CONTEXT + 24) + (CONTEXT + 25)
    ) * len(ZONES)
    assert bool(
        (
            panel.frame["feature_available_at_utc"]
            <= panel.frame["origin_timestamp"]
        ).all()
    )


def test_cwe_wide_panel_has_four_targets_and_zone_weather() -> None:
    index = _required_index()
    panel = build_origin_panel(
        {zone: _bank(zone, index) for zone in ZONES},
        _targets(index),
        delivery_days=[DAYS[0]],
        context_length=CONTEXT,
        layout="cwe_wide",
    )

    assert panel.frame["item_id"].eq("CWE").all()
    assert all(f"target_{zone.casefold()}" in panel.frame for zone in ZONES)
    assert all(f"{zone.casefold()}_temperature_fcst" in panel.frame for zone in ZONES)
    assert panel.audit["production_ready"] is True


def test_target_cache_and_atomic_panel_bundle(tmp_path: Path) -> None:
    index = _required_index()
    target_path = tmp_path / "target__test.csv.gz"
    pd.DataFrame(
        {"timestamp": index.astype(str), "value": np.arange(len(index), dtype=float)}
    ).to_csv(target_path, index=False, compression="gzip")
    target = load_target_cache(target_path, zone="FR")
    panel = build_origin_panel(
        _bank("FR", index),
        {"FR": target},
        delivery_days=[DAYS[0]],
        context_length=CONTEXT,
        layout="per_zone",
        zones=["FR"],
    )
    output, audit_path = write_origin_panel(panel, tmp_path / "panel.parquet")

    assert output.is_file() and audit_path.is_file()
    assert len(pd.read_parquet(output)) == CONTEXT + 24
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(audit["panel_sha256"]) == 64
    assert audit["target_sources"]["FR"]["source_sha256"] == target.attrs[
        "source_sha256"
    ]


def test_target_cache_identity_comes_from_live_contract(tmp_path: Path) -> None:
    (tmp_path / "chronos2_hourly_fr_mkonline_live_v1.yaml").write_text(
        "live:\n  base_config: base.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text(
        """data:
  cache_dir: data/cache
zones:
  FR:
    target:
      series: power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh
      naive_timezone: UTC
""",
        encoding="utf-8",
    )
    canonical, contract = _canonical_target_path(tmp_path, "FR")

    assert canonical.parent == (tmp_path / "data" / "cache" / "fr").resolve()
    assert canonical.name == "target__5dcf0bdf8c.csv.gz"
    assert contract["series"] == (
        "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
    )

    noncanonical = tmp_path / "data" / "cache" / "fr" / "target__wrong.csv.gz"
    noncanonical.parent.mkdir(parents=True)
    noncanonical.write_bytes(b"not-even-a-target")
    with pytest.raises(OriginPanelError, match="non canonique"):
        _resolve_target(
            tmp_path,
            "FR",
            {"FR": noncanonical},
            timezone="Europe/Paris",
        )
