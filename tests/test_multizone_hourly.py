from __future__ import annotations

from pathlib import Path

import pytest

from chronos2_hourly.zone_live import ZoneBundleError, load_zone_registry
from run_multizone_hourly import audit_training_candidate, build_training_command


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("zone", ["BE", "DE", "ES", "NL"])
def test_checked_in_training_candidates_are_valid(zone: str) -> None:
    registry, registry_dir = load_zone_registry(
        ROOT / "chronos2_hourly_live_zones.yaml"
    )
    audit = audit_training_candidate(
        registry,
        registry_dir=registry_dir,
        zone=zone,
    )
    assert audit["ready"], audit["blockers"]
    command = build_training_command(audit, local_files_only=True)
    assert command[3] == audit["config"]
    assert command[4:6] == ["--zone", zone]
    assert command[-1] == "--local-files-only"


@pytest.mark.parametrize("zone", ["GB", "IT"])
def test_incomplete_zones_fail_before_execution(zone: str) -> None:
    registry, registry_dir = load_zone_registry(
        ROOT / "chronos2_hourly_live_zones.yaml"
    )
    audit = audit_training_candidate(
        registry,
        registry_dir=registry_dir,
        zone=zone,
    )
    assert not audit["ready"]
    with pytest.raises(ZoneBundleError, match="entrainement refuse"):
        build_training_command(audit)


def test_uk_alias_maps_to_blocked_gb_bundle() -> None:
    registry, registry_dir = load_zone_registry(
        ROOT / "chronos2_hourly_live_zones.yaml"
    )
    audit = audit_training_candidate(
        registry,
        registry_dir=registry_dir,
        zone="UK",
    )
    assert audit["zone"] == "GB"
    assert not audit["ready"]

