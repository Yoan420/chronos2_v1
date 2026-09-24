"""Produce a sealed post-coupling spread diagnostic from captured public data."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/experiments/nyx_physical_p50_v1/forensics/2026-09-14/20260915T122535Z_89d3328e"
PANEL = ROOT / "runs/experiments/nyx_scarcity_v1/coherent_p50/snapshots/20260914T160104Z_d7366f80/panel.parquet"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SOURCE / "forensic_manifest.json").read_text(encoding="utf8"))
    for record in manifest["raw_http"]:
        if digest(SOURCE / record["path"]) != record["sha256"]:
            raise ValueError("Changed raw forensic input")
    shadow = pd.DataFrame(json.loads((SOURCE / "raw_http/002_shadowPrices.json").read_text())["data"])
    shadow["timestamp_utc"] = pd.to_datetime(shadow.dateTimeUtc, utc=True)
    expected = pd.date_range("2026-09-14T17:00Z", periods=4, freq="15min")
    event = shadow.loc[shadow.timestamp_utc.isin(expected)].copy()
    if set(event.timestamp_utc.unique()) != set(expected):
        raise ValueError("Missing quarter hour")
    for zone in ("DE", "BE", "NL"):
        event[f"spread_{zone}_FR_eur_mwh"] = event.shadowPrice * (event.hub_FR-event[f"hub_{zone}"])
    numeric = [name for name in event if name.startswith("spread_")]
    mtu = event.groupby("dateTimeUtc")[numeric].sum().reset_index()
    contributions = event.groupby(["cnecName", "contName"], dropna=False)[numeric].sum().div(4).reset_index()
    observed = pd.read_parquet(PANEL, columns=["zone", "timestamp_utc", "actual"])
    observed["timestamp_utc"] = pd.to_datetime(observed.timestamp_utc, utc=True)
    observed = observed.loc[observed.timestamp_utc.eq(expected[0])].set_index("zone").actual.to_dict()
    comparisons = [{"zone": zone, "reference_zone": "FR", "observed_spread_eur_mwh": observed[zone]-observed["FR"],
                    "active_cnec_dual_sum_eur_mwh": float(mtu[f"spread_{zone}_FR_eur_mwh"].mean()),
                    "unexplained_difference_eur_mwh": observed[zone]-observed["FR"]-float(mtu[f"spread_{zone}_FR_eur_mwh"].mean())}
                   for zone in ("DE", "BE", "NL")]
    initial = pd.DataFrame(json.loads((SOURCE / "initialComputation_rows.json").read_text(encoding="utf8")))
    initial = initial.loc[initial.dateTimeUtc.eq("2026-09-14T17:00:00Z") & initial.cneName.str.contains("Vigy|Gronau", case=False)]
    net = pd.DataFrame(json.loads((SOURCE / "raw_http/003_netPos.json").read_text())["data"])
    net_time = pd.to_datetime(net.dateTimeUtc, utc=True)
    if not pd.DatetimeIndex(net_time).sort_values().equals(pd.date_range("2026-09-14T16:00Z", periods=12, freq="15min")):
        raise ValueError("Incomplete net-position grid")
    payload = {
        "diagnostic_only": True, "allowed_as_model_input": False, "cutoff_utc": "2026-09-13T06:00:00Z",
        "shadow_publication_watermark_utc": manifest["results"]["shadowPrices"]["last_modified_utc"],
        "formula": "Sum_c shadowPrice_c * (hub_FR_c - hub_zone_c), then mean of four 15-min MTUs",
        "not_claimed": ["Exact marginal generating plant", "Full causal counterfactual", "Pre-08 availability of post-coupling fields", "Exact full spread reconstruction including all LTA/allocation/loss/rounding terms"],
        "observed_source_sha256": digest(PANEL), "observed_source": str(PANEL),
        "collector_manifest_sha256": digest(SOURCE / "forensic_manifest.json"),
        "raw_input_sha256": {record["path"]: record["sha256"] for record in manifest["raw_http"]},
        "observed_prices_eur_mwh": observed, "spread_comparison": comparisons,
        "quarter_hour_spread_diagnostics": mtu.to_dict("records"),
        "cnec_contributions_hour_mean": contributions.to_dict("records"),
        "active_cnecs_19h_paris": event[["dateTimeUtc", "cnecName", "contName", "shadowPrice", "hub_DE", "hub_BE", "hub_FR", "hub_NL", *numeric]].to_dict("records"),
        "initial_presolved_related_constraints_19h_paris": initial[["dateTimeUtc", "cneName", "contName", "direction", "ram", "fmax", "frm", "frefInit", "fcore", "ptdf_DE", "ptdf_BE", "ptdf_FR", "ptdf_NL"]].to_dict("records"),
        "net_positions_core_not_full_sdac_mw": net[["dateTimeUtc", "hub_DE", "hub_BE", "hub_FR", "hub_NL"]].to_dict("records"),
    }
    # Pandas' JSON conversion removes NaN in missing contingency names.
    encoded = json.dumps(json.loads(pd.Series(payload).to_json(date_format="iso")), ensure_ascii=False, indent=2, allow_nan=False)
    output = SOURCE / "spread_diagnostic.json"
    with output.open("x", encoding="utf8") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(output), "comparisons": comparisons,
                      "contributions": json.loads(contributions.to_json(orient="records"))}, indent=2))


if __name__ == "__main__":
    main()
