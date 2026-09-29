"""Diagnostic Saturn en lecture seule ; aucune admission de source ni prévision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from uuid import uuid4

import numpy as np
import pandas as pd

from chronos2_hourly.nyx_annual_saturn_source import ROOT, TARGET_CONTEXT_HOURS, cutoff, grid
from chronos2_modular.saturn import (
    AUDITED_EQUIVALENT_TARGET_FALLBACKS, create_saturn_client, fetch_saturn_series_from_client,
)
from experiment_console.security import redact
from run_nyx_annual_auction_prices_source import SERIES, load_plan


def diagnose(delivery_day, failed_day, zone="FR", *, progress=None):
    old_cutoff, current_cutoff = cutoff(failed_day), cutoff(delivery_day)
    if old_cutoff > current_cutoff or current_cutoff > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Les deux coupures doivent être passées et ordonnées.")
    end = grid(failed_day)[0]
    expected = pd.date_range(end - pd.Timedelta(hours=TARGET_CONTEXT_HOURS), end,
                             freq="h", inclusive="left", name="timestamp_utc")
    canonical = SERIES[zone]
    alternative = AUDITED_EQUIVALENT_TARGET_FALLBACKS[canonical]
    report = {"protocol": "nyx_saturn_target_diagnostic_v1", "diagnostic_only": True,
              "forecast_published": False, "source_admitted": False, "delivery_day": delivery_day,
              "failed_day": failed_day, "zone": zone, "old_cutoff_utc": old_cutoff.isoformat(),
              "current_cutoff_utc": current_cutoff.isoformat(), "probes": {}, "notes": []}
    client, plan, frames = None, {}, {}

    def safe_error(error):
        message = str(error)
        for value in (plan.get("saturn_url"), plan.get("saturn_author"), os.getenv("SATURN_AUTHOR")):
            if value:
                message = message.replace(str(value), "[connexion masquée]")
        return redact({"type": type(error).__name__, "message": message})

    probes = [
        ("canonical_old_long", canonical, expected, old_cutoff, "UTC", 0),
        ("canonical_old_context2048", canonical, expected[-2048:], old_cutoff, "UTC", 0),
        ("canonical_current_long", canonical, expected, current_cutoff, "UTC", 0),
        ("official_alternative_old_long", alternative["series"], expected, old_cutoff,
         alternative["naive_timezone"], alternative.get("request_padding_hours", 0)),
    ]
    try:
        plan = load_plan()
        client = create_saturn_client(plan["saturn_url"], os.getenv("SATURN_AUTHOR") or plan["saturn_author"])
        client.session.request = partial(client.session.request, timeout=60)
        for number, (name, series_name, hours, vintage, naive_timezone, padding) in enumerate(probes, 1):
            if progress:
                progress(f"Test {number}/4 : {name}")
            item = {"series": series_name, "revision_date_utc": vintage.isoformat(),
                    "requested_first_hour_utc": hours[0].isoformat(),
                    "requested_last_hour_utc": hours[-1].isoformat(), "required_hours": len(hours),
                    "naive_timezone": naive_timezone, "padding_hours": padding}
            try:
                series = fetch_saturn_series_from_client(client, series_name, hours[0], hours[-1], "UTC",
                    revision_date=vintage, naive_timezone=naive_timezone, incomplete_dst_policy="raise",
                    nocache=True, request_padding_hours=padding)
                if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None or not series.index.is_unique:
                    raise ValueError("Index Saturn ambigu ou sans fuseau horaire.")
                series = series.copy()
                series.index = series.index.tz_convert("UTC")
                selected = pd.to_numeric(series.reindex(hours), errors="coerce")
                finite = np.isfinite(selected.to_numpy(float))
                available = hours[finite]
                item.update(state="COMPLETE" if finite.all() else "INCOMPLETE", received_hours=len(series),
                    finite_hours=int(finite.sum()), missing_nonfinite_hours=int((~finite).sum()),
                    first_available_hour_utc=available[0].isoformat() if len(available) else None,
                    last_available_hour_utc=available[-1].isoformat() if len(available) else None,
                    first_missing_hours_utc=[stamp.isoformat() for stamp in hours[~finite][:5]])
                frames[name] = selected
            except Exception as error:
                item.update(state="ERROR", error=safe_error(error))
            report["probes"][name] = item
        if all(name in frames for name in ("official_alternative_old_long", "canonical_current_long")):
            paired = pd.concat([frames["official_alternative_old_long"], frames["canonical_current_long"]], axis=1)
            paired = paired.loc[np.isfinite(paired.to_numpy(float)).all(axis=1)]
            difference = (paired.iloc[:, 0] - paired.iloc[:, 1]).abs()
            report["posthoc_comparison"] = {"paired_hours": len(paired), "minimum_hours": 168,
                "enough_paired_hours": len(paired) >= 168,
                "max_abs_difference_eur_mwh": float(difference.max()) if len(paired) else None,
                "old_publication_proven": False, "source_admitted": False,
                "qualification_valid": False,
                "limit": "Comparaison après coup avec une révision récente : ne prouve ni disponibilité ni valeurs publiées à l'ancienne coupure."}
        states = {name: item["state"] for name, item in report["probes"].items()}
        if states["canonical_old_long"] != "COMPLETE" and states["canonical_old_context2048"] == "COMPLETE":
            report["notes"].append("La plage courte est disponible à l'ancienne coupure : examiner la couverture ou la taille de la demande longue.")
        elif states["canonical_old_long"] != "COMPLETE" and states["canonical_current_long"] == "COMPLETE":
            report["notes"].append("La plage longue est disponible à la coupure actuelle, mais sa demande à l'ancienne coupure échoue ou reste incomplète.")
        if states["official_alternative_old_long"] == "COMPLETE":
            report["notes"].append("La source officielle alternative répond à l'ancienne coupure ; cela ne l'autorise pas automatiquement pour l'entraînement.")
        report["state"] = "DIAGNOSTIC_COMPLETE"
    except Exception as error:
        report.update(state="ERROR", error=safe_error(error))
    finally:
        if client is not None:
            try:
                client.session.close()
            except Exception as error:
                report["session_close_error"] = safe_error(error)
    report["notes"].append("Aucun cache de calcul, lot de données ou modèle modifié. Aucune prévision publiée.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", default="2026-09-30")
    parser.add_argument("--failed-day", default="2024-06-18")
    parser.add_argument("--zone", choices=tuple(SERIES), default="FR")
    args = parser.parse_args(argv)
    report = diagnose(args.delivery_day, args.failed_day, args.zone, progress=lambda text: print(text, flush=True))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "runs/logs/nyx_annual_cpu" / f"saturn_targets_{stamp}_{uuid4().hex[:8]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, result in report["probes"].items():
        print(f"{name} : {result['state']} ({result.get('finite_hours', 0)}/{result['required_hours']} heures finies)")
    for note in report["notes"]:
        print(note)
    print(f"Diagnostic à transmettre : {path}", flush=True)
    return 0 if report["state"] == "DIAGNOSTIC_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
