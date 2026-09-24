"""Read-only, bounded clean fuel Saturn evidence; no production cache mutation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd
import requests
from tshistory_lite.client import unpack_series

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://saturn-energyscan.gem.myengie.com//api"
NAMES = [f"power.{zone}.price.everyday.cgc.da.index.eurmwh" for zone in ("fr", "de", "be", "nl", "it", "es")] + [
    "power.eu.price.everyday.ccc.da.index.eurmwh",
    "ccc.price.mid.api2.everyday.month.1.ice.eurmwh",
]
DAYS = ["2024-09-17", "2025-09-17", "2026-09-13", "2026-09-16"]


def probe(name):
    out = {"name": name, "asof": {}}
    with requests.Session() as session:
        for kind, endpoint, extra in (("metadata", "metadata", {"all": 1}),
                                       ("interval", "metadata", {"type": "interval"}),
                                       ("formula", "formula", {}),
                                       ("components", "formula_components", {})):
            try:
                response = session.get(BASE + "/series/" + endpoint, params={"name": name, **extra}, timeout=(10, 45))
                row = {"status": response.status_code}
                if response.status_code == 200:
                    value = response.json()
                    if isinstance(value, dict):
                        value = {k: v for k, v in value.items() if not re.search(r"token|password|secret|api.?key|credential", k, re.I)}
                    row["value"] = value
                else:
                    row["body"] = response.text[:300]
                out[kind] = row
            except Exception as exc:
                out[kind] = {"error": type(exc).__name__, "detail": str(exc)[:180]}
        for day in DAYS:
            cutoff = pd.Timestamp(day).tz_localize("Europe/Paris") + pd.Timedelta(hours=8)
            params = {"name": name, "format": "tshpack", "nocache": False,
                      "from_value_date": (cutoff - pd.Timedelta(days=10)).isoformat(),
                      "to_value_date": (cutoff + pd.Timedelta(days=1)).isoformat(),
                      "insertion_date": cutoff.isoformat()}
            try:
                response = session.get(BASE + "/series/state", params=params, timeout=(10, 45))
                row = {"status": response.status_code, "cutoff": cutoff.isoformat()}
                if response.status_code == 200:
                    series = unpack_series(name, response.content).dropna()
                    row.update(count=len(series), tail={str(i): float(v) for i, v in series.tail(4).items()})
                    index = pd.DatetimeIndex(series.index)
                    if index.tz is None:
                        index = index.tz_localize("Europe/Paris")
                    prior = series.loc[index < cutoff.normalize()]
                    row["previous_civil_day_last"] = None if prior.empty else {"time": str(prior.index[-1]), "value": float(prior.iloc[-1])}
                else:
                    row["body"] = response.text[:300]
                out["asof"][day] = row
            except Exception as exc:
                out["asof"][day] = {"error": type(exc).__name__, "detail": str(exc)[:180]}
    print(json.dumps({"name": name, "formula": out.get("formula"), "asof_status": {k: v.get("count", v.get("status")) for k, v in out["asof"].items()}}, ensure_ascii=True), flush=True)
    return out


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(probe, NAMES))
    output = ROOT / "tmp/clean_fuel_costs_probe_20260917.json"
    output.write_text(json.dumps({"queried_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True, "historical_formula_version_attested": False, "sources": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(str(output), flush=True)
