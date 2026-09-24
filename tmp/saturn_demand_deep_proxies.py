"""Bounded read-only Saturn proxy inventory. Never writes series or production files."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://saturn-energyscan.gem.myengie.com//api"


def catalogue():
    doc = json.loads((ROOT / "tmp/saturn_demand_deep_catalog.json").read_text(encoding="utf8"))
    return [dict(source=source, name=row[0], kind=row[1])
            for source, rows in doc.items() for row in rows]


def inspect_names(entries):
    categories = {
        "explicit_demand_response": r"effac|nebef|nebco|interrupt|load[._ ]?shed|demand[._ ]?(?:side|response)|(?:^|[._])(?:dsm|dsr)(?:[._]|$)|elastic|price[._]?sensitive|curtail.*(?:demand|load)",
        "industrial_power": r"^power\.(?=.*(?:industr|steel|alum|chemical|chlor|refin|cement|paper|fertili))",
        "industrial_gas": r"^(?:gas|gaz).*industr",
        "demand_destruction": r"demand[._]?destruc",
        "pump": r"pump|pompage|(?:^|[._])psh(?:[._]|$)",
        "hydrogen": r"electrolys|electrolyz|hydrogen|hydrogene",
        "storage_flex": r"^power\.(?=.*(?:battery|batteries|bess|flexib))",
        "balancing_price_quantity": r"^power\.(?=.*(?:balanc|afrr|mfrr|fcr|rrm|reserve))(?=.*(?:price|bid|offer|demand|volume|quantity|cost))",
        "marginal_storm": r"power\.(?:type|fuel)_marginal_costs.*\.(?:fr|de|be|nl)\.",
        "clean_gas_cost": r"(?:^|[._])cgc(?:[._]|$)|clean[._]?gas",
        "hourly_demand_variants": r"^power\.(?:fr|de|be|nl)\.(?:demand|load)\.",
    }
    for category, pattern in categories.items():
        compiled = re.compile(pattern, re.I)
        found = [r for r in entries if compiled.search(r["name"])]
        # Four CWE zones first, then explicit demand concepts, then other areas.
        selected = sorted(found, key=lambda r: (not bool(re.search(r"[._](?:fr|de|be|nl)[._]", r["name"])), r["name"]))
        print(json.dumps(dict(category=category, count=len(found), power_count=sum(x['name'].startswith('power.') for x in found), examples=selected[:25]), ensure_ascii=True), flush=True)
        if category == "explicit_demand_response":
            print("EXPLICIT_POWER", json.dumps([x for x in found if x['name'].startswith('power.')]), flush=True)
        if category == "pump":
            print("PUMP_FORECAST", json.dumps([x for x in found if 'fcst' in x['name'] or 'forec' in x['name']]), flush=True)


def probe(name):
    out = dict(name=name)
    with requests.Session() as session:
        for kind, endpoint, params in (
            ("metadata", "metadata", {"all": 1}),
            ("interval", "metadata", {"type": "interval"}),
            ("revisions", "insertion_dates", {
                "from_insertion_date": "2025-09-14T00:00:00Z",
                "to_insertion_date": "2026-09-16T00:00:00Z",
                "from_value_date": "2026-09-14T00:00:00Z",
                "to_value_date": "2026-09-15T00:00:00Z",
            }),
        ):
            try:
                response = session.get(BASE + "/series/" + endpoint,
                                       params={"name": name, **params}, timeout=(10, 40))
                item = {"status": response.status_code, "bytes": len(response.content)}
                if response.status_code == 200:
                    value = response.json()
                    if kind == "revisions":
                        dates = value.get("insertion_dates", []) if isinstance(value, dict) else value
                        item.update(count=len(dates), first=dates[:3], last=dates[-5:])
                    elif isinstance(value, dict):
                        safe = {k: v for k, v in value.items()
                                if not re.search(r"token|secret|password|credential|api.?key|authorization", k, re.I)}
                        item["data"] = safe
                    else:
                        item["data"] = value
                out[kind] = item
            except Exception as exc:
                out[kind] = {"error_type": type(exc).__name__, "error": str(exc)[:200]}
    return out


def samples(name, begin="2026-09-13T22:00:00Z", end="2026-09-14T21:59:59Z"):
    import pandas as pd
    from tshistory_lite.client import unpack_series
    out = {"name": name}
    with requests.Session() as session:
        for label, cutoff in (("latest", None), ("asof_20260913_0600Z", "2026-09-13T06:00:00Z")):
            params = {"name": name, "format": "tshpack", "nocache": False,
                      "from_value_date": begin,
                      "to_value_date": end}
            if cutoff:
                params["insertion_date"] = cutoff
            try:
                response = session.get(BASE + "/series/state", params=params, timeout=(10, 40))
                row = {"status": response.status_code, "bytes": len(response.content)}
                if response.status_code == 200:
                    series = unpack_series(name, response.content).dropna()
                    row.update(count=len(series), first=str(series.index.min()), last=str(series.index.max()),
                               minimum=float(series.min()) if len(series) else None,
                               maximum=float(series.max()) if len(series) else None,
                               head={str(i): float(v) for i, v in series.head(3).items()},
                               evening={str(i): float(v) for i, v in series.items() if i.hour in [17, 18, 19]})
                out[label] = row
            except Exception as exc:
                out[label] = {"error_type": type(exc).__name__, "error": str(exc)[:200]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--formula", action="store_true")
    parser.add_argument("--begin", default="2026-09-13T22:00:00Z")
    parser.add_argument("--end", default="2026-09-14T21:59:59Z")
    args = parser.parse_args()
    entries = catalogue()
    if not args.name:
        inspect_names(entries)
    else:
        available = {row["name"] for row in entries}
        if any(n not in available for n in args.name):
            raise ValueError("Only names actually observed in the shared catalogue may be probed")
        if len(args.name) > 12:
            raise ValueError("Maximum 12 bounded metadata probes")
        with ThreadPoolExecutor(max_workers=3) as pool:
            operation = formula if args.formula else (lambda n: samples(n, args.begin, args.end)) if args.samples else probe
            for result in pool.map(operation, args.name):
                print(json.dumps(result, ensure_ascii=True), flush=True)


def formula(name):
    response = requests.get(BASE + "/series/formula", params={"name": name}, timeout=(10, 40))
    return {"name": name, "status": response.status_code, "formula": response.text[:25000]}


if __name__ == "__main__":
    main()
