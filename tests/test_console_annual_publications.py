import hashlib
import json
import pytest
from test_console_api import api
from experiment_console.annual_publications import COUNTRIES, PROTOCOL


def publication(root):
    day = "2026-09-30"
    directory = root / "runs/nyx_annual_cpu_live" / day
    receipt = {"protocol": PROTOCOL, "status": "COMPLETE", "delivery_day": day,
               "future_labels_used": False, "Storm_used_as_model_input": False,
               "qualification_sha256": "a" * 64, "countries": {}}
    for zone in COUNTRIES:
        record = {"hours": 24}
        for kind, text in (("csv", "timestamp_utc,price_eur_mwh,p_negative\n"),
                           ("html", "<!doctype html><title>NYX annuel</title><p>Prix et probabilités</p>")):
            path = directory / "zones" / zone / f"forecast_{zone.lower()}_{day}_nyx_annual_cpu.{kind}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            record[kind + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt["countries"][zone] = record
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return directory


def test_scheduled_forecasts_visible_without_managed_run(api):
    directory = publication(api.project)
    rows = api.client.get('/api/annual-publications').json()
    assert rows["warnings"] == []
    assert [c["zone"] for c in rows["days"][0]["countries"]] == list(COUNTRIES)
    assert rows["days"][0]["delivery_day"] == directory.name
    assert api.manager.list_runs() == []
    response = api.client.get('/api/annual-artifact', params={"delivery_day":directory.name,"country":"DE","format":"html"})
    assert response.status_code == 200 and "Prix et probabilités" in response.text
    assert "sandbox" in response.headers["Content-Security-Policy"]


@pytest.mark.parametrize("failure", ["missing_country", "modified_csv", "unsealed"])
def test_incomplete_or_modified_publication_never_offered(api, failure):
    directory = publication(api.project)
    if failure == "modified_csv":
        next((directory / "zones/DE").glob("*.csv")).write_text("changed")
    elif failure == "unsealed":
        (directory / "receipt.json").unlink()
    else:
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        del receipt["countries"]["BE"]
        receipt_path.write_text(json.dumps(receipt))
    assert api.client.get('/api/annual-publications').json()["days"] == []
    response = api.client.get('/api/annual-artifact', params={"delivery_day":directory.name,"country":"DE","format":"csv"})
    assert response.status_code != 200


@pytest.mark.parametrize("day,zone,kind", [("../secrets", "DE", "csv"), ("2026-09-30", "../FR", "html"), ("2026-09-30", "FR", "py")])
def test_publication_route_rejects_other_files(api, day, zone, kind):
    publication(api.project)
    result = api.client.get('/api/annual-artifact', params={"delivery_day":day,"country":zone,"format":kind})
    assert result.status_code == 404
