import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_cpu_reporting import render_country_report, write_country_report
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid


@pytest.mark.parametrize("day,hours", [("2026-09-30",24), ("2026-03-29",23), ("2026-10-25",25)])
def test_report_preserves_all_physical_hours_and_has_no_remote_resources(day, hours):
    index = delivery_grid(day)[1]
    frame = pd.DataFrame({"price_eur_mwh": np.linspace(-10,100,hours),
                          "p_negative": np.linspace(1,0,hours)}, index=index)
    page = render_country_report(frame, country="DE", delivery_day=day, de_performance_exception=True)
    assert page.count("<tr>") == hours+1
    assert "DE : exception de performance" in page
    assert "<script" not in page and "https://" not in page and "http://" not in page
    assert page.count("<svg ") == 2
    for stamp in index:
        assert stamp.strftime("%Y-%m-%d %H:%M")+"Z" in page
    if hours == 25:
        assert "25/10/2026 02:00 <span class=\"offset\">UTC+02:00" in page
        assert "25/10/2026 02:00 <span class=\"offset\">UTC+01:00" in page


def test_report_refuses_partial_grid_invalid_probabilities_and_overwriting(tmp_path):
    day = "2026-09-30"
    index = delivery_grid(day)[1]
    frame = pd.DataFrame({"price_eur_mwh": np.zeros(len(index)), "p_negative": np.full(len(index), .5)}, index=index)
    with pytest.raises(ValueError, match="exact physical"):
        render_country_report(frame.iloc[:-1], country="FR", delivery_day=day)
    bad = frame.copy()
    bad.iloc[0,1] = 1.1
    with pytest.raises(ValueError, match="Finite prices"):
        render_country_report(bad, country="FR", delivery_day=day)
    destination = tmp_path/"report.html"
    write_country_report(frame, destination, country="FR", delivery_day=day)
    assert "exception de performance" not in destination.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_country_report(frame, destination, country="FR", delivery_day=day)
