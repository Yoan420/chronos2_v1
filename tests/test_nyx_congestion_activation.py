import numpy as np
import pandas as pd
import pytest
from nyx_congestion import activation


def fixture_panel(days=70):
    hours = pd.date_range('2025-09-15', periods=24*days, freq='h', tz='UTC')
    civil = hours.tz_convert('Europe/Paris').tz_localize(None).normalize()
    origins = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize('Europe/Paris').tz_convert('UTC')
    panel = pd.DataFrame([dict(timestamp_utc=h, forecast_origin_utc=o, zone=z, forecast_eligible=True,
        feature_eligible=True, feature_clean_gas_cost_ccgt_proxy_eur_mwh=100.,
        feature_fr_residual_load_gw=float(i%24), feature_hour_sin=float(np.sin(i)))
        for i, (h, o) in enumerate(zip(hours, origins)) for z in activation.ZONES])
    constraints = pd.DataFrame([dict(timestamp_utc=h, forecast_origin_utc=o, constraint_key=str(c),
        constraint_identified=True, feature_cnec_initial_ram_mw=float(100+c),
        feature_cnec_ptdf_FR=.2, feature_cnec_ptdf_DE=-.2, feature_cnec_ptdf_BE=-.1, feature_cnec_ptdf_NL=.1,
        label_active=bool((i+c)%5 == 0), label_shadow_price=20. if (i+c)%5 == 0 else 0.,
        label_eligible=True, label_available_at_utc=o+pd.Timedelta(hours=6))
        for i, (h,o) in enumerate(zip(hours, origins)) for c in range(2)])
    return panel, constraints


def test_features_never_include_labels_names_or_post_coupling():
    panel, c = fixture_panel(2)
    c['cne_name'] = 'revealing future constraint name'
    c['final_ram'] = 1000.
    data, x, names = activation.feature_frame(panel, c)
    assert not any(k in names for k in ('label_active', 'label_shadow_price', 'final_ram', 'cne_name'))
    assert x.dtype == np.float32
    assert data._ready.all()


def test_current_missing_labels_do_not_remove_forecast_support():
    panel, c = fixture_panel(2)
    c['label_eligible'] = False
    c['label_active'] = np.nan
    c['label_shadow_price'] = np.nan
    c['label_available_at_utc'] = pd.NaT
    data, _, _ = activation.feature_frame(panel, c)
    assert data._ready.all()


def test_late_fundamental_inputs_abstain_even_when_flags_true():
    p, c = fixture_panel(2)
    p['feature_available_at_utc'] = p.forecast_origin_utc+pd.Timedelta(seconds=1)
    data, _, _ = activation.feature_frame(p, c)
    assert not data._ready.any()


def test_producer_feature_names_match_aggregator():
    from nyx_congestion.data import FEATURE_COLUMNS
    assert {f'feature_cnec_ptdf_{z}' for z in activation.ZONES}.issubset(FEATURE_COLUMNS)


@pytest.mark.parametrize('mutation', ['duplicate', 'origin', 'future_feature', 'regional_mismatch', 'invalid_label'])
def test_contract_errors(mutation):
    p, c = fixture_panel(2)
    if mutation == 'duplicate': c = pd.concat([c, c.iloc[:1]])
    elif mutation == 'origin': c.loc[0, 'forecast_origin_utc'] += pd.Timedelta(hours=1)
    elif mutation == 'future_feature': c['feature_cnec_final_ram'] = 10.
    elif mutation == 'regional_mismatch': p.loc[0, 'feature_fr_residual_load_gw'] += 1.
    elif mutation == 'invalid_label': c.loc[0, 'label_available_at_utc'] = c.loc[0, 'forecast_origin_utc']
    with pytest.raises(ValueError): activation.feature_frame(p, c)


def test_prequential_future_label_invariance(monkeypatch):
    p, c = fixture_panel()
    def fit(x, active, shadow, fuel, v, cy, **kwargs):
        return {'mean':float(np.mean(shadow)), 'n':len(x)}
    def predict(s, x, fuel):
        return dict(activation_probability=np.full(len(x), .2), intensity_if_active=np.full(len(x), s['mean']),
                    expected_shadow_price=np.full(len(x), .2*s['mean']), climatology_probability=np.full(len(x), .2))
    monkeypatch.setattr(activation, 'fit_activation_intensity', fit)
    monkeypatch.setattr(activation, 'predict_activation_intensity', predict)
    a, fa, _ = activation.run_activation(p, c, threads=1)
    boundary = pd.Timestamp('2025-11-18', tz='Europe/Paris').tz_convert('UTC')
    changed = c.copy()
    changed.loc[changed.timestamp_utc.ge(boundary) & changed.label_active, 'label_shadow_price'] = 999.
    b, _, _ = activation.run_activation(p, changed, threads=1)
    columns = ['activation_probability', 'intensity_if_active', 'expert_ready', 'fit_day']
    pd.testing.assert_frame_equal(a.loc[a.timestamp_utc.lt(boundary), columns], b.loc[b.timestamp_utc.lt(boundary), columns])
    assert a.expert_ready.any()
    fitted = fa.loc[fa.status.eq('trained')]
    assert fitted.max_label_available_at_utc.le(fitted.fit_cutoff_utc).all()


def test_symmetric_aggregation_and_unqualified_hour_abstention():
    p, c = fixture_panel(1)
    c['expert_ready'] = True
    c['expected_shadow_price'] = 100.
    c['activation_probability'] = .25
    first = c.timestamp_utc.iloc[0]
    c.loc[c.timestamp_utc.eq(first), 'expert_ready'] = False
    signals = activation.aggregate_signals(p, c)
    assert signals.loc[p.timestamp_utc.eq(first), activation.SIGNALS].isna().all().all()
    common = pd.concat([p[['zone', 'timestamp_utc']], signals], axis=1)
    ready = common.loc[common.congestion_ready]
    assert ready.loc[ready.zone.eq('DE'), 'feature_congestion_net_pressure'].gt(0).all()
    assert ready.loc[ready.zone.eq('FR'), 'feature_congestion_net_pressure'].lt(0).all()
    np.testing.assert_allclose(ready.groupby('timestamp_utc').feature_congestion_net_pressure.sum(), 0., atol=1e-12)
