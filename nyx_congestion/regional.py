"""Prequential global FB-pressure expert, complementing the initial CNEC set.

G_z=D_z-min(D_FR,D_DE,D_BE,D_NL), D_z=sum_all_published_shadow(lambda*deltaPTDF).
This gauge-invariant LABEL is derived after coupling, never an input. It covers
published FB effects outside the matchable initial universe, not all market
coupling mechanisms. Only the forecasts of G enter the residual model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from nyx_scarcity import policy as base
from nyx_fundamental_stress.features import make_fundamental_features
from nyx_physical_p50.policy import _network_contract, FUEL
from .activation import ZONES
from .models import fit_activation_intensity, predict_activation_intensity

THRESHOLD = 50.
REGIONAL_SIGNALS = ['feature_congestion_regional_probability', 'feature_congestion_regional_intensity',
                    'feature_congestion_regional_expected_pressure']


def regional_targets(panel, labels):
    keys = ['zone', 'timestamp_utc']
    need = {*keys, 'label_directional_contribution_eur_mwh', 'label_eligible', 'label_available_at_utc'}
    if not need.issubset(labels) or labels.duplicated(keys).any():
        raise ValueError('Unique explicit zonal FB labels required.')
    labels = labels.copy()
    if not labels.label_eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError('Explicit boolean regional label eligibility required.')
    for name in ['timestamp_utc', 'label_available_at_utc']:
        if any(pd.Timestamp(v).tzinfo is None for v in labels[name].dropna()):
            raise ValueError('Timezone-aware zonal labels required.')
        labels[name] = pd.to_datetime(labels[name], utc=True)
    if labels.timestamp_utc.isna().any():
        raise ValueError('Missing zonal label timestamp.')
    if not set(labels.zone).issubset(ZONES):
        raise ValueError('Unexpected label zones.')
    values = labels.pivot(index='timestamp_utc', columns='zone', values='label_directional_contribution_eur_mwh').reindex(columns=ZONES)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Infinite zonal contribution.')
    masks = labels.pivot(index='timestamp_utc', columns='zone', values='label_eligible').reindex(columns=ZONES)
    times = labels.pivot(index='timestamp_utc', columns='zone', values='label_available_at_utc').reindex(columns=ZONES)
    known = masks.eq(True).all(axis=1) & values.notna().all(axis=1) & times.notna().all(axis=1)
    premium = values.sub(values.min(axis=1), axis=0).where(known, axis=0)
    # If even one country is unqualified, never choose a min over the others.
    available = times.max(axis=1).where(known)
    lookup = premium.stack(future_stack=True)
    index = pd.MultiIndex.from_arrays([panel.timestamp_utc, panel.zone])
    output = panel[['zone', 'timestamp_utc', 'forecast_origin_utc']].copy()
    output['realised_fb_premium'] = lookup.reindex(index).to_numpy(float)
    output['label_eligible'] = output.realised_fb_premium.notna()
    output['label_active'] = (output.realised_fb_premium >= THRESHOLD).astype(float).where(output.label_eligible)
    output['label_shadow_price'] = output.realised_fb_premium.where(output.label_active.eq(1), 0.).where(output.label_eligible)
    output['label_available_at_utc'] = panel.timestamp_utc.map(available)
    if (output.label_eligible & output.label_available_at_utc.le(output.forecast_origin_utc)).any():
        raise ValueError('Global congestion labels cannot be known at their own forecast origin.')
    return output


def run_regional(panel, network, labels, settings, *, threads=2, load_fit=None, save_fit=None, progress=None):
    augmented, physical, required, _ = make_fundamental_features(panel, variant='fundamental')
    network_names = _network_contract(panel, network)
    augmented = pd.concat([augmented, network], axis=1)
    names = physical+network_names
    params = base._parameters({**settings, 'threads':threads, 'feature_columns':names,
                               'required_feature_columns':required})
    data = base._prepare(augmented, params)
    output = regional_targets(panel, labels)
    original_index = output.index
    output = output.reset_index(drop=True)
    targets = output.set_index(['zone', 'timestamp_utc'])
    for name in ['label_active', 'label_shadow_price', 'label_eligible', 'label_available_at_utc']:
        data['_regional_'+name] = targets[name].reindex(pd.MultiIndex.from_frame(data[['zone','timestamp_utc']])).to_numpy()
    data['_ready'] = data._features_valid & data.network_eligible & np.isfinite(data[FUEL]) & data[FUEL].gt(0)
    for name in ['activation_probability', 'intensity_if_active', 'expected_shadow_price', 'climatology_probability']:
        output[name] = np.nan
    output['expert_ready'] = False
    output['fit_day'] = None
    zones = tuple(sorted(data.zone.unique()))
    records, fitted, state = [], None, None
    for day, current in data.groupby('_day', sort=True):
        cutoff = current.forecast_origin_utc.iloc[0]
        if fitted is None or (pd.Timestamp(day)-pd.Timestamp(fitted)).days >= 7:
            fitted = day
            first = (pd.Timestamp(day)-pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            split = (pd.Timestamp(day)-pd.Timedelta(days=28)).strftime('%Y-%m-%d')
            usable = data._day.ge(first) & data._day.lt(day) & data._ready & data._regional_label_eligible & data._regional_label_available_at_utc.le(cutoff)
            core, cal = data.loc[usable & data._day.lt(split)], data.loc[usable & data._day.ge(split)]
            record = dict(fit_day=day, status='fallback', reason='insufficient_regional_labels', fit_cutoff_utc=cutoff,
                core_rows=len(core), calibration_rows=len(cal), core_days=core._day.nunique(), calibration_days=cal._day.nunique(),
                max_label_available_at_utc=data.loc[usable, '_regional_label_available_at_utc'].max())
            sufficient = (core._day.nunique() >= 28 and cal._day.nunique() >= 14
                and min(core._regional_label_active.eq(1).sum(), core._regional_label_active.eq(0).sum()) >= 40
                and min(cal._regional_label_active.eq(1).sum(), cal._regional_label_active.eq(0).sum()) >= 10
                and core.loc[core._regional_label_active.eq(1), '_day'].nunique() >= 7
                and cal.loc[cal._regional_label_active.eq(1), '_day'].nunique() >= 3)
            metadata = dict(fit_day=day, fit_cutoff=cutoff, features=names, core_rows=len(core), calibration_rows=len(cal), threshold=THRESHOLD)
            state = load_fit(day) if load_fit and sufficient else None
            if state is not None:
                if any(state.get(k) != v for k,v in metadata.items()):
                    raise ValueError('Regional checkpoint mismatch.')
                record = {**state['record'], 'cache_reused':True}
            elif sufficient:
                if progress:
                    progress(f'Pression FB globale {day}: CORE={len(core)}, calibration={len(cal)}')
                model = fit_activation_intensity(base._matrix(core, names, zones), core._regional_label_active,
                    core._regional_label_shadow_price, core[FUEL], base._matrix(cal, names, zones), cal._regional_label_active,
                    threads=threads)
                record.update(status='trained', reason='', model_training_end_day=str(core._day.max()), calibration_start_day=split)
                state = {**metadata, 'model':model, 'record':record}
                if save_fit: save_fit(day, state)
            records.append(record)
        selected = current.loc[current._ready]
        if state is None or selected.empty: continue
        predicted = predict_activation_intensity(state['model'], base._matrix(selected, names, zones), selected[FUEL])
        # Conditional G>=50 has known support, even when the current CGC falls.
        predicted['intensity_if_active'] = np.maximum(THRESHOLD, predicted['intensity_if_active'])
        predicted['expected_shadow_price'] = predicted['activation_probability']*predicted['intensity_if_active']
        indexes = selected._row.to_numpy(int)
        for name, value in predicted.items(): output.loc[indexes, name] = value
        output.loc[indexes, 'expert_ready'], output.loc[indexes, 'fit_day'] = True, state['fit_day']
    signals = pd.DataFrame(index=panel.index)
    for name, column in zip(REGIONAL_SIGNALS, ['activation_probability', 'intensity_if_active', 'expected_shadow_price']):
        signals[name] = output[column].to_numpy(float)
    signals['regional_ready'] = output.expert_ready.to_numpy(bool)
    output.index = original_index
    audit = dict(target='G_z=D_z-min_four_CWE(D); D from all published FB shadows, not total market price',
                 activation_threshold_eur_mwh=THRESHOLD, training_window_cap_days=365, calibration_days=28,
                 label_mask_requires_all_four_zones=True, current_post_coupling_features=False,
                 expected_signal='p(G>=50)*E[G|G>=50]', features=names,
                 conditional_intensity_support_floor=THRESHOLD,
                 not_added_to_cnec_proxy=True, production_modified=False)
    return output, pd.DataFrame(records), signals, audit
