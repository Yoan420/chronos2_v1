"""Weekly, prequential prediction of individual initial-domain constraints.

Rows are never duplicated by country during fitting. Only verified matched
labels known at the fit origin are used; the full initial universe is predicted.
No constraint name, EIC, final-domain value or price label enters the features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .models import fit_activation_intensity, predict_activation_intensity

ZONES = ('FR', 'DE', 'BE', 'NL')
FUEL = 'feature_clean_gas_cost_ccgt_proxy_eur_mwh'
KEYS = ['timestamp_utc', 'forecast_origin_utc', 'constraint_key']
SIGNAL_NAMES = ['positive_pressure', 'negative_pressure', 'net_pressure',
                'largest_positive_pressure', 'top3_positive_pressure',
                'expected_active_count', 'maximum_activation_probability']
SIGNALS = ['feature_congestion_'+n for n in SIGNAL_NAMES]
MAX_CORE_ROWS = 200_000


def _utc(series, name, missing=False):
    if any(pd.Timestamp(v).tzinfo is None for v in series.dropna()):
        raise ValueError(name+': timezone required')
    result = pd.to_datetime(series, utc=True)
    if not missing and result.isna().any():
        raise ValueError(name+': complete timestamps required')
    return result


def feature_frame(panel, constraints):
    required = {*KEYS, 'constraint_identified', 'label_active', 'label_shadow_price', 'label_available_at_utc', 'label_eligible'}
    if constraints.empty or constraints.columns.has_duplicates or not required.issubset(constraints):
        raise ValueError('Nonempty initial-constraint panel with explicit label contract required.')
    data = constraints.copy(deep=True).reset_index(drop=True)
    if data.duplicated(KEYS).any():
        raise ValueError('Duplicate initial constraints.')
    for name in ('timestamp_utc', 'forecast_origin_utc', 'label_available_at_utc'):
        data[name] = _utc(data[name], name, missing=name.startswith('label'))
    civil = data.timestamp_utc.dt.tz_convert('Europe/Paris').dt.tz_localize(None).dt.normalize()
    origin = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize('Europe/Paris').dt.tz_convert('UTC')
    if not origin.eq(data.forecast_origin_utc).all() or not data.timestamp_utc.eq(data.timestamp_utc.dt.floor('h')).all():
        raise ValueError('Exact hourly origin D-1 08 h required.')
    data['_day'] = civil.dt.strftime('%Y-%m-%d')
    names = sorted(c for c in data if c.startswith('feature_cnec_'))
    if not names or any(any(t in c.lower() for t in ('final', 'shadow', 'label', 'actual', 'active', 'available', 'price', 'id')) for c in names):
        raise ValueError('Initial-domain feature allowlist required.')
    # Regional physical inputs have the same value in each zone row. Prove it
    # before collapsing, so a country-specific value cannot be silently chosen.
    physical = [f'feature_{z.lower()}_{field}' for z in ZONES for field in
                ('residual_load_gw', 'gas_available_gw', 'wind_generation_gw', 'solar_generation_gw', 'temperature_c')]
    physical += ['feature_fr_nuclear_generation_gw', 'feature_de_coal_available_gw',
                 'feature_de_lignite_available_gw', 'feature_be_nuclear_available_gw',
                 'feature_nl_coal_available_gw', FUEL, 'feature_hour_sin', 'feature_hour_cos', 'feature_weekday']
    physical = [c for c in physical if c in panel]
    grouped = panel.groupby('timestamp_utc', sort=True)
    if (grouped[physical].nunique(dropna=False) > 1).any().any():
        raise ValueError('Regional input differs across countries for the same hour.')
    common = panel.drop_duplicates('timestamp_utc').set_index('timestamp_utc')
    if not data.timestamp_utc.isin(common.index).all():
        raise ValueError('Constraint hour outside frozen forecast panel.')
    if (grouped.forecast_origin_utc.nunique(dropna=False).gt(1).any()
            or not data.timestamp_utc.map(common.forecast_origin_utc).eq(data.forecast_origin_utc).all()):
        raise ValueError('Constraint and fundamental forecast origins differ.')
    for name in physical:
        data[name] = data.timestamp_utc.map(common[name]).to_numpy()
    # Forecast ramps are differences inside the same forecast-origin/day only.
    for z in ZONES:
        name = f'feature_{z.lower()}_residual_load_gw'
        if name in common:
            ordered = common.sort_index()
            for lag in (1, 3):
                past = ordered[name].shift(lag)
                valid = ordered.forecast_origin_utc.eq(ordered.forecast_origin_utc.shift(lag))
                valid &= ordered.index.to_series().sub(ordered.index.to_series().shift(lag)).eq(pd.Timedelta(hours=lag))
                ramp = ((ordered[name]-past).where(valid))/lag
                feature = f'feature_regional_{z.lower()}_residual_ramp_{lag}h'
                data[feature] = data.timestamp_utc.map(ramp).to_numpy()
                physical.append(feature)
    names += physical
    values = data[names].astype(np.float32)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Infinite features forbidden.')
    # Target labels may be NaN at prediction time. Label eligibility must never
    # change whether a row can be predicted.
    baseline_ready = grouped.forecast_eligible.all() & grouped.feature_eligible.all()
    if 'feature_available_at_utc' in panel:
        available = _utc(panel.feature_available_at_utc, 'feature_available_at_utc', missing=True)
        available_ok = available.notna() & available.le(panel.forecast_origin_utc)
        baseline_ready &= available_ok.groupby(panel.timestamp_utc).all()
    data['_ready'] = data.timestamp_utc.map(baseline_ready).fillna(False).astype(bool)
    data['_ready'] &= np.isfinite(data[FUEL]) & data[FUEL].gt(0)
    if not data.constraint_identified.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError('Explicit physical-constraint identity mask required.')
    data['_ready'] &= data.constraint_identified
    if not data.label_eligible.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError('Boolean label mask required.')
    known = data.label_eligible
    if (known & (data.label_available_at_utc.isna() | data.label_available_at_utc.le(data.forecast_origin_utc))).any():
        raise ValueError('Congestion labels must be explicitly published after their own 08 h origin.')
    y = pd.to_numeric(data.label_active, errors='raise')
    intensity = pd.to_numeric(data.label_shadow_price, errors='raise')
    if (not y[known].isin([0, 1]).all() or not np.isfinite(intensity[known]).all()
            or intensity[known].lt(0).any() or ((y[known] == 0) & intensity[known].ne(0)).any()
            or ((y[known] == 1) & intensity[known].le(0)).any()):
        raise ValueError('Invalid qualified congestion labels.')
    return data, values.to_numpy(), names


def run_activation(panel, constraints, *, threads=2, load_fit=None, save_fit=None, progress=None):
    data, X, names = feature_frame(panel, constraints)
    output = constraints.copy(deep=True).reset_index(drop=True)
    for name in ('activation_probability', 'intensity_if_active', 'expected_shadow_price', 'climatology_probability'):
        output[name] = np.nan
    output['expert_ready'] = False
    output['fit_day'] = None
    records = []
    days = sorted(panel.timestamp_utc.dt.tz_convert('Europe/Paris').dt.strftime('%Y-%m-%d').unique())
    state, fitted = None, None
    for day in days:
        current = data.loc[data._day.eq(day)]
        cutoff = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize('Europe/Paris').tz_convert('UTC')
        if fitted is None or (pd.Timestamp(day)-pd.Timestamp(fitted)).days >= 7:
            fitted = day
            first = (pd.Timestamp(day)-pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            split = (pd.Timestamp(day)-pd.Timedelta(days=28)).strftime('%Y-%m-%d')
            usable = data._day.ge(first) & data._day.lt(day) & data._ready & data.label_eligible & data.label_available_at_utc.le(cutoff)
            core = data.loc[usable & data._day.lt(split)]
            cal = data.loc[usable & data._day.ge(split)]
            record = dict(fit_day=day, fit_cutoff_utc=cutoff, status='fallback', reason='insufficient_chronological_labels',
                          core_rows=len(core), calibration_rows=len(cal), core_days=core._day.nunique(),
                          calibration_days=cal._day.nunique(), training_window_cap_days=365,
                          max_label_available_at_utc=data.loc[usable, 'label_available_at_utc'].max())
            sufficient = (core._day.nunique() >= 28 and cal._day.nunique() >= 14
                          and core.loc[core.label_active.eq(1), '_day'].nunique() >= 7
                          and cal.loc[cal.label_active.eq(1), '_day'].nunique() >= 3
                          and min(core.label_active.eq(1).sum(), core.label_active.eq(0).sum()) >= 40
                          and min(cal.label_active.eq(1).sum(), cal.label_active.eq(0).sum()) >= 10)
            metadata = dict(fit_day=day, fit_cutoff=cutoff, features=names, source_core_rows=len(core), calibration_rows=len(cal))
            state = load_fit(day) if load_fit and sufficient else None
            if state is not None:
                if any(state.get(k) != v for k, v in metadata.items()):
                    raise ValueError('Activation checkpoint metadata mismatch.')
                record = {**state['record'], 'cache_reused': True}
            elif sufficient:
                if progress:
                    progress(f'Activation {day}: {len(core)} CNEC-heures CORE, calibration {len(cal)}')
                # Uniform deterministic cap, never class oversampling/weighting.
                selected = np.sort(np.random.default_rng(1729).choice(core.index, min(MAX_CORE_ROWS, len(core)), replace=False))
                sampled = data.loc[selected]
                if (min(sampled.label_active.eq(1).sum(), sampled.label_active.eq(0).sum()) < 20
                        or sampled.loc[sampled.label_active.eq(1), '_day'].nunique() < 7):
                    record.update(reason='insufficient_uniform_sample_classes', fitted_core_rows=len(selected))
                    records.append(record)
                    continue
                model = fit_activation_intensity(X[selected], data.loc[selected, 'label_active'],
                    data.loc[selected, 'label_shadow_price'], data.loc[selected, FUEL],
                    X[cal.index], cal.label_active, threads=threads)
                record.update(status='trained', reason='', fitted_core_rows=len(selected),
                              positive_core_rows=int(data.loc[selected, 'label_active'].sum()),
                              model_training_end_day=str(core._day.max()), calibration_start_day=split)
                state = {**metadata, 'model': model, 'record': record}
                if save_fit:
                    save_fit(day, state)
            records.append(record)
        selected = current.loc[current._ready]
        if state is None or selected.empty:
            continue
        predicted = predict_activation_intensity(state['model'], X[selected.index], selected[FUEL])
        for name, values in predicted.items():
            output.loc[selected.index, name] = values
        output.loc[selected.index, 'expert_ready'] = True
        output.loc[selected.index, 'fit_day'] = state['fit_day']
    return output, pd.DataFrame(records), dict(feature_columns=names, stage1_inference_uses_labels=False,
        chronological_calibration_days=28, core_maximum_rows=MAX_CORE_ROWS, uniform_sampling_seed=1729,
        intensity='conditional mean of hourly shadow price given any active MTU; normalized by CGC',
        aggregation_is_not_a_price_forecast=True, prediction_rows=len(output), ready_rows=int(output.expert_ready.sum()))


def aggregate_signals(panel, predictions):
    result = pd.DataFrame(index=panel.index)
    for name in SIGNALS:
        result[name] = np.nan
    result['congestion_ready'] = False
    by_hour = {h: block for h, block in predictions.groupby('timestamp_utc', sort=False)}
    for hour, part in panel.groupby('timestamp_utc', sort=False):
        c = by_hour.get(hour)
        if c is not None:
            c = c.loc[c.constraint_identified.eq(True)]
        if c is None or c.empty or not c.expert_ready.all():
            continue
        for name in ('expected_shadow_price', 'activation_probability'):
            if not np.isfinite(c[name]).all():
                raise ValueError('Nonfinite eligible congestion predictions.')
        for zone, indexes in part.groupby('zone').groups.items():
            # Average peer reference is symmetric (FR is not silently zero).
            ptdfs = c[[f'feature_cnec_ptdf_{z}' for z in ZONES]].to_numpy(float)
            if not np.isfinite(ptdfs).all():
                raise ValueError('Complete initial country PTDFs required for directional aggregation.')
            delta = ptdfs.mean(axis=1)-ptdfs[:, ZONES.index(zone)]
            contribution = c.expected_shadow_price.to_numpy()*delta
            positive, negative = np.maximum(contribution, 0), np.maximum(-contribution, 0)
            values = [positive.sum(), negative.sum(), contribution.sum(), positive.max(),
                      np.sort(positive)[-3:].sum(), c.activation_probability.sum(), c.activation_probability.max()]
            result.loc[indexes, SIGNALS] = values
            result.loc[indexes, 'congestion_ready'] = True
    return result
