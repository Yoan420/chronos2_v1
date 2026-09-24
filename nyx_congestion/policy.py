"""New residual probabilities conditioned on out-of-time congestion signals.

The network-only control has the exact same admissible training/calibration
rows and estimator classes. Congestion forecasts, not realised shadow labels,
enter the enriched detector and conditional residual CDF. No old zone rule or
day/hour-specific boost is retained. All interventions remain upward-only.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from nyx_scarcity import policy as base
from nyx_fundamental_stress.features import make_fundamental_features
from nyx_fundamental_stress.policy import _check_features
from nyx_physical_p50.policy import _network_contract, _bands, FUEL
from nyx_physical_p50.models import fit_physical_cdf, predict_physical_cdf
from nyx_coherent_p50.policy import _finalize
from .models import fit_probability, predict_probability
from .activation import SIGNALS as CNEC_SIGNALS
from .regional import REGIONAL_SIGNALS

SIGNALS = CNEC_SIGNALS + REGIONAL_SIGNALS

KEYS = ['zone', 'timestamp_utc', 'forecast_origin_utc']
STRATEGIES = {'control': ('congestion_control_direct', 'congestion_control_governed'),
              'congestion': ('congestion_direct', 'nyx_congestion')}
MODELS = tuple(v for pair in STRATEGIES.values() for v in pair)


def _fit(data, day, cutoff, p, zones, common_features, load_fit, save_fit, progress):
    first = (pd.Timestamp(day)-pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    split = (pd.Timestamp(day)-pd.Timedelta(days=28)).strftime('%Y-%m-%d')
    past = data.loc[data._day.ge(first) & data._day.lt(day) & data._features_valid & data._label_valid
                    & data.label_available_at_utc.le(cutoff) & np.isfinite(data[FUEL]) & data[FUEL].gt(0)]
    core, cal = past.loc[past._day.lt(split)], past.loc[past._day.ge(split)]
    span = min(365, (pd.Timestamp(day)-pd.Timestamp(data._day.min())).days)
    record = dict(fit_day=day, fit_cutoff_utc=cutoff, status='fallback', reason='insufficient_prequential_history',
        training_days=span, full_365_day_training=span == 365, training_rows=len(past),
        eligible_training_days_by_zone={z: past.loc[past.zone.eq(z), '_day'].nunique() for z in zones},
        eligible_training_hours_by_zone={z: int(past.zone.eq(z).sum()) for z in zones},
        core_rows=len(core), calibration_rows=len(cal),
        max_label_available_at_utc=past.label_available_at_utc.max(),
        model_training_end_day=str(core._day.max()) if len(core) else None,
        calibration_start_day=split, training_start_day=first)
    if (min(record['eligible_training_days_by_zone'].values()) < 90 or core.empty or cal.empty
            or cal._day.nunique() < 14 or not set(core.zone) == set(cal.zone) == set(zones)):
        return None, record
    thresholds = {z: max(p['minimum_threshold_eur_mwh'], float(np.quantile(core.loc[core.zone.eq(z), '_error'],
                                                                       p['threshold_quantile']))) for z in zones}
    u, cu = core.zone.map(thresholds).to_numpy(float), cal.zone.map(thresholds).to_numpy(float)
    y, cy = core._error.to_numpy() >= u, cal._error.to_numpy() >= cu
    if (min(y.sum(), (~y).sum()) < 30 or min(cy.sum(), (~cy).sum()) < 5
            or core.loc[y, '_day'].nunique() < 5 or cal.loc[cy, '_day'].nunique() < 3):
        record['reason'] = 'insufficient_distinct_calibration_events'
        return None, record
    metadata = dict(fit_day=day, fit_cutoff=cutoff, thresholds=thresholds, zones=tuple(zones),
                    common_features=list(common_features), core_rows=len(core), calibration_rows=len(cal))
    state = load_fit(day) if load_fit else None
    if state is not None:
        if any(state.get(k) != v for k, v in metadata.items()):
            raise ValueError('Residual checkpoint differs from chronological split.')
        return state, {**state['record'], 'cache_reused': True}
    models = {}
    for strategy in STRATEGIES:
        names = list(common_features)+(SIGNALS if strategy == 'congestion' else [])
        if progress:
            progress(f'P50 {day}/{strategy}: CORE={len(core)}, calibration={len(cal)}')
        X, V = base._matrix(core, names, zones), base._matrix(cal, names, zones)
        classifier = fit_probability(X, y, V, cy, threads=p['threads'])
        cdf = fit_physical_cdf(X, core._error.to_numpy(float), u, core[FUEL].to_numpy(float),
                              core.zone.to_numpy(), threads=p['threads'])
        models[strategy] = dict(features=names, classifier=classifier, cdf=cdf)
    priors = {z: float(y[core.zone.eq(z)].mean()) for z in zones}
    record.update(status='trained', reason='', thresholds_eur_mwh=thresholds, risk_prior_by_zone=priors,
                  training_tail_rows=int(y.sum()), calibration_tail_rows=int(cy.sum()),
                  control_and_congestion_same_training_rows=True)
    state = {**metadata, 'models': models, 'priors': priors, 'record': record}
    if save_fit:
        save_fit(day, state)
    return state, record


def _predict(state, current, strategy):
    if not current.forecast_origin_utc.ge(state['fit_cutoff']).all() or not current.congestion_ready.eq(True).all():
        raise ValueError('Residual inference requires causal fitted state and prequential signals.')
    m = state['models'][strategy]
    X = base._matrix(current, m['features'], state['zones'])
    probability = predict_probability(m['classifier'], X)
    threshold = current.zone.map(state['thresholds']).to_numpy(float)
    prior = current.zone.map(state['priors']).to_numpy(float)
    evaluated = probability > prior
    q = np.column_stack([current.q10-current.forecast, np.zeros(len(current)), current.q90-current.forecast])
    if evaluated.any():
        q[evaluated], _ = predict_physical_cdf(m['cdf'], X[evaluated], current.zone.to_numpy()[evaluated],
            probability[evaluated], threshold[evaluated], current[FUEL].to_numpy(float)[evaluated])
    raw = np.where(evaluated & (q[:, 1] > 0), q[:, 1], 0.)
    detail = current[KEYS].copy()
    for i, level in enumerate((10, 50, 90)):
        detail[f'mixture_error_q{level}'] = q[:, i]
    detail['risk_probability_gate'] = prior
    detail['cdf_evaluated'] = evaluated
    detail['proposal_reason'] = np.select([~evaluated, q[:, 1] <= 0],
        ['new_risk_not_above_core_prior', 'conditional_median_not_positive'], default='congestion_conditioned_median')
    return probability, raw, threshold, detail


def run_replay(panel, network, signals, settings, *, threads=2, load_fit=None, save_fit=None, progress=None):
    augmented, physical, required, feature_audit = make_fundamental_features(panel, variant='fundamental')
    _check_features(physical)
    network_names = _network_contract(panel, network)
    if (not signals.index.equals(panel.index) or set(signals) != {*SIGNALS, 'congestion_ready'}
            or signals.columns.has_duplicates or not signals.congestion_ready.map(lambda x: isinstance(x, (bool, np.bool_))).all()
            or np.isinf(signals[SIGNALS].to_numpy()).any()
            or signals.loc[~signals.congestion_ready, SIGNALS].notna().any().any()
            or signals.loc[signals.congestion_ready, SIGNALS].isna().any().any()):
        raise ValueError('Strict original-index OOS congestion signal contract required.')
    if set(network).intersection(augmented) or set(signals).intersection(augmented):
        raise ValueError('No overwritten source fields.')
    augmented = pd.concat([augmented, network, signals], axis=1)
    if (signals.congestion_ready & ~network.network_eligible).any():
        raise ValueError('Signals on an unqualified initial-domain hour forbidden.')
    common = physical+network_names
    params = base._parameters({**settings, 'threads': threads, 'feature_columns': common+SIGNALS,
                               'required_feature_columns': required+SIGNALS})
    memory, outputs, folds, governance, audits, interval_audits = {}, {}, [], [], {}, {}
    def load(day):
        return load_fit(day) if load_fit else memory.get(day)
    def save(day, state):
        if save_fit:
            save_fit(day, state)
        else:
            memory[day] = state
    for strategy, names in STRATEGIES.items():
        details = []
        def fit(data, day, cutoff, p, zones):
            return _fit(data, day, cutoff, p, zones, common, load, save, progress)
        def predict(state, current, p):
            probability, raw, threshold, detail = _predict(state, current, strategy)
            details.append(detail)
            return probability, raw, threshold
        result = base.run_policy(augmented, params, fit_callback=fit, predict_callback=predict)
        out = result.predictions.copy()
        for level in (10, 50, 90):
            out[f'mixture_error_q{level}'] = np.nan
        out['proposal_reason'] = 'prequential_warmup_or_missing_network'
        if details:
            detail = pd.concat(details).set_index(KEYS)
            if detail.index.has_duplicates:
                raise ValueError('Duplicate residual diagnostics.')
            for column in detail:
                out[column] = detail[column].reindex(pd.MultiIndex.from_frame(out[KEYS])).to_numpy()
        result = replace(result, predictions=out)
        for direct, model in zip((True, False), names):
            final = _finalize(result, direct=direct)
            if progress:
                progress(f'{model}: calibration chronologique des intervalles')
            bands, band_audit = _bands(final.predictions)
            outputs[model] = final.predictions.candidate_forecast
            outputs[model+'_q10'], outputs[model+'_q90'] = bands.candidate_q10, bands.candidate_q90
            outputs[model+'_interval_status'] = bands.interval_calibration_status
            interval_audits[model] = band_audit
        for name in ('spike_probability', 'threshold_eur_mwh', 'expert_ready', 'selected_weight',
                     'bounded_correction', 'raw_correction', 'cdf_evaluated', 'proposal_reason'):
            if name in out:
                outputs[strategy+'_'+name] = out[name]
        folds.append(result.folds.assign(strategy=strategy))
        governance.append(result.governance.assign(strategy=strategy))
        audits[strategy] = final.audit
    predictions = pd.concat([panel.copy(deep=True), signals, pd.DataFrame(outputs, index=panel.index)], axis=1)
    pd.testing.assert_frame_equal(predictions[panel.columns], panel, check_exact=True)
    for model in MODELS:
        q = predictions[[model+'_q10', model, model+'_q90']].to_numpy(float)
        if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
            raise ValueError('Invalid final ordered quantiles.')
    return dict(predictions=predictions, folds=pd.concat(folds, ignore_index=True),
        governance=pd.concat(governance, ignore_index=True), audit=dict(engine='nyx_congestion_v1',
        primary_model_fixed_before_evaluation='nyx_congestion', production_modified=False, activation_performed=False,
        diagnostic_only=True, post_coupling_features_used=False, detector_retrained=True,
        congestion_predictors_are_prequential_oos=True, control_has_exact_same_training_rows=True,
        feature_audit=feature_audit, signal_names=SIGNALS, policies=audits, interval_calibration=interval_audits,
        forecast_adjustment='mixture CDF median, not expected dual contribution added to NYX',
        electricity_prices_used_as_features=False, old_zonal_physical_gate_retained=False,
        secondary_training_eligibility='At least 90 distinct days with eligible OOF hours per country, not 90 complete days; core/calibration class and day minima also apply.',
        source_minimum_training_coverage_not_used_by_secondary_fitter=True,
        annual_non_regression_guaranteed=False, independent_validation=False,
        limitations=['Initial Presolved universe omits some constraints active after coupling.',
                     'CNEC predictions are conditional on historically matchable labelled constraints.',
                     'Historical API revision watermark is not a certified pre-08 real-time capture.',
                     'Progressive warmup: no complete 365-day training prefix.',
                     'This already examined year is exploratory; no production activation.']))
