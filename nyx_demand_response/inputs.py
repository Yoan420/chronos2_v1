"""Fail-closed contract for externally evidenced, pre-auction scenarios.

This adapter does NOT forecast tomorrow's bid curves from today's auction.
It accepts documented scenario forecasts or published flexibility offers,
never the realised curve of the auction being predicted. Scenario quantiles
are mathematical weighted quantiles, not a claim of statistical calibration.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .dispatch import solve_period, validate_period_inputs, DispatchQualificationError

KEYS = ['zone', 'timestamp_utc', 'forecast_origin_utc']
ZONES = ('FR', 'DE', 'BE', 'NL')
FLAGS = ('supply_scope_qualified', 'mobilisable_power_qualified', 'flexibility_curve_qualified',
         'network_scope_qualified', 'asof_certified')


def _tables(record):
    # A documented empty supply/flexibility list is valid, not a missing table.
    names = {'supply':['zone','segment','capacity_mw','bid_eur_mwh'],
             'flexibility':['zone','segment','capacity_mw','reservation_eur_mwh']}
    frames = {}
    for name in ('supply','demand','flexibility','network'):
        value = record[name]
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise InputContractError(name+': normalized row list required.')
        frames[name] = pd.DataFrame(value, columns=names[name]) if not value and name in names else pd.DataFrame(value)
    return frames


class InputContractError(ValueError):
    pass


def utc(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise InputContractError(name+': explicit timezone required')
    return stamp.tz_convert('UTC')


def validate_bundle(bundle):
    if (not isinstance(bundle, dict) or set(bundle) != {'schema_version','kind','sources','periods'}
            or bundle['schema_version'] != 1 or type(bundle['schema_version']) is not int
            or bundle['kind'] != 'ex_ante_demand_response_scenarios'):
        raise InputContractError('Schema1 ex-ante scenario bundle required.')
    sources = bundle['sources']
    if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s.strip() for s in sources):
        raise InputContractError('Explicit source evidence references required.')
    if not isinstance(bundle['periods'], list) or not bundle['periods']:
        raise InputContractError('No evidenced scenarios supplied.')
    identities, records = set(), []
    required = {'delivery_start_utc','forecast_origin_utc','inputs_available_at_utc','training_end_utc',
                'duration_minutes','scenario_id','scenario_weight','curve_evidence',
                'supply','demand','flexibility','network','qualification'}
    for record in bundle['periods']:
        if not isinstance(record, dict) or set(record) != required:
            raise InputContractError('Unexpected/missing period fields; no implicit inputs.')
        start = utc(record['delivery_start_utc'], 'delivery')
        origin = utc(record['forecast_origin_utc'], 'origin')
        expected = (pd.Timestamp(start.tz_convert('Europe/Paris').date())-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize('Europe/Paris').tz_convert('UTC')
        if origin != expected or utc(record['inputs_available_at_utc'], 'availability') > origin:
            raise InputContractError('Only inputs actually available by D-1 08:00 Paris are allowed.')
        if record['curve_evidence'] not in ('forecast_from_past_curves','published_flexibility_offers'):
            raise InputContractError('Realised/current auction curves and guessed price thresholds are forbidden.')
        if record['training_end_utc'] is None:
            if record['curve_evidence'] == 'forecast_from_past_curves':
                raise InputContractError('Forecast curves require last training-label availability.')
        elif utc(record['training_end_utc'], 'training label availability') > origin:
            raise InputContractError('Future labels used to forecast curves.')
        duration = record['duration_minutes']
        if type(duration) is not int or duration not in (15,60) or start.second or start.microsecond or start.minute % duration:
            raise InputContractError('Aligned 15- or 60-minute delivery periods required.')
        sid, weight = record['scenario_id'], record['scenario_weight']
        if not isinstance(sid, str) or not sid.strip() or isinstance(weight, bool) or not isinstance(weight, (int,float)) or not math.isfinite(weight) or weight <= 0:
            raise InputContractError('Named positive finite scenario weights required.')
        key = (start, origin, sid)
        if key in identities:
            raise InputContractError('Duplicate period/scenario.')
        identities.add(key)
        q = record['qualification']
        if (not isinstance(q, dict) or q.get('evidence_kind') != 'audited_inputs'
                or any(q.get(flag) is not True for flag in FLAGS)
                or not isinstance(q.get('evidence_reference'), str) or not q['evidence_reference'].strip()
                or record['network'] is None):
            raise InputContractError('Audited supply, flexibility, network and PIT qualification required; synthetic evidence forbidden here.')
        if q.get('duration_hours') != duration/60:
            raise InputContractError('LP duration differs from scenario duration.')
        demand = pd.DataFrame(record['demand'])
        if 'zone' not in demand or not set(ZONES).issubset(demand.zone):
            raise InputContractError('All four comparison zones required in each coupled scenario.')
        try:
            validate_period_inputs(**_tables(record), qualification=q)
        except DispatchQualificationError as exc:
            raise InputContractError(str(exc)) from exc
        records.append({**record, 'delivery_start_utc':start, 'forecast_origin_utc':origin})
    # Each scenario is a joint path across the four quarter-hours of an hour.
    # Never average marginal P50s or silently drop an unbalanced quarter-hour.
    frame = pd.DataFrame(records)
    frame['_hour'] = frame.delivery_start_utc.dt.floor('h')
    for (_, origin), hour in frame.groupby(['_hour','forecast_origin_utc']):
        expected_ids = None
        for _, part in hour.groupby('delivery_start_utc'):
            ids = set(part.scenario_id)
            if expected_ids is not None and ids != expected_ids:
                raise InputContractError('Joint scenario identities differ within an hour.')
            expected_ids = ids
        weights = []
        for _, scenario in hour.groupby('scenario_id'):
            scenario = scenario.sort_values('delivery_start_utc')
            if scenario.scenario_weight.nunique() != 1:
                raise InputContractError('Scenario weight changes within an hour.')
            durations = scenario.duration_minutes.to_numpy()
            if durations.sum() != 60 or len(set(durations)) != 1:
                raise InputContractError('Exactly one full hour per scenario required, without mixed resolutions.')
            expected_stamps = pd.date_range(scenario._hour.iloc[0], periods=60//int(durations[0]), freq=f'{durations[0]}min')
            if list(scenario.delivery_start_utc) != list(expected_stamps):
                raise InputContractError('Missing/overlapping scenario intervals.')
            weights.append(float(scenario.scenario_weight.iloc[0]))
        if not np.isclose(sum(weights), 1., rtol=0, atol=1e-9):
            raise InputContractError('Scenario weights must sum to one; no automatic renormalisation.')
    return records


def scenario_forecasts(bundle):
    records = validate_bundle(bundle)
    values, audits = [], []
    for record in records:
        solution = solve_period(**_tables(record), qualification=record['qualification'])
        if any(solution['served_demand_mw'][zone] <= 1e-6 for zone in ZONES):
            raise InputContractError('Fully curtailed or zero served demand: the balance dual is not a qualified transaction-price forecast.')
        if any(solution['price_brackets'][zone]['ambiguous'] for zone in ZONES):
            raise InputContractError('Non-unique marginal price at a curve kink; no arbitrary dual may become a forecast.')
        audits.append(dict(delivery_start_utc=record['delivery_start_utc'], scenario_id=record['scenario_id'], result=solution))
        for zone in ZONES:
            values.append(dict(zone=zone, timestamp_utc=record['delivery_start_utc'].floor('h'),
                forecast_origin_utc=record['forecast_origin_utc'], scenario_id=record['scenario_id'],
                scenario_weight=record['scenario_weight'], hours=record['duration_minutes']/60,
                price=solution['prices_eur_mwh'][zone]))
    values = pd.DataFrame(values)
    hourly = values.assign(weighted_price=values.price*values.hours).groupby(KEYS+['scenario_id'], as_index=False).agg(
        price=('weighted_price','sum'), scenario_weight=('scenario_weight','first'))
    rows = []
    for keys, group in hourly.groupby(KEYS):
        order = np.argsort(group.price.to_numpy(), kind='stable')
        p, w = group.price.to_numpy()[order], group.scenario_weight.to_numpy()[order]
        quantiles = p[np.minimum(np.searchsorted(np.cumsum(w), [.1,.5,.9], side='left'), len(p)-1)]
        rows.append(dict(zip(KEYS, keys), expert_q10=float(quantiles[0]), expert_price=float(quantiles[1]),
            expert_q90=float(quantiles[2]), expert_status='qualified_scenario_forecast_not_statistically_calibrated',
            scenario_count=len(p)))
    return pd.DataFrame(rows), audits
