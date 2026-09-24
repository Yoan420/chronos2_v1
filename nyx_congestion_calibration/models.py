"""Monotone, partially pooled calibration fitted on held-out classifier scores.

Minimise SUM log-loss + ridge towards raw logits (a=1, b=offsets=0).
This is a regularised point estimate, NOT a Bayesian uncertainty interval or
a guarantee of calibrated probabilities when events are absent.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from nyx_congestion.models import PARAMETERS, matrix, vector, logit

RIDGE = {'slope':10., 'intercept':2., 'zone':20.}


class CalibrationError(ValueError):
    pass


def _zones(values, n, zones):
    values = np.asarray(values)
    if (values.shape != (n,) or len(set(zones)) != len(zones) or not len(zones)
            or not set(values).issubset(zones)):
        raise ValueError('Explicit aligned known country codes required.')
    return np.array([zones.index(v) for v in values], dtype=int)


def fit_calibrator(raw, y, country, days, zones):
    zones = tuple(zones)
    y = vector(y, len(raw), probability=True)
    raw = vector(raw, len(y), probability=True)
    ids = _zones(country, len(y), zones)
    days = np.asarray(days)
    if not len(y) or not np.isin(y, [0,1]).all() or days.shape != y.shape or not all(isinstance(v, str) and v for v in days):
        raise ValueError('Nonempty binary chronological calibration sample required.')
    score = logit(raw).ravel()
    prior = np.r_[1., 0., np.zeros(len(zones))]
    penalty = np.r_[RIDGE['slope'], RIDGE['intercept'], np.full(len(zones), RIDGE['zone'])]
    def objective(theta):
        linear = theta[0]*score+theta[1]+theta[2:][ids]
        delta = theta-prior
        loss = np.sum(np.logaddexp(0., linear)-y*linear)+.5*np.dot(penalty, delta*delta)
        error = expit(linear)-y
        gradient = np.r_[np.dot(error, score), error.sum(), np.bincount(ids, weights=error, minlength=len(zones))]
        return float(loss), gradient+penalty*delta
    solved = minimize(objective, prior, jac=True, method='L-BFGS-B',
        bounds=[(0., None)]+[(None, None)]*(len(prior)-1),
        options={'maxiter':1000, 'ftol':1e-12, 'gtol':1e-7})
    if not solved.success or not np.isfinite(solved.fun) or not np.isfinite(solved.x).all() or solved.x[0] < 0:
        raise CalibrationError('Monotone probability calibration did not converge: '+str(solved.message))
    support = {}
    for i, z in enumerate(zones):
        selected = ids == i
        pos = selected & (y == 1)
        negatives = int((selected & (y == 0)).sum())
        positive_days = len(set(days[pos]))
        status = 'regularized_sparse_support' if pos.sum() < 5 or negatives < 5 or positive_days < 3 else 'regularized_supported'
        support[z] = dict(rows=int(selected.sum()), positives=int(pos.sum()), negatives=negatives,
                          positive_days=positive_days, eligible_days=len(set(days[selected])), status=status)
    return dict(zones=zones, coefficients=solved.x, ridge=dict(RIDGE), loss='sum_unweighted_log_loss',
        slope=float(solved.x[0]), intercept=float(solved.x[1]),
        offsets={z:float(solved.x[2+i]) for i,z in enumerate(zones)},
        support=support, converged=True, iterations=int(solved.nit),
        zero_slope=bool(solved.x[0] <= 1e-10), label_rows=len(y),
        prior='raw classifier logits, not climatology',
        uncertainty_intervals_provided=False, calendar_window_days=90)


def apply_calibrator(calibration, raw, country):
    raw = vector(raw, len(raw), probability=True)
    ids = _zones(country, len(raw), tuple(calibration['zones']))
    theta = np.asarray(calibration['coefficients'], dtype=float)
    if (theta.shape != (2+len(calibration['zones']),) or not np.isfinite(theta).all()
            or theta[0] < 0 or calibration.get('converged') is not True):
        raise CalibrationError('Invalid monotone calibrator state.')
    return expit(theta[0]*logit(raw).ravel()+theta[1]+theta[2:][ids])


def fit_probability(X, y, V, cy, *, cal_zones, cal_days, zones, threads=2):
    X, V = matrix(X), matrix(V)
    y = vector(y, len(X), probability=True)
    cy = vector(cy, len(V), probability=True)
    if (type(threads) is not int or threads not in (1,2) or X.shape[1] != V.shape[1]
            or not np.isin(y, [0,1]).all() or not np.isin(cy, [0,1]).all()
            or min(y.sum(), (1-y).sum()) < 30):
        raise ValueError('Sufficient binary CORE classes and matching calibration matrix required.')
    with threadpool_limits(limits=threads):
        model = HistGradientBoostingClassifier(**PARAMETERS).fit(X, y)
        calibration = fit_calibrator(model.predict_proba(V)[:,1], cy, cal_zones, cal_days, zones)
    return dict(model=model, calibration=calibration, columns=X.shape[1], threads=threads,
                prevalence=float(y.mean()), training_rows=len(y), calibration_rows=len(cy))


def predict_probability(state, X, zones):
    X = matrix(X, state['columns'])
    with threadpool_limits(limits=state['threads']):
        raw = state['model'].predict_proba(X)[:,1] if len(X) else np.empty(0)
    return apply_calibrator(state['calibration'], raw, zones), raw
