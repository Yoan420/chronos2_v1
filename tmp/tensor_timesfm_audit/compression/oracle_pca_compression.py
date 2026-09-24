"""ORACLE reconstruction diagnostic; never a forecast or predictive-gain estimate.

Fit centering/PCA on 180 training days only. Reconstruct the *actual test
vectors* using their perfect future coordinates in that frozen basis. Rank 4
is the trivial identity in four dimensions. No model/weight/data is fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ZONES = ('BE', 'DE', 'FR', 'NL')
SPLITS = ('train', 'validation', 'test')
SPREADS = (('FR', 'DE'), ('BE', 'NL'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(error):
    values = np.asarray(error, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite reconstruction error')
    return {'n': int(values.size),
            'mae_eur_mwh': float(np.mean(np.abs(values))) if values.size else None,
            'rmse_eur_mwh': float(np.sqrt(np.mean(values ** 2))) if values.size else None,
            'bias_eur_mwh': float(np.mean(values)) if values.size else None}


def load_aligned(path):
    raw = pd.read_csv(path)
    required = {'timestamp', 'zone', 'observed', 'frozen_actual', 'nyx', 'day', 'hour', 'split'}
    if not required.issubset(raw):
        raise ValueError(f'Missing columns: {required.difference(raw.columns)}')
    timestamps = [pd.Timestamp(value) for value in raw.timestamp]
    if any(value.tzinfo is None or pd.isna(value) for value in timestamps):
        raise ValueError('Explicit aware timestamps are mandatory')
    raw['timestamp'] = pd.to_datetime(timestamps, utc=True)
    if set(raw.zone) != set(ZONES) or set(raw.split) != set(SPLITS):
        raise ValueError('Expected exactly four countries and three supplied splits')
    if raw.duplicated(['timestamp', 'zone']).any():
        raise ValueError('Duplicate UTC country coordinates')
    grouped = raw.groupby('timestamp')
    if not grouped.size().eq(4).all():
        raise ValueError('Each UTC hour must contain all four countries')
    for field in ['day', 'hour', 'split']:
        if not grouped[field].nunique().eq(1).all():
            raise ValueError(f'Country metadata disagree: {field}')
    meta = raw.drop_duplicates('timestamp').set_index('timestamp').sort_index()[['day', 'hour', 'split']]
    expected = pd.date_range(meta.index[0], meta.index[-1], freq='h')
    if not meta.index.equals(expected) or len(meta) != 8760:
        raise ValueError('Expected all 8,760 consecutive physical UTC hours, without drops')
    local = meta.index.tz_convert('Europe/Paris')
    if list(meta.day) != list(local.strftime('%Y-%m-%d')) or list(meta.hour) != list(local.hour):
        raise ValueError('Local day/hour differ from Europe/Paris coordinates')
    days = meta.groupby('day')
    if not days.split.nunique().eq(1).all() or len(days) != 365:
        raise ValueError('Whole civil days must belong to one split')
    for day, data in days:
        start = pd.Timestamp(day).tz_localize('Europe/Paris')
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize('Europe/Paris')
        physical = pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC')
        if not data.index.equals(physical):
            raise ValueError(f'Incomplete physical day: {day}')
    descriptions = {}
    prior_end = None
    for split, expected_days in zip(SPLITS, [180, 95, 90]):
        subset = meta.loc[meta.split.eq(split)]
        if subset.day.nunique() != expected_days:
            raise ValueError(f'{split}: expected {expected_days} civil days')
        if not subset.index.equals(pd.date_range(subset.index[0], subset.index[-1], freq='h')):
            raise ValueError('Split is not contiguous')
        if prior_end is not None and subset.index[0] != prior_end + pd.Timedelta(hours=1):
            raise ValueError('Splits must be chronological and adjacent')
        prior_end = subset.index[-1]
        descriptions[split] = {'days': int(subset.day.nunique()), 'hours_per_country': len(subset),
                               'first_day': subset.day.iloc[0], 'last_day': subset.day.iloc[-1],
                               'first_utc': subset.index[0].isoformat(), 'last_utc': subset.index[-1].isoformat()}
    values = {}
    for field in ['observed', 'frozen_actual', 'nyx']:
        wide = raw.pivot(index='timestamp', columns='zone', values=field).sort_index().reindex(columns=ZONES)
        if not wide.index.equals(meta.index) or not np.isfinite(wide.to_numpy(float)).all():
            raise ValueError(f'Invalid aligned values: {field}')
        values[field] = wide.to_numpy(float)
    descriptions['civil_day_hour_counts'] = {str(k): int(v) for k, v in days.size().value_counts().sort_index().items()}
    return values, meta, descriptions


def evaluate_space(label, prices, nyx, train, test, residual):
    # Use the algebraic residual actual - forecast; do not inherit a source
    # column whose sign convention might instead be forecast - actual.
    full = prices - nyx if residual else prices
    fit = full[train]
    truth = full[test]
    actual = prices[test]
    forecast = nyx[test]
    center = fit.mean(axis=0)
    centered = fit - center
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    # Deterministic display convention only; signs have no reconstruction effect.
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    if not np.allclose(basis @ basis.T, np.eye(4), rtol=0, atol=1e-12):
        raise ValueError('PCA basis is not orthonormal')
    variance = singular ** 2 / (len(fit) - 1)
    fractions = variance / variance.sum()
    thresholds = {q: np.maximum(0.0, np.quantile(prices[train], q, axis=0)) for q in [0.95, 0.99]}
    masks = {'all': np.ones(actual.shape, dtype=bool), 'negative_price': actual < 0,
             'positive_peak_q95_train': actual > thresholds[0.95],
             'positive_peak_q99_train': actual > thresholds[0.99]}
    if residual:
        for q in [0.95, 0.99]:
            cut = np.quantile(np.abs(fit), q, axis=0)
            masks[f'absolute_residual_q{int(q*100)}_train'] = np.abs(truth) > cut
    result = {'diagnostic': 'ORACLE reconstruction, not a forecast', 'space': label,
              'definition': 'actual - nyx' if residual else 'actual price',
              'unit': 'EUR/MWh', 'standardized': False,
              'country_order': list(ZONES), 'train_center': center.tolist(),
              'train_eigenvalues_eur_mwh_squared': variance.tolist(),
              'train_explained_variance_fraction': fractions.tolist(),
              'train_cumulative_explained_variance_fraction': np.cumsum(fractions).tolist(),
              'train_component_loadings': basis.tolist(),
              'train_positive_price_thresholds': {str(q): dict(zip(ZONES, values.tolist())) for q, values in thresholds.items()},
              'rank_results': []}
    last_rmse = float('inf')
    for rank in [1, 2, 3, 4]:
        components = basis[:rank]
        # Deliberately ORACLE: future observed values form the coordinates.
        coordinates = (truth - center) @ components.T
        reconstructed = center + coordinates @ components
        error = reconstructed - truth
        implied_price = forecast + reconstructed if residual else reconstructed
        overall = score(error)
        if overall['rmse_eur_mwh'] > last_rmse + 1e-10:
            raise ValueError('Orthogonal projection RMSE must be nonincreasing in rank')
        last_rmse = overall['rmse_eur_mwh']
        if rank == 4 and np.max(np.abs(error)) > 1e-9:
            raise ValueError('Full-rank reconstruction should be the identity')
        records = {'rank': rank, 'train_variance_retained': float(fractions[:rank].sum()),
                   'test_reconstruction': overall,
                   'maximum_absolute_reconstruction_error': float(np.max(np.abs(error))),
                   'test_regimes': {name: score(error[mask]) for name, mask in masks.items()},
                   'countries': {}, 'spreads': {}}
        for j, zone in enumerate(ZONES):
            records['countries'][zone] = {name: score(error[:, j][mask[:, j]]) for name, mask in masks.items()}
        negative = masks['negative_price']
        lost = negative & (implied_price >= 0)
        records['negative_price_sign'] = {'negative_actual_cells': int(negative.sum()),
                                          'reconstructed_nonnegative_cells': int(lost.sum()),
                                          'fraction_lost': float(lost.sum() / negative.sum()) if negative.any() else None,
                                          'near_zero_tolerance_eur_mwh': 1e-9,
                                          'material_sign_losses': int((negative & (implied_price > 1e-9)).sum())}
        for left, right in SPREADS:
            a, b = ZONES.index(left), ZONES.index(right)
            spread = actual[:, a] - actual[:, b]
            recovered_spread = implied_price[:, a] - implied_price[:, b]
            spread_error = recovered_spread - spread
            threshold = float(np.quantile(np.abs(prices[train, a] - prices[train, b]), 0.95))
            extreme = np.abs(spread) > threshold
            material = np.abs(spread) > 1e-9
            records['spreads'][f'{left}-{right}'] = {
                'meaning': 'ORACLE reconstructed price spread, EUR/MWh',
                'all': score(spread_error), 'train_absolute_spread_q95': threshold,
                'test_large_absolute_spread': score(spread_error[extreme]),
                'actual_test_mean': float(spread.mean()), 'oracle_reconstructed_test_mean': float(recovered_spread.mean()),
                'material_actual_nonzero_hours': int(material.sum()),
                'direction_errors_or_erased_spreads': int((material & (np.sign(spread) != np.sign(recovered_spread))).sum())}
        result['rank_results'].append(records)
    return result


def make_note(output):
    split = output['alignment']['test']
    text = ['# Diagnostic ORACLE de compression PCA — aucune prévision', '',
            'Les coordonnées latentes du test sont calculées à partir des prix ou erreurs futurs effectivement observés. '
            'Ces reconstructions constituent un diagnostic de perte d’information dans une base figée ; elles ne sont ni '
            'des prévisions disponibles au cutoff, ni une mesure de gain de NYX.', '',
            '## Protocole et alignement', '',
            '- Quatre pays BE, DE, FR, NL ; 8 760 heures UTC communes, sans doublon ni heure supprimée.',
            '- Train : 180 jours, 4 321 heures par pays ; validation : 95 jours, 2 279 heures, inutilisée ; '
            f'test : 90 jours, {split["hours_per_country"]} heures, du {split["first_day"]} au {split["last_day"]}.',
            '- Les journées locales sont vérifiées contre leur grille physique ; le jour de 25 heures est conservé '
            'dans le train et celui de 23 heures dans la validation.',
            '- Centrage et SVD uniquement sur train, aucun écart-type appliqué : unité EUR/MWh conservée. '
            'Les pays de plus forte variance peuvent donc dominer la base.',
            '- Deux diagnostics principaux : prix révisés `observed`, et erreurs finales `observed − NYX`. '
            'Deux sensibilités indépendantes utilisent `frozen_actual` à la place des observations révisées.',
            '- Seuils de pics positifs : quantiles 95 % / 99 % des prix du train, séparément par pays '
            '(bornés à zéro). Les régimes du test servent uniquement à mesurer, jamais à ajuster.', '',
            '## Résultats ORACLE sur le test', '',
            '| Espace | Rang | Variance train retenue | RMSE globale | MAE globale | RMSE prix négatifs | RMSE pics q95 | RMSE pics q99 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['observed_price', 'observed_minus_nyx']:
        for row in output['spaces'][name]['rank_results']:
            r = row['test_regimes']
            f = lambda n: f'{n:.3f}' if n is not None else 'n/a'
            text.append(f'| {name} | {row["rank"]} | {100*row["train_variance_retained"]:.2f} % | '
                        f'{f(r["all"]["rmse_eur_mwh"])} | {f(r["all"]["mae_eur_mwh"])} | '
                        f'{f(r["negative_price"]["rmse_eur_mwh"])} | {f(r["positive_peak_q95_train"]["rmse_eur_mwh"])} | '
                        f'{f(r["positive_peak_q99_train"]["rmse_eur_mwh"])} |')
    text += ['', 'Toutes les erreurs sont des **erreurs de reconstruction ORACLE**, en EUR/MWh. '
             'Le diagnostic résiduel fournit déjà les erreurs finales futures exactes au projecteur.', '',
             '### Spreads et prix négatifs — reconstruction ORACLE', '',
             '| Espace | Rang | RMSE spread FR−DE | RMSE spread BE−NL | Prix négatifs reconstruits ≥ 0 |',
             '|---|---:|---:|---:|---:|']
    for name in ['observed_price', 'observed_minus_nyx']:
        for row in output['spaces'][name]['rank_results']:
            sign = row['negative_price_sign']
            text.append(f'| {name} | {row["rank"]} | {row["spreads"]["FR-DE"]["all"]["rmse_eur_mwh"]:.3f} | '
                        f'{row["spreads"]["BE-NL"]["all"]["rmse_eur_mwh"]:.3f} | '
                        f'{sign["reconstructed_nonnegative_cells"]}/{sign["negative_actual_cells"]} |')
    text += ['', 'Pour les résidus, les spreads et le signe sont calculés sur `NYX + résidu reconstruit`. '
             'Les erreurs de spread sont identiques à celles de la différence des deux résidus reconstruits.', '',
             '## Sensibilité aux observations révisées', '',
             '| Rang | RMSE prix observed | RMSE prix frozen_actual | RMSE résidu observed | RMSE résidu frozen_actual |',
             '|---|---:|---:|---:|---:|']
    for i in range(4):
        vals = [output['spaces'][key]['rank_results'][i]['test_reconstruction']['rmse_eur_mwh']
                for key in ['observed_price', 'frozen_price', 'observed_minus_nyx', 'frozen_minus_nyx']]
        text.append(f'| {i+1} | ' + ' | '.join(f'{value:.6f}' for value in vals) + ' |')
    text += ['', 'Les détails par pays, biais, effectifs, seuils train, charges des composantes, grandes erreurs '
             'résiduelles et grands spreads sont dans `oracle_compression.json`.', '',
             '## Constats de compression, sans conclusion prévisionnelle', '']
    price_rank3 = output['spaces']['observed_price']['rank_results'][2]
    residual_rank3 = output['spaces']['observed_minus_nyx']['rank_results'][2]
    regime_counts = price_rank3['test_regimes']
    text += [f'- Le rang 3 conserve {100*price_rank3["train_variance_retained"]:.2f} % de la variance des prix du train, '
             f'mais son erreur de reconstruction ORACLE du spread BE−NL reste '
             f'{price_rank3["spreads"]["BE-NL"]["all"]["rmse_eur_mwh"]:.2f} EUR/MWh de RMSE. '
             'La petite composante écartée conserve donc une information utile pour cette divergence entre pays.',
             f'- Pour les erreurs finales NYX, le rang 3 laisse '
             f'{residual_rank3["test_reconstruction"]["rmse_eur_mwh"]:.2f} EUR/MWh de RMSE de reconstruction ORACLE. '
             'Cela ne permet pas de déduire un gain réalisable : il faudrait ensuite prévoir les facteurs.',
             f'- Les régimes test comptent {regime_counts["negative_price"]["n"]} cellules de prix négatifs, '
             f'{regime_counts["positive_peak_q95_train"]["n"]} au-dessus du q95 train et '
             f'{regime_counts["positive_peak_q99_train"]["n"]} au-dessus du q99 train, sur '
             f'{regime_counts["all"]["n"]} cellules pays×heure. « q99 train » ne signifie pas les 1 % les plus hauts du test ; '
             'les seuils restent ceux appris avant ce test.', '',
             '## Interprétation autorisée', '',
             '- Le rang 4 est la reconstruction exacte triviale d’un vecteur à quatre coordonnées : aucune '
             'compression et aucune performance prévisionnelle démontrée.',
             '- Les rangs 1–3 mesurent ce que la projection supprime même avec des coordonnées futures parfaites. '
             'Un grand pourcentage de variance train retenue ne garantit pas une bonne préservation des pics, '
             'des prix négatifs ou des divergences de pays dans le test.',
             '- Pour cette base et ce décodeur linéaire figés, la projection orthogonale est la meilleure '
             'reconstruction quadratique dans le sous-espace choisi. Une prévision imparfaite des facteurs ajoute '
             'de l’erreur ; ce diagnostic n’est pas une borne universelle pour tout décodeur non linéaire.',
             '- La possibilité de compresser des erreurs contemporaines ne démontre pas que leurs facteurs '
             'soient prévisibles. Le test causal suivant doit estimer les facteurs sans aucune valeur future du test.',
             '- Aucun rang n’a été choisi sur ces résultats ; un choix de rang ou de normalisation pour une '
             'future comparaison doit utiliser la validation, puis un nouveau test réellement réservé.', '',
             '## Reproduction', '',
             '```powershell',
             "& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' 'tmp/tensor_timesfm_audit/compression/oracle_pca_compression.py'",
             '```', '',
             f'Entrée SHA256 : `{output["input_sha256"]}`. Le script contrôle que cette empreinte demeure '
             'inchangée après lecture. Il ne charge aucun modèle ni poids ; seuls les fichiers de ce diagnostic sont écrits.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('tmp/tensor_timesfm_audit/metrics/verified_hourly_pairs.csv.gz'))
    parser.add_argument('--output-dir', type=Path, default=Path('tmp/tensor_timesfm_audit/compression'))
    args = parser.parse_args()
    digest = sha256(args.input)
    values, meta, alignment = load_aligned(args.input)
    train, test = meta.split.eq('train').to_numpy(), meta.split.eq('test').to_numpy()
    output = {'diagnostic': 'ORACLE PCA RECONSTRUCTION ONLY — NOT A FORECAST OR A PREDICTIVE GAIN',
              'input_path': str(args.input.resolve()), 'input_sha256': digest,
              'script_sha256': sha256(__file__), 'alignment': alignment,
              'method': {'centering_fit': 'train only', 'pca_fit': 'train only', 'standardization': False,
                         'test_latents': 'actual future values projected into the frozen training basis',
                         'rank4': 'trivial four-dimensional identity, no compression',
                         'error_sign': 'reconstruction minus actual target', 'rank_selection': 'none'},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
              'spaces': {}}
    for label, field, residual in [('observed_price', 'observed', False), ('observed_minus_nyx', 'observed', True),
                                    ('frozen_price', 'frozen_actual', False), ('frozen_minus_nyx', 'frozen_actual', True)]:
        output['spaces'][label] = evaluate_space(label, values[field], values['nyx'], train, test, residual)
    if sha256(args.input) != digest:
        raise ValueError('Source changed during diagnostic')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'oracle_compression.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.output_dir / 'oracle_compression.md').write_text(make_note(output), encoding='utf-8')
    print(json.dumps({'diagnostic': output['diagnostic'], 'alignment': alignment,
                      'summary': {key: [{'rank': row['rank'], **row['test_reconstruction']} for row in value['rank_results']]
                                  for key, value in output['spaces'].items()}}, indent=2))


if __name__ == '__main__':
    main()
