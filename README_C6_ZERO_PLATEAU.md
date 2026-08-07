
# C6 — Zero Plateau Gate

## Définition explicite

Plateau near-zero :
- prix dans `[-3, +3] €/MWh` ;
- au moins `3 heures consécutives` ;
- fenêtre solaire par défaut `08:00–19:00`.

## Étapes

1. Features point-in-time :
   - residual load ;
   - minimum/largeur/profondeur de la vallée résiduelle ;
   - pente de la vallée ;
   - solaire direct si déjà présent dans le YAML ;
   - sinon proxy solaire horaire explicite ;
   - nucléaire forecast ;
   - `da_cap_system_asymmetry` si disponible ;
   - états MILP ;
   - `zp_milp_oversupply_pressure`.

2. CatBoost point-in-time par heure.

3. Décodeur séquentiel imposant un seul bloc cohérent `start/end`.

4. Probabilités ajoutées à Chronos comme covariables known-future.

5. C6A = Chronos + probabilités, sans correction finale.

6. C6B = même forecast C6A + soft zero gate en post-traitement.

L'expert near-zero n'est pas fixé arbitrairement à zéro : ses quantiles sont
appris uniquement sur les plateaux du train.

## Anti-leakage

Le CatBoost s'arrête avant la période évaluée, avec un buffer égal à :

`ceil(context_length / 24) + 7 jours`

Ainsi, même le contexte du premier backtest Chronos reçoit des probabilités
CatBoost hors-échantillon.

## Aucun rebuild structurel

Le pipeline ne relance pas :
- Saturn ;
- snapshots PIT ;
- Branch-and-Cut ;
- LP pricing.

Il réutilise `data/derived/structural_market_features.csv.gz`.

## Installation

```powershell
python -m pip install -r .\requirements_zero_plateau.txt
```

Puis :

```powershell
.\run_zero_plateau_experiment.ps1
```

## Sorties

- `data/derived/zero_plateau_predictions.csv.gz`
- `data/models/zero_plateau_catboost.cbm`
- `data/derived/zero_plateau_model_metadata.json`
- `data/derived/zero_plateau_feature_importance.csv`
- `<run C6>/fr/zero_plateau_detection_metrics.csv`
- `<run C6>/fr/zero_plateau_price_slices.csv`
- `<run C6>/fr/zero_plateau_evaluation.json`
- `<run C6>/fr/backtest_zero_plateau_soft_gate.csv`
