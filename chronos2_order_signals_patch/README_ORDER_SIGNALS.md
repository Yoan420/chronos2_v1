# Chronos-2 — scores OOF de structure de marché

Cette extension implémente le pipeline suivant :

```text
Prix Day-Ahead réalisés
    ↓
Labels ex post continus
    ↓
CatBoost multi-sorties entraîné seulement sur le passé
    ↓
Prévisions rolling out-of-fold
    ↓
Vintages PIT au cutoff D-1 08:00 Europe/Paris
    ↓
Injection dans Chronos-2 comme known_future/oracle
```

## Pourquoi les sorties sont des scores

Les données publiques ne permettent pas d'identifier avec certitude chaque
Load Gradient Order ou Block Order. Les labels sont donc des signatures de
prix continues entre 0 et 1 : rampe, plateau, jump, profil de bloc et rampe
linéaire compatible avec une contrainte de gradient.

Les labels ex post ne sont jamais lus par Chronos-2. Le modèle reçoit seulement
leurs prévisions strictement OOF ou live.

## Installation

Depuis la racine de `chronos2_v1`, décompresser le patch puis copier :

```powershell
Copy-Item .\chronos2_order_signals_patch\chronos2_order_signals `
    .\chronos2_order_signals -Recurse -Force

Copy-Item .\chronos2_order_signals_patch\build_order_signal_vintages.py `
    .\build_order_signal_vintages.py -Force

Copy-Item .\chronos2_order_signals_patch\run_chronos2_order_signals.py `
    .\run_chronos2_order_signals.py -Force

Copy-Item .\chronos2_order_signals_patch\run_order_signals_then_chronos.ps1 `
    .\run_order_signals_then_chronos.ps1 -Force

Copy-Item .\chronos2_order_signals_patch\chronos2_inputs_asof_jplus1_regime_order_signals.yaml `
    .\chronos2_inputs_asof_jplus1_regime_order_signals.yaml -Force

New-Item -ItemType Directory -Path .\tests -Force | Out-Null
Copy-Item .\chronos2_order_signals_patch\tests\test_order_signals.py `
    .\tests\test_order_signals.py -Force
```

Installer les dépendances :

```powershell
python -m pip install -r .\chronos2_order_signals_patch\requirements_order_signals.txt
```

Le fichier complet `chronos2_inputs_asof_jplus1_regime_order_signals.yaml`
est prêt à l'emploi. Le fichier `chronos2_order_signals_config_snippet.yaml`
reste disponible pour une fusion manuelle dans une autre configuration.

## Vérification

```powershell
python -m pytest .\tests\test_order_signals.py -q
```

## Backfill initial

À exécuter une seule fois après la création des vintages des fondamentaux :

```powershell
python .\build_order_signal_vintages.py `
    --config .\chronos2_inputs_asof_jplus1_regime_order_signals.yaml `
    --mode backfill `
    --zone FR `
    --start-day 2023-11-20
```

Le backfill :

- entraîne un modèle tous les 7 jours ;
- utilise au maximum les 730 jours précédents ;
- exige au moins 365 jours d'apprentissage ;
- prédit chaque bloc sans utiliser ses labels ;
- écrit un fichier PIT par score dans `data/pit/vintages`.

## Forecast opérationnel quotidien

Après `update_saturn_data.py` et avant Chronos-2 :

```powershell
python .\build_order_signal_vintages.py `
    --config .\chronos2_inputs_asof_jplus1_regime_order_signals.yaml `
    --mode live `
    --zone FR
```

Puis lancer Chronos-2 normalement :

```powershell
python .\run_chronos2_order_signals.py `
    --config .\chronos2_inputs_asof_jplus1_regime_order_signals.yaml `
    --zones FR `
    --local-files-only
```

Ne pas utiliser `--refresh-data` sur Chronos après la génération des scores si
les fondamentaux viennent déjà d'être actualisés, sauf si une nouvelle
actualisation Saturn est réellement souhaitée.

## Fichiers générés

```text
runs/order_signals/ex_post_labels.parquet
runs/order_signals/auxiliary_features.parquet
runs/order_signals/oof_signal_predictions.parquet
runs/order_signals/oof_signal_metrics.csv
runs/order_signals/live_signal_forecast.csv
runs/order_signals/models_oof/*.cbm
data/pit/vintages/fr_order_*_fcst.parquet
```

Chaque vintage contient au minimum :

```text
value_time_utc
snapshot_time_utc
revision_time_utc
value
```

Le snapshot d'une livraison D est horodaté exactement à D-1 08:00 en heure
locale, puis converti en UTC. Le loader PIT existant peut donc appliquer
`latest_before_asof` sans changement.

## Évaluation recommandée

Comparer quatre variantes sur exactement les mêmes origines :

```text
M0 prix seul
M1 fondamentaux
M2 fondamentaux + régimes de prix
M3 fondamentaux + régimes + scores OOF de structure
```

Ne pas se limiter à la MAE globale. Utiliser aussi la MAE sur les heures où le
label de rampe ou de jump dépasse 0,7, l'erreur sur le maximum journalier et le
timing du minimum/maximum.

## Correction oracle pour le forecast live

Le runner `run_chronos2_order_signals.py` corrige aussi le proxy `oracle` du
forecast live. Les valeurs PIT futures sont lues dans
`model_context_covariates`, dont l'index couvre J+1, plutôt que dans
`covariates`, qui est limité à l'historique de la cible. Le runner compose
automatiquement cette correction avec `chronos2_modular.regime` si le patch
de régime est installé.
