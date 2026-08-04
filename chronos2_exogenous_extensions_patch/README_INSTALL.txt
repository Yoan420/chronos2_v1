CHRONOS-2 — EXTENSIONS EXOGÈNES
================================

Ce patch ajoute :
- prix Day-Ahead DE/BE/NL/ES en lag24 et lag168 ;
- spreads FR-pays et facteurs agrégés voisins ;
- jours fériés FR/DE/BE/NL/ES, ponts, veille/lendemain de férié,
  heures de pointe, fins de mois/trimestre et transitions DST ;
- dispersion, dernière révision et ancienneté des forecasts PIT.

INSTALLATION
------------
Depuis C:\Users\BQ6757\chronos2_v1 :

Copy-Item .\chronos2_exogenous_extensions_patch\chronos2_modular\exogenous_extensions.py `
    .\chronos2_modular\exogenous_extensions.py -Force

Copy-Item .\chronos2_exogenous_extensions_patch\build_extended_exogenous_inputs.py `
    .\build_extended_exogenous_inputs.py -Force

Copy-Item .\chronos2_exogenous_extensions_patch\merge_exogenous_config.py `
    .\merge_exogenous_config.py -Force

Copy-Item .\chronos2_exogenous_extensions_patch\run_chronos2_extended_exogenous.py `
    .\run_chronos2_extended_exogenous.py -Force

Copy-Item .\chronos2_exogenous_extensions_patch\run_extended_pipeline.ps1 `
    .\run_extended_pipeline.ps1 -Force

New-Item -ItemType Directory -Path .\tests -Force | Out-Null
Copy-Item .\chronos2_exogenous_extensions_patch\tests\test_exogenous_extensions.py `
    .\tests\test_exogenous_extensions.py -Force

python -m pip install -r `
    .\chronos2_exogenous_extensions_patch\requirements_exogenous_extensions.txt

python .\chronos2_exogenous_extensions_patch\patch_order_signal_features.py

CRÉATION DE LA CONFIG
---------------------
python .\merge_exogenous_config.py `
    --input .\chronos2_inputs_asof_jplus1_regime_order_signals.yaml `
    --output .\chronos2_inputs_extended_exogenous.yaml

Les identifiants Saturn proposés pour les prix voisins suivent :
power.price.everyday.<pays>.hourly.eurmwh

S'ils diffèrent, modifie uniquement :
exogenous_extensions.neighbour_price_sources.countries

CONSTRUCTION DES ENTRÉES
------------------------
Avec actualisation Saturn :

python .\build_extended_exogenous_inputs.py `
    --config .\chronos2_inputs_extended_exogenous.yaml `
    --zone FR

Sans Saturn, en réutilisant les fichiers de prix voisins :

python .\build_extended_exogenous_inputs.py `
    --config .\chronos2_inputs_extended_exogenous.yaml `
    --zone FR `
    --local-only

TESTS
-----
python -m pytest .\tests\test_exogenous_extensions.py -q

RECONSTRUCTION DES SIGNAUX OOF
------------------------------
Les nouvelles variables sont automatiquement utilisées par le modèle
auxiliaire, car features.py lit data.known_future_columns.

Pour une comparaison propre :

Remove-Item .\runs\order_signals -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\data\pit\vintages\fr_order_*_fcst.parquet `
    -Force -ErrorAction SilentlyContinue

python .\build_order_signal_vintages.py `
    --config .\chronos2_inputs_extended_exogenous.yaml `
    --mode backfill `
    --zone FR `
    --start-day 2024-01-01

RUN CHRONOS-2
-------------
python .\run_chronos2_extended_exogenous.py `
    --config .\chronos2_inputs_extended_exogenous.yaml `
    --zones FR `
    --local-files-only

RUN QUOTIDIEN
-------------
.\run_extended_pipeline.ps1 `
    -Config .\chronos2_inputs_extended_exogenous.yaml `
    -Zone FR `
    -RefreshSaturn

POINT-IN-TIME
-------------
- Prix voisins : uniquement lag24 et lag168 ; aucun prix futur de livraison.
- Calendrier : déterministe, donc connu à l'avance.
- Incertitude : seules les révisions disponibles avant D-1 08:00 sont utilisées.

OPTIMISATION FEATURES.PY
------------------------
Dans chronos2_order_signals/features.py, ajoute "known_cal_" aux
excluded_prefixes de _add_path_features pour éviter les d1/d2 et statistiques
journalières inutiles des indicatrices calendaires.
