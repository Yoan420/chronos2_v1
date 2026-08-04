PATCH RÉGIME DE MARCHÉ — CHRONOS-2
=====================================

Contenu
-------
1. chronos2_modular/regime.py
   Calcule cinq covariables causales à partir de target.shift(1).

2. run_chronos2_regime.py
   Wrapper qui branche ces covariables sur le pipeline existant sans modifier
   run_chronos2_modular.py, data.py ou forecasting.py.

3. chronos2_inputs_asof_jplus1_regime.yaml
   Configuration complète incluant les exports nets et les régimes.

4. tests/test_market_regime.py
   Tests de non-fuite et de cohérence à l'origine.

Installation
------------
Depuis la racine de chronos2_v1, copiez les fichiers du patch en conservant
l'arborescence. Sous PowerShell :

Copy-Item .\chronos2_market_regime_patch\chronos2_modular\regime.py `
    .\chronos2_modular\regime.py -Force

Copy-Item .\chronos2_market_regime_patch\run_chronos2_regime.py `
    .\run_chronos2_regime.py -Force

Copy-Item .\chronos2_market_regime_patch\chronos2_inputs_asof_jplus1_regime.yaml `
    .\chronos2_inputs_asof_jplus1_regime.yaml -Force

Copy-Item .\chronos2_market_regime_patch\tests\test_market_regime.py `
    .\tests\test_market_regime.py -Force

Test
----
python -m pytest .\tests\test_market_regime.py -q

Exécution
---------
python .\run_chronos2_regime.py `
    --config .\chronos2_inputs_asof_jplus1_regime.yaml `
    --zones FR `
    --local-files-only `
    --refresh-data

Pour réutiliser les caches déjà actualisés, retirez --refresh-data.

Variables ajoutées
------------------
- regime_level_7d
- regime_volatility_7d
- regime_negative_rate_30d
- regime_spike_rate_30d
- regime_trend

Garantie point-in-time
----------------------
Pour une ligne historique t, chaque variable utilise target.shift(1), donc
uniquement les prix strictement antérieurs à t.

Pour chaque origine de backtest, le régime est recalculé sur
target.iloc[:origin_position], puis maintenu constant sur les 24 heures
futures. Aucun prix de l'horizon prédit n'est consulté.

Le recalcul à chaque origine évite aussi une erreur subtile : utiliser
simplement la dernière ligne de régime du contexte aurait ajouté un retard
involontaire d'une heure.
