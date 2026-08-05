CHRONOS-2 — SÉLECTION DE VARIABLES SANS FUITE TEMPORELLE
=========================================================

OBJECTIF
--------
Le patch réalise des ablations sur les colonnes réellement envoyées à Chronos-2,
après création des lags, variables oracle, calendriers, spreads et agrégats.

La procédure est volontairement temporelle : chaque famille est évaluée sur
plusieurs cutoffs opérationnels. Une matrice de corrélation seule n'est pas
suffisante pour un modèle non linéaire et ne mesure pas la valeur hors échantillon.

POINT IMPORTANT
---------------
Les vintages OOF des order signals doivent être construits UNE FOIS et rester
figés pendant toutes les ablations. Ne les reconstruis pas pour chaque config :
sinon la variable testée change de définition et les scores ne sont plus
comparables.

INSTALLATION
------------
Depuis la racine du projet :

Copy-Item .\chronos2_feature_selection\chronos2_modular\feature_selection.py `
    .\chronos2_modular\feature_selection.py -Force
Copy-Item .\chronos2_feature_selection\feature_groups.yaml `
    .\feature_groups.yaml -Force
Copy-Item .\chronos2_feature_selection\run_chronos2_selected.py `
    .\run_chronos2_selected.py -Force
Copy-Item .\chronos2_feature_selection\run_feature_selection_batch.py `
    .\run_feature_selection_batch.py -Force
Copy-Item .\chronos2_feature_selection\summarize_feature_selection.py `
    .\summarize_feature_selection.py -Force
Copy-Item .\chronos2_feature_selection\analyze_feature_redundancy.py `
    .\analyze_feature_redundancy.py -Force
Copy-Item .\chronos2_feature_selection\apply_selected_groups.py `
    .\apply_selected_groups.py -Force
Copy-Item .\chronos2_feature_selection\run_feature_selection.ps1 `
    .\run_feature_selection.ps1 -Force
New-Item -ItemType Directory -Path .\tests -Force | Out-Null
Copy-Item .\chronos2_feature_selection\tests\test_feature_selection.py `
    .\tests\test_feature_selection.py -Force
python -m pip install -r `
    .\chronos2_feature_selection\requirements_feature_selection.txt
python -m pytest .\tests\test_feature_selection.py -q

ÉTAPE 1 — SCREENING DES FAMILLES
--------------------------------
Mode rapide : 3 folds de 60 journées, modèle chargé une seule fois.

.\run_feature_selection.ps1 `
    -Config chronos2_inputs_extended_exogenous.yaml `
    -Level family `
    -Mode only `
    -BacktestWindows 60

Sorties :
- runs\feature_selection\family_only\selection_results.csv
- runs\feature_selection\family_only\summary\feature_selection_ranking.csv
- runs\feature_selection\family_only\summary\recommended_subjects.yaml

ÉTAPE 2 — ABLATION CONDITIONNELLE DES FAMILLES RETENUES
--------------------------------------------------------
Exemple à adapter à la sortie de l'étape 1 :

.\run_feature_selection.ps1 `
    -Config chronos2_inputs_extended_exogenous.yaml `
    -Level family `
    -Mode loo `
    -Subjects calendar,fr_fundamentals,neighbour_market,order_signals `
    -BacktestWindows 90

En mode loo, une contribution positive signifie que retirer la famille dégrade
le modèle complet. C'est plus fiable que l'importance standalone lorsque les
variables sont colinéaires.

ÉTAPE 3 — COMPOSANTS À L'INTÉRIEUR DES FAMILLES RETENUES
---------------------------------------------------------
Exemple pour détailler les prix voisins et l'incertitude :

.\run_feature_selection.ps1 `
    -Config chronos2_inputs_extended_exogenous.yaml `
    -Level component `
    -Mode both `
    -Subjects neighbour_prices_raw,neighbour_price_pair_spreads,` 
        neighbour_price_level_aggregates,neighbour_price_dispersion,` 
        neighbour_spread_aggregates,uncertainty_fr_residual,` 
        uncertainty_fr_nuclear,uncertainty_neighbours `
    -BacktestWindows 90

ANALYSE DE REDONDANCE
---------------------
À lancer sur le run complet :

python .\analyze_feature_redundancy.py `
    --input .\runs\feature_selection\family_only\fold_01_20250801_080000p0200\full\fr\model_covariates_selected.csv.gz `
    --output-dir .\runs\feature_selection\redundancy `
    --correlation-threshold 0.97

Cette analyse signale :
- constantes et quasi-constantes ;
- doublons exacts ;
- paires fortement corrélées ;
- clusters de corrélation.

Elle ne décide pas seule de la sélection. La décision finale doit rester basée
sur les gains hors échantillon des ablations temporelles.

CONSTRUIRE LA CONFIGURATION FINALE
---------------------------------

python .\apply_selected_groups.py `
    --config .\chronos2_inputs_extended_exogenous.yaml `
    --groups .\feature_groups.yaml `
    --selected .\runs\feature_selection\family_only\summary\recommended_subjects.yaml `
    --level family `
    --output .\chronos2_inputs_selected.yaml

Puis :

python .\run_chronos2_selected.py `
    --config .\chronos2_inputs_selected.yaml `
    --zones FR `
    --local-files-only

INTERPRÉTATION
--------------
1. Le critère principal est le MAE q50 sur plusieurs folds.
2. Le CRPS protège la qualité probabiliste.
3. Le ramp_mae protège les variations horaires.
4. Les métriques de prix négatifs et extrêmes doivent être contrôlées dans le
   CSV brut lorsque le nombre d'événements est suffisant.
5. Une famille n'est conservée que si son gain est stable ou si sa suppression
   dégrade clairement le modèle complet.

AUDIT DE DISPONIBILITÉ
----------------------
Avant toute sélection statistique, exclure les variables non disponibles à
08:00 à l'origine réelle. Une variable fuyarde peut sembler très importante.
En particulier, vérifie les exports nets observés lag24 pour les heures de J
postérieures à 08:00 ; leur disponibilité opérationnelle doit être démontrée.
