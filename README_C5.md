# C5 — Scarcity as Features

## Hypothèse testée

C4 a amélioré fortement le biais et la précision sur les extrêmes, mais a
dégradé MAE/RMSE et surtout les rampes. L'hypothèse C5 est donc :

> le signal scarcity contient de l'information utile, mais il ne faut pas
> l'imposer directement au niveau du prix.

C5 revient à l'ancre résiduelle de C3 :

    residual_t = actual_price_t - milp_structural_price_t

et ajoute les signaux scarcity comme covariables connues-futures.

## Features

- milp_reserve_pressure
- milp_ramp_pressure
- milp_startup_pressure
- milp_scarcity_probability_feature
- milp_scarcity_adder_feature
- milp_scarcity_probability_ewm
- milp_scarcity_adder_ewm
- milp_scarcity_delta
- milp_reserve_pressure_delta

Les EWMA sont causales (`adjust=False`) et les deltas utilisent uniquement
`t` et `t-1`.

## Pas de rebuild

Le run C5 ne fait aucun :
- refresh Saturn
- backfill PIT
- Branch-and-Cut
- LP pricing

Il réutilise :
- data/derived/structural_market_features.csv.gz
- data/derived/scarcity_layer_calibration.json

## Lancement

    .\run_structural_scarcity_features_experiment.ps1

Le résultat principal est :

    runs/structural_c3_c4_c5_comparison.csv

Le comparateur n'utilise que les métriques primaires et exclut volontairement
`baseline_*`, `*_gain_percent`, les compteurs et les "gains de gains".
