CHRONOS-2 — SCARCITY PRICING LAYER C4

Objectif
--------
Construire :
    P_adjusted = P_LP + ScarcityAdder
puis :
    residual = P_actual - P_adjusted

Chronos prévoit le résidu et le prix final est reconstruit par :
    P_hat = P_adjusted + residual_hat

AUCUN rebuild Saturn ou MILP n'est effectué.
Le patch réutilise :
    data/derived/structural_market_features.csv.gz
et :
    runs/structural_feature_build/fr/aligned_inputs.csv.gz

Colonnes ajoutées
-----------------
milp_structural_price_raw
milp_scarcity_probability
milp_scarcity_adder
milp_structural_price_adjusted

Calibration
-----------
Le split est temporel.
Par défaut, les N derniers jours sont réservés au holdout, avec :
    N = backtest.windows
Les paramètres sont calibrés uniquement avant ce holdout.

Paramètres calibrés :
alpha
tau_gw
beta_ramp
gamma_startup

La fonction objectif est une MAE pondérée :
- poids normal = 1
- poids supplémentaire sur |prix| >= 150 €/MWh
- pénalité de biais

Commandes
---------
Tests :
python -m pytest .\tests\test_structural_scarcity.py -q

Calibration seule :
python .\fit_structural_scarcity_layer.py `
    --config .\chronos2_selected_core_structural_covariates.yaml

Création C4 :
python .\install_structural_scarcity_layer.py `
    --config .\chronos2_selected_core_structural_residual.yaml

Run C4 :
python .\run_chronos2_structural_scarcity_residual.py `
    --config .\chronos2_selected_core_structural_scarcity_residual.yaml `
    --zones FR `
    --local-files-only

Tout en une commande :
.\run_structural_scarcity_layer.ps1

Fichiers de diagnostic :
data/derived/scarcity_layer_calibration.json
data/derived/scarcity_layer_holdout_predictions.csv.gz
runs/structural_scarcity_comparison.csv
