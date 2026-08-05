CHRONOS-2 — INTERACTIONS CALENDAIRES PIT
=========================================

Variables ajoutées
------------------
1. residual_load × morning_peak
2. residual_load × evening_peak
3. residual_load × public_holiday
4. residual_load × weekend
5. residual_load × bridge_day
6. residual_load × neighbour_holiday_count
7. residual_load_after_nuclear
8. residual_load_after_nuclear × morning_peak
9. residual_load_after_nuclear × evening_peak
10. residual_load_after_nuclear × public_holiday

Causalité
---------
Les variables continues utilisent :
- known_fr_residual_load_fcst_oracle
- known_fr_nuclear_generation_fcst_oracle

Dans ce pipeline, "oracle" signifie une prévision explicitement déclarée
connue à l'origine et matérialisée selon les vintages point-in-time. Les
interactions n'utilisent jamais la charge résiduelle ou le nucléaire réalisés
à J+1.

Le nombre de jours fériés voisins est calculé comme la somme des indicateurs
DE, BE, NL et ES. Le jour férié français est exclu de ce compteur.

Installation
------------
Copier tous les fichiers du ZIP à la racine du projet en conservant le
sous-dossier chronos2_modular.

Lancement
---------
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\run_calendar_interactions_ablation.ps1 `
    -ProjectRoot "C:\Users\BQ6757\chronos2_v1" `
    -PythonExe (Get-Command python).Source `
    -StartDay "2024-01-01" `
    -BaseConfig "chronos2_m1_calendar.yaml"

Sortie
------
runs/ablation_m1_calendar_interactions/
    chronos2_m1_calendar_interactions.html

Tests
-----
python -m pytest .\tests\test_calendar_interactions.py -q
