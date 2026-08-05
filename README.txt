M1 TARGETED ABLATIONS
=====================

Expériences
-----------
1. M1 + incertitude des forecasts uniquement.
2. M1 + capacités d'interconnexion uniquement.

La base est chronos2_m1_calendar.yaml. Les prix voisins et spreads restent
désactivés dans les deux expériences.

Important : capacités d'interconnexion
--------------------------------------
Les identifiants Saturn ne sont pas présents dans le dépôt. Ils ne sont donc
pas inventés. Renseigne interconnection_series.yaml avant le lancement.

Pour produire une liste de candidats depuis le catalogue Saturn :

python .\discover_saturn_interconnections.py `
    --config .\chronos2_m1_calendar.yaml

Puis ouvre :

saturn_interconnection_candidates.csv

et copie les identifiants pertinents dans interconnection_series.yaml.

Les capacités sont utilisées avec lag24 et lag168, pas en oracle. Cela évite
d'introduire une capacité ex post de J+1 dans le backtest.

Installation
------------
Copier tous les fichiers du paquet à la racine de chronos2_v1, en conservant
le sous-dossier chronos2_modular.

Lancement
---------
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\run_m1_targeted_ablations.ps1 `
    -ProjectRoot "C:\Users\BQ6757\chronos2_v1" `
    -PythonExe (Get-Command python).Source `
    -StartDay "2024-01-01"
