# Exogènes météo causales — Open-Meteo Previous Runs

Le matérialiseur `materialize_openmeteo_previous_runs.py` télécharge uniquement
des prévisions météo à lead fixe, sans prix, sans actuals météo et sans Storm.
Le modèle est épinglé à `ecmwf_ifs025` et le lead par défaut à 48 h
(`*_previous_day2`). Pour le cutoff day-ahead **D-1 08:00 Europe/Paris**, le
lead 24 h n'est pas sûr pour les heures tardives de D et est donc refusé.

Variables : température à 2 m, vent à 100 m et rayonnement solaire horizontal,
sur huit points répartis en France. Les points et des agrégats non pondérés
(moyenne, écart-type, minimum, maximum) sont conservés. Aucune pondération
n'est apprise sur une période d'évaluation.

Le calendrier est construit en UTC depuis les jours civils Europe/Paris : les
journées DST ont exactement 23 ou 25 heures. Toute heure ou valeur manquante
fait échouer le run. Il n'y a ni interpolation, ni resampling, ni remplissage.

## Contrôle sans réseau

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  .\materialize_openmeteo_previous_runs.py `
  --start-day 2024-08-12 --end-day 2025-08-11 `
  --output .\runs\tmp\openmeteo_d2\weather.parquet `
  --allow-large-range --dry-run
```

Pour 365 jours, `--chunk-days 14` planifie 27 requêtes multi-localisations
(8 points × 3 variables, environ 8 760 lignes). Selon la latence de l'API,
prévoir typiquement **5 à 15 minutes** et quelques dizaines de Mo au maximum
avec les JSON bruts. Cette commande ne télécharge rien grâce à `--dry-run`.

## Matérialisation

Après validation de la période de calibration (ne pas inclure le holdout final)
et des droits d'usage :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  .\materialize_openmeteo_previous_runs.py `
  --start-day 2024-08-12 --end-day 2025-08-11 `
  --output .\runs\tmp\openmeteo_d2\weather.parquet `
  --raw-dir .\runs\tmp\openmeteo_d2\raw `
  --allow-large-range
```

Le Parquet, le manifeste JSON, son sidecar `.sha256`, les paramètres sans clé,
le SHA-256 de chaque réponse et les coordonnées de grille résolues sont écrits
atomiquement. Une clé éventuelle se passe via `--api-key-env` et n'est jamais
écrite dans le manifeste.

## Provenance et licence

- Documentation : <https://open-meteo.com/en/docs/previous-runs-api>
- Licence des données : CC BY 4.0, attribution requise :
  <https://open-meteo.com/en/license>
- Conditions : <https://open-meteo.com/en/terms>

Le endpoint public gratuit est annoncé pour l'évaluation non commerciale.
Avant une utilisation de production chez une entreprise, confirmer un plan
commercial Open-Meteo approprié ou les droits d'une instance auto-hébergée.

