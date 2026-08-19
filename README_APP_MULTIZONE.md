# Forecast Control Room multi-zone

L'application Streamlit fournit un bouton de lancement des forecasts day-ahead
et une vue `Statistics` comparable au dashboard Storm. Elle lit directement
`chronos2_hourly_live_zones.yaml` : un pays n'est sélectionnable que si son
bundle est `enabled`, `production_ready` et validé par tous les garde-fous
locaux de `chronos2_hourly.zone_live.audit_zone_live_bundle`.

Les zones DE, BE, NL ou ES encore incomplètes restent visibles avec leurs
blocages. Elles deviennent automatiquement sélectionnables dès que leur bundle
de production est validé, sans modification de l'application.

Les bundles FR, DE, BE, NL et ES sont désormais activés. DE, BE, NL et ES
utilisent la recette autonome étendue : le challenger MKOnline propre à chaque
pays n'a pas franchi le seuil B1, donc aucun poids français ni fallback externe
n'est réutilisé. Les Statistics scellées sont visibles dès l'ouverture de
l'application, avant même le premier run live.

## Installation

Streamlit 1.58 est installé dans l'environnement `pricefm311`. Si cet
environnement doit être recréé, réinstallez les dépendances avec :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pip install -r 'C:\Users\BQ6757\chronos2_v1\requirements_app.txt'
```

## Lancement

Le moyen le plus simple sous Windows est de double-cliquer sur
`launch_forecast_app.cmd`. Le lanceur utilise l'environnement `pricefm311`,
démarre le serveur local et ouvre l'application dans le navigateur.

L'équivalent PowerShell est :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Start-ForecastApp.ps1'
```

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m streamlit run 'C:\Users\BQ6757\chronos2_v1\app_multizone.py'
```

Dans l'application :

1. vérifier l'état des garde-fous par pays ;
2. choisir un ou plusieurs pays, ou « Tous les pays disponibles » ;
3. choisir le jour de livraison et appuyer sur « Lancer le forecast » ;
4. suivre les logs, puis sélectionner le run ou le rapport à consulter ;
5. sélectionner de 1 à 5 pays dans « Comparaison des derniers forecasts » ;
6. comparer leurs P50 sur un graphe unique et, si nécessaire, afficher les
   bandes P10–P90 ;
7. télécharger le rapport HTML consolidé autonome ;
8. explorer la carte de couplage heure par heure ou sur la journée ;
9. visualiser le forecast P50 mono-pays et son intervalle P10–P90 ;
10. utiliser les filtres Daily, Weekly et Monthly dans `Statistics`.

Les lancements multi-pays sont séquentiels afin d'éviter la concurrence GPU et
les écritures simultanées dans les caches. Chaque job passe exclusivement par
`run_mkonline_live_zone.py`, avec une liste d'arguments et `shell=False`.

## Comparaison multi-pays et rapport consolidé

La comparaison charge uniquement les dernières archives `issued_live`, vérifie
leur identité, leur timeline 23/24/25 heures et tous leurs checksums, puis trace
les P50 sur un axe UTC commun. Les dates de livraison différentes sont refusées
par défaut. Le rapport HTML consolidé est un fichier autonome : le graphe SVG,
les valeurs horaires et la provenance SHA-256 y sont intégrés sans CDN ni
script externe.

## Carte de couplage

La carte représente les frontières réelles entre FR, DE, BE, NL et ES. En
l'absence de flux ou capacité causalement vérifié pour toute la journée, une
flèche représente seulement un **proxy économique** allant du prix P50 prévu le
plus bas vers le plus élevé. Ce proxy n'est ni un flux physique, ni un programme
d'échange, ni une capacité transfrontalière. Le module peut afficher un vrai
signal de flux/capacité uniquement après validation de la frontière, de l'unité
MW, de la couverture horaire, du cutoff et de la provenance.

## Contrat Storm

Storm est strictement un benchmark d'évaluation. L'application ne transmet
aucune série Storm au runner et ne lit Storm que dans le fichier de sortie
`statistics_history_hourly.csv.gz`. La colonne
`storm_dashboard_official__q50` n'est présentée comme « Storm officiel
dashboard » que si `statistics_history_audit.json` l'active explicitement.
La table calcule MAE, RMSE, MAPE, explained variance, R², écart-type et
corrélation, avec un win rate contre Storm pour chaque statistique. La MAPE
utilise la valeur absolue du prix réalisé au dénominateur et ignore uniquement
les observations quasi nulles (|prix| ≤ 1e-9 EUR/MWh).

Pour ES, Saturn ne fournit pas de série native Storm dashboard vérifiée. Le
rapport l'indique explicitement et n'affiche aucun proxy sous le nom Storm
officiel.
