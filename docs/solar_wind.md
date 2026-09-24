# SolarWind : solaire CWE + éolien DE/NL

Expérience séparée : aucun changement à `Forecast.ps1`, aux recettes de production,
à SolarCWE, à SolarCorrection ou au laboratoire CatBoost RMSE en cours.

## Modèles

On conserve la recette nucléaire, les cinq charges résiduelles et les quatre
prévisions solaires FR/DE/BE/NL. Deux entrées supplémentaires sont ajoutées :

- `power.de.generation.wind.hourly.gw.fcst`
- `power.nl.generation.wind.hourly.gw.fcst`

Les séries solaires DE/NL restent exactement
`power.de.generation.solar.hourly.gw.fcst` et
`power.nl.generation.solar.hourly.gw.fcst`.
Toutes ces prévisions horaires en GW entrent dans le contexte et l'horizon futur
de Chronos-2, le correcteur résiduel et les covariables du Kalman.
La série néerlandaise générique reste la source principale. Seules les deux
heures manquantes autorisées ci-dessous peuvent utiliser un secours ECMWF.
On ne retranche pas à nouveau le solaire ou l'éolien de la charge résiduelle.
Les hyperparamètres restent ceux de la référence nucléaire : ce test ne change
ni la perte du correcteur, ni son plafond, ni les paramètres du Kalman.

Deux rapports par pays, au format habituel, sont produits :
`solar_wind_autonomous` et `solar_wind_kalman`. Comparaison avec le modèle
nucléaire de même famille et Storm, sur des observations appariées. Cette
comparaison contre le nucléaire mesure l'ajout solaire + éolien ensemble ; elle
ne constitue pas à elle seule une ablation isolant l'éolien contre solaire seul.

## Lancement et reprise

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarWind.ps1' -Action Run -DeliveryDay 2026-09-22 -Zones DE,NL -Threads 2 -Workers 2
```

Le premier lancement recalcule l'historique Chronos avec les nouvelles entrées,
puis le correcteur et le Kalman. Les checkpoints solaires seuls ne sont pas
compatibles. Les 365 jours de backtest du rapport sont précédés de l'historique
de calibration de la référence nucléaire ; ils ne sont pas remplacés par les
scores d'un ancien modèle. Relancer la même commande reprend les checkpoints
compatibles après une interruption, sans interrompre les autres expériences.

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarWind.ps1' -Action Status -DeliveryDay 2026-09-22 -Zones DE,NL
& 'C:\Users\BQ6757\chronos2_v1\SolarWind.ps1' -Action Report -DeliveryDay 2026-09-22 -Zones DE,NL
```

`Audit` vérifie les références existantes sans réseau ni entraînement.
`Sync` prépare les sources dans `data/pit/solar_wind_v1`.
`Prepare` fige les sources et les références sans entraîner les modèles.
`Report` exige les calculs terminés et ne refait aucun entraînement.
Les paramètres sont dans `config/solar_wind.yaml` ; seuls DE et NL sont lancés.

Résultats : `runs/experiments/solar_wind_v1/2026-09-22/`.
L'index final s'appelle `solar_wind_DE_NL_index.html`.
Les caches, modèles, statistiques et rapports opérationnels restent intacts.

## Limites et traçabilité

La journée du 22 septembre a motivé le test après lecture des prix du marché.
Il s'agit donc d'une reconstitution rétrospective, pas d'une preuve prospective
ni d'une autorisation de promotion. Une baisse simultanée du solaire/éolien peut
être informative, mais ne démontre pas seule la cause du spike ni que le modèle
saura en prévoir l'amplitude.

Les prévisions sont demandées à Saturn as-of J-1 08 h, heure civile. Ce timestamp
de requête ne certifie pas la publication d'origine, que l'API ne fournit pas.
Au changement d'heure d'automne, les formules civiles peuvent retourner une
seule valeur pour l'heure répétée. La politique explicite `wind_dst_policy:
duplicate` autorise seulement sa répétition sur les deux heures physiques,
avec audit des dates et valeurs concernées. Elle n'autorise ni interpolation
des autres trous ni imputation des observations. `raise` permet de refuser
ces heures ambiguës au lieu de les répéter. Aucun zéro nocturne n'est supposé
pour l'éolien. Les prix observés de livraison ne servent qu'à l'évaluation,
jamais à sa prévision ; les observations indisponibles restent vides.

### Exception NL approuvée : deux heures de printemps

La politique explicite `wind_gap_policy: nl_ecmwf_spring_2025_2026` autorise
uniquement les instants **2025-03-30 02:00 UTC** et **2026-03-29 02:00 UTC**
(04 h locale NL). Il s'agit bien d'heures physiques existantes : la composante
Meteologica de la formule Saturn générique manque de ces deux valeurs.
Si la valeur native est absente, on utilise la prévision ECMWF NL disponible
au **même cutoff J-1 08 h locale**, convertie de MW en GW. Ce n'est ni une
interpolation, ni un remplacement de toute la courbe par ECMWF.

Toute valeur native valide reste prioritaire. Tout autre trou reste bloquant.
Les heures, séries, cutoff, unités et valeurs de chaque substitution figurent
dans les audits journaliers, l'audit global des sources et les rapports HTML.
La politique `null` interdit ces substitutions. Les journées natives déjà
téléchargées restent inchangées et réutilisables lors de la reprise.
