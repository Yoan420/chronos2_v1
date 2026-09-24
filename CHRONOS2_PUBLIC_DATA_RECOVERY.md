# Sources publiques pour la reprise Chronos-2 / LoRA

La recherche porte sur les prévisions connues à J−1 08:00 Europe/Paris,
pour les journées de livraison nécessaires aux 365 jours d'entraînement,
365 jours de calibration hors échantillon et 365 jours d'évaluation finale.
Les modèles rang 8 et rang 16 déjà évalués restent les références de comparaison.

## Archive météo retenue pour le premier essai

[NOAA GFS sur AWS](https://registry.opendata.aws/noaa-gfs-bdp-pds/) fournit les
cycles opérationnels d'origine, accessibles sans compte. Le cycle 00 UTC de
J−1 est retenu si la date de dépôt de chaque fichier est antérieure au cutoff.
Le bucket public est `noaa-gfs-bdp-pds`. Exemple de l'index sondé avec succès :

`gfs.20230907/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx`

Les sondages de métadonnées couvrent des exemples de 2023, 2024, 2025 et 2026.
La couverture quotidienne complète doit encore être vérifiée par le téléchargement.

Les premières variables à extraire sont la température à 2 m, les composantes
du vent à 100 m et le rayonnement solaire descendant. Les messages nécessaires
sont sélectionnés dans l'index et téléchargés par plages d'octets. Une même
grille sert aux quatre pays FR, DE, BE et NL. Les agrégats géographiques
utilisent les points déjà déclarés dans le laboratoire météo.

Le rayonnement GFS est une moyenne sur un intervalle pouvant couvrir plusieurs
heures. Sa conversion en moyenne horaire doit respecter les bornes de cet
intervalle. Pour une livraison horaire [h,h+1), la température et le vent sont
pris à h et le rayonnement couvre [h,h+1). Les jours de changement d'heure
doivent conserver leurs 23 ou 25 heures physiques.

L'archive conserve les messages bruts, leurs empreintes, l'initialisation du
modèle, les échéances, les dates de dépôt et la transformation appliquée. La
preuve de dépôt historique et une capture prospective locale restent deux
informations distinctes. L'extracteur ne prend aucune décision de promotion.

Source et attribution : NOAA / NCEP, GFS 0,25° ; décodage avec ECMWF ecCodes.
[Inventaire officiel des produits GFS](https://www.nco.ncep.noaa.gov/pmb/products/gfs/).

## Séries électriques examinées

| Source | Utilisation envisageable | Point à résoudre |
|---|---|---|
| RTE consommation D−2 | Prévision de charge FR, avec `updated_date` contrôlé | Authentification RTE et vérification de chaque publication ; aucun identifiant disponible dans les variables d'environnement examinées |
| RTE prévisions de production | Prévisions éoliennes/solaires aux horizons admissibles | Horizon disponible selon la filière, historique et horodatages à vérifier |
| ENTSO-E charge et production prévisionnelle | Comparaison et exploration | Le produit éolien/solaire day-ahead est figé à 18:00 ; il ne prouve pas une disponibilité à 08:00 |
| NED / TenneT | Prévisions NL récentes | Historique recalculable ; prévision de charge ajoutée en novembre 2025, sans archive de cycles documentée pour 2023 |

Sources : [RTE consommation](https://data.rte-france.org/web/guest/catalog/-/api/doc/user-guide/Consumption/1.2),
[RTE production](https://data.rte-france.org/web/guest/catalog/-/api/doc/user-guide/Generation%2BForecast/3.1),
[ENTSO-E production day-ahead](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648445340180-Generation-Forecasts-for-Wind-and-Solar-14-1-D),
[NED changelog](https://ned.nl/nl/changelog-api).

Open-Meteo dispose d'archives utiles, mais ses cycles complets ECMWF commencent
en mars 2024. L'accès gratuit ne couvre pas la recherche interne non publiée
d'une entreprise. Aucun abonnement ni compte n'a été créé.
[Couverture](https://open-meteo.com/en/docs/historical-forecast-api),
[conditions d'utilisation](https://open-meteo.com/en/terms).

## Conséquence pour le modèle

Le vent en m/s et le rayonnement en W/m² deviennent de nouvelles variables.
Ils ne portent pas les noms des prévisions Saturn de production en GW.
Une reconstruction de la charge résiduelle exige des modèles de charge et de
production entraînés uniquement sur des données antérieures à chaque origine.
L'autre piste est un adaptateur LoRA consommant directement les variables météo
publiques avec les autres fondamentaux disponibles.

Ces deux pistes définissent de nouveaux candidats : leurs schémas, données,
entraînements et calibrations doivent être distincts. Les résultats rang 8 et
rang 16 existants ne peuvent pas être attribués à ces nouvelles entrées.
Leur évaluation conservera une fenêtre commune de 365 jours et sera suivie,
si les preuves sont admissibles, d'une nouvelle période shadow prospective.

## Validation réalisée le 7 septembre 2026

Le connecteur `auxiliary_lab/noaa_gfs.py` et la commande autonome
`materialize_noaa_gfs_weather.py` ont été exécutés sur les archives réelles.
Les quatre jeux sont conservés dans
`runs/experiments/chronos2_exogenous_public_weather_v1/samples/`.

| Livraison | Heures physiques | Dernière publication contrôlée (UTC) | Cutoff (UTC) |
|---|---:|---|---|
| 2023-09-08 | 24 | 2023-09-07 03:46:22 | 2023-09-07 06:00 |
| 2024-03-31 | 23 | 2024-03-30 03:53:55 | 2024-03-30 07:00 |
| 2024-10-27 | 25 | 2024-10-26 03:47:22 | 2024-10-26 06:00 |
| 2026-09-06 | 24 | 2026-09-05 04:01:21 | 2026-09-05 06:00 |

Chaque jeu contient trois variables par pays pour FR, DE, BE et NL. Les 16
ingestions pays/journée ont également été vérifiées avec la banque d'entrées
Chronos existante : complètes, zéro violation de causalité, statut recherche.
Les SHA du Parquet, du manifeste brut et de chaque message GRIB ont été
revérifiés sur disque. Les deux changements d'heure conservent bien 23/25 heures.
Les quatre journées représentent environ 300 Mo de messages bruts au total.
Cela ne constitue pas encore une vérification de couverture quotidienne sur
365 ou 1 095 jours ; la collecte complète n'a pas été lancée.

Validation automatisée : **80 tests réussis** le 7 septembre 2026 :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest `
  tests/test_noaa_gfs_weather.py tests/test_noaa_gfs_decoding.py `
  tests/test_exogenous_feature_bank.py tests/test_exogenous_pit_evidence.py -q
```

Ces tests couvrent notamment les intervalles solaires, les valeurs manquantes,
les bornes physiques, les coordonnées, la publication avant cutoff, les hashes,
la reprise du cache sans réseau, les sorties concurrentes et l'absence de
promotion implicite. Le téléchargement est parallèle ; le décodage ecCodes
reste séquentiel pour éviter les arrêts natifs constatés sous Windows.

## Utilisation et reprise

Les dépendances optionnelles sont figées dans `requirements_public_weather.txt`.
Elles ont été installées dans un répertoire d'expérience isolé ; l'environnement
opérationnel n'a pas été modifié. Depuis la racine du projet, exemple d'une
nouvelle sortie réutilisant intégralement le cache déjà contrôlé :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  '.\materialize_noaa_gfs_weather.py' `
  --start-day 2023-09-08 --end-day 2023-09-08 `
  --zones FR DE BE NL --workers 2 `
  --output '.\runs\experiments\chronos2_exogenous_public_weather_v1\samples\2023-09-08_recheck.parquet' `
  --cache-dir '.\runs\experiments\chronos2_exogenous_public_weather_v1\raw_cache' `
  --dependency-directory '.\runs\experiments\chronos2_exogenous_public_weather_v1\deps'
```

Adapter `--start-day`, `--end-day` et `--output` pour une nouvelle période.
Utiliser des chemins de sortie distincts pour les expériences : les résultats
existants ne sont jamais écrasés. `--ca-bundle` accepte uniquement le chemin
d'un vrai certificat local si le proxy le nécessite ; TLS reste vérifié.

En cas d'erreur Python, les réservations sont nettoyées et le cache brut permet
la reprise. Après une interruption brutale du processus, un `.publish.lock`
ou un Parquet sans manifeste peut subsister. Ne pas supprimer une réservation
d'un processus encore actif : choisir un **nouveau `--output` avec le même
`--cache-dir`**, puis vérifier le manifeste final. Un Parquet seul n'est pas un
résultat achevé. Aucun nettoyage automatique des résultats antérieurs n'est fait.

Les configurations opérationnelles, les checkpoints rang 8/rang 16 et leurs
rapports de comparaison n'ont pas été modifiés par cette intégration. Avant
une collecte complète et un nouvel entraînement, il reste à choisir le candidat
qui utilisera ces variables publiques et à figer son protocole d'évaluation.
