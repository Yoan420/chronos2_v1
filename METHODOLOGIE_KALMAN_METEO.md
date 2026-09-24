# Note méthodologique — Kalman météo en rolling 365 jours

## Objectif

La couche Kalman corrige la médiane du forecast autonome déjà corrigé par le
modèle résiduel. Elle ne remplace ni Chronos-2 ni le correcteur résiduel. Son
rôle est d'apprendre une correction dynamique lorsque l'erreur récente varie
avec les prévisions météorologiques ou renouvelables disponibles avant
l'enchère day-ahead.

## Données utilisées

Pour chaque journée de livraison `D`, les covariables Saturn sont figées à
`D-1 08:00` dans le fuseau civil de la zone :

- température 2 m : `meteo.nrjscan.<zone>.t_2m.index.fcst.d` ;
- production éolienne : `power.<zone>.generation.wind.hourly.gw.fcst` ;
- production solaire : `power.<zone>.generation.solar.hourly.gw.fcst`.

Pour NL, le vent utilise
`power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache`, converti de MW
en GW. Les données portent leur heure de livraison, leur cutoff et leur
origine/révision. Aucune observation météo réalisée, donnée Storm ou valeur
MKOnline n'entre dans l'apprentissage.

La température quotidienne est répétée sur les 23, 24 ou 25 heures physiques
du jour. Le vent et le solaire restent horaires. Aucun trou n'est interpolé.
Les variables dérivées sont déterministes : degrés de chauffage/refroidissement
et rampes vent/solaire.

## Modèle

Les covariables sont standardisées de manière robuste dans chaque fenêtre
d'entraînement. La configuration opérationnelle ajuste trois filtres gouvernés
en parallèle :

- `linear_weather` : température et degrés de chauffage/refroidissement ;
- `linear_renewables` : niveaux et rampes éolien/solaire ;
- `linear_fundamental` : combinaison des deux groupes.

Les familles `ekf_scale`, `ukf_scale`, biais et marché restent disponibles dans
le laboratoire d'expérimentation, mais ne sont pas activées dans ce challenger
opérationnel météo.

La gouvernance compare leurs erreurs sur une période passée et applique un
poids discret uniquement si le gain absolu et relatif minimal est atteint.
Sinon, le forecast autonome reste inchangé. Les coefficients représentent une
association conditionnelle utile à la prédiction, pas une causalité économique.

## Protocole causal et rolling 365

Pour chaque journée évaluée `D`, tous les états du filtre sont recréés depuis
zéro et entraînés sur exactement les journées civiles `D-365` à `D-1`, avec
leurs timelines DST complètes. Le prix observé de `D` n'est jamais assimilé
avant la prédiction de `D`.

Le préfixe manquant du correcteur résiduel est reconstruit de manière
préquentielle : une recette figée est réentraînée par blocs en utilisant
uniquement les labels antérieurs au premier jour du bloc. Lorsque le forecast
`residual_corrected` officiel est déjà publié avec une origine contractuelle
`D-1 08:00`, cette valeur publiée est réutilisée pour conserver une comparaison
directe avec les rapports actuels.

Les hyperparamètres sont sélectionnés sur la validation avec le même rolling
exact. La configuration gagnante est ensuite rejouée séparément sur les 365
derniers jours de test. Les métriques et coefficients sont écrits dans un
rapport HTML autonome, avec audits de fenêtres, couvertures et provenance.

## Statut expérimental hors du mode `All`

`kalman_weather` (clé interne `residual_kalman_weather`) et
`kalman_hybrid` ne font plus partie de `Forecast.ps1 -Action Run -Mode All`.
Ils restent des challengers reproductibles via le laboratoire, avec leurs
propres sorties, rapports et audits. Leurs anciens rapports sont conservés et
aucune de leurs colonnes n'est ajoutée à l'archive live scellée.

Le template modifiable est
`config/kalman_weather_operational.yaml`. Il fixe le rolling 365, les familles
de candidats et les transformations météo, mais pas les checksums quotidiens.
Le pipeline expérimental dédié suit l'ordre causal suivant :

1. prolonger les trois Parquet Saturn de chaque pays jusqu'au jour livré ;
2. vérifier le préfixe `residual_corrected` construit par refits préquentiels ;
3. générer un sidecar runtime par pays avec les SHA-256 courants ;
4. rejouer les 365 origines, chacune entraînée sur exactement D-365..D-1 ;
5. produire le CSV et le rapport HTML FINAL365 dans `runs/exports`.

Le mode `All` publie désormais exactement deux chaînes par pays :
`autonomous`, soit Chronos-2 + LoRA + correcteur résiduel, et `kalman`, soit
la même base suivie du Kalman standard. Il n'inclut pas le blend ; MKOnline
reste accessible séparément avec `Blend` pour FR et NL. Pour `-Action Run`,
`Both` produit les deux vues autonome/Kalman sur l'incumbent lorsque LoRA
est inactif, ou sur LoRA promu et activé. `All` est fail-closed et
refuse le batch tant qu'un bundle LoRA final promu, épinglé et couvert par les
preuves PIT/shadow requises n'est pas activé pour les deux chaînes de chaque
pays demandé. Il ne retombe jamais silencieusement sur l'incumbent.

La calibration `q_over_r=0,001` vient du test FR ; les autres pays restent
explicitement en évaluation shadow et ne sont pas promus automatiquement.

## Banque gouvernée marché–météo–combustibles

`kalman_hybrid` est un challenger distinct, également appliqué après le
correcteur résiduel autonome. Pour chaque journée `D`, ses états sont recréés
depuis zéro sur les 365 jours civils complets `D-365` à `D-1`. La banque
évalue onze candidats : biais lent, profil harmonique, charge résiduelle
européenne, météo, renouvelables, fondamentaux météo-énergie, gaz/CO2,
marché+météo, marché+météo+combustibles, échelle linéaire et UKF d'échelle
bornée. L'identité — aucune correction Kalman supplémentaire — reste un
garde-fou séparé et toujours disponible.

La gouvernance utilise les 90 derniers jours disponibles dans chaque fenêtre.
Les 76 premiers choisissent un candidat et un poids de contraction entre 0 et
1, par pas de 0,05. Le même couple candidat-poids, sans réoptimisation, doit
ensuite confirmer son gain sur les 14 jours les plus récents. Il n'est activé
que si l'amélioration de MAE atteint, séparément sur la sélection et la
confirmation, le maximum de 0,10 EUR/MWh et de 1 % de la MAE de l'identité.
Dans tous les autres cas, la correction est nulle. La correction brute est
bornée à +/-20 EUR/MWh et le même déplacement est appliqué aux trois quantiles,
ce qui conserve leur ordre et leur largeur.

Les entrées supplémentaires sont strictement point-in-time : prévisions de
charge résiduelle FR/DE/BE/NL/ES, température, vent et solaire Saturn au cutoff
civil `D-1 08:00`, ainsi que les dernières observations connues de TTF M1 et
d'EUA première échéance décembre, leurs variations à 1/5 jours et leur
volatilité sur 20 jours. Aucun prix futur de combustible, aucune météo
observée, aucune prévision Storm ou MKOnline n'est utilisée. À chaque run, les
Parquet, audits et préfixe préquentiel `residual_corrected` sont copiés dans un
bundle de sources immuable, adressé par SHA-256, puis le sidecar opérationnel
référence exactement ce bundle. La source PIT commune des cinq charges
résiduelles remplace autoritativement les cinq colonnes natives sur les heures
qu'elle couvre : elles ne sont jamais mélangées avec des vintages d'archives
différents. Les écarts, recouvrements et remplacements sont conservés par pays
dans l'audit.

La formule Saturn de charge résiduelle NL est horodatée sur une grille naïve
UTC. Lors du passage à l'heure d'été, son état point-in-time omet de façon
reproductible les deux produits correspondant à 04:00 et 06:00 locales. Ces
deux valeurs seulement sont reconstruites par moyenne des voisins UTC immédiats
du même forecast déjà disponible au cutoff. La règle exige une journée physique
de 23 heures, exactement ces deux trous, aucun doublon ou point supplémentaire
et des voisins finis ; toute autre signature échoue. L'audit versionné liste
chaque heure réparée, ses deux donneurs et la méthode. Aucune observation ni
révision ultérieure n'est utilisée.

La banque ne moyenne pas simultanément les onze modèles : elle choisit au plus
un candidat par jour et contracte sa correction. Ses coefficients mesurent une
association adaptative avec l'erreur résiduelle, pas un effet causal ni une
importance structurelle. La confirmation séparée réduit le surajustement lié
au nombre de candidats sans l'éliminer ; toute comparaison doit donc rester
appariée sur les mêmes 365 jours et les mêmes observations.

## Résultat FR au 28 août 2026

Le test scellé couvre 365 jours et 8 760 heures physiques, du 29 août 2025 au
28 août 2026. `q_over_r=0,001` est sélectionné sur les 60 jours de validation.
La comparaison ci-dessous utilise comme baseline le forecast
`residual_corrected` effectivement publié et fourni au Kalman :

| Métrique test | Baseline résiduelle | Kalman météo | Gain |
| --- | ---: | ---: | ---: |
| MAE (EUR/MWh) | 12,5903 | 12,4990 | +0,0913 (-0,72 %) |
| RMSE (EUR/MWh) | 20,1876 | 20,0297 | +0,1579 |
| Biais (EUR/MWh) | -1,1679 | -0,8132 | +0,3546 en valeur absolue |

Sur les 365 jours, la gouvernance retient les renouvelables 192 jours, la
température 79 jours, l'ensemble fondamental 55 jours et conserve l'identité
39 jours. Une correction non nulle est donc appliquée 326 jours. Le signal est
présent, mais le Kalman opérationnel actuel reste meilleur sur la même fenêtre
(MAE 12,3862 EUR/MWh). La variante météo demeure donc un challenger shadow et
n'est pas promue automatiquement.

Les 14 coefficients finaux publiés servent à expliquer l'état du dernier
refit. Ils sont standardisés, fortement dépendants des colinéarités
température/HDD/CDD et niveau/rampe, et ne doivent pas être lus comme des effets
causaux isolés.

## Limites

La température Saturn est un indice national quotidien : elle décrit le niveau
thermique mais pas sa forme intrajournalière. L'option Open-Meteo/ECMWF apporte
des variables horaires supplémentaires, mais elle reste expérimentale et doit
utiliser un endpoint/licence compatible avec l'usage commercial. L'intégration
opérationnelle d'une nouvelle source n'est promue qu'après un gain hors
échantillon et une couverture PIT complète.

Le matérialiseur compact Saturn archive le cutoff explicite de la requête as-of
dans `snapshot_time_utc` et `revision_time_utc`. Il ne reçoit pas, via cet
endpoint, l'horodatage d'insertion fournisseur de la révision servie ; cette
limite forensique est déclarée dans chaque sidecar d'audit.
