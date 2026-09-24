# P50 cohérent avec le risque de forte hausse

Laboratoire indépendant : aucune modification de Forecast.ps1, des prévisions
actives, des anciens laboratoires ou des rapports publiés sous runs/exports.

## Pourquoi ce changement

Le précédent POC pouvait estimer une probabilité supérieure à 50 % d'une erreur
NYX supérieure à 50 EUR/MWh, tout en prédisant une médiane d'erreur négative.
Deux estimateurs indépendants donnaient des conclusions incompatibles.

Cette version conserve **exactement les probabilités calibrées du détecteur figé**
et remplace uniquement l'estimation de distribution/amplitude. Elle n'est pas
une nouvelle calibration opportuniste des probabilités sur les pics observés.

## Distribution d'erreur

Pour e = observé − NYX et le seuil u appris sur le core du pays :

`F(e|X) = (1-p) F_normal(e|e<u,X) + p F_spike(e|e>=u,X)`.

Les erreurs sont normalisées par u pour partager l'estimation entre pays. Deux
forêts estiment les distributions conditionnelles des régimes normal et extrême.
Le régime normal conserve aussi les erreurs négatives et nulles : il n'est pas
remplacé par une masse artificielle à zéro. Les supports sont strictement séparés.

Le P50 d'erreur est l'inverse généralisée à gauche de cette distribution :

- p ≤ 0,5 : quantile `0,5/(1-p)` du régime normal ;
- p > 0,5 : quantile `1-0,5/p` du régime extrême.

Ainsi, p > 0,5 implique un P50 brut d'erreur au moins égal à u. Quand p reste
inférieur à 0,5, le P50 peut remonter dans le régime normal, mais ne doit pas être
forcé dans la queue des spikes. Les probabilités zéro/un et les atomes sont
traités explicitement ; pas d'interpolation linéaire entre observations pour
inverser la CDF, pas de moyenne des médianes, pas de formule p × sévérité.

Les poids des observations sont la moyenne des poids uniformes dans les feuilles
des arbres. Chaque régime utilise 48 arbres, profondeur maximale 6,
feuille minimale 20 observations, max_features 0,7, sans bootstrap, seed 1729.
L'imputation médiane est apprise sur le core uniquement. Les prix électriques,
NYX, ses quantiles et Storm ne sont pas des variables explicatives des forêts.

Cette construction s'inspire des [Quantile Regression Forests de Meinshausen,
JMLR 2006](https://jmlr.org/papers/v7/meinshausen06a.html). Notre version finie,
à deux régimes et profondeur fixée, ne revendique pas une garantie automatique
de calibration ou les résultats asymptotiques du papier.

## Décision et quantiles finaux

Une proposition positive nécessite la porte physique du POC précédent, une
probabilité supérieure à la prévalence passée du pays et une médiane positive.
Le seuil de 50 % n'est pas un réglage optimisé sur les dates étudiées : c'est
une propriété structurelle de la médiane du mélange.

- `forest` : proposition directe, candidat principal expérimental déclaré avant le nouveau backtest.
- `forest_governed` : même proposition, gouverneur quotidien existant sur les 90 jours précédents.
- `empirical` / `empirical_governed` : témoins avec distributions empiriques par pays/régime,
  sans forêt ; pool normalisé régional si moins de 10 observations dans un régime du pays.

Les témoins utilisent les mêmes probabilités, périodes, labels et règles de décision.
Leurs résultats ne déclenchent aucun changement automatique de candidat ou de pays.

Pour garder P10 ≤ P50 ≤ P90 après plafond et gouverneur :

`Q_final(a) = (1-w) Q_NYX(a) + w [P50_NYX + min(Q_erreur(a), 400)]`.

C'est une **interpolation des fonctions quantiles**, pas un mélange arithmétique
des CDF de NYX et de l'expert. w vaut 1 en direct et appartient à {0, 0,25, 0,5, 1}
en gouverné. Il vaut zéro quand les portes refusent la correction ou que la médiane
n'est pas positive. Dans ce cas les trois quantiles NYX sont conservés exactement.

Les rapports distinguent le P50 brut et final. La probabilité affichée appartient
au détecteur/expert brut : elle n'est pas présentée comme la probabilité de la
distribution finale après plafonnement et interpolation. L'ordre des quantiles
est garanti par construction ; leur calibration future ne l'est pas.

## Rejouer, reprendre et afficher

```powershell
& 'C:\Users\BQ6757\chronos2_v1\CoherentP50.ps1' -Action Run
& 'C:\Users\BQ6757\chronos2_v1\CoherentP50.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\CoherentP50.ps1' -Action Backtest
& 'C:\Users\BQ6757\chronos2_v1\CoherentP50.ps1' -Action Report
```

Run prépare un nouveau snapshot et lance les deux replays. Backtest reprend le
dernier snapshot préparé en réutilisant les workers terminés et vérifiés. Report
ne refait aucun entraînement ni actualisation des sources. `-RunDirectory` cible
explicitement un snapshot pour Backtest, Report ou Status ; `-DryRun` ne calcule rien.

Ce launcher rejoue les dates présentes dans le détecteur figé. Il ne produit pas
encore une nouvelle date de forecast absente de ce snapshot : celle-ci nécessite
d'abord l'inférence du détecteur à la bonne origine. Ce laboratoire ne remplace
donc pas la commande opérationnelle Forecast.ps1.

Configuration : config/nyx_coherent_p50.yaml. Code : nyx_coherent_p50/.
Résultats : runs/experiments/nyx_scarcity_v1/coherent_p50/snapshots/.
Les deux ensembles de rapports de production-format sont reports/ (direct)
et reports_governed/ ; coherent_p50_comparison.html compare toutes les variantes.

## Protocole et limites

Même snapshot que le POC précédent : 365 jours du 15/09/2025 au 14/09/2026,
8 735 heures appariées à Storm par pays. Livraison du 15/09 affichée mais exclue
des Statistics. Aucun remplissage d'heures Storm manquantes, aucun décalage DST.

Les CDF sont ajustées exactement sur le core historique du détecteur, hors ses
28 jours de calibration et avec vérification des dates de disponibilité des labels.
La fenêtre est plafonnée à 365 jours avec entraînement hebdomadaire et amorçage
progressif ; ce panel n'offre pas 365 jours de calibration avant chaque jour évalué.

Année déjà examinée : résultats exploratoires, pas validation prospective indépendante.
Les proxys de parc restent incomplets, les températures sont lacunaires après
le 4 septembre et aucun domaine JAO qualifié à 08 h n'est ajouté. Les requêtes
as-of ne prouvent pas toutes les publications historiques. Les corrections
restent uniquement haussières et ne réparent pas les surestimations du 25 juin.

Sources, sorties et code sont scellés par SHA ; les modifications après Prepare
interdisent la reprise d'un fit partiel avec un autre code. Aucun pickle externe
n'est chargé. Les modèles locaux doivent passer par runner.load_model.

## Résultats du replay du 14 septembre 2026

Snapshot : `20260914T160104Z_d7366f80`. Scores sur les mêmes 34 940 heures
appariées à Storm (4 pays × 8 735 heures), du 15/09/2025 au 14/09/2026.
Aucun réglage n'a été changé après lecture des résultats.

| MAE (EUR/MWh) | NYX initial | Storm | Forêt directe | Forêt gouvernée | Témoin empirique direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ensemble | 11,36750 | 11,31759 | 11,34645 | 11,35394 | 11,33371 |
| BE | 11,15039 | 10,94667 | 11,10909 | 11,12666 | 11,11147 |
| DE | 11,07819 | 11,45784 | 11,03426 | 11,06117 | 11,03180 |
| FR | 12,07889 | 11,85856 | 12,10817 | 12,07889 | 12,07294 |
| NL | 11,16253 | 11,00728 | 11,13428 | 11,14903 | 11,11864 |

Le gain annuel de la forêt directe reste faible : 0,02105 EUR/MWh (environ
0,19 %). L'intervalle exploratoire à 95 %, groupé par journée, du gain de MAE
est [-0,02190 ; +0,06400] : il comprend zéro. Ce résultat ne démontre donc pas
une amélioration robuste. Hors les dates déjà identifiées du 24–26 juin et du
14 septembre, le gain n'est plus que 0,00611 EUR/MWh.

Sur les 1 % d'heures de prix observé les plus élevés du support commun
(seuil calculé ex post 232,37 EUR/MWh, 352 heures à cause des égalités), la MAE
passe de 89,17 à 85,22 EUR/MWh, soit environ -4,43 %. Storm reste meilleur
sur ce sous-ensemble à 66,92 EUR/MWh. Cette sélection ex post sert uniquement
à l'analyse, jamais à déclencher les corrections.

La forêt directe corrige 779 heures : 440 améliorées, 339 aggravées, dont
300 hausses ajoutées alors que NYX surestimait déjà le prix. Elle améliore
BE/DE/NL mais dégrade FR. Le gouverneur laisse FR inchangé et réduit le gain
global. Le témoin empirique obtient une meilleure MAE annuelle que la forêt :
la valeur ajoutée de l'amplitude dépendant des fondamentaux, au-delà de leur
utilisation par le détecteur, n'est pas démontrée ici.

### Le P50 remonte effectivement le 14 septembre à 19 h

| Pays | NYX initial | Forêt directe | Observé | Storm |
| --- | ---: | ---: | ---: | ---: |
| BE | 286,78 | 339,16 | 441,74 | 370,58 |
| DE | 324,59 | 389,02 | 697,31 | 421,70 |
| FR | 284,41 | 291,28 | 298,00 | 296,47 |
| NL | 307,18 | 326,32 | 400,00 | 412,32 |

La contradiction entre probabilité de pic et médiane est résolue, mais pas
toute la sous-estimation des pics. Le 24 juin à 19 h reste notamment mal
détecté : correction DE +2,33, NL +3,36 EUR/MWh et aucune correction BE
(porte physique fermée). Relever mécaniquement le P50 ne remplace pas un
meilleur signal de risque quand la probabilité estimée reste faible.

### Intervalles : point de vigilance avant toute utilisation opérationnelle

P10 ≤ P50 ≤ P90 est respecté, mais la couverture observée n'atteint pas
les 80 % nominaux. Sur toutes les heures, elle est de 74,46 % pour la forêt
directe, contre 74,53 % pour NYX. Sur les 779 heures corrigées, elle passe
de 68,42 % (NYX sur ces mêmes heures) à 65,21 %. Le P10 y est dépassé vers
le bas trop souvent : 17,72 % des observations sont au-dessous ou égales
au P10, au lieu des 10 % visés. Cohérence des quantiles et calibration
sont deux propriétés différentes.

Conclusion : conserver ce laboratoire hors production. Les priorités sont
la calibration chronologique des intervalles sur les interventions, un
signal physique plus informatif pour les événements mal détectés et une
validation sur de nouvelles journées. Le choix d'une variante ou d'une règle
par pays après lecture de cette année serait une nouvelle hypothèse à tester,
pas une validation indépendante.

Audit chiffré reproductible : `independent_evaluation_audit.json` dans le snapshot.
Tests automatisés du laboratoire : 124 réussis, 1 ignoré (permission Windows
nécessaire pour créer un lien symbolique).
