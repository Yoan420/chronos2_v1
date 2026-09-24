# CWE PnL : Quantile Based et Unlimited Bid

Le rapport utilise désormais uniquement l'historique complété et vérifié. Les
onglets « Cache only » et « Completed history » sont remplacés par **Quantile
Based** et **Unlimited Bid**. Les anciennes simulations 1 MW / 4 MWh et les
PnL VPS publiés/estimés ne participent plus aux tableaux affichés.

## Règles communes

Référence : Lipiecki et Weron, *Foundation models for electricity price
forecasting and battery arbitrage: Can they replace market-specific forecasting
models?*, arXiv:2609.00089v1, §6.2, pages 15–17.

- Batterie : 1 MWh de capacité interne, initialement vide, un cycle au maximum
  par journée ; achat avant vente, batterie vide après le cycle.
- Rendement : 0,95 à la charge et 0,95 à la décharge, soit 0,9025 aller-retour.
  Achat réseau : 1/0,95 MWh ; vente réseau : 0,95 MWh.
- Coût : 25 euros par cycle effectivement exécuté, incluant les coûts
  opérationnels/dégradation du scénario de l'article ; aucun coût si pas de
  transaction ou si le loop bid est rejeté.
- Les heures sont choisies **avant d'utiliser le réalisé**, parmi toutes les
  paires `achat < vente`, en maximisant :

  `profit prévu = 0,95 × prévision_vente − prévision_achat / 0,95 − 25`.

- Aucune proposition si le profit prévu maximal est inférieur ou égal à zéro.
  En cas d'égalité entre paires optimales, première heure d'achat puis première
  heure de vente : règle déterministe sans information future.
- Le profit exécuté est calculé sur les prix observés, avec les mêmes volumes
  et le même coût. Une perte réalisée reste négative.

**Unlimited Bid** exécute systématiquement chaque paire proposée, sans limite
de prix. « Unlimited » ne signifie pas une capacité ou un nombre de cycles
illimités.

**Quantile Based**, à `alpha = 80 %`, utilise P90 de l'heure d'achat comme prix
limite d'achat et P10 de l'heure de vente comme prix limite de vente. La paire
est exécutée ensemble ou rejetée ensemble. C'est l'un des cinq niveaux du
papier ; les autres exigent des quantiles NYX non archivés. Aucun P05/P95,
P15/P85, P20/P80 ou P25/P75 n'est interpolé ou inventé.

## Convention explicite pour les loop bids

L'article décrit le caractère tout-ou-rien, mais ne fournit pas l'inégalité
exacte de sa simulation de clearing, ni de code de réplication identifié.
Le rapport retient une simulation preneuse de prix fondée sur le surplus
global du loop :

`(limite_achat − réalisé_achat) / 0,95 + 0,95 × (réalisé_vente − limite_vente) >= 0`.

Une jambe individuellement défavorable peut être compensée par l'autre.
Cette convention est cohérente avec le principe de familles de loops décrit
dans la [spécification officielle des smart blocks HUPX](https://hupx.hu/uploads/Kereskedes/Keresked%C3%A9si%20rendszer/DAM/Smart%20block%20changes_20240510.pdf).
Les rejets paradoxaux et les contraintes de clearing d'une bourse ne sont pas
simulés. Il ne s'agit donc ni d'un PnL de trading exécuté ni d'une reproduction
certifiée du code des auteurs.

## Quantiles Storm estimés, autorisés par l'utilisateur

Les archives NYX contiennent les P10/P50/P90 natifs de la sortie Kalman finale.
Les archives Storm ne contiennent qu'une prévision centrale. Dans l'onglet QB,
**Storm calibré** désigne une variante avec intervalles empiriques estimés à
partir des erreurs `prix observé − prévision Storm`.

Pour chaque jour D et chaque heure locale, les erreurs proviennent uniquement
de journées civiles complètes dans `[D−365,D)`. Minimum : 60 journées complètes
et 30 observations de l'heure concernée. Les quantiles empiriques, calculés par
interpolation linéaire de l'échantillon d'erreurs, sont recentrés sur la médiane
historique des erreurs :

`Storm_Pp = Storm_point + Qp(erreurs passées) − Q50(erreurs passées)`, p = 10 %, 90 %.

Cette construction conserve exactement la prévision centrale Storm dans les
deux stratégies ; elle estime la dispersion sans ajouter de correction de
biais. Les intervalles ne sont pas des quantiles fournis par Storm et leur
couverture nominale n'est pas garantie. Aucun prix observé du jour D n'entre
dans leur estimation. L'audit quotidien conserve fenêtre, dates, effectifs et
statut de calibration.

## Échantillon et affichage

- MAE, biais, RMSE, hit rate et R² utilisent les mêmes prix et les mêmes heures
  appariées dans les deux onglets. Le changement de stratégie ne les modifie pas.
- Le PnL utilise **exactement les mêmes journées pour NYX, Storm, QB et UB** :
  observations et prévisions complètes, P10/P90 NYX disponibles et ordonnés,
  calibration Storm disponible. Le début d'historique consacré à la calibration
  est donc exclu également d'UB pour rendre la comparaison des stratégies équitable.
- Le PnL moyen est le total divisé par le nombre de jours admissibles, en
  incluant les journées sans transaction ou rejetées à zéro. Il n'est pas
  divisé par le seul nombre de transactions exécutées. Total, nombre de cycles
  et dates admissibles sont conservés dans les données du rapport.
- Une absence de données reste indisponible, jamais remplacée par zéro. Les
  jours incomplets ne sont pas imputés. La fréquence « DAY » ne change que
  l'agrégation des erreurs de prévision ; le trading reste horaire.
- Les journées de changement d'heure gardent leurs 23/25 heures physiques,
  distinctes en UTC. C'est une adaptation par rapport à la normalisation sur
  24 heures employée dans l'article.

Les prévisions, observations archivées, modèles et calculs en cours restent
inchangés. La génération des rapports est locale et ne collecte plus la
métrique VPS Saturn.
