# Prévision directe à 15 minutes — résultats du 16 septembre 2026

**Résultat mitigé : Chronos-2 à 15 minutes réduit légèrement l'erreur par rapport au même modèle horaire, mais le gain n'est pas statistiquement établi. NYX complet reste nettement meilleur.** Aucun modèle n'a été activé dans l'application.

[Rapport HTML](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_quarterhour_v1/20260916T085805Z_bdd9f9b3/report.html) · [Protocole et reproduction](C:/Users/BQ6757/chronos2_v1/docs/NYX_QUARTERHOUR.md).

## Ce qui a effectivement été testé

Nouvelles prévisions Chronos-2 produites avec les poids locaux gelés, pour **BE, DE, FR et NL**, du **18/06 au 15/09/2026**. Deux recettes fixées avant les scores : contexte de 2 048 prix horaires ou de 8 192 prix natifs à 15 minutes, donc la même durée de 2 048 heures. Mêmes six fondamentaux horaires prévus, mêmes fonctions calendaires, aucun fine-tuning. Le candidat à 15 minutes produit chaque quart avant de moyenner les quatre prévisions ponctuelles par heure.

**Le contrôle horaire et le candidat à 15 minutes n'ont ni CatBoost ni Kalman. NYX complet, avec ces correcteurs, est une référence distincte.** Ce test isole le changement de résolution dans Chronos-2 ; il ne teste pas une réimplémentation de toute la chaîne NYX à 15 minutes. Les fondamentaux horaires sont explicitement constants sur les quatre quarts.

Archive native : quatre séries primaires EPEX/UTC, 350 jours, 134 400 prix, sans interpolation. Le score porte sur **8 640 points pays-heure par modèle**, parfaitement appariés. Les 34 560 prévisions natives du candidat sont agrégées en 8 640 points horaires. La cible commune est la moyenne des quatre prix natifs : elle correspond aux observations horaires NYX sur ce test, à l'arrondi numérique près.

## Scores globaux

| Modèle | MAE €/MWh ↓ | RMSE €/MWh ↓ | Biais €/MWh |
|---|---:|---:|---:|
| NYX complet | **14,1914** | **30,1960** | −1,8524 |
| Chronos-2 horaire, contrôle comparable | 15,7640 | 32,0991 | +1,0985 |
| Chronos-2 à 15 minutes, moyenne horaire | 15,4489 | 31,7106 | −1,2478 |

Face au contrôle horaire, le candidat réduit la MAE de **0,3151 €/MWh**, soit **1,9987 %**. L'intervalle bootstrap à 95 % du delta MAE est **[−1,2903 ; +0,8377] €/MWh** : il inclut des améliorations comme des dégradations. Le RMSE baisse de 0,3886 €/MWh, avec un intervalle **[−1,1250 ; +0,8270]**, également non concluant. Ces intervalles utilisent 1 000 tirages de blocs communs de sept jours.

Face à NYX complet, la MAE du candidat se dégrade de **1,2575 €/MWh**, soit **8,86 %**. L'intervalle du delta MAE **[+0,4257 ; +2,1672]** reste entièrement positif. Son RMSE est également plus élevé de **1,5145 €/MWh**, intervalle **[+0,7738 ; +2,6891]**.

Le gain exact contre le contrôle est juste sous le seuil préétabli de 2 %, mais cette proximité n'est pas le point décisif : l'incertitude n'établit pas un gain et la référence complète NYX conserve un avantage net.

## Détail par pays

| Pays | MAE NYX | MAE Chronos horaire | MAE Chronos 15 min → heure | Gain MAE du 15 min contre le contrôle |
|---|---:|---:|---:|---:|
| BE | 16,1076 | 17,4664 | 17,6040 | −0,79 % |
| DE | 13,0686 | 15,2855 | 14,5776 | +4,63 % |
| FR | 13,6537 | 14,6169 | 14,1285 | +3,34 % |
| NL | 13,9356 | 15,6872 | 15,4857 | +1,28 % |

Les quatre intervalles pays du contraste avec le contrôle horaire incluent zéro. L'amélioration moyenne est donc descriptive, même en Allemagne et en France.

Les heures à prix négatifs constituent un signal intéressant à examiner : contre Chronos horaire, la MAE du candidat baisse de **27,5 % en BE, 43,7 % en DE, 41,9 % en FR et 38,3 % en NL**. Ce sont des diagnostics conditionnels descriptifs, pas des tests de significativité séparés. NYX conserve néanmoins une meilleure MAE dans ce régime ; les dégradations du candidat contre NYX atteignent **41,5 % en BE, 36,5 % en DE et 23,0 % en NL**. Les garde-fous sur les pics allemands et néerlandais échouent également contre NYX. Tous les régimes prévus disposent d'au moins 30 heures sur cinq jours.

## Décision et portée

**Conserver NYX complet dans l'application.** La recette testée ne franchit pas les critères fixés avant le calcul. Elle apporte un signal faible en faveur d'une fréquence plus fine pour Chronos seul, particulièrement sur les prix négatifs, mais ne démontre pas un gain global fiable.

Ce résultat ne permet pas de conclure sur un candidat qui réentraînerait ensuite CatBoost et Kalman sur les sorties du modèle à 15 minutes, ni sur l'apport de nouveaux fondamentaux eux-mêmes natifs à 15 minutes. Ces hypothèses demanderaient une expérience distincte, avec sélection et évaluation séparées.

Les prix sont des observations historiques révisées, sans certificat de leur version publiée à chaque cutoff. Les deux modèles admettent les prix day-ahead de D−1 jusqu'à la fin de D−1, et aucun prix de D dans les entrées. Les fondamentaux conservent les limites de preuve des archives NYX. La période avait déjà été examinée lors d'audits précédents : l'étude reste exploratoire, non prospective. Une moyenne de quatre q50 n'est pas présentée comme un quantile horaire calibré.

Des écarts entre moyennes natives et anciens labels horaires existent avant le test (155 heures par pays, avec un motif de bordure documenté dans l'audit des sources), mais **aucun sur les 90 jours évalués**. Les labels et les sorties de NYX n'ont pas été modifiés.

## Exécution et contrôles

Run `20260916T085805Z_bdd9f9b3`, terminé avec statut `complete`. Durée de l'exécuteur : **4 943,83 secondes, soit 82,4 minutes**, hors collecte, préparation et benchmarks matériels. CPU, 8 threads, lots de 64 ; aucune modification des réglages après consultation des scores. Le processeur était également partagé avec un autre run NuclearKalman.

Les **103 tests logiciels** passent. Les entrées, poids, versions, code, prévisions quotidiennes et sorties sont scellés par empreinte. Les 720 origines pays/jour/résolution enregistrent le contexte utilisé ; les reprises refusent les changements de données, de dépendances ou de modèle. Les tests synthétiques servent à vérifier le logiciel, pas à mesurer la performance prédictive.
