# Prévision directe à 15 minutes, puis moyenne horaire

Cette expérience répond à la deuxième variante : faire produire à Chronos-2 les prix de chaque quart d'heure, puis calculer une prévision ponctuelle horaire par moyenne arithmétique des quatre points. Elle crée de nouvelles prévisions avec les poids locaux ; elle ne recycle pas des prix observés comme prévisions.

Premier essai réel terminé le 16/09/2026 : **gain moyen d'environ 2 % contre Chronos horaire, non statistiquement établi ; erreur de 8,86 % supérieure à NYX complet**. Voir les [résultats et limites](C:/Users/BQ6757/chronos2_v1/docs/research/NYX_QUARTERHOUR_2026-09-16.md). Le protocole ci-dessous a été fixé avant ces scores.

## Protocole fixé avant le calcul des scores

Deux candidats utilisent le **même modèle Chronos-2 gelé**, la même horloge, les mêmes six fondamentaux horaires prévus et la même durée d'historique :

| Élément | Contrôle horaire | Candidat natif à 15 minutes |
|---|---|---|
| Contexte | 2 048 prix horaires | 8 192 prix à 15 minutes |
| Durée physique du contexte | 2 048 heures | 2 048 heures |
| Cible historique | Moyenne de quatre prix natifs | Prix natif de chaque quart |
| Horizon d'un jour civil | 23/24/25 points | 92/96/100 points |
| Fondamentaux | Six prévisions horaires archivées | Mêmes valeurs, constantes sur les quatre quarts |
| Calendrier | Heure, semaine, année, weekend, offset UTC | Mêmes fonctions, heure fractionnaire |
| Apprentissage supplémentaire | Aucun | Aucun |
| CatBoost / Kalman | Absents | Absents |

La comparaison entre ces deux candidats isole le changement de résolution dans Chronos-2. **NYX complet, avec ses correcteurs existants, est une troisième référence distincte. Ce premier test ne porte pas toute la chaîne NYX à 15 minutes.** Aucun quantile horaire calibré n'est déduit des quantiles par quart : seule la moyenne de quatre prévisions ponctuelles q50 est évaluée.

Le test porte sur **BE, DE, FR et NL, du 18 juin au 15 septembre 2026**, soit 90 journées et 8 640 points pays-heure par méthode. Les réglages ne sont pas sélectionnés sur ces scores. La période a déjà été consultée dans les audits précédents ; il s'agit d'un test exploratoire, pas d'une preuve prospective.

L'inférence est CPU, sans téléchargement de poids, sans fine-tuning, avec `cross_learning=False`, seed 42 et huit threads. Le contexte à 15 minutes atteint la limite de 8 192 points du checkpoint ; un contexte plus long est refusé plutôt que tronqué silencieusement. Le batch matériel peut être mesuré sur une fixture synthétique sans utiliser les prix futurs pour choisir la recette.

## Données et temporalité

Quatre séries primaires EPEX de Saturn, toutes UTC, sans formule de remplissage : **BE 60451, DE 60452, FR 60454, NL 60453**. Archive du 01/10/2025 au 15/09/2026 : **350 jours, 33 600 quarts par pays, 134 400 lignes complètes**. Les jours de changement d'heure restent physiques ; aucune interpolation ni répétition de prix horaires n'est admise.

Le contexte d'une livraison D se termine à la fin de D−1 : la recette admet les prix day-ahead de D−1, conformément à l'information de marché supposée connue au cutoff D−1 à 08:00 Paris. Les observations extraites sont cependant les dernières versions historiques disponibles, **sans certification de chaque version publiée au cutoff**. Cette convention est identique pour les deux candidats. Les prix réalisés de D n'entrent jamais dans leurs contextes ni dans leurs entrées futures.

Les fondamentaux proviennent des archives NYX vérifiées du bundle de livraison du 16/09/2026. Les répéter dans une heure n'ajoute pas d'information native à 15 minutes. Leurs limites de preuve rétrospective restent celles du bundle initial.

La cible de score commune aux trois méthodes est la moyenne des quatre prix natifs. Les écarts avec les observations horaires du reporting NYX sont conservés dans un audit ; les labels originaux ne sont pas modifiés. La prévision NYX demeure exactement celle du bundle figé.

## Évaluation et décision

MAE, RMSE, biais, pays, heures et régimes de prix. Intervalles appariés : 1 000 bootstrap de blocs communs de sept jours, seed 20260916. Les seuils de prix élevés sont les q95 des observations d'apprentissage figées de NYX jusqu'au 13/03/2026 inclus, avant la période évaluée. Les pays et heures ne sont jamais filtrés selon leurs erreurs.

Un résultat encourageant exige : gain MAE d'au moins 2 % contre NYX et le contrôle horaire ; IC 95 % du delta MAE entièrement négatif ; RMSE globale au plus 1 % moins bonne ; MAE par pays au plus 5 % moins bonne. Les régimes négatifs et au-dessus du q95, s'ils comportent au moins 30 heures sur cinq jours, ne doivent pas se dégrader de plus de 5 %. Une preuve partielle sur les régimes est explicitement signalée. Aucun résultat ne déclenche une activation en production.

Les trois grilles doivent être complètes et identiques. Un quart manquant, un doublon, un NaN ou une heure hors fenêtre provoquent une erreur ; aucune intersection silencieuse ni substitution par NYX ne masque un problème.

## Exécution et traçabilité

Configuration : [nyx_quarterhour.yaml](C:/Users/BQ6757/chronos2_v1/config/nyx_quarterhour.yaml).

```powershell
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' run_nyx_quarterhour.py
```

Les sorties restent sous `runs/experiments/nyx_quarterhour_v1`. Le runner copie les entrées, verrouille une empreinte de protocole, écrit les nouvelles prévisions par jour et résolution, puis calcule le rapport. `--resume <dossier>` exige les mêmes données, le même code et la même recette ; les résultats partiels sont vérifiés par empreinte. Le contexte UTC, le cutoff déclaré, la durée d'historique et l'absence de cible future sont audités par origine. Les tests avec données synthétiques vérifient ces propriétés et ne servent pas de mesure de qualité prédictive.
