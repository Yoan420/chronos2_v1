# NYX horaire enrichi par des profils à 15 minutes — premier résultat réel

La variante est implémentée et évaluée. **Ce premier pilote n'améliore pas NYX et ne justifie pas le passage à un modèle prédisant directement les quarts d'heure.** Aucun modèle n'a été activé dans le lancement principal.

Run : `20260916T082626Z_9b3d391b`, terminé en 38,26 secondes après acquisition des données. [Rapport HTML complet](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_intrahour_v1/20260916T082626Z_9b3d391b/report.html) · [Protocole et reproduction](C:/Users/BQ6757/chronos2_v1/docs/NYX_INTRAHOUR.md).

## Périmètre et comparaison

Le pilote utilise une source native à 15 minutes vérifiée : la prévision solaire belge Elia, série primaire Saturn 23259. Les 365 jours du 16/09/2025 au 15/09/2026 sont complets : 35 040 quarts, 8 760 heures, dont 4 851 heures avec variation interne. Les journées de changement d'heure conservent leurs 92 ou 100 quarts. Les séries horaires simplement répétées à 15 minutes ont été exclues.

Les prévisions restent horaires pour BE, DE, FR et NL. La variante ajoute un correcteur ridge après NYX complet ; elle ne réentraîne pas Chronos-2. Le contrôle horaire utilise les mêmes entrées, sans les cinq caractéristiques décrivant la forme interne à l'heure. Cette comparaison sépare l'effet du nouveau fondamental de l'effet propre de sa résolution.

180 jours initiaux, validation du 15/03 au 16/06/2026, embargo le 17/06, test du **18/06 au 15/09/2026**. Les fenêtres de fit respectent D−2 et les réglages restent ceux choisis avant le test. La validation avait déjà retenu **NYX actuel** ; les deux challengers choisissent alpha=100. Aucun réglage n'a été modifié après lecture des scores.

## Résultats du test : 90 jours, 8 640 points pays-heure

| Variante | MAE, €/MWh ↓ | RMSE, €/MWh ↓ | Évolution MAE contre NYX |
|---|---:|---:|---:|
| NYX actuel | **14,1914** | **30,1960** | Référence |
| Contrôle : moyenne solaire horaire | 14,7294 | 30,2250 | +3,79 % |
| Moyenne et forme à 15 minutes | 14,7550 | 30,2379 | +3,97 % |

Le delta MAE de la variante à 15 minutes contre NYX est **+0,5636 €/MWh**, avec un intervalle bootstrap à 95 % **[+0,2753 ; +0,8719]**. La dégradation est donc assez nette dans ce test rétrospectif. Contre le contrôle horaire, le delta n'est que **+0,0256 €/MWh**, intervalle **[−0,0134 ; +0,0587]** : aucun apport de la forme à 15 minutes n'est démontré. Les intervalles utilisent 1 000 rééchantillonnages de blocs communs de sept jours.

| Pays | MAE NYX | MAE contrôle horaire | MAE profils à 15 minutes |
|---|---:|---:|---:|
| Belgique | 16,1076 | 16,8659 | 16,9278 |
| Allemagne | 13,0686 | 13,8696 | 13,9069 |
| France | 13,6537 | 13,6543 | 13,6659 |
| Pays-Bas | 13,9356 | 14,5277 | 14,5194 |

Tous les régimes critiques prévus disposent d'au moins 30 heures sur cinq jours. Sur les prix négatifs, la MAE du candidat se dégrade contre NYX de **12,8 % en Allemagne, 26,2 % en France et 9,5 % aux Pays-Bas**. Ces garde-fous échouent aussi. Il n'y a aucun repli lié à des données manquantes : le taux de repli est 0 %.

La dégradation globale se retrouve presque entièrement dans le contrôle horaire. Elle ne peut donc pas être attribuée uniquement aux caractéristiques à 15 minutes. Le résultat rejette cette recette de correction, avec cette source, sur cette période ; il ne prouve pas que toute modélisation à 15 minutes serait inutile.

## Décision et limites

**Conserver NYX actuel ; ne pas lancer la deuxième variante sur la base de ce résultat.** Le prototype, l'archive et les scores restent disponibles pour une comparaison reproductible.

Ce pilote ne couvre que le solaire belge, utilisé comme entrée commune des quatre sorties. Les lectures sont des requêtes historiques as-of D−1 08:00 Paris, sans preuve indépendante de l'heure de publication originale. Les labels figés ne certifient pas non plus chaque version historiquement disponible. Enfin, la période de test a déjà été consultée dans l'audit antérieur du projet : ce résultat est exploratoire, pas une validation prospective.

Les artefacts conservent les données exactes, les prédictions appariées, les métriques, les intervalles, les régimes critiques, les audits d'apprentissage, les versions et les empreintes des fichiers. Les 101 tests logiciels couvrent séparément les calculs, la temporalité, les changements d'heure, les données manquantes et l'isolation des sorties ; ils ne constituent pas une preuve de gain prédictif.
