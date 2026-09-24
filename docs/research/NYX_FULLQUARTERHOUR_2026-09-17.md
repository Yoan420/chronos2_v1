# NYX complet à 15 minutes — première analyse provisoire

Instantané du 17 septembre 2026. [Rapport HTML](nyx_fullquarterhour_2026-09-17/report.html) · [Métriques détaillées](nyx_fullquarterhour_2026-09-17/metrics.csv) · [Analyse reproductible](nyx_fullquarterhour_2026-09-17/analyze.py).

## État de l'exécution

Les quatre chaînes natives à 15 minutes sont terminées et scellées : Chronos-2, CatBoost quotidien, puis Kalman gouverné. La dernière s'est terminée le 16 septembre vers 20 h 47. Le témoin complet horaire s'est interrompu vers 21 h, pendant CatBoost, après 64 à 70 des 264 journées selon le pays. Aucun processus Python ne tournait au contrôle du 17 septembre à 15 h 04. La cause de l'arrêt ne peut pas être établie à partir des fichiers disponibles.

Le témoin a été repris en arrière-plan à 15 h 05, avec les mêmes paramètres, versions et empreintes de code. Les quatre résultats natifs complets sont réutilisés après vérification. Les nouveaux journaux sont dans `tmp/nyx_fullquarterhour_resume_20260917/`. Aucun suivi actif n'est requis pour la poursuite du calcul.

Le `report.html` et le `summary.json` situés directement dans le dossier source reflétaient encore l'échec initial de lancement Windows, antérieur à la reprise réussie du 16 septembre. Ils ne décrivent pas les quatre résultats natifs achevés. Le rapport provisoire ci-dessus est un artefact distinct ; il ne remplace pas le futur rapport final du calcul.

## Résultats mesurés

Population commune : BE, DE, FR et NL, du 18 juin au 15 septembre 2026, soit 90 jours, 2 160 heures par pays et 8 640 points par famille. Les prévisions à 15 minutes sont moyennées par groupes de quatre quarts. Toutes les erreurs sont mesurées à l'heure, en €/MWh.

| Prévision | MAE | RMSE | Biais |
|---|---:|---:|---:|
| NYX actuel archivé | 14,191 | 30,196 | −1,852 |
| Chronos seul horaire | 15,500 | 31,860 | −0,864 |
| Chronos seul à 15 minutes | 15,216 | 31,577 | −2,257 |
| Chronos + CatBoost à 15 minutes | 14,399 | 30,251 | −1,867 |
| NYX complet à 15 minutes | 14,431 | 30,286 | −1,826 |

La chaîne complète à 15 minutes a une MAE supérieure de **1,69 %** à NYX actuel. L'écart est de **+0,240 €/MWh**, avec un intervalle bootstrap à 95 % **[−0,402 ; +0,899]**. Il contient zéro : aucun gain global n'est démontré, et la dégradation observée n'est pas non plus statistiquement établie par ce protocole. La RMSE augmente de 0,30 %.

CatBoost améliore la MAE du Chronos natif de **5,37 %**. Kalman la dégrade ensuite légèrement de **0,032 €/MWh** sur cette période. La gouvernance sélectionne l'identité dans 347 des 360 journées-pays ; seules 13 reçoivent une correction de poids positif. Ce constat ne suffit pas à choisir une nouvelle architecture sur cette même période.

| Pays | MAE NYX actuel | MAE NYX complet 15 min | Variation MAE |
|---|---:|---:|---:|
| BE | 16,108 | 16,480 | +2,31 % |
| DE | 13,069 | 13,497 | +3,28 % |
| FR | 13,654 | 13,174 | −3,51 % |
| NL | 13,936 | 14,574 | +4,58 % |

Les intervalles de l'écart MAE contiennent zéro dans chacun des quatre pays. La baisse française reste donc un signal exploratoire. La RMSE française augmente légèrement malgré la baisse de MAE.

Sur les prix négatifs, la MAE baisse dans les quatre pays : BE −27,1 %, DE −15,8 %, FR −10,8 %, NL −17,7 %. À l'inverse, au-dessus du q95 historique en Allemagne, elle augmente de **5,74 %**, au-delà de la limite de 5 % fixée au protocole. Les q95 sont calculés sur des observations antérieures au 14 mars ; ce ne sont pas les percentiles de la période testée. Les analyses de sous-groupes ne font pas l'objet d'une correction pour tests multiples.

## Interprétation et limites

Les critères de poursuite préétablis ne sont pas satisfaits face à NYX actuel : absence de gain MAE d'au moins 2 %, intervalle non entièrement favorable et dégradation trop importante sur les prix élevés en Allemagne. **Le témoin horaire complet manque encore** : il serait prématuré d'attribuer ces différences au seul passage à 15 minutes.

Les deux nouvelles chaînes disposent de 174 à 263 jours de résidus antérieurs, contre un historique plus long pour NYX archivé. Les fondamentaux restent horaires, répétés à l'intérieur de l'heure. Les observations sont rétrospectives, la période a été consultée auparavant, et les moyennes de prévisions ponctuelles ne sont pas des quantiles horaires calibrés.

Suite nécessaire : achever le témoin complet horaire à historique égal, comparer les deux chaînes, puis confirmer tout signal favorable sur une nouvelle période. Aucune activation en production.

## Traçabilité

- Source : `runs/experiments/nyx_fullquarterhour_v1/20260916T111425Z_dff7f605`.
- Protocole complet : `6d9552cd7a066ec1617461b6c81569c65c9d3f13fe0079dc655fb90095fc3b4d`.
- Empreintes des entrées, du brut, du code et des quatre workers natifs vérifiées avant l'évaluation.
- Agrégations strictes, sans suppression de points, interpolation ou substitution par NYX.
- Bootstrap apparié commun : blocs de sept jours, 1 000 répétitions, graine 20260916.
- La sortie machine reste explicitement `insufficient_data` / `missing_matched_control` ; cinq familles sont disponibles, au lieu des sept prévues pour l'analyse finale.
