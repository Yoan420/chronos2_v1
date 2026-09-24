# Retour d’expérience — Ajout des Clean Fuel Costs dans NYX

L’expérience visait à améliorer la prévision des prix de l’électricité en apportant à NYX une information explicite sur les coûts de production thermique. Cinq variables ont été ajoutées : les **Clean Gas Costs** français, allemand, belge et néerlandais, ainsi qu’un **Clean Coal Cost** fondé sur le charbon API2. Ces indices incluent déjà le coût du CO₂ et la conversion en euros par MWh électrique.

Le test portait sur l’ensemble de la chaîne : contexte et covariables de Chronos-2, réentraînement du correcteur résiduel, puis recalibration de Kalman. Les coûts retenus étaient les dernières cotations disponibles selon le protocole de coupure à J−1, 08 h ; leur valeur quotidienne était répétée sur les heures de livraison. Aucun plancher de prix n’était imposé. Les hyperparamètres étaient hérités du modèle de référence, sans optimisation sur l’année évaluée.

**Résultats.** La comparaison ci-dessous concerne la sortie finale avec Kalman, face à la référence NYX intégrant le nucléaire français. Les statistiques des rapports couvrent le **19 septembre 2025 au 18 septembre 2026**, soit **8 760 heures par pays**, avec exactement les mêmes observations et heures pour les deux modèles. La MAE mesure l’erreur absolue moyenne ; la RMSE accorde davantage de poids aux grandes erreurs.

| Pays | MAE de référence (€/MWh) | MAE avec Clean Fuel (€/MWh) | Variation MAE | Variation RMSE |
|---|---:|---:|---:|---:|
| France | 12,136 | 12,142 | +0,05 % | +0,12 % |
| Allemagne | 11,123 | 11,153 | +0,27 % | +0,53 % |
| Belgique | 11,226 | 11,451 | +2,00 % | +1,48 % |
| Pays-Bas | 11,271 | 11,355 | +0,75 % | +0,13 % |

Une variation positive indique une dégradation. Sur les quatre pays réunis, avec le même poids par heure et par pays, la MAE augmente de **0,75 %** et la RMSE de **0,63 %**.

**Pourquoi cette configuration n’a pas été retenue.** L’ajout des variables ne produit aucun gain annuel sur ces deux métriques finales, dans aucun des quatre pays. Les améliorations observées sur certains épisodes ne suffisent pas à justifier leur intégration systématique : en France, la MAE diminue de **15,6 % sur les seuls 24–26 juin 2026**, mais cet avantage local ne se traduit pas par une amélioration annuelle. Les dépendances de données et la complexité supplémentaires ne sont donc pas compensées par un bénéfice prédictif démontré sur la période étudiée.

**Enseignement.** Ces résultats ne remettent pas en cause le rôle économique des combustibles dans la formation des prix. Ils montrent que **leur ajout direct, sous la forme et avec la recette testées, n’apporte pas de valeur prédictive supplémentaire à NYX sur cette évaluation**. Une information déjà partiellement captée par les entrées existantes, ou la faible granularité temporelle des indices quotidiens, sont des explications possibles ; l’expérience ne les démontre pas séparément.

Les résultats restent descriptifs, sans test de significativité. Le protocole est rétrospectif et les horodatages fournisseur ainsi que le versionnement historique des formules ne sont pas certifiés indépendamment. Une éventuelle réutilisation ciblée des coûts thermiques demanderait donc une nouvelle validation. La variante est restée expérimentale et n’a pas remplacé le modèle de production.

---

Sources : expérience **nyx_clean_fuel_full_v1**, livraison du 18/09/2026 ; [protocole](C:/Users/BQ6757/chronos2_v1/CLEAN_FUEL.md) et [index des rapports](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_clean_fuel_full_v1/2026-09-18/clean_fuel_FR_DE_BE_NL_index.html).

Chiffres issus des comparaisons archivées : [France](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_clean_fuel_full_v1/2026-09-18/fr/4bd46fb699b407a7/reports/clean_fuel_incumbent_comparison.json), [Allemagne](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_clean_fuel_full_v1/2026-09-18/de/453bf870f6b6ec02/reports/clean_fuel_incumbent_comparison.json), [Belgique](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_clean_fuel_full_v1/2026-09-18/be/63ba76e0d4291d63/reports/clean_fuel_incumbent_comparison.json), [Pays-Bas](C:/Users/BQ6757/chronos2_v1/runs/experiments/nyx_clean_fuel_full_v1/2026-09-18/nl/f2f08c39d8b3684d/reports/clean_fuel_incumbent_comparison.json). La fenêtre « Statistics », qui inclut le 18/09 observé, est distincte du backtest précédant la livraison (18/09/2025–17/09/2026). Aucun recalcul du modèle ni changement de référence d’observation n’a été réalisé pour cette synthèse.
