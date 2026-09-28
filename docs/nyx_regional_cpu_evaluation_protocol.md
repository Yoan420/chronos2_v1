# Évaluation avant activation de NYX régional CPU

Cette recette est nouvelle. Elle utilise des variables récupérables par le
poste de travail et des modèles CatBoost sur CPU. Les RMSE des essais GPU de
septembre 2026 ne sont pas transférables à ce code.

## Fenêtre et entraînement

- Prévisions horaires physiques UTC pour FR, DE, BE et NL, du 24 septembre 2025
  au 23 septembre 2026 inclus, si chaque source permet cette fenêtre complète.
- Origines hebdomadaires à partir du 24 septembre 2025 pour choisir les
  candidats. Du 6 mai au 23 septembre 2026, confirmation avec un entraînement
  à chaque journée de livraison, comme le lancement opérationnel. Chaque
  entraînement ne lit que les 365 jours civils antérieurs ; la journée prédite
  reste hors de l'entraînement et de la calibration. Les variables de chaque
  jour suivent sa propre coupure D−1 08 h.
- La construction des variables suit, pour chaque journée de livraison, sa
  propre coupure civile D−1 à 08 h. Aucune observation de cette journée, ni
  Storm, n'entre dans les variables ou dans la sélection à l'inférence.
- Les candidats de prix sont préfixés : `absolute`,
  `residual_prior_day_mean`, `blend50`. Le classifieur de prix négatifs est
  évalué séparément sur l'événement `prix < 0` et calibré avec des données
  antérieures au pli prédit. Pour la probabilité négative, la sélection compare
  par pays le Brier de la sortie calibrée, de la sortie brute et de la
  fréquence observée sur les 365 jours passés.

## Choix et contrôle

Les candidats sont choisis par RMSE du 24 septembre 2025 au 5 mai 2026
inclus. Du 6 mai au 23 septembre 2026, les choix figés sont confirmés avec
réentraînement quotidien sur de nouvelles heures. La séparation coïncide avec
une origine hebdomadaire. Le
rapport donne, par pays et sur exactement les mêmes heures que
Storm, RMSE, MAE, nombre et proportion de victoires strictes. Il donne aussi
le Brier, la calibration et le nombre d'événements pour la probabilité de
prix négatif. Les heures absentes restent absentes ; aucun zéro ou valeur
interpolée ne les remplace.

Le reçu chiffre les divergences entre la cible Saturn canonique utilisée pour
l'entraînement et l'extraction EPEX de référence utilisée pour noter les prix
et les événements négatifs. Les contrôles de causalité attestent les grilles
physiques et les requêtes Saturn à la coupure ; ils ne certifient pas la date
de publication historique chez chaque fournisseur.

Le reçu contient les versions, paramètres, plages UTC, nombre de plis et
d'heures, empreintes des entrées et des sorties, contrôles de disponibilité
des données et résultats de chaque candidat. La configuration de lancement
reste `pending_backtest` tant que ce reçu n'est pas complet et vérifié.
Chaque pays qui ne bat pas Storm en RMSE ou en taux de victoires est signalé
explicitement avant une décision de mise en production.

Un essai de deux semaines sur les anciennes matrices 123/503 démontre
seulement que CatBoost tourne sur CPU et que ses points diffèrent du GPU. Il
ne qualifie pas la présente recette, ses nouvelles variables ou ses scores.
