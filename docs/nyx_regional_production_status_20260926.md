# FR/BE/NL — état de préparation pour les prévisions futures

La sélection fixe est implémentée dans `chronos2_hourly/nyx_regional_live_selection.py`.
Sur les sorties historiques scellées, elle reproduit exactement les 10 944 points
FR et les 8 760 points BE/NL (égalité bit à bit). Elle n'utilise ni prix réalisé
de l'heure prédite ni Storm. Aucun pays DE n'est autorisé.

Cette pièce ne produit pas encore de prévision autonome dans NYX. Les trois
points exigent les experts et la référence suivants pour chaque livraison :

| Pays | Entrées | Règle |
|---|---|---|
| FR | résiduel pooled original, référence | résiduel si désaccord >= 20 EUR/MWh, sinon référence |
| BE | résiduel compact JAO, absolu exchange JAO, référence | moyenne 50/50 si désaccord >= 20, sinon référence |
| NL | résiduel compact JAO, absolu exchange JAO, référence | moyenne 50/50 sur toutes les heures |

Le replay annuel employait CatBoost GPU et des caches de variables sous
`runs/experiments/nyx_improvement_to20260923`, absents du dépôt Git. La
référence annuelle et les variables JAO actualisées ne disposent pas encore
d'un producteur prospectif pour une nouvelle date de livraison. Le poste de
travail sans GPU exige un réentraînement CPU : un premier fit CPU à 1 000
arbres sur les 129 entrées compactes s'est terminé, mais ses scores annuels,
sa durée de production et sa conformité aux critères historiques ne sont pas
établis. Un modèle CPU doit porter une nouvelle identité et être évalué avant
de remplacer les sorties canoniques de l'application.

Le dépôt GitHub est public. Les archives et modèles locaux ne sont donc pas
ajoutés à cette branche. L'activation dans le lanceur NYX doit attendre un
producteur de données prospectif, le replay CPU audité et un essai de bout en
bout sur un environnement équivalent au poste de travail.
