# Expert anti-pics zonal — laboratoire isolé

Le programme `ScarcityZonal.ps1` ne modifie ni `Forecast.ps1`, ni ses modes,
ni les modèles, séries ou rapports opérationnels. Il ne comporte aucune action
de promotion. Les résultats sont écrits sous
`runs/experiments/nyx_scarcity_v1/zonal/snapshots/`.

## Utilisation

Depuis n’importe quel dossier PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\ScarcityZonal.ps1' -Action Run
```

`Run` prépare un nouveau snapshot, entraîne les trois ablations et génère les
rapports. `Prepare` fige les données sans entraîner. `Backtest` reprend le dernier
snapshot préparé ; les variantes terminées et vérifiées ne sont pas réentraînées.
`Report` régénère les rapports du dernier snapshot terminé, sans apprentissage.
`Status` lit l’état du dernier snapshot préparé. `-RunDirectory` permet de choisir
explicitement un snapshot pour Backtest/Report/Status. `-DryRun` affiche seulement
la commande. Ne relancez pas Run pour reprendre un calcul interrompu : utilisez
Backtest. Un changement de code impose un nouveau Prepare pour un calcul inachevé.

La configuration dédiée est `config/nyx_scarcity_zonal.yaml`. Le runtime XGBoost
privé 3.2.0 est réutilisé ; rien n’est installé dans l’environnement opérationnel.
Les modèles entraînés sont enregistrés dans `latest_model.joblib` avec empreinte
SHA-256 et identité de snapshot. `runner.load_model` vérifie ces identités et le
runtime avant chargement ; ne chargez jamais un pickle provenant d’un tiers.
L’artefact est le dernier ajustement du backtest, pas une autorisation de live.

## Hypothèses fixées avant le nouveau backtest

Trois ablations : `regional_hiercal` (variables initiales, calibration zonale),
`zonal_context` (variables locales/voisins, calibration initiale),
`zonal_hiercal` (les deux, candidat principal pré-déclaré).

Le contexte zonal comporte 38 signaux et leurs indicateurs de données manquantes.
Les mêmes règles physiques s’appliquent à chaque pays, sans exception française
ajoutée après lecture du résultat. Le modèle compare notamment charge résiduelle,
production/disponibilité sélectionnée, rampes et tension des voisins. Les prix
prévus des voisins sont alignés à la même heure ET à la même origine de prévision.
Ni les prix observés ni Storm ne sont des entrées de l’expert.

Le proxy d’offre sélectionnée utilise gaz + nucléaire prévu en FR, gaz + charbon +
lignite en DE, gaz + nucléaire disponible en BE, gaz + charbon + nucléaire
disponible en NL. Ce mélange de prévision de production et de disponibilités
n’est ni une marge de réserve complète, ni un bilan réseau, ni une capacité
d’importation. Les ratios utilisent un plancher de dénominateur fixé à 1 GW ;
une donnée manquante reste manquante, sans somme partielle inventée.

Après le Platt global, un décalage logit régularisé est estimé par pays sur les
28 jours réservés à la calibration. La pente reste partagée :

`p_z = sigmoid(logit(p_global) + delta_z)`

Chaque delta minimise la SOMME des log-loss pondérées par `24 / heures du jour`,
augmentée de `delta² / 2`, avec bornes fixes ±3. Les journées de 23/25 heures sont donc
normalisées. La pénalité rapproche le décalage de zéro, mais ne certifie pas la
calibration future. Aucun événement récent n’implique pas une probabilité nulle.
Une absence complète de données pays produit un décalage nul explicitement audité.

L’algorithme de gravité, le seuil de sous-estimation (max de 50 EUR/MWh et Q95 du
résidu sur l’entraînement), la porte `p > 0.6`, le plafond de proposition
400 EUR/MWh et les hyperparamètres de l’expert initial sont conservés.
La gravité est toutefois réentraînée sur les nouvelles variables pour les
variantes contextuelles. Les offsets changent aussi le quantile de sévérité du
mélange (`1 - 0.5/p`) : ils peuvent modifier l’amplitude, pas seulement l’alerte.

## Deux politiques distinctes — ne pas les confondre

1. **Proposition expérimentale fixe 25 %**, principale : prix NYX + 25 % de la
   correction proposée et bornée lorsque le détecteur franchit sa porte. Le
   gouverneur strict de non-régression n’est PAS appliqué à cette série.
2. **Sortie gouvernée stricte**, témoin : même détecteur, banque de poids et
   contrôles chronologiques existants, qui peuvent retenir zéro.

Les deux séries sont sauvegardées et évaluées. Aucun poids n’est sélectionné par
pays sur les résultats de l’année. Les intervalles de la proposition fixe sont
recalculés avec ses propres erreurs antérieures connues, pas ceux du gouverneur.
Faute d’historique suffisant, une enveloppe de repli est explicitement marquée
non calibrée. Aucune couverture future P10–P90 n’est garantie.

## Période, contrôles et limites

Source figée : suite terminée du 14 septembre 2026, panel inchangé. Évaluation
du 15 septembre 2025 au 14 septembre 2026 : 365 jours représentés, 8 760 heures
physiques/pays, dont 8 735 heures appariées à Storm. Les 25 trous Storm restent
vides, même si une autre archive contient une valeur. La livraison du 15 septembre
2026 est affichée séparément et exclue des Statistics, même si l’enchère est connue.

Entraînement progressif 90 → 365 jours, réajustement tous les 7 jours, avec les
28 derniers jours réservés à la calibration. Ce panel ne contient PAS 365 jours
d’entraînement avant le premier des 365 jours évalués. Le repli NYX initial reste
dans tous les scores annuels : la couverture réellement entraînée est affichée.
Cutoff D−1 08 h Europe/Paris, labels uniquement disponibles avant le réajustement
ou la décision. Les règles internes sont auditées ; les forecasts historiques et
les références de ce replay ne constituent pas une certification indépendante
PIT/neural-OOF. JAO post-cutoff reste exclu.

Comparer simultanément la MAE annuelle de chaque pays, celle des plus hauts prix,
les heures dont l’erreur est aggravée, leur fréquence rapportée aux corrections,
les vrais pics manqués et le nombre de journées indépendantes. Diminuer le nombre
de fausses corrections en ne corrigeant plus rien n’est pas une amélioration
suffisante. Les épisodes déjà analysés (notamment le 14 septembre) ne sont pas un
test indépendant. Il faut une phase prospective avant toute décision de remplacement.

Les rapports pays utilisent le vrai moteur HTML existant (Statistics, prix moyens,
calendrier, heures, mode nuit). P10/P50/P90 sont sauvegardés ; les déciles
intermédiaires/CRPS affichés par ce moteur sont interpolés, pas de nouveaux
quantiles appris. Aucune attribution causale du prix n’est inventée à partir des
probabilités du classificateur.
