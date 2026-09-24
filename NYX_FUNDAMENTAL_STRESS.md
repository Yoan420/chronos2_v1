# Expert fondamental — laboratoire isolé

Cet outil ajoute un expert à la prévision `nuclear_kalman` figée, sans changer
`Forecast.ps1`, les modèles actifs ou les rapports publiés sous `runs/exports`.
Les rapports utilisent le moteur HTML de production, mais restent expérimentaux.

## Utilisation

Depuis n'importe quel dossier PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\FundamentalStress.ps1' -Action Run
& 'C:\Users\BQ6757\chronos2_v1\FundamentalStress.ps1' -Action Status
```

- `Run` fige une nouvelle expérience, entraîne les deux variantes, compare et génère les HTML.
- `Prepare` fige uniquement les entrées, paramètres, code et versions de dépendances.
- `Backtest` reprend le dernier snapshot préparé ; les variantes terminées et vérifiées sont réutilisées.
- `Report` régénère les rapports du dernier résultat terminé, sans réentraîner ni actualiser les observations.
- `-RunDirectory <snapshot>` sélectionne explicitement un résultat pour Backtest, Report ou Status.
- `-DryRun` affiche la commande sans lancer de calcul.

Tous les résultats restent sous `runs/experiments/nyx_scarcity_v1/fundamental`.
Deux processus maximum, deux threads par processus. Aucune installation automatique,
aucune API appelée, aucun entraînement Chronos-2 et aucune promotion en production.
Le runtime XGBoost privé déjà installé est utilisé, pas une nouvelle dépendance du pipeline.

## Méthode

L'expert apprend une association entre la situation physique prévue à D-1 08 h
et les erreurs futures du NYX figé. NYX et l'observé définissent la **cible supervisée**,
mais ne sont pas des entrées du classificateur ou du correcteur.

1. Variables : charge résiduelle, disponibilités sélectionnées du parc,
   prévisions nucléaires françaises, vent, solaire, rampes horaires, températures,
   coûts du gaz/CO2, tensions locales et chez les trois pays voisins, calendrier.
2. Classificateur XGBoost non pondéré : erreur observé − NYX supérieure ou égale
   à max(50 EUR/MWh, quantile 95 % des erreurs du bloc d'entraînement du pays).
3. Probabilité calibrée par régression logistique sur les 28 derniers jours
   strictement antérieurs, exclus de l'entraînement des deux estimateurs.
4. Correcteur HGB : médiane conditionnelle de **toutes** les erreurs signées,
   incluant les surestimations de NYX, et non sévérité des seuls pics.
5. Proposition positive seulement si p dépasse la prévalence de l'événement
   dans l'entraînement du pays et si la porte physique est ouverte :
   FR/DE tension locale au-dessus du Q90 historique et rampe de charge résiduelle
   sur trois heures positive ; BE/NL tension des voisins au-dessus du Q90.
   Tous les seuils sont recalculés sur le bloc d'entraînement, hors calibration.
6. Proposition bornée à 400 EUR/MWh ; poids 0, 25, 50 ou 100 % décidé par
   le gouverneur existant à partir de résultats hors échantillon déjà connus
   sur les 90 jours précédents. Aucune garantie de non-régression future.

Les quatre pays partagent l'estimation, avec identité zonale en entrée. Le protocole
est déclaré avant ce nouveau backtest ; aucun réglage pays par pays n'est choisi
après lecture des résultats. Il reste néanmoins exploratoire, car l'année a déjà
été examinée lors des expériences précédentes.

## Témoins et comparaison

- `fundamental` : candidat principal gouverné.
- `fundamental_25` : même proposition physique avec poids fixe de 25 %, sans gouverneur,
  uniquement pour diagnostiquer le signal et les mauvaises corrections.
- `calendar` / `calendar_25` : vrais témoins calendrier, sans fondamentaux dans
  les matrices apprises et **sans porte physique**. Ils suivent les mêmes dates et métriques.
- NYX, Storm, ancien HGB et ancien expert régional à 25 % : prévisions déjà figées,
  jamais réentraînées ou complétées a posteriori par ce laboratoire.

Les HTML `reports/index.html` présentent la version gouvernée par pays ;
`reports_fixed25/index.html` montre le diagnostic fixe ;
`fundamental_comparison.html` compare toutes les variantes, les erreurs annuelles,
les grandes sous-prévisions, les corrections aggravantes et les profils horaires.

## Limites importantes

- Période héritée du snapshot source : 15/09/2025 au 14/09/2026, 365 jours.
  Livraison du 15/09/2026 affichée mais exclue des Statistics, même si observée.
- Les heures sans Storm restent manquantes : 8 735 heures appariées par pays.
- Il n'existe pas 365 jours de calibration antérieurs à chaque jour évalué dans
  ce panel. Entraînement progressif de 90 à 365 jours, réentraînement hebdomadaire,
  NYX conservé pendant le démarrage ou quand un estimateur n'est pas exploitable.
- Les Pmax sont des proxys journaliers, pas une production ou une flexibilité
  horaire garantie. Les parcs sont incomplets ; imports réalisables, stockage,
  réserves et engagement des unités ne sont pas modélisés.
- La charge résiduelle est déjà nette des renouvelables concernés : aucune
  deuxième soustraction du vent ou du solaire.
- Températures manquantes après le 4 septembre ; révisions de prévisions non
  renseignées dans ce snapshot. Aucune valeur future ou manquante n'est inventée.
- Réseau JAO non qualifié pour le cutoff strict de 08 h : exclu des inputs.
- Les requêtes historiques as-of ne prouvent pas à elles seules une publication
  effectivement disponible à 08 h ; le NYX figé n'est pas certifié neural OOF
  indépendant par cette expérience.
- La validation prospective reste nécessaire avant toute décision de promotion.

## Code et intégrité

Configuration : `config/nyx_fundamental_stress.yaml` ; variables :
`nyx_fundamental_stress/features.py` ; estimateurs et portes physiques : `policy.py` ;
exécution scellée : `runner.py` ; adaptation HTML : `reporting.py`.

Changer du code ou une dépendance après Prepare interdit la reprise des fits
partiels : créer un nouveau snapshot. Les artefacts joblib ne doivent être chargés
que via `runner.load_model` à partir des résultats locaux de confiance vérifiés.
Le laboratoire n'importe pas d'artefacts pickle fournis par un tiers.

## Résultat de la première expérience

Snapshot `20260914T152452Z_41529943`, achevé le 14/09/2026.
Même support de 34 940 heures-pays :

| Prévision | MAE annuelle, EUR/MWh |
|---|---:|
| NYX nucléaire + Kalman figé | 11,367500 |
| Storm figé | 11,317589 |
| Ancien expert régional à 25 % | 11,347755 |
| Expert fondamental gouverné | 11,367672 |
| Expert fondamental à 25 %, diagnostic | 11,366934 |
| Calendrier seul à 25 %, diagnostic | 11,364819 |

Le candidat principal ne justifie **aucune promotion** : 9 corrections appliquées,
toutes en NL, dont 5 aggravent l'erreur absolue. La variante fixe intervient
411 fois, dont 191 corrections aggravantes. Son gain annuel de 0,000567 EUR/MWh
est trop faible pour constituer une amélioration exploitable ; le témoin calendrier
fait même mieux en MAE agrégée. Il n'y a donc pas de démonstration d'une valeur
ajoutée physique suffisante avec cette spécification et ces données.

Cette conclusion sur la **correction de prix** ne signifie pas que la détection
physique est inutile. Sur les 5 231 heures hors échantillon exploitables par pays,
le classement du détecteur fondamental est meilleur que le témoin calendrier :
en DE, ROC-AUC 0,901 contre 0,652 et average precision 0,309 contre 0,0108.
Il s'agit toutefois de cette année exploratoire, pas d'une validation prospective.

La faiblesse principale est la conversion du risque en amplitude : les deux
estimateurs indépendants peuvent être incohérents. En DE, les 11 heures où le
classificateur attribue plus de 50 % de probabilité à une grande sous-prévision
ont malgré tout une médiane d'erreur estimée non positive ; 8 étaient de vrais
événements. Une même distribution conditionnelle cohérente ne devrait pas produire
simultanément ces deux affirmations. Le correcteur HGB a donc effacé un signal
du détecteur, avant même le gouverneur.

À 19 h le 14 septembre, les portes de risque et de stress sont ouvertes dans
les quatre pays, mais les médianes signées estimées restent négatives : BE −8,40,
DE −10,46, FR −7,93 et NL −10,69 EUR/MWh. Aucune correction n'est proposée,
y compris à poids fixe. La prochaine expérience devra chercher une distribution
conditionnelle cohérente entre probabilité et amplitude ; simplement desserrer
le gouverneur ne résout pas ce défaut. Ce constat ne modifie pas les résultats
figés ni les paramètres du présent backtest.

Le 25 juin illustre aussi une limite distincte : les grandes erreurs à 19 h en
BE/DE/NL sont des **surestimations**, qu'une correction positive ne peut réparer.
L'amélioration des sous-prévisions de pics et la correction des excès de prix
doivent être évaluées séparément. Aucun seuil ni poids n'a été retouché après
lecture des résultats de cette expérience.
