# Expert des mouvements extrêmes et décision économique gouvernée

Ce laboratoire **ne modifie pas les forecasts opérationnels**. Il réutilise
le forecast, P10 et P90 de `nuclear_kalman`, et produit une nouvelle alternative
de **position simulée**, `nuclear_kalman_extreme_governed`. `Forecast.ps1`,
`Both`, `All` et `Complete` ne sont ni modifiés ni appelés. Aucun ordre n'est passé.

## Lancer

Depuis n'importe quel dossier PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Run
```

Le lanceur fige les entrées, entraîne l'expert chronologiquement, évalue les
positions et produit un rapport HTML indépendant. Il ne télécharge pas de
données et n'entraîne pas Chronos-2, LoRA, le correcteur résiduel ou le Kalman.

```powershell
# Contrôle en lecture seule / affichage de commande sans aucun calcul
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Audit
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -DryRun

# Séparer le gel des données et l'évaluation
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Prepare
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Backtest

# Lire l'état ou régénérer uniquement le rapport déjà calculé
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\EconomicExpert.ps1' -Action Report
```

`-RunDirectory` permet de cibler un snapshot pour Backtest/Report/Status.
Un Backtest terminé n'est pas réentraîné. Un calcul interrompu est relancé
de façon déterministe depuis ses entrées figées ; cette version ne reprend
pas un estimateur à mi-entraînement. Les anciens rapports ne sont pas écrasés.

## Comparaison figée et reproductible

La recette par défaut est dans `config/economic_extreme_policy.yaml`.
Elle référence le snapshot EVA `20260910T085138Z_13a598c8` :

- Évaluation : **11 septembre 2025 au 10 septembre 2026**, 365 jours civils.
- Support d'entraînement et d'évaluation : 11 septembre 2024 au 10 septembre 2026.
- Pays : FR, DE, BE, NL, **100 MW au total**, 25 MW fixes par pays.
- Comparateurs : règle initiale `nuclear_kalman`, nouvelle politique, Storm,
  absence de position. Les alternatives ne s'additionnent pas.
- Prix observés, prévisions Storm, référence veille, forecast et quantiles de
  base sont identiques au snapshot initial, y compris leurs valeurs manquantes.
- Même masque horaire, mêmes coûts : 0,5 EUR/MWh de transaction et 0,5 de
  slippage. Le seuil de la règle de base est 5 + 1 = 6 EUR/MWh.

Les résultats vont dans
`runs/experiments/economic_extreme_policy_v1/snapshots/<identifiant>/`.
`latest.json` désigne le dernier rapport terminé ; `latest_prepared.json`,
le dernier jeu d'entrées préparé. Modifier les sources nécessite un nouveau
Prepare/Run ; modifier le code entre Prepare et Backtest est refusé.

Pour expérimenter une autre recette, copier le YAML et utiliser `-Config`.
Ne pas sélectionner une recette sur son meilleur résultat annuel puis
présenter cette même année comme validation indépendante.

## Méthode

### Expert direct des mouvements

La cible est `prix observé − référence veille`, pas une correction du forecast
Kalman. Aucun forecast Storm ni score réalisé futur n'entre dans les features.
Le forecast Kalman intervient ensuite uniquement dans la décision de base.

Les entrées initiales sont les prévisions de charge résiduelle des quatre pays,
la génération nucléaire française prévue, les différences régionales,
le calendrier et le proxy de prix de la veille. Les sources Saturn déjà
matérialisées sont vérifiées par empreinte et filtrées au cutoff **D−1 08 h civil**.
Les réparations DST amont restent explicitement documentées dans les audits.
Pas de variables réseau dont la qualification est encore bloquée.

À chaque réentraînement, l'expert utilise exactement les **365 dates civiles
précédentes**, avec au moins 95 % d'heures utilisables. Les labels doivent être
déclarés disponibles au cutoff. Une histoire plus courte provoque un repli
vers la règle de base, jamais un entraînement raccourci silencieux.
Le réentraînement est hebdomadaire par défaut ; l'inférence et la gouvernance
sont quotidiennes. Il ne s'agit donc pas de 365 réentraînements par an.

Trois régimes sont appris par gradient boosting : mouvement inférieur ou égal
à −50 EUR/MWh, régime intermédiaire, mouvement supérieur ou égal à +50 EUR/MWh.
Deux régressions apprennent les amplitudes des queues ; une moyenne historique
d'entraînement est utilisée si moins de 30 événements sont disponibles.
L'espérance de mouvement combine les probabilités de régime et les amplitudes.
Les sorties du classifieur sont des **scores non calibrés**, pas des probabilités
garanties de profit. La contribution du régime calme utilise sa moyenne passée.

### Trois règles préspécifiées, pas une recherche du meilleur seuil annuel

1. `baseline` : conserver la position issue du forecast nucléaire + Kalman.
2. `reduce_opposite` : diviser par deux la position si elle est opposée au
   mouvement extrême anticipé.
3. `blend_tail` : mélanger à 50/50 la fraction de position de base et le sens
   indiqué par l'expert. Deux sens opposés peuvent donc donner une position nulle.

Les règles 2/3 ne proposent une intervention que si le score extrême atteint
0,6 et que l'espérance de mouvement dépasse le seuil net en valeur absolue.
Les fractions restent dans `[-1, 1]`, donc sans levier supplémentaire.

### Gouvernance uniquement sur des décisions historiques hors entraînement

Chaque jour, les propositions produites les jours précédents sont comparées
à la règle de base, après coûts et uniquement lorsque leurs observations
sont devenues disponibles. Storm n'intervient pas dans cette sélection.

La fenêtre est de 60 jours, avec au moins 28 journées complètes et 7 journées
où la proposition aurait modifié une position. Le gain moyen journalier
normalisé est pénalisé par une marge d'incertitude et une régularisation.
Sans gain suffisamment étayé, la règle de base est conservée. Le démarrage
est inclus dans les résultats annuels, sans inventer un passé OOF de Kalman.

Le poids de gouvernance vaut 0 ou 1 dans cette version : il désigne le choix
de politique, pas nécessairement une intervention. Le fractionnement des
positions est effectué par les règles 2/3. Un poids de 1 peut conserver
exactement la position initiale si le déclencheur extrême n'est pas actif.

Cette prudence ne garantit ni non-dégradation future ni EVA positive. La
marge d'incertitude n'est pas un test de significativité post-sélection.

## Lire le rapport

La section spécifique affiche l'EVA contre Storm, le gain contre la règle
de base, les interventions, les choix de gouvernance et les résultats pendant
les spikes. Les autres Statistics conservent leurs définitions antérieures.
Le hit ratio utilise le **sens de la position effectivement retenue** ; les
P10/P90 et la confiance descriptive restent ceux du forecast d'origine.

Les fichiers `folds.parquet` et `governance.parquet` tracent les fenêtres
d'entraînement, dates de disponibilité et décisions de sélection.
`decisions.parquet` conserve les propositions et leurs décisions effectives.
Les manifestes scellent entrées, sorties, code et versions des bibliothèques.
Le rapport est régénérable sans les caches sources une fois le snapshot terminé.

## Limites indispensables

- La référence est le prix day-ahead de la veille : **non négociable pour la
  livraison prévue**. Ce n'est pas un P&L de trading réalisable.
- Les heures de publication des labels sont une hypothèse conservatrice
  (veille de livraison à 18 h), pas une preuve de publication historique.
- Les caches peuvent contenir des révisions. Le PIT historique et le caractère
  OOF neuronal de la baseline ne sont pas certifiés.
- L'année affichée a déjà servi au diagnostic qui a motivé l'expert. Le
  calcul est chronologique, mais ne constitue pas un test final indépendant.
- Le gain éventuel est celui d'une **politique de décision différente**,
  pas une nouvelle mesure de précision du forecast à règle identique.
- Les trous DST et les heures non appariées restent présents. Le rapport
  n'extrapole pas en EUR/MW/an une année qui n'est pas entièrement appariée.
- Les options météo/combustibles sont désactivées dans le premier test.
  Les activer crée une nouvelle expérience ; les données manquantes restent
  explicites et ne doivent pas être remplacées par des valeurs futures.

Avant un usage réel : une période prospective intacte, des vintages attestés,
un prix négociable du même produit, et des contraintes de portefeuille réalistes
restent nécessaires. Aucune promotion opérationnelle n'est incluse ici.

## Premier résultat — 10 septembre 2026

Snapshot : `20260910T094048Z_191882db`.

Les **212 réentraînements** (53 par pays) ont réussi, sans fenêtre raccourcie.
L'expert dispose de 35 032 heures-pays d'inférence valides ; le masque économique
commun, qui exige aussi Storm, retient 35 028 heures-pays.

La gouvernance n'a retenu **aucune intervention** : les propositions n'ont pas
atteint le minimum de 7 journées modifiées dans leurs fenêtres de validation
passées de 60 jours. Les 28 premières journées par pays sont le démarrage
normal de la gouvernance. Les poids nuls ne signifient pas que l'entraînement
n'a pas été effectué.

| Politique | EVA simulée vs Storm | Gain vs règle de base |
|---|---:|---:|
| Nucléaire + Kalman, règle initiale | −412 867,50 EUR | 0 EUR |
| Expert extrêmes gouverné | −412 867,50 EUR | 0 EUR |

**Aucune amélioration économique n'est démontrée par ce premier test.** Aucun
seuil n'a été modifié après lecture du résultat. Il faut examiner la rareté et
la qualité des propositions, puis concevoir une nouvelle expérience validée
chronologiquement, et non activer une politique pour obtenir un meilleur score
sur cette même année.

Le diagnostic indépendant des propositions explique ce résultat : l'expert
déclenche 3 150 heures-pays de risque extrême, mais rejoint la position initiale
dans environ 98,2 % de ces situations. La réduction ne propose que 28 heures
modifiées ; le mélange, 57. Les fenêtres de gouvernance contiennent au maximum
2 à 5 journées modifiées selon le pays. Même sans ce critère de fréquence,
leurs bornes de gain régularisées restent insuffisantes.

À titre de **diagnostic ex post seulement**, imposer la réduction sans
gouvernance aurait perdu 6 949,03 EUR de plus que la baseline ; imposer le
mélange aurait perdu 13 503,34 EUR de plus, sur le même masque et après frais.
Ces variantes n'ont pas été activées. Ces montants n'ont pas servi à choisir
les politiques quotidiennes, calculées auparavant sur leur seul passé.
