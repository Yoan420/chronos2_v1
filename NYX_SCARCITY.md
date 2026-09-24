# NYX Scarcity — challenger indépendant des pointes de prix

Ce laboratoire compare le NYX `nuclear_kalman` existant, Storm et un expert
conditionnel de forte sous-estimation. **Il ne modifie ni Forecast.ps1, ni le
mode Complete/Both/All, ni les modèles, caches ou rapports opérationnels.**
Il ne déclenche aucun entraînement Chronos et ne permet aucune activation.

## Lancer

Depuis n'importe quel répertoire PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Run -RefreshSources
```

Cette commande sélectionne la dernière livraison commune des exports nucléaires
FR/DE/BE/NL, copie les sources nécessaires dans un espace privé, complète les
suffixes manquants autorisés à partir de Saturn, fige les entrées, entraîne le
challenger et génère son rapport HTML. Deux requêtes matérialisatrices au plus
sont lancées simultanément, avec délais et tentatives bornés. Aucun téléchargement
de 730 jours n'est implicitement déclenché : Refresh exige un historique local
déjà qualifié et refuse un suffixe trop long (30 jours par défaut).
Les disponibilités et prévisions vent/solaire sont complétées d'abord, puis
TTF/EUA. Ce second rafraîchissement vérifie aussi 20 jours de recouvrement
avant d'ajouter le suffixe, avec un seul processus réseau. Les prix historiques
du fichier source restent inchangés ; les copies et leurs logs sont conservés.

Pour figer précisément la comparaison discutée le 14 septembre :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Run -RefreshSources `
    -Countries FR,DE,BE,NL -DeliveryDay 2026-09-15 -EndDay 2026-09-14
```

Sans `-RefreshSources`, Run ne fait **aucune requête réseau** : il utilise les
caches locaux. Une entrée obligatoire manquante impose alors le retour exact
à NYX pour l'heure concernée. La couverture figure dans le rapport.

Autres actions :

```powershell
# Lecture seule : disponibilité, dates, provenance et éligibilité.
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Audit

# Préparer, puis calculer séparément sur les mêmes entrées figées.
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Prepare -RefreshSources
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Backtest
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\Scarcity.ps1' -Action Report
```

`Backtest`, `Report` et `Status` acceptent `-RunDirectory` pour viser un snapshot
précis. Leurs pays, dates et paramètres restent ceux du snapshot. Un backtest
terminé est réutilisé sans réentraînement. Après interruption, Backtest recommence
déterministement le calcul de l'expert sur le snapshot ; il ne reprend pas un
réseau neuronal et ne prétend pas reprendre un fit au milieu de ses itérations.
Un changement de code avant la fin exige un nouveau Prepare.

`-Action Refresh` actualise seulement les copies privées et fournit le chemin
de leur configuration. Si une source échoue, Run avec RefreshSources s'arrête
avant l'entraînement ; le diagnostic et les logs sont conservés. Pour tester
explicitement les données partielles, utiliser cette configuration avec
`-Config <chemin fourni>` **sans** `-RefreshSources` : les trous restent visibles,
jamais remplacés par zéro ou par des valeurs réalisées.

## Où modifier le modèle

`config/nyx_scarcity.yaml` contient les paramètres :

- `data` : chemins de remplacement des seules séries autorisées, hypothèses
  d'efficacité gaz et facteur d'émission ; un chemin ne change pas l'identité
  de série exigée par son audit.
- `policy` : fenêtre d'apprentissage, séparation temporelle de calibration,
  complexité des arbres, seuil de forte erreur, probabilité de déclenchement,
  plafond de correction et gouvernance.
- `refresh` : nombre de processus, suffixe maximal et délais réseau.

Pour comparer des recettes, copier ce YAML et conserver un `output_root` sous
`runs/experiments/nyx_scarcity_v1/`. Chaque Run/Prepare crée un nouveau snapshot ;
ne jamais retoucher un snapshot scellé. Une autre liste de pays constitue une
autre expérience, car l'apprentissage mutualise les pays sélectionnés.

## Méthodologie

### Entrées : information disponible au cutoff

Les prévisions NYX, leurs quantiles, les observations et Storm sont lus dans
les exports existants avec leurs contrôles d'axes horaires et de provenance.
Leur sélection est figée avant l'entraînement. Storm et les prix réalisés du
jour ne sont jamais des variables explicatives.

L'expert ajoute les prévisions de charges résiduelles FR/DE/BE/NL, nucléaire
FR, vent et solaire séparés, disponibilités Pmax gaz/charbon/lignite/nucléaire
pertinentes, températures disponibles et coûts gaz calculés à partir de TTF/EUA.
Il construit des indicateurs centrés sur la zone : demande résiduelle rapportée
au gaz disponible, écarts entre zones, tension simultanée et rampes du soir.
Les variations horaires sont calculées **dans la même prévision de journée**,
pas en mélangeant des origines différentes.

Ces marges sont des **proxies**, pas une réserve physique complète : le périmètre
national du parc, l'hydraulique flexible, les batteries, les réserves et les
imports ne sont pas entièrement couverts. Le vent et le solaire déjà soustraits
dans la charge résiduelle ne le sont pas une seconde fois. Une disponibilité
nulle est une vraie valeur, distincte d'une donnée manquante ; le plancher de
dénominateur sert seulement aux ratios et ne crée aucun MW de capacité.

Le CGC est ici un proxy `(TTF + facteur_CO2 × EUA) / rendement`, hors coûts
variables d'exploitation. **La série CGC native Saturn n'est pas branchée**
tant que son identifiant et sa convention exacte ne sont pas qualifiés.

Les révisions et amplitudes entre vintages ne sont calculées que si plusieurs
vintages antérieurs à 08 h existent. Un historique avec une seule prévision
par origine ne devient pas un ensemble météorologique : aucune distribution
météo artificielle n'est présentée comme observée. Les ensembles probabilistes
de demande/vent/solaire restent une extension à qualifier.

**JAO est exclu**, y compris des features, tant que la référence du domaine et
ses frontières ne sont pas qualifiées au cutoff strict de 08 h. Aucune publication
10 h 30 ni donnée post-coupling n'est substituée.

### Détection et amplitude

La cible est `prix observé - prévision NYX`. Dans chaque bloc d'apprentissage,
le seuil de queue est le maximum de 50 €/MWh et du percentile95 des erreurs
antérieures du pays. Ce seuil n'utilise pas les dates de calibration ultérieures.

Un classifieur à arbres estime la probabilité de dépasser ce seuil, sans
suréchantillonnage ni poids modifiant la fréquence des classes. Une calibration
logistique est apprise sur les 28 derniers jours de l'historique, exclus du fit
du classifieur. Un deuxième modèle estime la sévérité logarithmique conditionnelle
à la queue positive. Trop peu d'événements entraîne une abstention explicite.

La proposition ponctuelle est une médiane d'un mélange simplifié « correction
nulle / erreur positive extrême ». Ce n'est **pas** probabilité × amplitude
présenté abusivement comme un P50, ni une distribution complète des prix.
La probabilité doit dépasser0,6 par défaut ; la proposition est ensuite plafonnée
à400 €/MWh. Ce plafond propre au challenger ne modifie pas les plafonds du
correcteur résiduel ou du Kalman opérationnel.

### Gouvernance et intervalles

Les poids0/0,25/0,5/1 sont comparés quotidiennement par pays sur les prévisions
véritablement produites hors échantillon aux origines précédentes. Le poids0
reste toujours disponible. Une correction requiert suffisamment de jours
modifiés, un gain sur la queue et un garde-fou sur la MAE globale. La marge
d'incertitude utilise des observations agrégées par jour, pas quatre pays
supposés indépendants. Aucun résultat Storm ni EVA ne choisit ces poids.

Les intervalles du candidat sont estimés séparément à partir des erreurs
signées hors échantillon déjà connues, par régime prédit si le support le permet,
sinon sur l'ensemble des heures passées. Les états et les replis sont tracés.
En repli total, les intervalles NYX restent inchangés. Il n'existe aucune
garantie de couverture future ou de non-dégradation annuelle.

### Fenêtre et limites de validation

Le rapport score exactement les 365 dernières journées retenues ; les lignes
live non observées sont affichées séparément et ne contribuent pas aux scores.
Les comparaisons NYX/candidat/Storm utilisent les mêmes heures ; une heure Storm
manquante n'est pas remplie. Les heures sans intervention restent dans la MAE.

L'historique gelé contient365 jours de prévisions NYX, **pas un second historique
de365 jours avant cette année**. L'apprentissage commence après90 jours, puis
utilise un historique croissant plafonné à365 jours à chaque refit hebdomadaire.
La calibration fait partie de ce support, pas un supplément. Les premiers jours
conservent NYX. Mettre `minimum_training_days: 365` impose le support complet,
mais ne crée pas l'année manquante. Une vraie évaluation365 avec365 jours
d'entraînement à chaque origine nécessite d'abord cet historique NYX préalable.

Les horodatages des labels sont une hypothèse explicite D-1 à18 h ; le replay
et les révisions finales ne prouvent pas les vintages opérationnels originaux.
Le laboratoire n'est donc pas éligible à une promotion automatique. Juin et
le14 septembre ont déjà été vus : ils restent des diagnostics, pas un test
final indépendant. Une validation prospective est nécessaire après sélection.

## Résultats et audit

Chaque snapshot contient le panel figé, la recette, les SHA256 des entrées et du
code, les prévisions, les blocs d'entraînement, les décisions de gouvernance,
les métriques, les agrégats horaires/journaliers et `nyx_scarcity_report.html`.
`latest.json` indique le dernier résultat terminé. Les sous-périodes actives
et les propositions non gouvernées sont clairement séparées du score annuel.

Le HTML est autonome, sans CDN : prix/quantiles par date et pays, Statistics
des prix moyens, MAE et RMSE, moyennes horaires, calendrier d'erreur, queues
top1/top5 **définies ex post**, Brier/précision/rappel et couverture des intervalles.
Le mode nuit adapte également les graphiques.

### Premier test réalisé le 14 septembre 2026

Snapshot : `20260914T125027Z_ed1dc982`. Période d'évaluation :
15 septembre 2025 au 14 septembre 2026 inclus ; livraison affichée séparément :
15 septembre 2026. Les 8 760 heures par pays sont conservées. La comparaison
triple utilise 8 735 heures communes par pays : 25 heures sans Storm dans les
entrées retenues ne sont pas reconstruites. Il ne s'agit pas d'une absence de
prévision NYX ou du challenger.

| MAE, EUR/MWh | NYX nucléaire + Kalman | Challenger gouverné | Storm |
| --- | ---: | ---: | ---: |
| FR | 12,078891 | 12,078891 | 11,858560 |
| DE | 11,078194 | 11,078194 | 11,457842 |
| BE | 11,150389 | 11,150389 | 10,946671 |
| NL | 11,162528 | 11,165486 | 11,007282 |

**Conclusion : non concluant, aucune activation.** Une seule correction a été
autorisée : NL, 1er juillet à 20 h locale, +25,838888 EUR/MWh. Elle a augmenté
l'erreur. Les corrections aux heures extrêmes du 14 septembre à 19 h sont
nulles : p=0,4848 en DE et p=0,5903 en BE, sous le seuil de 0,6 ; en NL,
p=0,6037 mais la gouvernance rejette les poids non nuls.

Le détecteur a proposé 32 alertes sur les heures évaluées où il était disponible :
12 vraies, 20 fausses et 202 événements manqués. Précision 37,5 %, rappel 5,61 %.
Ces événements sont les fortes erreurs positives de NYX, pas la définition DWT
du papier. La calibration et la discrimination doivent progresser ; abaisser
le seuil après avoir vu le 14 septembre ne constituerait pas une validation.

32 blocs ont été entraînés, 21 se sont abstenus (historique initial ou événements
de calibration insuffisants). Les données obligatoires sont complètes sur tout
le panel. La borne 365 jours d'entraînement n'est jamais atteinte dans ce
premier replay : les blocs entraînés utilisent 91 à 364 jours avant leur
origine, dont 28 jours de calibration séparée.

Validation logicielle : 159 tests dédiés passent, un test de création de lien
symbolique est ignoré car indisponible sur cet environnement Windows. Les tests
des lanceurs opérationnels ont également passé sans modification de ces fichiers.
Le HTML et ses contrôles interactifs sont testés automatiquement ; la vérification
visuelle par le navigateur automatisé n'a pas pu être réalisée (erreur de
création des ressources du navigateur).

## Références de conception

- [Prévision de spikes et marge offre-demande,2024](https://www.mdpi.com/2571-9394/6/1/7).
- [Prévisions probabilistes de demande et renouvelables,2026](https://www.sciencedirect.com/science/article/pii/S0960148126006701).
- [Distributions des prix day-ahead,2025](https://www.sciencedirect.com/science/article/pii/S0140988325008187).
- [Processus flow-based Core](https://www.entsoe.eu/bites/ccr-core/day-ahead/).

Ces références motivent des essais ; leurs gains ne sont pas des performances
mesurées ou promises pour NYX.

### Article fourni le 14 septembre : Ma, Chen et Meng (2026)

*An interpretable machine learning approach for forecasting the occurrence of
extreme electricity prices in the day-ahead market*, Journal of the Operational
Research Society, [DOI 10.1080/01605682.2026.2660989](https://doi.org/10.1080/01605682.2026.2660989).
Le PDF fourni a été lu sans modification. Cette note distingue les pistes du
papier de ce qui est effectivement implémenté dans le premier candidat.

Le travail combine un seuil dynamique global/local pondéré par la volatilité
(DWT), deux classifieurs XGBoost pondérés pour les extrêmes hauts/bas, et SHAP.
Les entrées comprennent charge résiduelle, charge et production prévues,
vent/solaire, prix passés, gaz, charbon, CO2 et indice de risque géopolitique.
La séparation est chronologique (75 % apprentissage, 25 % test), avec validation
temporelle pour le réglage des hyperparamètres. Ce n'est pas notre replay
glissant de 365 jours, ni une estimation directe de la taille d'un spike.

Trois précautions pour une transposition :

1. **Information à 08 h.** Les équations du DWT utilisent une référence globale
   sur la série entière et une fenêtre locale incluant l'observation courante.
   Pour un signal utilisable avant l'enchère, références, normalisation et
   fenêtres devront être recalculées uniquement sur les observations déjà
   publiées au cutoff. Les clôtures combustibles de D-1 ne sont pas disponibles
   à D-1 08 h ; il faut le dernier cours réellement publié à cet instant.
2. **Probabilité et déséquilibre.** La pondération XGBoost peut aider la
   discrimination des classes rares, mais son score brut n'est pas une
   probabilité calibrée. Il faudrait une recalibration sur un bloc temporel
   non pondéré avant de l'utiliser pour une correction ou un signal économique.
3. **Ne pas confondre les métriques.** Dans la figure 4, scénario 30 jours,
   753 vrais spikes sont détectés pour 1 137 fausses alertes et 129 spikes
   manqués : précision des alertes = 753/(753+1137), soit 39,8 %, rappel =
   753/(753+129), soit 85,4 %. La précision pondérée 93,9 % du tableau 6 ne
   signifie donc pas que 93,9 % des alertes de spike sont correctes. Nos
   comparaisons doivent montrer la classe extrême séparément et le coût des
   fausses interventions, en plus des courbes PR, du Brier et de la MAE.

À tester ensuite dans des recettes séparées, sans retoucher le premier test :
un classifieur de régime de prix élevé en complément du détecteur d'erreur NYX,
un seuil DWT causal, un classifieur pondéré puis recalibré, et des explications
SHAP du classifieur. Les poids SHAP d'un risque ne sont ni des causes physiques
ni des poids de contribution au prix final. Le charbon et les prix retardés
peuvent être évalués par ablation ; l'indice géopolitique n'est pas prioritaire,
son importance étant faible dans ce travail. Ces variantes ne sont pas
présentées comme déjà entraînées ou validées par notre premier rapport.
