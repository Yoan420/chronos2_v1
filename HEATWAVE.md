# Candidat températures et chaleur persistante

Expérience indépendante : aucun changement à `Forecast.ps1`, aux modes Both /
Complete, à NuclearCWE, aux modèles activés ou à leurs exports.

## Lancement

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Heatwave.ps1' -Action Run -Zones FR
```

Cette commande produit deux candidats : `heatwave_autonomous` et
`heatwave_kalman`. La date par défaut, **11 septembre 2026**, est volontairement
figée pour comparer exactement la même période que NuclearCWE.
`-Zones FR` désigne le pays du **prix prédit** ; les cinq températures FR, DE,
BE, NL et ES sont toutes utilisées, même pour cette seule zone de sortie.

Actions disponibles :

- `Prepare` : complète les sources manquantes, copie et scelle les entrées et
  comparateurs, puis vérifie les données réelles sans lancer Chronos.
- `Run` : Prepare, replay Chronos, correcteur, Kalman, attribution et rapports.
  Les checkpoints de ce candidat sont réutilisés après une interruption.
- `Report` : régénère les HTML à partir des résultats figés, sans réentraîner
  et sans rafraîchir les observations ou Storm.
- `Status` : état du processus, étape et nombres de checkpoints. Les nombres
  de fichiers ne constituent pas une validation des résultats.
- `Sync` : sources de température seulement, sans exiger de modèle de référence.
- `-DryRun` : affiche la commande sans calcul ni création de fichiers.

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Heatwave.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\Heatwave.ps1' -Action Report -Zones FR
```

Le premier replay est coûteux : les prédictions historiques doivent réellement
être recalculées avec les nouveaux inputs. Le launcher utilise uniquement les
poids locaux Chronos-2 ; il n'entraîne pas de LoRA. Un verrou empêche deux
lancements simultanés sur le même espace d'expérience.

Les sorties attendues après achèvement sont dans :

```text
runs/experiments/heatwave_v1/2026-09-11/fr/reports/
  forecast_fr_2026-09-11_heatwave_autonomous.html
  forecast_fr_2026-09-11_heatwave_kalman.html
  heatwave_report_audit.json
  heatwave_incumbent_comparison.json
```

## Base et paramètres

La configuration est dans `config/heatwave.yaml`. Par défaut, le candidat
repart du **modèle nucléaire FR de référence**, pas de NuclearCWE : cela isole
l'apport des températures après le résultat insatisfaisant du test CWE.
Les deux chaînes sont :

1. Chronos-2 + nucléaire FR + températures/chaleur → correcteur résiduel.
2. La même chaîne → banque Kalman gouvernée.

Les 17 nouvelles variables entrent dans le contexte et le futur connu de
Chronos, les features du correcteur et les covariables du Kalman, notamment le
groupe `market` réellement consommé par `linear_market`. Les prix historiques,
charges résiduelles, recettes, graines et hyperparamètres restent ceux du run
de référence. Les autres candidats de la banque Kalman restent disponibles.

Pour tester une autre configuration, copier le YAML et choisir un **nouvel**
`output_root` sous `runs/experiments/heatwave_v1/`, puis utiliser `-Config`.
Un snapshot existant refuse les changements de recette/code/sources ; il ne
réécrit pas silencieusement un résultat déjà scellé.

Une expérience cumulative est possible avec `baseline_kind: nuclear_cwe`,
`baseline_root: runs/experiments/nuclear_cwe_v1` et un nouvel output_root.
Elle conserve les capacités Pmax BE/NL dans les trois étages. Les hyperparamètres
ne sont pas optimisés sur l'année évaluée. Pour une autre date ou un autre pays
de sortie, un run de référence complet et figé doit déjà exister ; sinon le
préflight refuse de présenter une comparaison incomplète.

## Sources et causalité

Séries Saturn : `meteo.nrjscan.{fr,de,be,nl,es}.t_2m.index.fcst.d`.
Ce sont des **indices quotidiens de température prévus à 2 m**, en °C,
diffusés sans changement aux 23/24/25 heures physiques du jour. Ce ne sont pas
des observations, des températures horaires, des Tmax/Tmin ou des nuits chaudes.
La pondération géographique exacte de l'indice n'est pas certifiée ici.
L'unité est issue du catalogue météo existant ; les métadonnées Saturn
n'exposent pas de champ d'unité directement vérifiable.

Le jeu isolé contient 804 jours du 30 juin 2024 au 11 septembre 2026 pour chaque
pays. Les 37 journées manquantes ont été collectées ; les cinq caches météo
existants et leurs audits sont restés identiques (SHA vérifiés).

Chaque jour D utilise sa prévision interrogée **à D−1 08 h civil**. Aucune
interpolation ni substitution par de la météo réalisée n'est autorisée.
Les requêtes historiques as-of sont auditées, mais les dates de publication
originales du fournisseur ne sont pas exposées : cela reste un diagnostic
historique, pas une preuve de disponibilité opérationnelle certifiée.

## Définition des 17 variables

Pour chacun des cinq pays : température prévue `T_D`, excès de chaleur et
durée de persistance, soit 15 variables. On ajoute deux indices régionaux.

Le seuil quotidien est `max(plancher_pays, quantile90(T_[D−365,D)))` : le jour
courant est exclu de la référence. Les planchers préfixés sont 20 °C pour
FR/DE/BE/NL et 24 °C pour ES. Avant 60 jours d'historique, seul le plancher est
utilisé. Le préfixe météo antérieur au replay est conservé pour calculer ces
seuils ; pendant l'année évaluée, 365 jours antérieurs sont disponibles.

- Excès : `max(T_D − seuil_D, 0)`.
- Persistance : nombre de jours consécutifs d'excès positif, plafonné à 7.
  Chaque jour passé conserve sa propre prévision et son propre seuil de l'époque.
  On n'attend pas des jours futurs pour qualifier le jour courant.
- Fraction régionale : part des cinq pays avec au moins 3 jours consécutifs
  d'excès prévu.
- Refroidissement : moyenne non pondérée des `max(T_D − 22 °C, 0)` des cinq pays.

Ces seuils sont des heuristiques fixées avant le test, **pas une définition
météorologique officielle de canicule**. Ce diagnostic permet de tester le
signal thermique, sans promettre qu'il explique tous les spikes de prix.

## Évaluation et lecture des rapports

Le protocole conserve l'ancre froide du 9 septembre 2024 et la calibration
glissante de 365 jours. Il réévalue 365 jours, sur les mêmes observations et le
même snapshot Storm que le modèle de référence. Les observations du jour de
livraison ne sont injectées que dans le reporting, jamais dans sa prédiction.

Pour le snapshot du 11 septembre déjà observé, les Statistics couvrent
**12 septembre 2025 – 11 septembre 2026**, soit 8 760 heures observées.
Storm couvre 8 759 heures appariées avec son exception DST auditée. La fenêtre
technique FINAL365 avant livraison est distincte : 11 septembre 2025 –
10 septembre 2026. Si la livraison n'est pas encore observée, elle reste vide.

Les deux HTML reprennent Statistics, prix moyens, calendrier, mode nuit,
comparaison horaire avec Storm et attribution des variables. Une section
complémentaire compare : année complète, jours de chaleur persistante prévue,
autres jours, et 24–26 juin 2026 à titre exploratoire. Chaque variant a sa propre
table. Les trois modèles comparés à Storm utilisent exactement les mêmes heures.

Un gain de MAE/RMSE sur quelques jours ne suffit pas : vérifier d'abord la
performance annuelle, puis la couverture et les gains sur les journées chaudes.
Ne pas choisir des seuils après avoir observé les erreurs de cette année.
L'attribution explique une sensibilité de la prédiction, pas un effet causal
physique ; sur le rapport Kalman, elle est celle du modèle autonome amont.
Aucune promotion ni activation automatique n'est effectuée.
