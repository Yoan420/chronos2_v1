# NYX Congestion — laboratoire indépendant

Cette expérience cherche à prévoir **quelles contraintes réseau vont devenir actives**, avec quelle intensité, ainsi que la **pression régionale globale** des contraintes publiées, puis à employer ces prévisions pour corriger la distribution d'erreur de NYX. Elle ne remplace ni le modèle opérationnel, ni `Forecast.ps1`, ni ses rapports.

## Utilisation

La configuration est `config/nyx_congestion.yaml`. Le chemin `source_suite` désigne le snapshot physique figé utilisé comme comparaison. Toutes les sorties restent sous `runs/experiments/nyx_congestion_v1`.

```powershell
# Vérifier la commande sans données, fichiers ni entraînement
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestion.ps1' -Action Run -DryRun

# Collecte historique explicite, indépendante de la production
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestion.ps1' -Action Collect `
  -StartDay 2025-09-15 -EndDay 2026-09-14

# Préparer un nouveau snapshot immuable
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestion.ps1' -Action Prepare

# Entraîner chronologiquement, évaluer et générer le rapport
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestion.ps1' -Action Run

# État / vérification / nouveau rendu sans réentraînement
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestion.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\NyxCongestion.ps1' -Action Report
```

`-RunDirectory` permet de sélectionner explicitement un snapshot existant pour `Run`, `Backtest`, `Report` ou `Status`. `Backtest` suit la même chaîne que `Run`. Une reprise réutilise uniquement les checkpoints liés aux mêmes sources, paramètres, code et environnement. Si ceux-ci ont changé, créer un nouveau snapshot avec `Prepare` ; ne jamais réécrire les sceaux d'un ancien entraînement. `Run` et `Prepare` ne collectent pas silencieusement de nouvelles données.

Les données incomplètes peuvent empêcher l'apprentissage : une erreur d'insuffisance d'historique n'autorise ni à transformer des labels inconnus en zéros, ni à supprimer le contrôle chronologique. La collecte peut être longue ; les jours complets déjà scellés sont réutilisables.

## Modèles comparés

| Variante | Contenu |
| --- | --- |
| `congestion_control_direct` | Nouveau détecteur et distribution d'erreur, fondamentaux et domaine initial, sans signaux CNEC ni pression régionale prévus |
| `congestion_control_governed` | Même proposition, avec gouvernance passée |
| `congestion_direct` | Même protocole avec les signaux de congestion prévus hors entraînement |
| `nyx_congestion` | Variante congestion avec gouvernance ; candidat primaire fixé avant l'évaluation |

Comparateurs conservés : NYX opérationnel `nuclear_kalman`, Storm, `network_fuel_direct` et `nyx_physical_p50` du snapshot antérieur. Les deux nouvelles chaînes utilisent les mêmes lignes admissibles d'entraînement et de calibration. Les replis sur NYX sont inclus dans les scores : ils ne disparaissent pas du dénominateur.

## Étape 1 — activation et intensité

Pour chaque CNEC-heure présent et identifié dans le domaine initial, le classifieur estime la probabilité d'au moins une unité de marché (MTU) active. Une activation signifie un prix dual strictement supérieur à `1e-9`. L'intensité cible est la moyenne des prix duaux de l'heure, y compris les MTU inactives, conditionnellement à cet événement horaire. Le pas de publication historique est respecté : une MTU horaire avant le 1er octobre 2025, quatre quarts horaires à partir de cette date.

L'identité stricte tient compte de l'élément, de la contingence et de la direction. Une paire de contingences inverse n'est pas assimilée à la paire originale. Les variables d'entrée sont issues des fondamentaux et de **l'initialComputation** disponibles avant 08 h. Les prix duaux et le domaine final servent seulement à constituer des cibles historiques qualifiées, après leur disponibilité.

Un second modèle apprend la moyenne conditionnelle positive avec une perte de Poisson, sur `prix dual horaire / CGC`. Le CGC courant restitue ensuite l'unité. Le signal attendu est :

`E[λ | informations] = P(activation | informations) × E[λ | activation, informations]`.

Il s'agit d'un **signal explicatif**, pas d'un prix électrique ni d'un P50. Les sensibilités PTDF du domaine initial permettent d'en construire des expositions directionnelles par zone, avec une référence symétrique égale à la moyenne des PTDF des quatre pays CWE : la France n'est donc pas forcée à zéro. Cela ne résout pas un dispatch complet et ne constitue pas une estimation certifiée d'import disponible.

Les fits ont lieu chaque semaine sur au plus 365 jours passés. Les 28 derniers jours sont réservés à une calibration chronologique des probabilités ; des minima de jours, de classes et d'événements distincts conditionnent l'apprentissage. Un échantillon uniforme déterministe peut limiter le volume CORE, sans sur-échantillonner les activations. Le comparateur « prior historique » est la fréquence d'activation dans cet échantillon CORE, pas un nouveau modèle recalibré sur le test.

## Étape 1 bis — pression régionale globale

L'expert CNEC ne peut apprendre directement les activations de contraintes absentes de son univers initial. Un second signal cible donc une pression **agrégée sur toutes les contributions duales publiées qualifiées**, y compris hors de cet univers. Après couplage, on calcule `D_pays = somme λ × (PTDF_FR − PTDF_pays)`, puis :

`G_pays = D_pays − min(D_FR, D_DE, D_BE, D_NL)`.

Les **quatre pays CWE doivent être qualifiés simultanément**. On ne recalcule jamais le minimum à partir de trois pays. Cette transformation est invariante à un changement commun de référence : si les quatre D augmentent de la même constante, G ne change pas.

Elle supprime donc aussi la composante commune : **G ne permet pas à lui seul de détecter une hausse de prix identique et simultanée dans les quatre pays**. Ce risque reste à porter par les fondamentaux, la tension offre/demande et le CGC ; l'expert régional complète ces signaux sans les remplacer.

La cible de classification est fixée à `G ≥ 50 €/MWh`. Une régression de moyenne conditionnelle positive, normalisée par le CGC, apprend `E[G | G ≥ 50]`. Le signal attendu est `P(G ≥ 50) × E[G | G ≥ 50]` : une estimation de la **pression sévère**, pas de `E[G]` toutes pressions confondues. Pour les métriques d'intensité de cette étape, les faibles G ont une cible nulle ; le tableau du 14 septembre affiche aussi G réalisé non tronqué.

À l'inférence, la moyenne conditionnelle est bornée inférieurement à **50 €/MWh**, ce qui respecte le support de l'événement même lorsque le CGC courant baisse. Il ne s'agit pas d'un plancher sur le prix NYX ni sur le signal attendu : ce dernier peut rester inférieur à 50 lorsque la probabilité est faible.

Cet expert n'utilise à l'inférence que les fondamentaux et informations réseau pré-08 h. Ses labels agrégés sont des résultats historiques post-coupling, jamais des variables du forecast courant. Il suit un apprentissage hebdomadaire, une calibration chronologique de 28 jours et des sorties OOF. Probabilité, intensité conditionnelle et pression sévère attendue deviennent trois variables supplémentaires du correcteur aval. **Aucune de ces valeurs n'est ajoutée manuellement à NYX et aucune n'est appelée P50.**

Le prior régional présenté dans les diagnostics est la fréquence d'événement CORE **groupée sur les quatre pays**, et non une fréquence spécifique à chaque pays. Une partie du gain de classification face à ce prior peut donc venir des différences géographiques ; la comparaison des prévisions de prix au contrôle apparié reste essentielle. Lorsque les événements positifs d'un pays sont très rares, ses scores ne fournissent pas à eux seuls une preuve robuste.

## Étape 2 — une correction qui reste un P50

Seuls les signaux des étapes 1 et 1 bis **produits antérieurement hors entraînement** sont autorisés comme variables historiques du correcteur. On ne recalcule pas ses lignes d'entraînement avec un expert qui a déjà vu leur résultat. L'étape 2 exige au moins 90 jours distincts avec des heures OOF admissibles par pays ; il ne s'agit pas d'une garantie de 90 journées complètes.

Le nouveau détecteur estime le risque d'une erreur positive extrême. La distribution d'erreur à deux régimes, conditionnée par les variables, produit ensuite son quantile 0,5. Le transport de la queue par le CGC reprend le mécanisme physique précédent. Le produit `probabilité × intensité` n'est **pas** ajouté directement à NYX et une moyenne n'est pas renommée médiane.

La proposition reste à la hausse, bornée, et peut être réduite à zéro par la gouvernance sur les résultats historiquement disponibles. Les intervalles sont calibrés chronologiquement. Les limites du modèle et des données peuvent provoquer une abstention ; elles ne garantissent ni non-régression annuelle ni couverture future des intervalles.

## Lecture du rapport

- **Prix** : MAE, RMSE, taux de victoire contre Storm par heure, MAE quotidienne et prix moyen quotidien, prix moyens. Les huit modèles partagent exactement la même intersection d'heures physiques sur les 365 derniers jours ; les jours incomplets sont exclus des seuls scores quotidiens. La livraison future n'est pas évaluée.
- **EVA** : politique inchangée du laboratoire économique, même allocation de capacité, seuils et coûts. La référence reste le prix observé D−1 à la même heure civile, un proxy non exécutable à 08 h. Ce n'est pas un P&L de trading démontré.
- **Activation** : Brier, average precision (AP, pas l'aire PR trapézoïdale), courbe précision–rappel et fiabilité en dix classes fixes. Expert et prior partagent les mêmes CNEC-heures qualifiés. Si l'échantillon ne contient pas les deux classes, AP n'est pas présentée comme une mesure de discrimination.
- **Intensité** : MAE et RMSE conditionnelles sur les heures actives ; MAE/RMSE du signal attendu sur toutes les heures qualifiées. Le zéro constant est montré comme contrôle trivial. Une moyenne se juge notamment par son erreur quadratique ; la MAE seule peut favoriser la médiane nulle lorsque les activations sont rares.
- **Cas du 14 septembre** : prix par zone, probabilité prévue et activation observée, intensités prévues/réalisées. Les contraintes mises en évidence sont choisies pour l'affichage uniquement ; cette sélection ne revient jamais dans l'apprentissage.

Les CNEC-heures sont comptés une seule fois au niveau du réseau, **pas multipliés par les quatre pays**. Leur dénominateur diffère des heures-pays des scores de prix. Le filtre pays n'altère donc pas les scores du détecteur réseau.

Un tableau distinct quantifie la part des contributions absolues ex post couverte par l'univers initial, annuellement et au 14 septembre à 19 h. Il emploie `|λ × (PTDF FR − PTDF pays)|`, avec la France comme référence **de diagnostic uniquement**, contrairement à la référence symétrique CWE des variables du modèle. La couverture est un ratio des sommes de contributions, pas une moyenne des ratios horaires. Son dénominateur est nul pour FR : le ratio reste absent, pas égal à 0 %. Les heures non qualifiées sont comptées séparément et exclues des sommes comparables.

## Limites et critère de poursuite

Le sous-ensemble initial `Presolved` omet parfois un CNEC qui devient actif après le couplage. Le cas Vigy exact du 14 septembre à 19 h illustre cette limite : sa paire inverse ne permet pas de reconstituer son identité manquante. Les contraintes hors univers initial, les heures non publiées et les correspondances ambiguës doivent rester visibles dans l'audit. Un label manquant ou non qualifié n'est jamais remplacé par « aucune congestion ».

La collecte actuelle fournit surtout des preuves de watermark historique, pas des captures immuables faites à 08 h en temps réel. Les vintages de prix de référence et l'historique d'échauffement présentent également des limites ; la fenêtre évaluée de 365 jours n'implique pas 365 jours d'entraînement disponibles avant son premier jour.

Le rapport mesure aussi les disponibilités tardives : un label est compté comme tardif si son horodatage audité dépasse **08 h Europe/Paris le jour de sa livraison**, l'origine du forecast de livraison suivante. Les nombres d'heures-pays, d'heures physiques distinctes, le retard maximal et le détail mensuel sont calculés sur les données de la période sélectionnée, sans valeurs figées. Une date tardive peut correspondre à une révision groupée de l'archive, et non à la première publication effective au marché. Le replay n'anticipe pas cette date : il attend que le label soit admissible. Ainsi, historique entièrement couvert, fenêtres effectivement entraînables et prévisions d'experts prêtes sont trois notions différentes. L'échauffement et les labels tardifs peuvent réduire les calibrations possibles et provoquer des abstentions ; ils ne doivent pas être contournés pour améliorer artificiellement un score.

L'année déjà examinée est exploratoire. Pour poursuivre, vérifier le gain du candidat contre son **contrôle apparié**, les fausses corrections, les erreurs pendant les pics, la calibration et l'EVA à politique fixe. Geler ensuite le candidat sur de nouvelles journées sans choisir une règle pays/heure après lecture des résultats. **Aucune promotion automatique, aucun changement du pipeline de production.**

## Premier replay terminé — 15 septembre 2026

Snapshot : `20260915T150242Z_5b87c16a`. Évaluation du 15/09/2025 au 14/09/2026, FR/DE/BE/NL, sur 34 940 heures-pays communes. Les démarrages et replis restent inclus.

| Prévision | MAE €/MWh | RMSE €/MWh |
| --- | ---: | ---: |
| NYX opérationnel | 11,367500 | 21,875529 |
| Contrôle physique direct | 11,462477 | 21,944129 |
| Congestion directe | 11,442775 | 21,926902 |
| Congestion gouvernée | 11,367491 | 21,875198 |
| Réseau + CGC précédent, direct | 11,337640 | 21,583696 |
| Storm | 11,317589 | 20,787627 |

L'ajout des signaux de congestion améliore légèrement le contrôle direct apparié, **mais les deux prévisions directes dégradent NYX**. La version gouvernée reste pratiquement identique à NYX ; son gain économique simulé est de seulement 17,81 € sur le portefeuille alternatif de 100 MW, avec une référence D−1 hypothétique. Ce n'est pas un gain économique convaincant ni un motif de promotion.

Les deux experts amont sont néanmoins informatifs : AP CNEC 0,478 ; AP régionale 0,568. Leur prior est une référence historique simple, regroupée entre pays pour l'expert régional. Le 14/09 à 19 h, le risque régional DE vaut 73,76 %, mais l'intensité conditionnelle estimée de 147,22 €/MWh reste inférieure à G réalisé de 398,50 €/MWh. La contrainte réciproque Vigy mise en avant par le modèle n'est pas le CNEC réellement actif : ne pas compter cette confusion comme une identification exacte réussie.

Le correcteur aval ne dispose que de 19 entraînements admissibles sur 53 tentatives et de 9 748 heures-pays prêtes. Sa tentative du 14/09 est refusée faute d'événements de calibration distincts suffisants ; les quatre P50 de 19 h restent donc ceux de NYX, malgré les signaux amont disponibles. Le rapport montre séparément ces deux états.

Prochaine hypothèse à tester dans une expérience séparée : une calibration chronologique adaptée à la rareté des événements (historique plus long ou calibration hiérarchique avec repli explicite), puis une meilleure estimation de leur intensité. Le protocole doit être fixé avant le nouveau test ; ne pas simplement relâcher un garde-fou pour obtenir une correction sur la journée déjà observée. Le candidat actuel reste hors production.
