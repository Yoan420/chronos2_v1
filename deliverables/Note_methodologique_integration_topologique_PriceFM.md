# Note méthodologique — intégration topologique inspirée de PriceFM

Date : 21 août 2026  
Périmètre : France, Allemagne, Belgique, Pays-Bas et Espagne

## 1. Objectif

L’objectif était de tester si l’information provenant des marchés électriques voisins pouvait améliorer la prévision de prix day-ahead de Chronos-2, sans fragiliser le modèle actuel et sans introduire Storm ou MKOnline dans les données d’apprentissage.

Le travail s’inspire de **PriceFM**, qui montre l’intérêt d’un contexte spatial parcimonieux : seules les zones reliées au marché cible sont autorisées à contribuer. Dans l’ablation publiée, le masque topologique obtient une MAE de 14,28, contre 16,41 sans masque et 17,05 avec un masque aléatoire. Le papier montre toutefois aussi que les voisins ne sont pas toujours utiles : le rayon optimal est nul pour certains pays, notamment la France et l’Allemagne.

Sources :

- Papier : https://arxiv.org/abs/2508.04875
- Dépôt des auteurs : https://github.com/runyao-yu/PriceFM
- Fiche du modèle : https://huggingface.co/RunyaoYu/PriceFM

## 2. Adaptation retenue

L’intégration est une réimplémentation indépendante, dite *clean-room*. Aucun code, poids de modèle ou jeu de données PriceFM n’a été copié dans Chronos-2.

La nouvelle couche intervient après la prévision autonome et avant un éventuel blend MKOnline :

1. Chronos-2 produit les quantiles P10, P50 et P90.
2. Le correcteur résiduel autonome actuel produit sa prévision.
3. La couche topologique estime une petite correction commune aux trois quantiles.
4. Si la variante autonome est validée, MKOnline peut ensuite être mélangé en dernier.
5. Si une validation échoue, la prévision actuelle est conservée à l’identique.

Appliquer la même correction à P10, P50 et P90 préserve l’ordre des quantiles et la largeur de la bande. P10 et P90 restent des scénarios bas et haut, sans garantie de couverture probabiliste.

## 3. Informations utilisées

La couche utilise uniquement des informations disponibles au moment causal de la prévision :

- les prévisions de charge résiduelle des cinq pays, archivées avec leur date de révision ;
- le prix day-ahead du jour précédent, avec un retard physique de 24 heures ;
- des variables calendaires déterministes ;
- un graphe de voisinage limité aux interconnexions représentées dans le périmètre étudié.

Il n’y a ni interpolation, ni remplissage silencieux des valeurs absentes. Storm est chargé seulement après le gel complet du candidat, uniquement pour l’évaluation. MKOnline n’est jamais une variable d’entrée de la couche topologique.

## 4. Protocole de validation

La fenêtre scellée couvre exactement 365 jours, du 12 août 2025 au 11 août 2026. Elle est découpée chronologiquement en apprentissage initial, sélection, développement, B1, B2 et test final.

Le rayon topologique et l’intensité de la correction sont choisis avant l’ouverture des périodes formelles. Une variante n’est retenue que si elle respecte simultanément les règles suivantes :

- gain de MAE d’au moins 0,05 €/MWh face au modèle autonome actuel ;
- gain positif sur chacune des deux moitiés chronologiques ;
- borne basse de l’intervalle bootstrap journalier à 95 % strictement positive.

Une période suivante n’est ouverte que si la précédente passe toutes les règles. Storm ne participe à aucune sélection, aucun réglage et aucune décision de promotion.

## 5. Résultats

| Pays | Variante choisie | Étape de décision | Gain MAE vs autonome | Gain MAE vs Storm | Win rate journalier vs Storm | Décision |
|---|---:|---:|---:|---:|---:|---|
| France | rayon 0, échelle 0,25 | B1 | +0,011 €/MWh | +1,418 €/MWh | 56,7 % | rejetée : gain autonome insuffisant et robustesse non démontrée |
| Allemagne | rayon 1, échelle 0,50 | B1 | +0,051 €/MWh | +1,712 €/MWh | 63,3 % | rejetée : une moitié et le bootstrap échouent |
| Belgique | rayon 1, échelle 0,25 | final | +0,089 €/MWh | −4,035 €/MWh | 30,0 % | validée face à l’autonome, mais ne bat pas Storm |
| Pays-Bas | rayon 0, échelle 0,25 | final | +0,032 €/MWh | −3,183 €/MWh | 33,3 % | rejetée au test final |
| Espagne | rayon 0, échelle 0,50 | B1 | +0,094 €/MWh | N/A | N/A | rejetée : robustesse chronologique non démontrée |

Le résultat important est donc nuancé : l’information des voisins apporte un signal significatif en Belgique face au modèle autonome, mais elle ne permet pas encore de battre Storm sur cette période finale. Pour les quatre autres pays, le mécanisme de sécurité conserve le modèle actuel.

## 6. Rapport aligné sur les 365 jours

Le rapport annuel utilise exactement la même fenêtre scellée que les modèles actuels : du **12 août 2025 au 11 août 2026**, soit **365 jours locaux et 8 760 heures physiques**. Cette fenêtre contient une journée de 25 heures et une journée de 23 heures ; aucune heure n'est interpolée.

Le candidat topologique pur ne possède pas de prévision scellée sur les 365 jours complets. Le rapport ne complète donc jamais artificiellement les périodes manquantes sous son nom. Il présente deux lectures distinctes :

- la **stratégie séquentielle gouvernée**, utilisée comme résultat annuel principal : le modèle autonome reste actif jusqu'à ce qu'une gate formelle antérieure autorise la topologie ;
- le **shadow causal formel**, présenté comme diagnostic secondaire : il utilise les prédictions topologiques uniquement sur les holdouts formels effectivement ouverts, sans réécrire les décisions de validation.

| Pays | MAE autonome 365 j | MAE stratégie gouvernée | Gain annuel | Topologie active | Gain shadow formel | Storm apparié | MKOnline production |
|---|---:|---:|---:|---:|---:|---:|---:|
| France | 12,165 | 12,165 | 0,000 €/MWh | 0 h | +0,001 €/MWh | 11,653 | 11,231 |
| Allemagne | 11,240 | 11,240 | 0,000 €/MWh | 0 h | +0,004 €/MWh | 11,153 | N/A |
| Belgique | 11,312 | 11,284 | +0,028 €/MWh | 2 160 h / 90 j | +0,040 €/MWh | 10,334 | N/A |
| Pays-Bas | 11,175 | 11,147 | +0,028 €/MWh | 2 160 h / 90 j | +0,040 €/MWh | 10,614 | 10,648 |
| Espagne | 9,728 | 9,728 | 0,000 €/MWh | 0 h | +0,008 €/MWh | N/A | N/A |

Pour la Belgique et les Pays-Bas, le taux de victoire sur les seuls jours où la topologie est active est respectivement de **71,1 %** et **64,4 %**. Sur les jours de fallback, la stratégie est exactement égale au modèle autonome ; le rapport les compte comme des égalités et affiche séparément le taux de victoire des jours actifs.

Storm est comparé sur l'intersection exacte de **8 759 heures** en France, Allemagne, Belgique et aux Pays-Bas, avec une heure DST manquante laissée absente. Storm n'est pas disponible pour l'Espagne. Le blend MKOnline de production est affiché uniquement comme référence pour la France et les Pays-Bas ; ses poids ne sont ni recalculés ni utilisés par la topologie.

La génération initiale du rapport annuel a été effectuée avec :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Report365 -Countries FR,DE,BE,NL,ES
```

Il est publié sous :

`runs/experiments/pricefm_topology_annual_365_v1/reports/topology_annual_365_index.html`

Le dossier est immuable : une relance vers le même emplacement refuse de l'écraser. Cette opération ne fait aucun nouvel apprentissage, aucun nouveau calcul de prédiction et aucune écriture dans `runs/live`.

## 7. Backtest avec réentraînement glissant sur 365 jours

Un second protocole a été ajouté pour mesurer la couche dans les mêmes conditions de durée que les modèles actuels, mais avec un **véritable réentraînement quotidien**. Pour chacune des 365 journées évaluées, un nouveau correcteur est ajusté uniquement sur les 365 journées locales immédiatement précédentes. Cela représente 365 ajustements par pays, soit 1 825 entraînements au total.

La période évaluée reste strictement identique : du **12 août 2025 au 11 août 2026**, soit **8 760 heures physiques**. Les fenêtres d’apprentissage respectent les changements d’heure : leur longueur physique peut donc être de 8 759, 8 760 ou 8 761 heures tout en contenant toujours exactement 365 journées locales.

Le rayon, l’échelle de correction et les hyperparamètres proviennent de la recette déjà scellée. Ils ne sont jamais resélectionnés à partir des résultats rolling. Le correcteur rolling part de la prévision OOF `ensemble`, seule prévision autonome disponible sans trou sur les 730 jours nécessaires à l’amorce et à l’évaluation. Il est ensuite comparé au correcteur autonome actuel `residual_corrected` sur les mêmes 8 760 heures.

| Pays / variante | MAE rolling365 | MAE modèle actuel | Gain MAE | Win rate journalier | Intervalle bootstrap 95 % du gain |
|---|---:|---:|---:|---:|---:|
| France autonome | 12,608 | 12,165 | −0,442 €/MWh | 42,2 % | [−0,690 ; −0,195] |
| France + MKOnline | 11,562 | 11,231 | −0,331 €/MWh | 37,5 % | [−0,451 ; −0,211] |
| Allemagne autonome | 11,608 | 11,240 | −0,369 €/MWh | 41,4 % | [−0,628 ; −0,107] |
| Belgique autonome | 11,416 | 11,312 | −0,104 €/MWh | 48,2 % | [−0,331 ; +0,122] |
| Pays-Bas autonome | 11,666 | 11,175 | −0,491 €/MWh | 37,0 % | [−0,707 ; −0,279] |
| Pays-Bas + MKOnline | 11,064 | 10,648 | −0,416 €/MWh | 32,9 % | [−0,538 ; −0,292] |
| Espagne autonome | 9,878 | 9,728 | −0,150 €/MWh | 46,0 % | [−0,342 ; +0,040] |

Le résultat rolling365 est défavorable dans les cinq pays. Les gains positifs observés sur certains holdouts courts ne se généralisent pas à l’ensemble de l’année lorsque le correcteur est réestimé chaque jour. Le signal topologique doit donc rester en shadow et ne doit remplacer aucun modèle actuel. Pour FR et NL, les poids MKOnline de production sont appliqués après la prévision rolling sans aucun recalibrage ; MKOnline ne devient jamais une variable d’apprentissage.

Storm reste un comparateur Statistics chargé seulement après le scellement des 365 prévisions rolling. Sur les 8 759 heures appariées, seule la variante France rolling + MKOnline devance légèrement Storm en MAE (+0,091 €/MWh), mais elle reste nettement moins bonne que le blend FR actuellement en production (−0,331 €/MWh). Ce résultat ne justifie donc aucune promotion.

Commande simple :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Rolling365 -Countries FR,DE,BE,NL,ES -Workers 5
```

Rapport et artefacts scellés :

`runs/experiments/pricefm_topology_rolling365_v1/reports/topology_rolling365_index.html`

Chaque pays contient les 8 760 prévisions horaires, les 365 fenêtres d’apprentissage auditées, les fichiers Statistics, les rapports HTML et leurs empreintes SHA-256. Aucun fichier sous `runs/live` n’est modifié.

## 8. Conséquence pour MKOnline

Le blend n’est évalué qu’après validation de la variante autonome correspondante. La France et les Pays-Bas n’ayant pas passé toutes les étapes autonomes, leurs poids MKOnline n’ont pas été recalibrés et leur blend de production reste inchangé. Cela évite de modifier un blend sur la base d’un correcteur non validé.

## 9. Utilisation quotidienne

La calibration de référence est figée et son modèle opérationnel est scellé sous :

`runs/experiments/pricefm_topology_operational_v1`

Le lanceur unique permet ensuite d’appliquer la recette au forecast d’une journée précise, sans réentraîner le modèle et sans modifier les archives live :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Apply -Countries FR,DE,BE,NL,ES -Mode Production
```

Le mode `Production` applique la correction validée à la Belgique, conserve l’autonome actuel pour l’Allemagne et l’Espagne, et conserve exactement le blend MKOnline courant pour la France et les Pays-Bas. Pour obtenir côte à côte les sorties autonomes et blendées :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\Forecast.ps1' -Action Topology -TopologyStage Apply -Countries FR,DE,BE,NL,ES -Mode Both -DeliveryDay 2026-08-22
```

Les modes disponibles sont `Autonomous`, `Blend` et `Both`. `Blend` est limité à FR et NL, car ce sont les deux seuls blends MKOnline promus. Comme les correcteurs FR et NL ont été rejetés par les gates autonomes, ces deux sorties blendées sont volontairement des copies contrôlées du blend actuel : aucun poids n’est recalculé et aucun gain topologique n’est revendiqué.

## 10. Livrables et statut

Les résultats et rapports détaillés sont publiés sous :

`runs/experiments/pricefm_topology_v1`

Le dossier contient les prévisions scellées, les décisions par étape, les contrôles de causalité, les empreintes SHA-256 et un rapport HTML détaillé par pays. Une application quotidienne vérifiée a également été produite sous :

`runs/experiments/pricefm_topology_daily/2026-08-22/both`

Elle contient cinq rapports autonomes et deux rapports blendés, ainsi qu’un index HTML. Les archives live actuelles n’ont pas été modifiées.

Le rapport comparatif demandé sur 365 jours est disponible sous :

`runs/experiments/pricefm_topology_annual_365_v1`

Il contient une série annuelle auditée par pays, cinq rapports HTML détaillés, un index multi-pays et les manifests de checksums associés. Le rapport distingue explicitement la stratégie mixte gouvernée de la performance d'un modèle topologique pur, qui reste déclarée non estimable sur 365 jours à partir des seuls artefacts scellés.

Le backtest avec réentraînement quotidien est disponible sous :

`runs/experiments/pricefm_topology_rolling365_v1`

Statut recommandé : conserver cette intégration en shadow et ne promouvoir aucun des correcteurs rolling365. Les résultats annuels montrent que la couche topologique réentraînée chaque jour est moins performante que les modèles actuels. Toute nouvelle version devra modifier ou enrichir la représentation du contexte, puis être réévaluée par le même protocole causal avant une éventuelle promotion.
