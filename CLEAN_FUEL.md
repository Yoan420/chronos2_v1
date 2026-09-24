# NYX nucléaire + Clean Fuel Costs — variante indépendante

`CleanFuel.ps1` injecte les CGC/CCC dans **toute la chaîne**, comme le nucléaire
français : contexte et covariables connues de Chronos-2, apprentissage du
correcteur résiduel, puis covariables et recalibration de Kalman.

Deux sorties : `clean_fuel_autonomous` et `clean_fuel_kalman`.
**Forecast.ps1, Both/All/Complete, les modèles actifs et leurs exports restent inchangés.**

## Commandes

```powershell
& 'C:\Users\BQ6757\chronos2_v1\CleanFuel.ps1' -Action Audit -DeliveryDay 2026-09-18
& 'C:\Users\BQ6757\chronos2_v1\CleanFuel.ps1' -Action Run -DeliveryDay 2026-09-18 -Zones FR,DE,BE,NL
& 'C:\Users\BQ6757\chronos2_v1\CleanFuel.ps1' -Action Status -DeliveryDay 2026-09-18 -Zones FR,DE,BE,NL
```

Le run nucléaire de cette date doit être achevé, avec inputs, observations et
Storm de reporting figés. `Run` collecte les indices manquants, fige une copie
séparée des entrées, recalcule Chronos, réentraîne le correcteur puis Kalman,
et génère les rapports. `Prepare` s'arrête après préparation des données ;
`Sync` après collecte ; `Report` ne fait aucun nouvel entraînement.

Configuration : `config/nyx_clean_fuel_full.yaml`. Options utiles :
`-Device auto|cpu|cuda`, `-Threads 4`, `-Workers 2`, `-SkipAttribution`.
Les hyperparamètres et l'ancre de calibration sont hérités du run nucléaire
figé : aucune sélection sur les résultats de l'année de test.

La première exécution est longue : nouveau replay neuronal sur environ 730
jours, refits journaliers du correcteur puis Kalman. Elle reprend les checkpoints
validés après interruption. Les caches sont propres à ce candidat. Changer de
jour réutilise les entrées et calculs identiques ; les clés vérifient les données,
la recette et le code. Ne pas supprimer le verrou d'un processus actif.

## Rapports

Sous `runs/experiments/nyx_clean_fuel_full_v1/<livraison>/<pays>/<identité>/reports` :
rapports autonomes et Kalman au même format que les rapports nucléaires :
prévision day-ahead, quantiles, Statistics, prix moyens, calendrier, comparaison
Storm, performances horaires, variables et mode nuit. Un index commun est
publié après les pays demandés. Aucun export de production n'est remplacé.

Comparaisons par famille : candidat autonome contre `nuclear_autonomous`,
candidat Kalman contre `nuclear_kalman`, mêmes observations et heures physiques.
Statistics porte sur les 365 derniers jours observés et inclut le jour affiché
si son prix est déjà publié. Sans observation, il reste vide. Le backtest historique
couvre les 365 jours précédant la livraison ; le jour de livraison est distinct.
L'attribution est recalculée pour les nouveaux inputs ; en cas d'échec elle est
déclarée indisponible, jamais remplacée par celle de l'ancien modèle. Sur le
rapport Kalman, la neutralisation concerne Chronos+correcteur et conserve le
décalage Kalman : ce ne sont pas des poids causaux complets du filtre.

## Indices utilisés

Tous sont des indices Saturn natifs, déjà en **EUR/MWh électrique** :

| Indice | Hub / série | Formule native |
| --- | --- | --- |
| CGC France | PEG / `power.fr.price.everyday.cgc.da.index.eurmwh` | `2 × gaz DA + 0.368 × EUA` |
| CGC Allemagne | THE / `power.de.price.everyday.cgc.da.index.eurmwh` | idem |
| CGC Belgique | ZEE / `power.be.price.everyday.cgc.da.index.eurmwh` | idem |
| CGC Pays-Bas | TTF / `power.nl.price.everyday.cgc.da.index.eurmwh` | idem |
| CCC API2 M1 | `ccc.price.mid.api2.everyday.month.1.ice.eurmwh` | `2.63 × (API2 USD/t / EURUSD / 6.9776 + 0.34 × EUA)` |

CGC : rendement implicite 50 %, CO₂ 0.368 t/MWh électrique.
CCC : rendement 1/2.63 ≈ 38.02 %, 6.9776 MWh thermique/t,
CO₂ 0.34 t/MWh thermique, donc 0.8942 t/MWh électrique.
**Ni second ajout de CO₂, ni deuxième division par le rendement.** Pas de VOM
ajouté. CCC API2 n'est pas un coût du lignite. PSV/Italie n'entre pas dans cette
première variante FR/DE/BE/NL.

## Note méthodologique

Pour chaque livraison D, Saturn est interrogé tel qu'il était à **D−1 08:00
heure locale**. On conserve la dernière cotation finie datée strictement avant
la journée du cutoff, jamais sa clôture du soir. Âge maximal : 176 h. La valeur
connue est diffusée sur les 23/24/25 heures physiques de D. Il s'agit d'un
**proxy de coût connu à l'origine**, pas d'une prévision parfaite du futur gaz
ou charbon. Les dates de cotation restent dans la banque quotidienne auditée.

Les cinq canaux sont ajoutés aux inputs nucléaires existants sans changer les
algorithmes. Les nouvelles prévisions Chronos servent à recalibrer le résidu sur
les 365 jours antérieurs, quotidiennement. Kalman reçoit ce nouvel upstream et
les coûts directement, sans réutiliser l'ancien historique corrigé. Aucun prix
observé de D ni Storm n'entre dans les inputs. Aucun plancher de prix n'est imposé.
Le correcteur conserve son décalage commun des quantiles ; la calibration de
leur couverture n'est pas modifiée par cette expérience.

L'amorçage du premier historique est diagnostique, comme pour la référence.
La requête as-of Saturn est contrôlée, mais le versionnement historique des
formules et les timestamps fournisseur ne sont pas attestés indépendamment.
Il faut donc mesurer le gain annuel/par pays et sur les spikes, puis le valider
sur de nouvelles journées. Aucune promotion automatique et aucun gain présumé.

La première ablation limitée au correcteur reste disponible via
`run_clean_fuel.py` et `config/nyx_clean_fuel.yaml` ; elle n'est **pas** la variante
complète lancée par `CleanFuel.ps1` et ses résultats éventuels restent dans
`runs/experiments/nyx_clean_fuel_v1`.
