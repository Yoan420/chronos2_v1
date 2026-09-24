# Expert de prix `nuclear_kalman_extreme`

Expérience indépendante : elle ne modifie ni `Forecast.ps1`, ni les modes Both / All / Complete, ni les archives opérationnelles. Elle ne réentraîne pas Chronos-2, LoRA ou Kalman et n'active aucun modèle.

## Lancer

```powershell
& 'C:\Users\BQ6757\chronos2_v1\PriceExpert.ps1' -Action Run
```

`Audit` vérifie les entrées sans écrire ; `Prepare` fige les entrées et la recette ; `Backtest` exécute le dernier snapshot préparé ; `Report` régénère le HTML ; `Status` lit l'état du dernier snapshot préparé. Pour cibler un résultat : `-RunDirectory 'C:\Users\BQ6757\chronos2_v1\runs\experiments\nuclear_kalman_extreme_v1\snapshots\IDENTIFIANT'` avec Backtest, Report ou Status. `-DryRun` affiche seulement la commande. Les chemins relatifs se résolvent depuis le projet, pas le répertoire courant du terminal.

Paramètres : `config/nuclear_kalman_extreme.yaml`. Une modification se teste avec un **nouveau** Run ; les snapshots terminés ne sont pas réentraînés. Un Backtest interrompu repart des entrées gelées avec la même graine ; il n'y a pas de reprise de chaque arbre individuellement. Les verrous empêchent les évaluations concurrentes du même snapshot, les SHA vérifient les entrées et les résultats, et les rapports sont publiés atomiquement.

## Ce qui change par rapport au premier expert économique

Le premier expert prédisait `observé − référence de la veille` et modifiait une position simulée. L'ajouter directement à `nuclear_kalman` compterait deux fois une partie du signal.

Ce nouvel expert apprend **`observé − prévision nuclear_kalman`**, avec une régression par arbres à gradient (perte absolue, recette fixe). Il utilise les fondamentaux déjà audités — charges résiduelles FR/DE/BE/NL, nucléaire FR, calendrier et prix connu de la veille — et le contexte de la prévision de base (niveau, écart à la référence, largeur d'intervalle). Storm n'est ni une variable ni une cible de sélection.

La proposition s'applique uniquement si le résidu prédit atteint au moins 10 EUR/MWh en valeur absolue. La correction brute est plafonnée à ±100 EUR/MWh ; les poids candidats sont 0, 0,25 et 0,5. Le candidat final est donc `prix de base + poids × correction proposée`. Ces paramètres sont figés avant le test, pas ajustés pour reproduire les épisodes de juin déjà observés.

La même règle économique transforme ensuite chaque forecast en BUY / SELL / FLAT. Les positions du premier POC ne sont **pas** réutilisées. Seuil, coûts, référence et allocation totale de 100 MW (25 MW par pays) restent identiques à la comparaison d'origine.

## Limite importante de l'historique

Le snapshot contient les **365 jours de prévisions nuclear_kalman du 11/09/2025 au 10/09/2026**, mais pas 365 jours supplémentaires de ce modèle pour entraîner le nouvel expert en amont. Les premières valeurs `identity` d'anciens replays correspondent au modèle autonome : elles ne sont pas des prévisions Kalman interchangeables et ne sont pas utilisées.

Le diagnostic utilise donc **90 jours de démarrage, puis une calibration progressive plafonnée à 365 jours**. Il affiche les dates et effectifs réellement utilisés par chaque entraînement. Pendant le démarrage, le candidat est strictement égal à la baseline. Mettre `minimum_training_days: 365` impose une calibration pleine : avec ce snapshot, cela signifie zéro entraînement et retour à la baseline sur toute l'année, pas un faux historique complété.

Un vrai test de 365 jours avec 365 jours complets de calibration préalable nécessitera une année supplémentaire de prévisions historiques qualifiées de la même baseline. Ce travail n'est pas dissimulé derrière les 730 jours de données fondamentales disponibles.

## Chronologie et gouvernance

À chaque origine, seules les journées strictement passées et les observations dont la disponibilité déclarée précède le cutoff D−1 08 h Europe/Paris sont utilisées. Réentraînement hebdomadaire : le premier créneau atteignant le minimum de 90 jours tombe après 91 jours dans cette recette. Les journées de 23 / 25 heures conservent leurs identités UTC.

La gouvernance compare les propositions réellement émises hors entraînement sur les 60 derniers jours. Elle requiert 28 journées complètes et 7 journées avec corrections proposées, un gain moyen de MAE non négatif et un gain économique net passé supérieur à une marge d'incertitude et de régularisation. Sinon, poids nul. Les prix observés du jour ne choisissent jamais son poids. Les premières interventions ne peuvent donc pas commencer immédiatement après le lancement de l'apprentissage.

Ces contrôles réduisent le risque ; ils ne **garantissent pas** la non-dégradation annuelle. Le bilan annuel est affiché tel quel, même défavorable, sans réécrire les mauvais jours en baseline après coup.

## Lecture du rapport

- Comparaison de prix baseline / candidat / Storm : MAE, RMSE, biais, détails quotidiens, heures corrigées et épisodes extrêmes.
- Comparaison économique : P&L simulé, EVA, drawdown, heures, saisons, spikes et confiance, avec la même période annuelle et le démarrage inclus.
- Les métriques de prix utilisent un masque commun de prix disponibles ; l'EVA exige en plus la référence disponible. Leurs effectifs peuvent donc différer. La MAE portefeuille regroupe les erreurs horaires de tous les pays ; elle n'est pas l'erreur du prix moyen des quatre zones.
- Sur les heures corrigées, P10/P90 restent vides : modifier un point de prévision ne crée pas des intervalles calibrés. Sur les heures de repli, ceux de la baseline sont conservés.
- Le sous-ensemble `expert_ready` distingue la phase où un expert existe ; ce n'est pas un substitut opportuniste au bilan des 365 jours.

**Qualification :** résultat de recherche rétrospectif sur une année déjà examinée, référence veille non négociable, disponibilité des labels fondée sur l'hypothèse documentée D−1 18 h, PIT et OOF neuronal de la baseline non certifiés. Les contrôles causaux du nouveau correcteur ne certifient pas rétroactivement les modèles amont. Aucune preuve de profit exécutable ni promotion en production.

## Premier résultat — 10 septembre 2026

Snapshot : `runs/experiments/nuclear_kalman_extreme_v1/snapshots/20260910T102502Z_fb61f194` ; rapport `nuclear_kalman_extreme_report.html`.

Les quatre pays ont été évalués jusqu'au bout : 160 entraînements, utilisant effectivement 91 à 364 jours passés ; 52 créneaux en repli pendant le démarrage. Les 35 040 heures-pays sont conservées. Les métriques de prix comparent 35 036 heures-pays ; les métriques économiques 35 028, en raison des heures manquantes de Storm et du proxy de référence aux changements d'heure.

**Aucune correction n'a été autorisée par la gouvernance.** Le candidat reste donc strictement égal à `nuclear_kalman` : MAE annuelle regroupée 11,277454 EUR/MWh ; P&L net hypothétique 19 260 868,56 EUR ; EVA vs Storm **−412 867,50 EUR**. Ce résultat ne démontre aucune amélioration, mais ne dégrade pas la baseline.

Le nouvel expert a bien été appris : il propose 389 corrections horaires sur 93 dates distinctes, mais ces corrections changent seulement 18 positions avec le poids 0,25 et 31 avec le poids 0,5. Toutes les bornes de gain économique utilisées par la gouvernance restent au plus à −0,02 EUR/MWh après marge et régularisation. Le refus ne vient donc pas seulement du minimum de sept jours modifiés.

Diagnostic **non gouverné** des deux propositions fixées avant le run, sans les sélectionner après lecture de l'année :

| Variante examinée | MAE (EUR/MWh) | Gain net vs baseline (EUR) | EVA vs Storm (EUR) |
|---|---:|---:|---:|
| Baseline / candidat gouverné | 11,277454 | 0,00 | −412 867,50 |
| Proposition poids 0,25 | 11,269035 | +3 325,56 | −409 541,94 |
| Proposition poids 0,5 | 11,265347 | −515,06 | −413 382,56 |

Ces diagnostics ne justifient pas de choisir rétroactivement 0,25 ni d'assouplir les garde-fous pour rendre le résultat positif. L'amélioration économique de cette proposition reste très modeste et insuffisamment étayée dans les fenêtres de validation passées.

Sur les 24–26 juin 2026, aucune des deux propositions ne change le P&L. Leur petit gain de MAE concerne les Pays-Bas uniquement. Les résidus prédits ne récupèrent pas encore l'amplitude des grands spikes : ce test ne résout donc pas la faiblesse ciblée sur cet épisode.

Vérifications : 250 tests unitaires/intégration réussis sur les suites économiques existantes et nouvelles ; QA JavaScript des 11 graphiques, dix sélections et mode nuit, concordance HTML/Parquet. Pas de validation visuelle par navigateur revendiquée. Empreintes des fichiers opérationnels et du code expérimental inchangées pendant le calcul ; aucune activation effectuée.
