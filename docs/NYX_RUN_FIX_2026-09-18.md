# NYX — échec du 18 septembre 2026 corrigé

## Incident

Le lancement `cf75a1d7b8d243a9a932787293a1cf86`, demandé le 18 septembre à 08:59 pour la livraison du **19 septembre 2026**, s'est arrêté à 09:01. Les sources communes avaient été synchronisées ; les quatre pays ont ensuite échoué pendant l'actualisation des observations, avant le calcul des modèles.

Erreur : `PostAuctionObservationDivergenceError`, « la source post-enchere diverge de la cible canonique ». Sur le recouvrement de sept jours, les sources ENTSO-E et EPEX divergeaient le 13 septembre : maxima de 7,1425 €/MWh en BE, 20,7975 en FR et 0,005 en DE/NL. Aucun problème d'affichage ni de lancement PowerShell n'a été identifié.

## Cause et correction

La lecture réelle des deux sources a confirmé que les observations canoniques étaient complètes jusqu'au 18 septembre inclus. Seules les 24 heures du 19 septembre manquaient dans les deux sources, ce qui est normal avant publication de l'enchère. EPEX ne pouvait donc ajouter **aucune** observation.

`_fetch_latest_observed_snapshot` vérifiait néanmoins l'équivalence du complément avant de déterminer si ce complément contenait une valeur utilisable. Une divergence historique d'une source inutilisée bloquait ainsi le run.

Le lecteur détermine désormais les valeurs utilisables après validation des fuseaux et des doublons. Lorsqu'aucune observation manquante ne peut être ajoutée, il conserve la série canonique et ses manques, avec un audit explicite `not_applied_no_available_observations`, zéro observation appliquée et aucune équivalence revendiquée.

Dès qu'une seule valeur de secours serait ajoutée, les contrôles d'équivalence et de couverture restent intégralement obligatoires, avec la tolérance initiale de 1e-9 €/MWh. Les trous historiques restent bloquants et un jour de livraison partiel reste entièrement masqué. Aucun prix observé, cache de modèle ou paramètre scientifique n'a été corrigé artificiellement.

## Vérification

- 25 nouveaux tests dans `tests/test_observed_unused_fallback.py` : absence de publication, représentations des manques, prix déjà connus préservés, refus des divergences dès un ajout, trous historiques et timelines invalides.
- 42 tests existants de rafraîchissement, reprises de lecture et trous internes réussis.
- Deux tests existants de complément post-enchère dans le lanceur réussis.
- Contrôle réel complet `refresh_nuclear_reporting_sources` puis `verify_refreshed_observations` pour BE, DE, FR et NL : 8 784 heures par pays, historique complet, 24 heures du 19 septembre en attente, zéro complément utilisé, une seule tentative et vérification des instantanés réussie.

Les preuves de diagnostic sont dans `tmp/nyx_observation_diagnostic_20260918/summary.json` ; celles de validation réelle dans `tmp/nyx_observation_fix_validation_20260918/summary.json`.

## Relance

Le calcul a été relancé depuis l'API locale habituelle de NYX pour la même livraison, sans lancer de doublon. Nouvel identifiant application : `409a79a775bd4a2a96b9d3ec2c02369d`. Lot : `20260918T071131_414061Z_a103f5b4`. Il a terminé avec succès le 18 septembre à **12:38**, pour les quatre pays.

Cette correction ne résout pas la divergence entre fournisseurs le 13 septembre. Elle empêche uniquement une source inutilisée de bloquer le calcul ; toute utilisation future de cette source continuera à exiger sa cohérence avec la cible canonique.

## Complément : échec du lancement de 14:02

Le lancement `7b4232b7b37d4a099e424b33a2d7182f`, lot `20260918T120258_877749Z_d505dace`, a rencontré un cas différent après publication de l'enchère. EPEX disposait alors des 24 prix du 19 septembre, tandis que la source canonique s'arrêtait encore au 18 septembre. Le même désaccord historique empêchait cette fois l'utilisation effective du complément. Le correctif du matin, limité à un complément inutilisable, ne couvrait pas ce cas.

Les lectures réelles de 14:08 ont confirmé, pour les quatre pays : zéro heure historique manquante, 24 heures du jour de livraison manquantes côté canonique et présentes côté EPEX, avec les mêmes écarts historiques le 13 septembre. Preuve : `tmp/nyx_observation_diagnostic_20260918_afternoon/summary.json`.

La politique de rapport autorise désormais un rejet explicite du complément **après trois lectures** lorsque toutes les heures manquantes appartiennent au seul jour de livraison. Les observations de référence restent inchangées et le jour de livraison reste sans scores. L'audit conserve les trois diagnostics de divergence, les valeurs et heures en désaccord, le nombre de prix candidats et zéro prix appliqué. Le lecteur partagé garde un comportement strict par défaut ; seul le rafraîchissement NYX active cette possibilité en dernière tentative.

Un manque avant le jour de livraison, un problème de couverture, un fuseau invalide, un doublon ou une erreur de transport conserve son traitement bloquant. La tolérance d'équivalence ne change pas. Les rapports distinguent désormais « prix non validés : sources en désaccord » et « prix indisponibles » ; ils ne présentent pas une observation rejetée comme un prix réalisé.

Les quatre bundles de prévisions du matin ont été vérifiés et peuvent être republiés sans entraîner Chronos, CatBoost ou Kalman. Leurs 36 empreintes sont enregistrées avant la relance dans `tmp/nyx_afternoon_fix_20260918/frozen_before.json`.

### Validation et résultat final de la correction de l'après-midi

- Sélection de régressions existantes : 75 tests réussis.
- Nouvelle politique de rejet du seul jour de livraison : 23 nouveaux cas et neuf tests de reprise existants réussis ; comportement strict par défaut, trois diagnostics conservés, historique manquant toujours bloquant, jours DST de 23/25 heures et résolution transitoire au deuxième essai couverts.
- Rapports nucléaires, CWE, reprises historiques et avertissements : 101 tests réussis, dont sept nouveaux cas ciblés ; deux rendus HTML lourds exclus de cette suite sur fixtures. Les rapports réels ont ensuite été générés par le run complet.

Ces suites comportent des tests communs : leurs nombres ne s'additionnent pas en un total de cas distincts.

La relance application `a20f5ad7a4aa4f06b4ad51216ef4a4f3`, lot `20260918T121754_668108Z_99a0d4f7`, a **terminé avec succès à 14:22:20**, heure de Paris, le 18 septembre. Les sources communes, BE, DE, FR, NL et le rapport CWE sont tous `complete`.

Rapport global : `runs/reports/model_storm/CWE_Model_Storm_2026-09-19.html`. Les rapports par pays sont dans `runs/exports/2026-09-19/<pays>/nuclear_kalman/`. Le nouveau message distingue les prix du jour non validés pour désaccord entre sources et affiche la dernière extraction. Les prévisions NYX et Storm restent disponibles, mais les observations et scores du 19 septembre restent vides jusqu'à validation des prix.

La cause amont (divergence ENTSO-E/EPEX sur le 13 septembre) n'a pas été corrigée artificiellement : aucun prix de secours divergent n'a été admis. Le logiciel peut désormais publier les résultats de prévision et expliquer séparément pourquoi leur évaluation du jour n'est pas encore disponible.
