# NYX — référence EPEX pour les résultats

À la demande de l'utilisateur, les nouvelles publications NYX pour BE, DE, FR et NL prennent exclusivement les prix EPEX comme référence des prix réalisés et des scores. L'accès utilise les séries horaires Saturn existantes `power.price.<zone>.euromwh.h.obs.epex`.

## Périmètre

- Chaque nouvelle extraction de reporting couvre la totalité de D−365 à D depuis une seule série EPEX. Les scores NYX et Storm utilisent exactement les mêmes observations.
- La comparaison d'égalité ENTSO-E/EPEX et le complément entre ces sources ne font plus partie de ce chemin. Un écart historique entre fournisseurs ne bloque donc plus ces rapports.
- L'identité de source est explicite : politique `epex_only_v1`, origine EPEX, fournisseur Saturn, fréquence horaire et unité EUR/MWh. Les nouveaux audits portent la version 2 ; les anciens audits restent lisibles sous leur identité d'origine.
- Les contrôles de couverture, fuseau, grille horaire, doublons, prix finis, changements d'heure et empreintes des fichiers restent actifs. Une journée D incomplète reste entièrement sans scores ; un trou historique reste une erreur explicite, sans substitution ENTSO-E.
- Les rapports affichent la référence de prix et la date d'extraction. Les archives historiques déjà publiées ne sont pas rebaptisées EPEX.

## Séparation du modèle et de l'évaluation

Les prix de reporting ne complètent plus les entrées d'un nouveau calcul. `refreshed_target_snapshot` relit séparément la série cible canonique configurée pour l'entraînement, conserve uniquement les heures strictement antérieures à D et enregistre sa propre provenance. Il refuse une actualisation d'un snapshot déjà figé.

Les prévisions existantes, les données scellées et les paramètres des modèles sont conservés. La migration de l'historique d'entraînement n'est pas incluse dans cette modification.

## Vérification

La lecture réelle du 18 septembre a trouvé, pour chacun des quatre pays, 8 784 heures EPEX : un historique complet de 8 760 heures et les 24 prix du 19 septembre. Preuves isolées : `tmp/nyx_epex_reference_20260918/source_check.json` et les quatre fichiers Parquet associés.

La suite d'intégration du lecteur, de la provenance, du rafraîchissement, des comportements historiques et de l'isolation des entrées du modèle a passé 140 tests (dix tests PowerShell hors modification exclus). Les tests incluent des journées de 23 et 25 heures, la corruption de provenance/fichiers, les prix manquants, et l'interdiction d'utiliser les prix de D pour l'entraînement.

Les rapports et leurs lecteurs ont également passé 111 tests, dont dix nouveaux tests de référence EPEX ; deux rendus Plotly lourds sur données synthétiques ont été exclus, les rapports complets étant ensuite générés dans le run réel. Les suites partagent des fixtures et certains cas : ces nombres ne doivent pas être additionnés comme un total de cas distincts.

Cette politique élimine le conflit de référence à l'origine de l'incident du 18 septembre. Elle ne garantit pas qu'un flux EPEX/Saturn ne puisse jamais être retardé ou incomplet.

## Publication validée

Le run application `ddf5bda0ea464f51a7c3f006de72dde2` a terminé avec succès le 18 septembre à **14:55:45, heure de Paris**, avec retour 0. Les quatre pays et le rapport CWE du 19 septembre ont été publiés.

La vérification réelle confirme 24 prix EPEX, 24 prévisions NYX et 24 prévisions Storm pour chacun des quatre pays, la référence EPEX visible dans les rapports, 12 fichiers publiés conformes à leurs empreintes et les 36 fichiers de prévisions figées inchangés. Aucun réentraînement n'a été effectué.

Rapport : `runs/reports/model_storm/CWE_Model_Storm_2026-09-19.html`. Détail des contrôles : `tmp/nyx_epex_reference_20260918/verification.json`. Les 32 régressions supplémentaires sur les anciens compléments et les trous historiques ont également réussi.
