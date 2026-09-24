# Dossier de recherche NYX / Tensor-TimesFM — 15 septembre 2026

Le [rapport décisionnel](C:/Users/BQ6757/chronos2_v1/docs/research/NYX_Tensor_TimesFM_2026-09-15.md) présente la recommandation, les constats, les options, l'architecture et les critères de poursuite.

Ce dossier est une copie durable des travaux isolés réalisés pendant l'audit. Aucun nouveau modèle n'est activé dans NYX. Les sources téléchargées sont conservées pour lecture et traçabilité ; leurs scripts complets n'ont pas été exécutés. Les poids TimesFM et les jeux de données amont n'ont pas été téléchargés.

## Contenu

| Pièce | Usage |
|---|---|
| [nyx_architecture.md](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/nyx_architecture.md) | Chaîne opérationnelle, temporalité, variantes, 59 références locales |
| [upstream_evidence.md](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/upstream_evidence.md) | Dépendances, versions, licences, résultats annoncés et réserves |
| [MEASURED_FINDINGS.md](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/metrics/MEASURED_FINDINGS.md) | Mesures NYX et expérience causale légère |
| [oracle_compression.md](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/compression/oracle_compression.md) | Projection des véritables valeurs futures : diagnostic, jamais prévision |
| [rideshare_context_check.json](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/rideshare_context_check.json) | Contrôle miniature des indices et gradients, sans TimesFM |
| [source_manifest.json](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/upstream/source_manifest.json) | 25 fichiers amont, URLs au commit figé, SHA256 |
| [extraction_manifest.json](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/metrics/extraction_manifest.json) | Origines des bundles NYX, observations et Storm, empreintes, grilles temporelles |
| [validation_selection_locked.json](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/metrics/validation_selection_locked.json) | Treize configurations et choix effectué sur validation avant le test |
| [experiment_summary.json](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/metrics/experiment_summary.json) | Protocole, coûts mesurés, contrôles de causalité, limites |
| [package_manifest.json](C:/Users/BQ6757/chronos2_v1/research/tensor_timesfm_20260915/package_manifest.json) | Empreintes du dossier livré |

`metrics/verified_hourly_pairs.csv.gz` contient les 35 040 points annuels, les observations figées/réactualisées, NYX, ses ablations et Storm. `locked_test_predictions.csv.gz` contient les prédictions des petits modèles. Les CSV de scores sont accompagnés de leurs effectifs ; les JSON conservent les paramètres et les 151 contrôles de refit.

`upstream/source/` conserve le commit Tensor-TimesFM `130fd4787f268cbdd0d46ac5b840290b7c3440dc`. `google_timesfm/` conserve deux fichiers officiels lus au commit `8cb0628371af142e16b8c232cc9fbf667ffb12f9`. `eps_published_metrics/` contient trois petites traces de résultats publics et leur provenance lorsqu'elles sont présentes.

## Reproduction des expériences légères

Environnement de l'audit : `C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe`. Les scripts écrivent leurs résultats dans ce dossier de recherche ; ils ne lancent ni NYX ni TimesFM et n'accèdent pas aux services de données. Exécuter depuis `C:/Users/BQ6757/chronos2_v1`.

Pour reproduire les corrections à partir de la matrice livrée, sans relire les archives de production :

```powershell
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' 'research/tensor_timesfm_20260915/metrics/experiments.py'
```

Pour reproduire les deux diagnostics complémentaires :

```powershell
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' 'research/tensor_timesfm_20260915/compression/oracle_pca_compression.py'
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' 'research/tensor_timesfm_20260915/rideshare_context_check.py'
```

Le diagnostic oracle reçoit les observations du test pour mesurer la perte de projection, après ajustement de la base sur train. Il n'entre dans aucune sélection de recette de prévision. Le diagnostic Rideshare réimplémente seulement les opérations concernées dans un petit réseau CPU ; il ne reproduit pas le benchmark publié.

`metrics/extract_metrics.py` permet de refaire l'extraction si les archives locales référencées existent. Il recherche les sources vérifiées les plus récentes du run 16/09 : si ces sources évoluent, les nouvelles empreintes peuvent différer. Pour reproduire exactement l'analyse livrée, partir de la matrice conservée, dont le SHA256 est `0548b72b6fadcb700d1b863db93fd95d0076670fb5e11bf4ad1c1fe1413a1188`.

Les durées et mémoires mesurées varient entre exécutions ; le manifeste du dossier décrit les fichiers à la livraison et ne sera plus identique après régénération des sorties.

## Portée scientifique

Test des corrections : train initial 180 jours, validation 95 jours, test séparé 90 jours ; transformations sur passé éligible uniquement, labels au plus récents à D−2, refits hebdomadaires, paramètres sélectionnés sur validation. Les observations passées du test sont assimilées selon ce protocole fixé sans nouvelle sélection. Le test est désormais consulté ; les futures hypothèses nécessitent une autre réserve temporelle.

Les bandes statistiques proviennent de 2 000 rééchantillonnages appariés par blocs de sept jours. Elles ne représentent ni l'incertitude des vintages ni toutes les décisions de recherche antérieures. Les archives sont rétrospectives : aucune amélioration prospective ni aucun gain Tensor-TimesFM sur NYX n'est démontré.
