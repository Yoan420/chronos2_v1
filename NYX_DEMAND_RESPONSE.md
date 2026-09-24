# Réponse de la demande — laboratoire de faisabilité isolé

Ce laboratoire prépare un expert de prix incorporant une demande flexible. **Il ne remplace ni NYX, ni ses modes opérationnels.** Par défaut, aucun historique d'offres d'achat et aucun système complet offre/demande/réseau n'est importé : l'intégration reste bloquée et le prix expert est laissé vide.

## Utilisation

Depuis le projet :

```powershell
& '.\DemandResponse.ps1' -Action DryRun
& '.\DemandResponse.ps1' -Action Validate
& '.\DemandResponse.ps1' -Action Prepare
& '.\DemandResponse.ps1' -Action Run
& '.\DemandResponse.ps1' -Action Report
& '.\DemandResponse.ps1' -Action Status
& '.\DemandResponse.ps1' -Action Demo
```

La configuration dédiée est `config/nyx_demand_response.yaml`. Son `bundle_path: null` signifie qu'aucun bundle physique qualifié n'est fourni ; cela ne demande pas d'inventer des données ou de lancer un entraînement automatique. Les résultats restent dans `runs/experiments/nyx_demand_response_v1`. Les snapshots et sources historiques ne sont pas écrasés. Consulter l'aide du lanceur pour sélectionner explicitement un snapshot lors d'une régénération.

## Contrat d'un futur bundle d'entrée

L'importeur `nyx_demand_response/inputs.py` attend un JSON normalisé, sans champs implicites. Il ne collecte ni ne prévoit automatiquement les courbes manquantes. Le niveau racine contient exactement `schema_version: 1`, `kind: ex_ante_demand_response_scenarios`, `sources` (références documentaires non vides) et `periods` (liste de scénarios). Aucun exemple déclaré « audité » avec des données fictives n'est fourni.

Chaque scénario de `periods` contient les champs suivants :

| Groupe | Champs / colonnes |
| --- | --- |
| Identité et temporalité | `delivery_start_utc`, `forecast_origin_utc`, `inputs_available_at_utc`, `training_end_utc`, `duration_minutes` |
| Scénario et preuve | `scenario_id`, `scenario_weight`, `curve_evidence` |
| `supply` | `zone`, `segment`, `capacity_mw`, `bid_eur_mwh` |
| `demand` | `zone`, `demand_mw` — demande brute avant réponse au prix |
| `flexibility` | `zone`, `segment`, `capacity_mw`, `reservation_eur_mwh` — réduction volontaire supplémentaire |
| `network` | `constraint_id`, `ram_mw`, une colonne `ptdf_<zone>` pour chaque zone modélisée |
| `qualification` | Preuves de périmètre, mobilisabilité, flexibilité, réseau, antériorité et référence réseau explicite |

Les dates portent explicitement un fuseau. L'origine est exactement D−1 08 h Europe/Paris, et les entrées ainsi que les derniers labels d'entraînement doivent déjà être disponibles. `curve_evidence` vaut `forecast_from_past_curves` (avec disponibilité des derniers labels d'entraînement obligatoire) ou `published_flexibility_offers` ; une courbe réalisée de l'enchère à prévoir est interdite.

Les périodes durent 15 ou 60 minutes et doivent couvrir une heure physique entière par scénario, sans trou, chevauchement ni mélange de résolutions. Les quatre pays FR/DE/BE/NL sont présents dans chaque système couplé ; d'autres zones ne peuvent être omises si elles sont nécessaires au domaine. Les identités et poids des trajectoires restent identiques entre les quatre quarts d'heure ; les poids positifs totalisent exactement 1. Le prix horaire de chaque trajectoire est calculé avant ses quantiles : on ne moyenne pas quatre médianes.

`qualification` inclut `evidence_kind: audited_inputs`, `evidence_reference`, `evidence_description`, `physical_inputs_qualified`, `demand_basis: before_price_response`, `flexibility_basis: additional_voluntary_reduction`, `duration_hours`, puis `supply_scope_qualified`, `mobilisable_power_qualified`, `flexibility_curve_qualified`, `network_scope_qualified` et `asof_certified`. Tous les indicateurs requis doivent être vrais. Le sous-contrat `network` précise `qualified`, `basis: reference_net_positions`, `balanced_zones`, `reference_evidence` et les `reference_net_positions_mw` équilibrées du domaine complet.

**Ces déclarations ne constituent pas un audit indépendant.** Les contrôles informatiques vérifient la structure, les cohérences et les horodatages déclarés ; ils ne prouvent pas à eux seuls la véracité, la complétude économique ou l'antériorité de la source. Il faut conserver les preuves externes, leur provenance et la méthode produisant les courbes prévisionnelles. Ne pas remplacer les données absentes par des valeurs supposées ni faire passer des hypothèses synthétiques pour `audited_inputs`. Une situation infaisable ou un prix marginal ambigu est refusé, sans pénalité de pénurie inventée.

Une fois un vrai bundle documenté disponible, renseigner son chemin dans `bundle_path`, puis lancer `Validate`, `Prepare` et `Run`. `Prepare` fige une copie du JSON et ses empreintes ; `Run` calcule les scénarios admissibles sans activer l'expert dans NYX. Les P10/P50/P90 obtenus sont des quantiles pondérés de scénarios, **pas des probabilités statistiquement calibrées démontrées**.

## Ce que démontre le moteur, et ce qu'il ne démontre pas

Dans une enchère, la dernière quantité achetée peut fixer le prix par sa disposition à payer, comme une offre de production peut le fixer par son prix demandé. Les bourses décrivent explicitement des ordres de consommation dépendants ou indépendants du prix : [Nord Pool, ordres par période](https://www.nordpoolgroup.com/en/trading/Day-ahead-trading/Order-types/single-periodic-order/). La formation du prix résulte des offres d'achat et de vente et du couplage, pas obligatoirement du seul coût de la dernière centrale : [EPEX SPOT, principes du marché](https://www.epexspot.com/en/basicspowermarket).

Les démonstrations synthétiques peuvent ainsi produire un prix égal à un palier de disposition à payer, par exemple **350 ou 650 €/MWh**. Ce sont des hypothèses illustratives, **pas des seuils calibrés ou des prévisions du 14 septembre**. Ces exemples valident les équations et les bilans du solveur, pas la qualité prédictive sur le marché réel.

Le moteur est un **programme linéaire mono-période à offres par paliers**, et non une reproduction exacte du couplage SDAC : il n'intègre pas les blocs multi-périodes, l'engagement des centrales, les rampes, les budgets d'énergie du stockage ni les rebonds de consommation. Les courbes par période décrites par Nord Pool peuvent être interpolées linéairement entre points prix-volume ; les transformer en paliers impose une discrétisation dont l'erreur et le domaine d'application doivent être qualifiés. Un futur bundle ne peut pas déclarer ces contraintes absentes satisfaites par de simples booléens.

Trois objets doivent rester distincts :

1. **Mécanisme économique contrôlé** : offres synthétiques, solution du solveur, sensibilité à la demande, bilans et contraintes.
2. **Proxy statistique de tension** : des paramètres ajustés sur les prix peuvent aider à prévoir des pics, mais leurs volumes latents ne sont pas des MW d'effacement observés.
3. **Réponse réelle de la demande** : son estimation demande des offres d'achat, des activations vérifiées ou une référence contrefactuelle, ainsi qu'un traitement de l'endogénéité du prix. Une régression consommation/prix ne suffit pas ; même certains estimateurs instrumentaux peuvent être biaisés avec les séries autocorrélées : [Tiedemann, Sgarlato et Hirth, 2023](https://arxiv.org/abs/2306.12863).

Une prévision de charge peut déjà incorporer une réponse au prix. Lui retrancher un effacement estimé sans définir son point d'ancrage peut compter deux fois le même comportement. Il faut aussi distinguer consommation supprimée, consommation reportée avec rebond, stockage et délestage involontaire.

## Pourquoi la qualification ne peut pas être contournée

Le précédent expert marginal V2 conserve des réserves sur le périmètre du parc national, les imports, la référence du RAM et les frontières externes. Un déficit d'une pile partielle n'est pas une pénurie réelle. Ajouter une demande flexible pour absorber ce déficit peut réduire artificiellement le prix d'une pénalité de slack, sans identifier un comportement de consommateurs.

La qualification des sources et du domaine doit donc précéder toute interprétation réelle. Un fichier JSON de scénarios ne constitue pas à lui seul cette preuve ; son contenu et les preuves d'antériorité avant **D−1 08 h Europe/Paris** doivent être vérifiés. Les courbes d'achat de l'enchère à venir, les prix observés et les paramètres post-couplage ne peuvent pas devenir des entrées contemporaines du replay.

## Lecture du rapport annuel

Le rapport conserve **365 jours calendaires** de références figées — NYX, Storm et, si fournies, les variantes antérieures — avec le support effectivement évalué. Les cas réels et les démonstrations synthétiques sont dans des sections distinctes.

Quand aucune heure n'est qualifiée :

- la ligne expert affiche `—` pour les métriques et zéro heure évaluée ;
- les cas réels gardent un prix expert manquant ;
- aucune intervention ni gain empirique n'est revendiqué ;
- la copie de NYX en repli n'est jamais présentée comme un expert ayant réussi le backtest.

**Zéro intervention ne démontre pas une non-dégradation obtenue par un nouvel expert : il signifie ici que le test n'a pas pu être activé.** Le rapport refuse les affirmations d'intégration ou de gain démontré dans ce laboratoire de faisabilité.

Lorsqu'un bundle fournit des prix admissibles, une section **distincte** compare l'expert et toutes les références sur leur intersection horaire stricte, pour la même fenêtre annuelle. Les références y sont recalculées sur ce support, et non reprises de leurs scores annuels complets. Cette comparaison est exploratoire et non validée indépendamment. Sans bundle évaluable, la section indique explicitement qu'aucune évaluation expert n'est possible.

## Protocole d'un futur candidat évaluable

Figer avant le test les paramètres et sources, les unités, la base de demande, les stocks et rebonds éventuels, les offres de consommation, la référence réseau et toutes les frontières. Garder les inconnues manquantes, sans assimiler absence de donnée et absence de flexibilité.

Sur une fenêtre annuelle commune, comparer le candidat brut, son contrôle rigide, NYX, puis une intervention gouvernée uniquement sur des résultats antérieurs disponibles au cutoff. Publier séparément la couverture physique, les replis, les interventions favorables/défavorables et leurs effets sur MAE, RMSE, prix moyens et épisodes extrêmes. Les seuils d'événement sont fixés à l'avance ou appris sur le passé uniquement ; aucun réglage par pays après lecture du 14 septembre.

Le score annuel doit inclure les replis, mais l'effet propre de l'expert doit aussi être mesuré sur les heures réellement admissibles et sur les interventions. Une comparaison des seules heures favorables ou une pénalité de déficit artificiellement plafonnée ne démontre aucun progrès. Une validation prospective gelée reste nécessaire avant toute promotion ; la causalité de l'effacement et l'utilité prédictive sont deux questions distinctes.
