# Saturn — recherche approfondie de demande flexible, 16 septembre 2026

## Conclusion

Des variables pertinentes existent bien. La recherche initiale par noms était insuffisante pour conclure à leur absence. La nouvelle recherche n'a toutefois pas identifié de courbe électrique day-ahead prix-volume d'achat/vente ou d'effacement FR/DE/BE/NL exploitable par le moteur de dispatch. Ce constat est limité au catalogue et à l'accès examinés ; ce n'est pas une preuve d'absence dans toutes les instances Saturn ou chez les fournisseurs connectés.

Recherche en lecture seule ; aucun modèle, forecast, rapport scellé ni configuration opérationnelle modifié. Seuls des diagnostics sont enregistrés dans `tmp`.

## Périmètre réellement contrôlé

- Catalogue `series/catalog?allsources=True` : 166 580 entrées dans neuf sources (energyscan, prim, oil, romania, hydro, stp, power, prices, gas). Le nombre désigne des entrées, pas des séries uniques.
- Catalogue des groupes : HTTP 200, résultat vide pour cet accès.
- Synonymes dans les noms : courbes, enchères, achats/ventes, offres, effacement, demande flexible, industrie, stockage, pompage, prix de réservation et destruction de demande.
- Métadonnées de 2 076 familles à identifiant numérique : 1 343 familles energyscan hors familles power/stp, puis 733 familles power/stp dédupliquées. Pour les contrats suffixés, un représentant par identifiant de base a été interrogé.
- Toutes ces lectures ont répondu HTTP 200. Parmi elles, 2 013 fournissent un label et une description Mercure ; 63 ne fournissent que des informations techniques. Cette dernière catégorie reste une limite documentaire.
- Lecture ciblée de formules, unités, intervalles, révisions et échantillons à l'état connu au cutoff du 13/09/2026 à 06:00 UTC, soit 08 h Paris, pour préparer la livraison du 14/09.
- Swagger et interface Series Quick-View consultés : l'interface utilise le même catalogue. Pas de recherche serveur par description documentée ; la tentative `/series/find` répond 404.

Les données brutes du diagnostic sont dans `saturn_demand_deep_catalog.json`, `saturn_demand_deep_numeric_energyscan.jsonl`, `saturn_demand_deep_power_stp_metadata.json` et `saturn_demand_deep_direct_metadata.json`, dans ce même dossier. Les scripts de sondage portent le préfixe `saturn_demand_deep_`.

## Variable nouvelle la plus intéressante : aluminium

`power.eu.breakeven.aluminium.price.eurmwh`

La formule consultée correspond à :

    prix aluminium USD/t / EURUSD / 13,5 MWh/t

Dépendances primaires résolues : `metal.aluminium.usdt.close` et `forex.everyday.eurusd.close`.

C'est l'équivalent du revenu brut du métal par MWh consommé. Malgré le mot « breakeven », ce n'est pas un seuil complet d'arrêt d'usine : autres coûts, contrats, contraintes d'arrêt/redémarrage et volumes effaçables ne sont pas représentés. C'est un proxy économique candidat, pas une courbe d'offres d'effacement certifiée.

Les sources historiques remontent au 03/01/2000. Sur la fenêtre 15/09/2025–14/09/2026, la dernière version de la formule comporte 253 points. Au cutoff 13/09/2026 06:00 UTC, elle en comporte 251 ; dernière date disponible : 10/09/2026, valeur 209,208864 EUR/MWh. La valeur datée 14/09, 208,457004 EUR/MWh, n'était pas encore disponible. Un modèle doit donc conserver la dernière valeur connue ainsi que son âge. Ce sondage ne certifie pas à lui seul les 365 extractions historiques quotidiennes.

## Séries « demand_destruction » présentes, mais gazières

Observée : `gas.de.demand.industrial.demand_destruction.mcmd.obs`.

La formule est la différence entre consommation industrielle de gaz observée et prévision d'un modèle LGBM météo mm-mos. Ce résidu ne permet pas d'attribuer causalement toute différence à une réponse au prix, et ce n'est pas un volume électrique disponible à l'effacement.

Prévision : `gas.de.demand.industrial.model.demand_destruction.mcmd.daily.fcst`.

Métadonnées « handcrafted » ; dates-valeurs 01/11/2021–31/12/2032. La valeur du 14/09/2026 est −15 millions de m³/j, déjà visible au cutoff ; insertion pertinente du 08/07/2026 à 12:59 UTC. Cela peut fournir du contexte industriel quotidien, mais ni un prix de réservation ni des MW horaires électriques.

## Séries proches, mais non équivalentes

| Série / famille | Constat vérifié |
| --- | --- |
| `power.stp.da_volume.de.mw.qh.obs.epex` | Volume de clearing scalaire, pas courbe prix-volume. Échantillon du 14/09 présent dans la dernière version, absent au cutoff précédent. |
| Anciennes séries EPEX `DA.volume.cleared` FR/DE/BE/NL | Couverture arrêtée au 30/09/2025 ; ne couvrent pas la période récente. |
| `power.prod.pump.de.mw.h.fcst.3mv.storm.da.basecase` | Prévision de pompage disponible au cutoff, mais issue de Storm : introduirait une dépendance au benchmark dans un expert présenté comme indépendant. |
| `power.de.consumption.hydro.pumpedstorage.entsoe.hourly.gw.obs` | Observation de pompage, inconnue pour le jour cible avant enchère. Seulement des retards documentés seraient admissibles. |
| `power.fr.consumption_steel_industry.monthly.odre.gwh`, `power.fr.consumption_paper_board.monthly.odre.gwh` | Consommation industrielle mensuelle, arrêtée en décembre 2023. |
| `power.stp.flag_curtailment.nl.1min` | Couverture limitée au 10–17 août 2023 ; pas de courbe d'effacement DA établie. |
| `power.stp.bid_afrr_up_1000_mw_priority.de.eurmwh.qh.obs` | Offre d'équilibrage aFRR, pas marché day-ahead ; couverture arrêtée fin 2025. |
| Identifiants `51066` / `51074` | Labels ELIA_SYS_IMBALANCE_BIDSUP/BIDSDOWN : volumes MW de réglage, pas courbes d'achat DA. |
| `power.type_marginal_costs.ccgt.de.mw.h.fcst.3mv.storm.da.basecase` et `power.fuel_marginal_costs.nat_gas.de.mw.h.fcst.3mv.storm.da.basecase` | Composants Storm ; unité/définition à clarifier, nom `.mw` ambigu. Ne pas les appeler seuils de demande. |
| `power.nrjscan.cwe.capa.bess.mw.gma.base.h` | Somme de scénarios 2024Q1 incluant aussi ES, IT et UK : ne pas assimiler à la capacité physique CWE. |

## Conséquence pour NYX

Une expérimentation statistique peut étudier le proxy aluminium et son âge, conjointement aux fondamentaux de tension offre-demande et réseau déjà disponibles à 08 h. Elle doit rester distincte d'un expert physique identifiant des volumes réellement effaçables par palier de prix. Ni un seuil automatique de P50, ni un gain sur les spikes, ni une causalité d'effacement ne sont démontrés par cette recherche de données.
