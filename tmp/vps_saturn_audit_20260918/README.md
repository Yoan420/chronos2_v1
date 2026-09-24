# Réconciliation du Daily P&L VPS Storm

Diagnostic du 18 septembre 2026, fenêtres se terminant le 19 septembre 2026.
Aucun forecast, paramètre de production ou rapport CWE existant n'a été modifié.

## Résultat

Les séries Saturn `power.vps.{zone}.euromwh.h.da.pnl.storm` permettent de
reproduire les 16 montants Storm affichés dans le
[dashboard STP](https://dashboards.gems.myengie.com/stp-dashboard/day-ahead/rolling-models-performance).
Il n'est donc pas nécessaire d'approcher ce résultat en recalculant les arbitrages Storm.

| Zone | 7 jours | 30 jours | 60 jours | 90 jours |
|---|---:|---:|---:|---:|
| BE | 230 € | 443 € | 432 € | 494 € |
| DE | 231 € | 447 € | 437 € | 482 € |
| FR | 247 € | 450 € | 421 € | 429 € |
| NL | 213 € | 447 € | 435 € | 490 € |

Toutes les valeurs sont des moyennes calendaires, arrondies à l'euro comme le dashboard.

## Agrégation vérifiée

Pour une fenêtre de N jours civils, date de fin comprise :

`Daily P&L = somme des cashflows horaires VPS disponibles dans la fenêtre / N`.

Les bornes sont définies dans le fuseau de livraison du pays, puis converties en UTC.
La fenêtre ne s'arrête pas à la dernière valeur publiée si celle-ci précède la date du rapport.
Les valeurs sources manquantes ne sont pas remplacées dans le fichier brut : la division
par N reste néanmoins identique, même si des journées sont absentes du numérateur.

Exemple France 90 jours : 38 578,24 € / 90 = 428,6471 €, affiché 429 €.
Seulement 56 jours sont renseignés. Une moyenne sur ces seuls jours donnerait
688,8971 €, ce qui n'est pas la convention du dashboard.
Sur les 7 jours, deux journées FR sont renseignées : 1 726,36 € / 7 = 246,6229 €.

## Source et vérifiabilité

- Lectures Saturn : `nocache=True`, `live=False`, `_keep_nans=True`.
- Données brutes : `vps_raw_hourly.parquet`.
- Empreinte SHA256 du brut : `1311bd703718fcd83e870f38b04785e0ae04cb956460d296309da55e3c5c5023`.
- Agrégats, couvertures et paramètres de requête : `window_summary.json`.
- Séries `...pnl.observed` également relevées pour le diagnostic, jamais utilisées comme forecast.
- Dernière livraison publiée dans ces extractions : 17 septembre 2026.

## Intégration recommandée

À chaque génération, lire et archiver le P&L VPS Storm disponible, appliquer
l'agrégation ci-dessus et afficher la date d'extraction, la dernière publication
et la couverture. En cas d'indisponibilité complète, indiquer « indisponible » :
ne pas substituer silencieusement une simulation locale à la série VPS.

Cette identité est vérifiée sur 16 cellules du snapshot étudié ; elle ne garantit
pas l'immuabilité des données ou du dashboard externe. L'option 365 jours resterait
une extension non vérifiée sur cette interface.

Pour NYX, aucun P&L VPS officiel n'a été identifié. Une comparaison équitable
nécessite le même protocole d'arbitrage et les mêmes journées disponibles, et non
un P&L NYX complet comparé à un P&L Storm partiellement publié. Conserver en parallèle
le diagnostic interne avec son protocole explicite et son historique complet.

## Hypothèses locales écartées

Les essais de supports cache/natif, rendements, cycles, inventaires, stratégies
discrètes, valorisation prévue/observée et décalages UTC n'ont pas fourni une
approximation uniforme fiable avant l'identification de la source VPS directe.
Aucun coefficient zonal ou ajustement arbitraire n'a été ajouté aux rapports.

## Approximation du moteur pour NYX

L'analyse ultérieure des cashflows horaires publiés permet une approximation bien
plus proche : capacité 4 MWh, puissance 1 MW, rendement charge 0,85 et décharge 1,
stock initial 2 MWh et final 1 MWh, sans limite de cycles, planning optimisé avec
le forecast puis valorisé aux observations. Ces paramètres sont une inférence,
pas une documentation du contrat VPS.

Sur 227 journées renseignées, 5 003 des 5 448 cashflows horaires correspondent
exactement au centime, et l'erreur absolue moyenne du total journalier est de
6,32 €. L'agrégation sur le même support publié donne par exemple sur 90 jours :
BE 496,77 € contre 494,06 € ; DE 484,13 € contre 481,88 € ; FR 429,34 € contre
428,65 € ; NL 492,14 € contre 490,21 €.

Les différences restantes concernent notamment la première heure de la journée
et les arbitrages entre heures de prévisions égales. Les flux publiés semblent
favoriser la vente à minuit, alors que l'optimiseur peut préférer une heure
matinale plus chère. Ne pas transformer cette observation en règle arbitraire
avant validation supplémentaire.

Le passage de 2 à 1 MWh consomme 1 MWh d'inventaire initial : ce diagnostic de
cashflow n'est pas un cycle journalier à inventaire constant ni un bénéfice net
autofinancé. Pour NYX, conserver un libellé « VPS estimé », appliquer les mêmes
journées que Storm et préserver le diagnostic économique interne à stock
initial/final identique. Aucun moteur de production n'a été remplacé.

Diagnostics reproductibles : `runs/tmp/diagnose_dashboard_vps_inventory.py`,
`runs/tmp/diagnose_saturn_vps_soc_boundaries.py`,
et résultats `runs/tmp/saturn_vps_soc_boundary_diagnostic.json`.

### Dernier contrôle : convention de première heure

Dans les 227 journées observées, le cashflow publié de minuit correspond toujours
à la vente de 1 MW, alors que cette heure n'est pas toujours optimale. Tous les
forecasts de minuit de cet échantillon sont positifs. En imposant cette seule
contrainte supplémentaire, 5 224/5 448 cashflows correspondent au centime (95,9 %).
L'écart absolu moyen du total journalier renseigné tombe à 2,706 €.
Sur les 16 moyennes calendaires du dashboard, l'écart moyen est de 0,921 €/jour,
et le maximum de 2,288 €/jour. Les quatre valeurs arrondies à 7 jours sont identiques.

Cette variante de compatibilité ne doit pas être présentée comme un arbitrage
économiquement optimal. Son comportement lorsque le prix prévu de minuit est
négatif n'est pas validé. Ne pas affirmer avoir retrouvé le code officiel.
La variante sans vente forcée reste l'approximation physiquement optimisée ;
son écart moyen sur les 16 moyennes est de 2,736 €/jour, maximum 4,341 €/jour.

Script : `runs/tmp/diagnose_saturn_vps_first_hour_constraint.py` ; résultats :
`runs/tmp/saturn_vps_first_hour_constraint_diagnostic.json`.
