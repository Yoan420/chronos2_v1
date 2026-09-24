# KPI — comparaison des modèles et gain économique simulé

## Lancer

Depuis n’importe quel dossier PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\KPI.ps1'
```

Le launcher affiche le chemin du nouveau `KPI.html`. Il lit les derniers snapshots complets des laboratoires Fundamental, CoherentP50 et StressGuard, ainsi que leur référence NYX nucléaire + Kalman vérifiée contre les rapports opérationnels archivés. Aucun entraînement, forecast, téléchargement, promotion ou changement à `Forecast.ps1`.

```powershell
# Consulter les modèles disponibles
& 'C:\Users\BQ6757\chronos2_v1\KPI.ps1' -Action List

# Retrouver le dernier rapport et vérifier ses empreintes
& 'C:\Users\BQ6757\chronos2_v1\KPI.ps1' -Action Status

# Restreindre les pays ; les filtres HTML couvrent les pays retenus
& 'C:\Users\BQ6757\chronos2_v1\KPI.ps1' -Countries FR,DE,BE,NL

# Vérifier seulement la commande, sans calcul ni écriture
& 'C:\Users\BQ6757\chronos2_v1\KPI.ps1' -DryRun
```

`-EndDay YYYY-MM-DD` permet une clôture plus ancienne. La date ne peut pas dépasser le dernier jour d’évaluation commun disponible. `-Models` accepte les identifiants renvoyés par `List` ; la référence de production doit rester incluse. Pour comparer des sous-ensembles sans changer le support, préférer le filtre de famille dans le HTML.

Chaque génération crée un dossier indépendant sous `runs/reports/kpi/snapshots/`. Les anciens rapports restent intacts. `latest.json` n’est actualisé qu’après vérification des sources et finalisation du rapport.

## Métriques de prix

Le HTML propose 365, 90, 30 et 7 jours, par pays ou en agrégé CWE, avec mode jour/nuit. La période par défaut couvre les 365 jours calendaires se terminant au dernier jour commun d’évaluation ; les lignes de forecast live restent exclues.

- **MAE et RMSE horaires** : mêmes heures physiques pour tous les modèles, Storm et les observations.
- **Win rate horaire** : part des heures avec une erreur absolue inférieure à celle de Storm.
- **Win rate jours MAE** : part des jours avec une MAE horaire quotidienne inférieure à Storm.
- **MAE prix moyen / jour** : moyenne des erreurs absolues entre prix moyens journaliers prévus et observés.
- **Win rate jours prix moyen** : part des jours où cette erreur de prix moyen est inférieure à celle de Storm.
- **Prix moyen** : moyenne sur les mêmes heures physiques appariées ; le prix observé et Storm sont affichés aussi.

Les égalités (tolérance 1e-9 EUR/MWh) restent au dénominateur et ne comptent pas comme victoires. Les jours incomplets ne participent pas aux métriques journalières. Les journées de changement d’heure exigent leurs 23 ou 25 heures réelles. L’agrégat additionne les heures-pays et les jours-pays, sans moyenner naïvement des pourcentages. Aucun prix manquant n’est remplacé par zéro.

## Gain économique : diagnostic hypothétique, pas P&L négociable

Le tableau économique reprend les hypothèses de `config/economic_value.yaml`, sans les ajuster au backtest. Configuration initiale : portefeuille alternatif de **100 MW au total**, réparti en **25 MW par pays** FR/DE/BE/NL, conservés même si un seul pays est sélectionné.

Référence : prix day-ahead observé de la veille à la même heure civile. Position positive ou négative selon le signe de `forecast - référence`, uniquement si l’écart absolu dépasse 5 EUR/MWh + 0,5 EUR/MWh de coûts + 0,5 EUR/MWh de slippage, soit un seuil strict de 6 EUR/MWh. Storm utilise exactement la même règle.

Pour chaque heure :

```text
P&L net = position MW × durée h × (prix observé − référence)
          − |position MW| × durée h × (coûts + slippage)
Gain vs Storm = P&L net du modèle − P&L net de Storm
Gain/MWh = gain vs Storm / somme des MW alloués × heures admissibles
```

Le dénominateur du gain/MWh est le **volume potentiel commun**, pas le volume engagé propre à chaque stratégie. Les alternatives ne sont jamais additionnées entre elles. Sans position, le P&L est nul. Aucun chiffre n’est annualisé.

Une référence manquante ou ambiguë lors d’un changement d’heure exclut l’heure pour tous les modèles et Storm. L’agrégat économique exige tous les pays sélectionnés simultanément et ne redistribue jamais leur capacité. Sa couverture, affichée séparément, peut être plus faible que celle des KPI de prix.

Ce prix de livraison de la veille n’est **pas** un prix exécutable pour la livraison prédite. La disponibilité de la référence repose sur une convention de publication, pas sur une preuve historique certifiée. Les résultats ne constituent donc ni une EVA financière certifiée ni une rentabilité réalisable. Le tableau mesure seulement une valeur simulée du signal sous ces hypothèses.

## Traçabilité et limites

Le dossier du rapport contient `kpi_metrics.json` (calculs détaillés, dont résultats journaliers), `source_audit.json` (sources, configuration économique et dépendances) et `report_manifest.json` (empreintes SHA). Les sources sont vérifiées avant et après génération ; un changement concurrent interrompt la publication.

La comparaison porte sur les dernières configurations sauvegardées. Si leurs évaluations n’ont pas été relancées, le rapport ne crée pas de nouveaux jours. Le modèle opérationnel est comparé dans la version figée des mêmes sources, et non mélangé à une autre version réentraînée ultérieurement.

Cette année a déjà servi à explorer les variantes : ce tableau est descriptif, pas une validation indépendante ni une sélection automatique pour la production. Un recalibrage des intervalles seul ne change pas le P50, donc pas les KPI ponctuels ni ce signal économique.

Code isolé : `kpi_report/data.py` (adaptateurs), `metrics.py` (prix), `economic.py` (simulation), `render.py` (HTML), `runner.py` (orchestration). Pour ajouter une famille, étendre l’adaptateur et ses tests sans modifier le pipeline opérationnel.
