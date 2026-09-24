# NYX Solar Ramp — laboratoire isolé

Extension diagnostique du modèle `nuclear_kalman`, sans modification de
`Forecast.ps1`, des configurations opérationnelles ni des exports existants.
La décision par défaut est de conserver NYX. Un gain rétrospectif n'autorise pas
une activation, et la non-dégradation future ne peut jamais être garantie.

## Lancement

Depuis n'importe quel dossier PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarRamp.ps1' -Action Run
& 'C:\Users\BQ6757\chronos2_v1\SolarRamp.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\SolarRamp.ps1' -Action Report
```

`Run` crée une nouvelle expérience figée et entraîne les variantes.
`Prepare` fige les sources sans entraîner ; `Backtest` entraîne l'expérience
préparée. `Report` relit uniquement les prévisions et métriques scellées : aucun
entraînement, chargement de modèle exécutable ou appel aux fournisseurs.
`-RunDirectory '<dossier snapshot>'` sélectionne une expérience précise pour
Backtest/Report/Status. Les résultats terminés sont réutilisés. Un entraînement
achevé est aussi réutilisé après un échec ultérieur de l'évaluation. Une
interruption au milieu de l'entraînement nécessite sa reprise depuis le début.

`Prospective` expose le statut de qualification ; il ne simule pas une émission
réelle et n'affirme pas qu'un prix serait encore inconnu. Cette phase reste
bloquée tant que les contrôles de publication frais et les captures d'entrées
avant enchère ne sont pas disponibles.

Paramètres : `config/nyx_solar_ramp.yaml`. Mettre `enabled: false` produit
exactement les prix et intervalles de référence. Un changement de recette ou de
code exige un nouveau `Prepare`. Toutes les sorties restent sous
`runs/experiments/nyx_solar_ramp_v1` ; les chemins redirigés sont refusés.

## Baseline et information

La baseline provient des rapports opérationnels du **18 septembre 2026**.
L'année évaluée est **18/09/2025–17/09/2026**, soit 365 jours civils. La livraison
du 18 septembre, dont les prix sont déjà connus, est présentée séparément.
Chaque pays conserve ses 8 760 heures physiques d'évaluation, y compris les
changements d'heure et les replis sans expert. Storm reste un comparateur,
jamais une entrée.

Les features viennent d'un snapshot historique existant : prévisions solaires,
éoliennes, charge résiduelle, température, disponibilités et nucléaire. Les
fondamentaux s'arrêtent au 15 septembre ; les dates suivantes restent NYX.
Traitement en UTC, affichage local explicite ; les heures indiquent le **début**
d'intervalle. Le 14/09 à 19 h en Allemagne correspond donc à 17 h UTC.

Origine visée : **J−1 08 h civile**. Les archives attestent des requêtes as-of,
pas des horodatages originaux de publication fournisseur. Les labels hérités
comportent eux aussi une hypothèse de disponibilité ; le laboratoire impose en
plus un délai conservateur à la fin de livraison + deux jours. Ce délai ne
certifie pas les révisions historiques. Aucune certification PIT opérationnelle
n'est revendiquée, y compris pour les prévisions NYX historiques figées.

Pas de réalisé solaire, de flux réalisés ou de résultats post-coupling en entrée.
Les données JAO sont exclues faute de qualification à 08 h. Le périmètre est
FR/DE/BE/NL, pas une assertion sur tous les sens historiques de « CWE » ou sur
les agrégats internes STORM. Les prix sont horaires : un pic de quart d'heure
peut être atténué dans cette agrégation.

## Méthode

- Contrôles : calendrier, charge résiduelle, vent, disponibilités, températures,
  rampes non solaires, prévision NYX et largeur de son intervalle.
- Ablations cumulatives : solaire local ; solaire des trois autres pays ;
  baisses solaires 1 h/3 h ; interactions avec RL et pression système.
- La RL contient déjà une déduction des renouvelables : le solaire n'en est
  **pas soustrait une seconde fois**. Son ajout teste une représentation
  complémentaire, pas une information complètement absente de NYX.
- Pression = RL / disponibilité sélectionnée, avec dénominateur plancher.
  L'offre n'est pas exhaustive ; ce ratio ne mesure ni une congestion, ni une
  marge physique ferme, ni un coût marginal observé.
- Expert simple : gradient boosting peu profond, classification du prix
  horaire ≥ 300 EUR/MWh et régression de la médiane du résidu NYX. Le P50
  n'est jamais assimilé à « probabilité × amplitude du spike ».
- Refit hebdomadaire, fenêtre antérieure maximale de 365 jours, minimum de
  120 jours éligibles. L'archive ne fournit **pas 365 jours d'apprentissage à
  chaque origine**. Les 28 derniers jours connus sont réservés : première
  moitié pour calibration Platt si événements suffisants ; seconde pour seuils
  d'alerte et erreurs d'intervalle. Sinon, probabilité brute explicitement marquée.
- Budget de fausses alertes réglé à 1 % sur les négatifs de calibration par
  pays. Ce n'est pas une garantie hors échantillon : le taux effectivement
  observé est publié. Les ex aequo au seuil n'activent pas une alerte.
- Gouverneur par pays : utilise seulement les propositions hors échantillon
  antérieures dont les labels sont déjà admissibles. Poids 0/0,25/0,5/1, gain
  requis sur les pics et absence de dégradation MAE globale/hors pics, avec un
  garde bootstrap. Données/preuves insuffisantes : prix et intervalles NYX.
- Les intervalles du candidat sont calibrés sur erreurs chronologiques ; ceux
  des interventions gouvernées utilisent les anciennes propositions OOS.

## Validation et lecture

Exploration, sélection de 60 jours et diagnostic final de 90 jours sont
chronologiques. Cette année et le cas du 14 septembre ayant déjà été examinés,
le dernier segment **n'est pas un test indépendant vierge**. La recette est
figée avant son exécution ; aucun hyperparamètre n'est choisi d'après ce cas.

Le rapport distingue année complète avec replis, support OOS commun, sélection,
diagnostic final et livraison historique séparée. MAE/RMSE/biais, erreurs et
sous-estimation des pics, précision/rappel/PR-AUC, taux de fausses alertes,
épisodes/timing, Brier/log-loss et pinball/couverture sont sauvegardés.
Les seuils statistiques q99 sont estimés dans le train et définissent un
**autre régime** : la probabilité du seuil métier 300 n'est pas abusivement
évaluée comme une probabilité q99.

IC : bootstrap apparié par blocs de 7 jours, conservant ensemble toutes les
heures et zones. L'évaluation utilise des blocs non circulaires ; le garde
historique utilise des blocs circulaires sur calendrier réindexé sans imputer
les jours absents. Les IC ne corrigent ni la recherche de modèles, ni les
ruptures structurelles.

L'analyse physique recherche aussi des baisses solaires fortes sans spike et
des spikes sans baisse. Les strates heure/saison/RL/vent/pression ne suppriment
pas tous les facteurs confondants : les résultats sont des associations, pas
une preuve causale. Aucun poids de variable n'est interprété comme une part
causale du prix.

## Artefacts

Chaque snapshot contient les inputs figés et leurs SHA256, les versions
logicielles/code, `predictions.parquet` et CSV, `folds.json`, `governance.json`,
`models.joblib`, `metrics.json`, `source_audit.json`, les références vérifiées et
les manifestes de résultats. Ne charger un joblib que depuis une source locale
de confiance. Le rapport n'en a pas besoin.

`reports/index.html` : analyse interactive autonome à quatre pays, prix,
solaire, rampes, RL, pression, probabilités, cas connu et contre-exemples.
`reports/standard/` : quatre pages utilisant le moteur HTML opérationnel,
explicitement expérimentales. Aucune page n'est copiée dans les exports de prod.
Les déciles intermédiaires du renderer standard sont interpolés seulement pour
son CRPS d'affichage ; les vrais P10/P50/P90 sauvegardés restent inchangés.

Références et leurs limites : [note scientifique](nyx_solar_ramp_literature.md).
Tests : `python -m pytest -q tests/test_nyx_solar_ramp_*.py` (utiliser le Python
du venv `pricefm311` ; selon le shell, expliciter les six fichiers plutôt que le
glob).
