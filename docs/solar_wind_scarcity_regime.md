# NYX — expérience isolée de régime de rareté DE/NL

Autorisation du 23 septembre 2026. Aucun déploiement, modification de production,
resynchronisation de sources ou recalcul Chronos/Kalman. Les fichiers des anciennes
expériences restent inchangés. L'expérience lit les sorties finales scellées
`interaction_40` du 22 septembre : DE `48d7c14fd8bc2e4f`, NL `8f246d6bbd54b511`.
Les colonnes `residual_kalman__q10/q50/q90` sont la référence finale ; les colonnes
génériques `q*` ne le sont pas.

## Protocole fixé avant les résultats

- Cible résiduelle : prix réalisé moins P50 final interaction40 gelé.
- Régime rare : résidu strictement supérieur à 50 EUR/MWh. Ce seuil ne désigne
  pas un prix réalisé de 200 ou 300 EUR/MWh ; ces seuils servent au scoring.
- Modèle commun DE/NL avec indicateur pays, prévisions de vent/solaire/charge
  résiduelle des deux pays, composantes séparées, dépassement de tension non
  saturé à 1 et rampes internes à la journée de livraison. Aucune observation
  contemporaine ni valeur du lendemain n'entre dans les variables.
- Normalisations sur les seuls jours antérieurs, 365 jours au maximum ; les
  14 premiers jours des sources sont un warm-up explicite, jamais imputés.
- Gate CatBoost de classification sans surpondération des classes ; calibration
  chronologique Platt sur les 14 derniers jours si le support le permet.
- Deux experts CatBoost MultiQuantile pour les résidus ordinaires et rares.
  Repli empirique tracé quand un régime manque d'exemples. Les quantiles sont
  réordonnés et la CDF du mélange est inversée : ce n'est ni une moyenne des
  médianes, ni une prime moyenne présentée comme P50. Les quantiles résiduels
  sont ajoutés au P50 de référence, pas aux trois quantiles préexistants.
- Pas de plafond additionnel de correction, pas de hausse forcée. Cette méthode
  ne garantit pas une extrapolation correcte hors du support historique.
- Fenêtre de calibration mobile maximale 365 jours ; 90 jours initiaux exclus.
  Refit tous les sept jours, 120 arbres, profondeur et régularisation fixées dans
  le module. Chaque bloc exclut tous les prix de sa date d'origine et au-delà.
- Évaluation du 21 décembre 2025 au 21 septembre 2026 : 275 jours, 6 599 heures
  par pays (heure d'été incluse), strictement appariées à la référence.
- Refit distinct à l'origine du 22 septembre, 24 heures par pays. Ses observations
  sont lues uniquement après écriture de toutes les prévisions. Cet événement a
  inspiré l'hypothèse : diagnostic post-hoc, PAS test indépendant.
- Mesures : MAE, RMSE, biais, pinball 10/50/90, fréquences sous les quantiles,
  couverture, spikes à 200/300, faux positifs/négatifs et Brier du régime.
  Aucun réglage automatique sur ces résultats, aucune promotion automatique.

Les archives héritées ne certifient pas les publications d'origine PIT ; deux
substitutions historiques de vent NL aux passages à l'heure d'été sont héritées.
L'absence de fuite dans le nouveau traitement ne certifie pas rétroactivement ces
sources. Une validation prospective intacte reste nécessaire avant déploiement.

## Exécution et reprise

Python `pricefm311`, `-B -u`, un processus et un thread, priorité BelowNormal pour
préserver la prévision de production concurrente. Avant chaque fit, attendre au
besoin que la mémoire disponible dépasse 3,5 GiB (réserve visée 3 GiB, marge de
croissance 0,5 GiB ; ce n'est pas une réservation OS). Aucun processus n'est tué.

`--action validate` est strictement en lecture seule ; il vérifie les 24 artefacts
de chaque référence interaction40 et les huit fichiers de chaque bundle SolarWind,
ainsi que la correspondance du manifeste à l'expérience parente. L'identité
comprend les SHA des sources, du nouveau code, du protocole et les dépendances.
`--action run --expected-identity <identité>` refuse une identité différente.

Un verrou système évite les doublons. Chaque bloc terminé écrit un checkpoint
et un reçu SHA. Une reprise de la même identité réutilise seulement les blocs
valides ; un checkpoint partiel non scellé est conservé et exige une inspection.
Les statuts donnent PID/date de création, phase, lots terminés et ETA indicative.
Les journaux de lancement sont distincts. Tous les résultats résident exclusivement
sous `runs/experiments/solar_wind_scarcity_regime_v1/2026-09-22`.
