# NYX horaire enrichi par des profils de prévision à 15 minutes

Cette variante expérimentale conserve les prix de sortie à l'heure et la chaîne NYX complète comme référence. Elle ajoute un correcteur ponctuel léger après Chronos-2, CatBoost et Kalman. Elle ne remplace pas les entrées de Chronos et n'active aucun modèle dans le lancement principal de l'application.

Premier essai réel terminé le 16/09/2026 : **aucun gain**, MAE globale de 14,1914 à 14,7550 €/MWh. Voir les [résultats et limites](C:/Users/BQ6757/chronos2_v1/docs/research/NYX_INTRAHOUR_2026-09-16.md). La recette ci-dessous a été fixée avant ce résultat.

## Les trois variantes comparées

1. **NYX actuel** : P50 final figé du parcours NuclearKalman.
2. **Contrôle horaire** : correction ridge utilisant NYX, les calendriers, les six fondamentaux déjà consommés par NYX et la moyenne horaire du nouveau profil solaire.
3. **Profils intra-horaires** : même modèle et mêmes informations, avec les caractéristiques de forme calculées sur les quatre quarts d'heure.

Le contrôle horaire est essentiel : un éventuel gain peut venir de la nouvelle source solaire plutôt que de sa résolution à 15 minutes. La variante doit donc être comparée à NYX **et** à ce contrôle.

## Source réellement admise dans le premier pilote

Prévision solaire belge Elia, série primaire Saturn **23259**, en MW, sous-jacente à `power.stp.da_solar_production.be.mw.qh.fcst.elia`. La lecture utilise directement la série primaire UTC, sans passer par la formule enveloppe qui rééchantillonne la série. Les métadonnées et des lectures à plusieurs saisons confirment des quarts d'heure successifs et des variations internes à l'heure.

Certaines séries de vent Meteologica nommées `qh` sont au contraire des données horaires répétées par une formule `resample/ffill`. Elles sont exclues. Le pilote ne prétend pas disposer de fondamentaux natifs à 15 minutes pour les quatre pays. Le solaire belge est testé comme information commune sur chacune des quatre sorties pays.

La collecte historique demande explicitement l'état disponible **D−1 à 08:00 Europe/Paris** pour chaque journée D. Les colonnes `snapshot_time_utc` et `revision_time_utc` enregistrent ce cutoff d'interrogation, pas l'heure de publication originale du fournisseur. L'expérience est donc un rejeu historique as-of, sans preuve d'archives réellement capturées en temps réel.

La disponibilité effective, les jours incomplets et les erreurs de collecte sont inscrits dans `collection_audit.json`. Une étiquette 15 minutes ou une déclaration dans un manifeste ne suffit pas, à elle seule, à établir une résolution native.

## Caractéristiques et garde-fous

Pour chaque heure physique complète, après conversion MW→GW :

| Caractéristique | Définition |
|---|---|
| Moyenne | Moyenne arithmétique des quatre quarts |
| Dispersion | Écart-type population, ddof=0 |
| Amplitude | Maximum moins minimum |
| Pente | `(q45 − q00) / 0,75`, en GW/h |
| Excursion haute | Maximum moins moyenne |
| Excursion basse | Minimum moins moyenne |

Les quatre valeurs doivent être présentes et finies. Aucun remplissage, interpolation, remplacement par une ancienne valeur finie ou utilisation de réalisation future n'est autorisé. Le dernier état admissible est choisi par révision puis snapshot, les deux bornés au cutoff. Un dernier NaN invalide l'heure.

Les heures sont conservées en UTC : 23/24/25 heures, donc 92/96/100 quarts par jour civil. Une heure manquante ne disparaît pas du score ; le candidat se replie sur NYX. Les quantiles NYX ne sont pas décalés ni présentés comme de nouveaux intervalles calibrés : cette expérience ne produit qu'une prévision ponctuelle.

## Évaluation fixée avant le premier résultat

Le premier run prend comme référence le bundle complet pour la livraison du **16/09/2026**, dont le backtest couvre le 16/09/2025 au 15/09/2026. Ce choix est explicite pour ne pas basculer vers un calcul principal encore en cours.

- 180 jours initiaux d'apprentissage ; 94 jours de validation ; un jour d'embargo ; 90 jours de test.
- Le jour d'embargo rend les derniers labels de sélection disponibles au premier cutoff du test selon la règle conservatrice D−2.
- Fenêtre de fit de 180 jours au maximum, refit tous les sept jours, minimum 720 lignes d'apprentissage complètes par pays. Les labels du test deviennent utilisables au fil du temps selon D−2, sans nouvelle sélection des paramètres.
- Ridge avec régularisations 10 et 100 ; normalisation et imputation du train calculées seulement sur le passé autorisé. Les caractéristiques courantes manquantes provoquent un repli, pas une imputation.
- Sélection de paramètres et famille par MAE de validation sur les labels figés ; comparaison test sur observations vérifiées. Les trois familles sont documentées même si la validation préfère l'identité NYX.
- MAE, RMSE, biais, pertes par pays/heure, prix négatifs et prix supérieurs au q95 du train. Intervalles appariés par 1 000 bootstrap de blocs communs de sept jours.

Cette période historique a déjà été examinée dans l'audit antérieur de NYX. Le test est séparé de la sélection de cette nouvelle recette, mais **n'est pas un historique globalement vierge de recherche**. Tout résultat favorable reste exploratoire et nécessite de nouvelles prévisions réellement émises.

Les prix utilisés pour apprendre sont les observations figées dans le bundle NYX. Leur utilisation respecte D−2 dans le calcul, mais ces valeurs rétrospectives ne certifient pas la version effectivement publiée à chaque date historique. Les observations de reporting plus récentes servent uniquement au score. Cette distinction empêche de présenter le backtest comme une validation intégrale en conditions réelles.

« Encourageant » exige une réduction de MAE d'au moins 2 % contre les deux références, un intervalle à 95 % du delta MAE entièrement négatif, une RMSE globale au plus 1 % moins bonne et une MAE par pays au plus 5 % moins bonne. Les régimes critiques suffisamment représentés doivent également ne pas se dégrader de plus de 5 %. Leur preuve exige au moins 30 heures réparties sur cinq jours ; l'insuffisance d'événements est signalée. Un taux de repli supérieur à 5 %, un apprentissage insuffisant ou une forme intra-horaire sans variation interdisent une conclusion favorable. Aucun résultat ne déclenche automatiquement une activation ou le passage à la variante qui prédit directement 96 prix.

## Utilisation et fichiers

Les commandes ci-dessous sont destinées à la reproduction ; l'expérience peut être exécutée par l'assistant et son rapport consulté directement.

```powershell
# Acquisition isolée : requêtes de lecture seulement, maximum deux simultanées.
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' materialize_nyx_intrahour.py --start-day 2025-09-16 --end-day 2026-09-15 --output-dir data/pit/nyx_intrahour/be_solar_elia_20250916_20260915

# Contrôle des données, sans ajuster de modèle.
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' run_nyx_intrahour.py --action audit

# Comparaison et rapport HTML autonome.
& 'C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe' run_nyx_intrahour.py
```

`--resume` reprend une acquisition interrompue dans son dossier dédié, avant gel. Les expériences copient ensuite les entrées exactes et leurs empreintes. Une expérience existante n'est pas écrasée ; chaque lancement crée un dossier distinct sous `runs/experiments/nyx_intrahour_v1`.

Le manifeste des sources vérifie l'empreinte du parquet. Le lecteur NYX vérifie les bundles figés et les observations de reporting. Le runner conserve panneau d'entrée, profils horaires, prévisions, métriques, différences appariées, sélection, audits de refit, versions, empreintes du code et rapport HTML. Les sources sont recontrôlées en fin de traitement ; un changement invalide les résultats.

Les tests synthétiques servent exclusivement à vérifier le code, la causalité et les cas limites. Ils ne constituent pas une preuve de performance sur NYX. Les résultats métier doivent toujours provenir de l'archive réelle collectée.

Code : [caractéristiques](C:/Users/BQ6757/chronos2_v1/nyx_intrahour/features.py), [évaluation](C:/Users/BQ6757/chronos2_v1/nyx_intrahour/evaluation.py), [lecture des données](C:/Users/BQ6757/chronos2_v1/nyx_intrahour/data.py), [exécution](C:/Users/BQ6757/chronos2_v1/run_nyx_intrahour.py), [configuration](C:/Users/BQ6757/chronos2_v1/config/nyx_intrahour.yaml).
