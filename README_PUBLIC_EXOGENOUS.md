# Variables exogènes publiques — protocole strict FR

Cette branche ajoute deux matérialiseurs isolés du modèle de production :

- `materialize_openmeteo_previous_runs.py` : température 2 m, vent 100 m et
  rayonnement solaire à lead fixe 48 h sur huit points français. Cette source
  passe le contrat PIT au cutoff civil J-1 08:00 Europe/Paris.
- `materialize_eco2mix_features.py` : profils physiques Eco2Mix D-2/D-7.
  Le dataset consolidé est révisé a posteriori ; ces variables sont donc
  étiquetées `exploratory_non_pit` et ne peuvent pas être promues.

Le cribleur `screen_public_exogenous_strict.py` conserve exactement le blend
figé :

```text
0.4977609282924196 × modèle autonome
+ 0.5022390717075804 × MKOnline 41551_native
```

Il apprend les corrections uniquement sur A, sélectionne une famille sur B1
si elle gagne globalement et sur les deux moitiés de 30 jours, puis utilise B2
uniquement comme veto. Le holdout final de 365 jours n'est jamais chargé.

## Résultats de l'écran du 13 août 2026

- Open-Meteo D-2 : aucune famille ne passe B1. Le meilleur bloc combiné donne
  une MAE B1 de 10,22639 contre 10,18533 pour le blend figé.
- Eco2Mix D-2/D-7 : aucune famille ne passe B1. La demande apprend une
  correction nulle (MAE inchangée à 10,18533) ; la meilleure correction
  non nulle donne 10,18670. Elle reste de toute façon non promouvable sans
  archive de vintages contemporaines.

Le modèle de production n'est donc pas modifié par ces écrans.

## Réexécution

Voir `README_OPENMETEO_EXOGENOUS.md` pour la matérialisation météo. Puis :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  .\screen_public_exogenous_strict.py `
  --phase b1 `
  --exogenous-a-b1 .\runs\tmp\openmeteo_d2_ab1b2\weather.parquet `
  --exogenous-a-b1-manifest .\runs\tmp\openmeteo_d2_ab1b2\weather.manifest.json `
  --output-json .\runs\tmp\openmeteo_d2_ab1b2\screen_b1.json
```

Ne lancer la phase B2 que si `selection_passed=true`. Les conditions du
endpoint public Open-Meteo doivent être vérifiées avant un usage commercial.
