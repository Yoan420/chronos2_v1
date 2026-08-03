# Chronos-2 — Power Markets

Dépôt minimal pour exécuter `amazon/chronos-2` sur le prix Day-Ahead France,
avec covariables point-in-time et forecast complet de J+1.

## Garantie anti-fuite

Pour une livraison D, chaque forecast exogène est sélectionné parmi les
révisions disponibles au plus tard à D-1 08:00 (`Europe/Paris`). Le mot
`oracle` dans la configuration signifie uniquement que cette valeur forecast,
déjà matérialisée point-in-time, est connue sur l'horizon futur. Il ne lit pas
la réalisation finale future.

Le forecast live utilise les prix connus jusqu'à J 23:00, puis prédit
J+1 00:00–23:00. Le run échoue si la cible de J est incomplète.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .equirements.txt
```

`tshistory_lite` doit être disponible dans l'environnement ENGIE pour Saturn.

## Exécution

```powershell
.un_chronos2_modular.ps1 -RefreshData
```

Sortie :

```text
runs/chronos2_asof_jplus1_fr/fr/day_ahead_forecast_native.csv
```
