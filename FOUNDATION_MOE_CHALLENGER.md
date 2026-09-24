# Challenger Foundation-MoE pour l'autonome Chronos-2

## Statut

Le module est implémenté et évalué comme challenger isolé. Il ne modifie ni
`runs/live`, ni les recettes autonomes officielles, ni les exports de
production.

Le papier fourni est un preprint non relu par les pairs, sans code public ni
hyperparamètres suffisants pour une reproduction exacte. Sa contribution
principale n'est pas LoRA seul : c'est une couche de routage résiduel sparse
autour de Chronos2-LoRA.

## Ce qui est implémenté

`chronos2_hourly/models/foundation_moe.py` fournit :

- une banque générique de forecasts directs déjà matérialisés et alignés ;
- trois experts résiduels entraînables (`spike_up`, `spike_down`, `robust`) ;
- des features de niveau/dispersion de courbe, de désaccord inter-experts,
  de marché et d'horizon ;
- un MLP avec embeddings marché/horizon et routage Top-2 par heure ;
- le gain résiduel borné `gamma` autour de l'ancre ;
- les losses Huber, balance, prix élevés, sous-prédiction, oracle et regret ;
- le scheduler CLRS de température, bruit d'exploration et revival ;
- le biais train par horizon et les gates validation `qD=.60`, `qU=.40`,
  `wD=1.00`, `wU=.35` du papier ;
- les modes `paper_balanced`, `mae_downward_only`, `anchor_bias` et `routed` ;
- un déplacement identique de q10/q50/q90, qui conserve largeur et ordre ;
- une sauvegarde/reprise complète du routeur, de la normalisation et de la
  calibration.

Les choix absents du papier (largeur du MLP, learning rate, poids des losses,
seuils CLRS, lissage et initialisation) sont tous explicites dans
`config/foundation_moe_challenger.yaml`.

## Protocole causal utilisé

Le runner `run_foundation_moe_challenger.py` vérifie les SHA256 des cinq runs
autonomes figés avant de lire les données. Il utilise uniquement les forecasts
`residual_corrected`, Chronos-2, ensemble, CatBoost et LEAR présents dans les
artefacts OOF :

- train : 12/08/2025 au 18/03/2026, 219 jours par zone ;
- validation : 19/03/2026 au 30/05/2026, 73 jours par zone ;
- test : 31/05/2026 au 11/08/2026, 73 jours par zone ;
- cinq zones, journées DST physiques de 23/24/25 heures conservées ;
- 26 285 tokens train, 8 755 validation, 8 760 test.

L'ancre est l'autonome actuel `residual_corrected`, et non Chronos-2 brut. Cela
évite de présenter comme un gain une méthode qui ne ferait que rattraper le
correcteur déjà en production.

## Résultat observé

La calibration finale du papier, adaptée à notre banque d'experts, ne passe
pas le test scellé :

| Variante | MAE validation | MAE test | Gain test vs autonome |
|---|---:|---:|---:|
| Autonome actuel | 12,3822 | 14,2622 | - |
| `paper_balanced` | 12,3769 | 14,3200 | -0,0578 EUR/MWh |
| `routed` avant gates fixes | 12,3437 | 14,2452 | +0,0170 EUR/MWh |

La sélection du checkpoint est volontairement MAE-only
(`validation_tail_weight: 0`) pour correspondre à l'objectif demandé ; la
MAE de pointe reste mesurée et publiée. Le point estimé de `routed` est
publié comme diagnostic, et non comme nouveau choix après ouverture du test.
Le challenger sélectionné est rejeté ; son bootstrap circulaire par blocs de
sept dates, avec les cinq marchés conservés ensemble, donne un intervalle 95 %
de [-0,1142 ; -0,0017] EUR/MWh. `routed` est meilleur dans les cinq zones,
mais son gain poolé reste faible (+0,12 %) et son propre intervalle traverse
zéro : [-0,0187 ; +0,0555] EUR/MWh. Il ne franchit pas le seuil de promotion
de +0,05 EUR/MWh.

Conclusion : le routeur est utilisable comme expérience, mais aucune variante
n'est promue automatiquement. La calibration fixe du papier, conçue pour
quatre marchés provinciaux chinois à 15 minutes, ne se transfère pas telle
quelle à cette banque européenne horaire.

## Pourquoi la reproduction reste partielle

Les artefacts locaux ne contiennent pas les experts réellement utilisés dans
le papier : TimesFM-2.5, Moirai-1.1, Chronos-T5, DLinearProxy et de vrais OOF
Chronos2-LoRA. Les experts actuels sont fortement corrélés, et CatBoost/LEAR
sont nettement moins précis que l'ancre. Le routeur a donc peu de diversité
utile à exploiter.

Une reproduction complète exige de générer, pour chaque fold, des forecasts
strictement causaux de chaque modèle et de chaque adapter LoRA. Un adapter
entraîné une fois sur toute l'histoire puis rejoué sur cette même histoire
créerait une fuite de cible.

## Exécution

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_foundation_moe_challenger.py' `
  --config 'C:\Users\BQ6757\chronos2_v1\config\foundation_moe_challenger.yaml'
```

Le répertoire de sortie est immuable et limité à
`runs/experiments/foundation_moe_autonomous_v3`. Il contient le checkpoint,
les prédictions test, les diagnostics Top-2, les métriques validation/test,
les bootstraps sélectionné et par mode, les forecasts live expérimentaux, les
contrats de source et un manifest SHA256.

## Tests

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' -m pytest `
  tests\test_foundation_moe.py `
  tests\test_foundation_moe_runner.py -q
```

Les tests couvrent la séparation chronologique, le routage Top-2, la
normalisation des poids, les jours locaux, l'ordre/largeur des quantiles, les
marchés/horizons inconnus, l'immuabilité des sorties et l'identité après
sauvegarde/rechargement.
