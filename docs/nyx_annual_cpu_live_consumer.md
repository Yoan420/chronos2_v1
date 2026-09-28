# Consommateur prospectif annuel NYX CPU

`run_nyx_annual_cpu_live.py` consomme un bundle quotidien déjà produit. Il ne
télécharge aucune donnée et ne reconstruit aucune variable. Son lancement reste
bloqué par `forecast_enabled: false` dans
`config/nyx_annual_cwe_historical.json` et par l'absence d'une qualification
annuelle des experts CPU.

## Entrées exigées

`nyx_annual_live_preflight.inspect_bundle` contrôle pour la journée demandée :

- les 12 matrices ordonnées : `features/{fr_residual_1000,cwe_residual_2000,cwe_absolute_2000}/{FR,DE,BE,NL}.parquet` ;
- les 4 baselines `baseline/{FR,DE,BE,NL}.parquet`, avec 365 jours de prix réalisés, NYX `q50` et aucune étiquette future ;
- les références `reference/{FR,BE,NL}.parquet` pour la journée future ;
- les neuf reçus dans `source_receipts/`, chacun lié par empreinte à ses artefacts et à la coupure D−1 08 h.

Le consommateur recontrôle et hache chaque entrée avant et après l'entraînement.
Un changement interrompt la publication. Aucun ancien fichier de
`runs/experiments` ne sert de prévision future.

## Modèles et sorties

Les trois fits de prix utilisent `nyx_pooled_cpu_price_model.fit_pooled_block` :
résiduel 1000 arbres avec 449 variables, résiduel 2000 arbres avec 123
variables et absolu 2000 arbres avec 503 variables. Les trois fits de
probabilité utilisent `nyx_negative_probability_cpu.fit_predict_block` sur
les 123 variables FR, BE et NL, avec historique de 365 jours et calibration
chronologique. Les prix emploient exactement 8 threads CPU et les
classifieurs 2, sans option CLI pour les modifier. Le reçu annuel doit
attester les deux valeurs. Si un classifieur passe en repli `one_class` ou
`constant_features` et ne produit aucun CBM, le run s'arrête explicitement.

Les prix horaires sont composés avec les choix figés : FR utilise le résiduel
1000 si son écart absolu à la référence atteint 20 EUR/MWh, sinon la
référence exacte ; BE applique la même règle à la moyenne des deux experts
2000 ; NL utilise cette moyenne à toutes les heures. Les sorties CSV et
Parquet sous `runs/nyx_annual_cpu_live/<jour>/zones/` contiennent le prix,
les probabilités négatives, la référence et les points experts. Le reçu du run
contient les empreintes des entrées, modèles, sorties et du code.

## Verrou de qualification

Le CLI ne propose aucun argument pour remplacer le manifeste ou ignorer le
verrou. Pour activer plus tard, le manifeste devra porter
`forecast_enabled: true` et un bloc `cpu_annual_qualification` avec le chemin
fixe `config/nyx_annual_cpu_qualification_receipt.json` et son SHA-256. Ce
reçu devra confirmer 53 origines annuelles, les trois experts et compositions
figées, les métriques FR/BE/NL sur exactement 8 759 heures communes à Storm, une
RMSE inférieure à Storm et plus de 50 % de victoires strictes dans chaque
pays, les scores Brier des probabilités sur exactement 8 760 heures, les
empreintes du code consommé et les SHA-256 des deux `receipt.json` de replay
prix et négatif. Ces deux reçus sous `runs/` sont ignorés par Git : un clone
propre vérifie l'empreinte du reçu de qualification, qui lie ces SHA, mais ne
peut pas recalculer lui-même les deux SHA sans leurs fichiers. L'outil de
qualification doit vérifier les deux reçus originaux avant de sceller le reçu
agrégé suivi dans Git.

`qualify_nyx_annual_cpu.py` prépare ce reçu à partir des replays prix et
probabilité négative complets. Il vérifie les 159 checkpoints de prix, reconstruit
les trois prix annuels, recalcule leurs scores sur EPEX/Storm, contrôle les trois
séries négatives contre les archives hachées et vérifie les versions de code et
de dépendances enregistrées. Il ne crée le fichier canonique que si la règle de
prix fixée avant le replay passe dans chaque pays : RMSE strictement inférieure
à Storm et plus de 50 % de victoires horaires strictes sur les 8 759 heures
communes. Le replay négatif doit couvrir exactement 8 760 heures par pays et
reproduire ses métriques scellées.

Ce reçu atteste seulement les experts CPU **conditionnellement aux entrées
historiques**. Le replay utilise un NYX q50 et des références issus d'une chaîne
historique GPU/archive ; leur génération prospective équivalente depuis un clone
propre n'est pas qualifiée. Le builder inscrit donc
`price_expert_replay_qualified: true`, `full_input_chain_qualified: false` et
`qualified: false`. Le consommateur exige explicitement les trois validations
positives et les mêmes versions de Python et des bibliothèques que le replay
qualifié. Le manifeste reste à `forecast_enabled: false`. Le replay négatif
ne porte pas dans son propre reçu le SHA du code effectivement exécuté ; le
builder revérifie ses séries, ses sources, ses métriques et le code actuel, sans
prétendre certifier cette provenance manquante.

Contrôle du builder, sans écriture :

```powershell
python qualify_nyx_annual_cpu.py --price-replay runs/experiments/nyx_selected_cwe_cpu_20260928 --negative-replay runs/nyx_negative_annual_replay/fr_be_nl_20260928 --preflight
```

Sans `--preflight`, le même appel écrit une fois
`config/nyx_annual_cpu_qualification_receipt.json` si les deux replays passent.
Le replay prix actuellement disponible est partiel ; l'appel doit donc rester
bloqué et ne produire aucun reçu.
Les scores GPU historiques et le replay des classifieurs seuls ne satisfont
pas ce verrou.

Commande de contrôle en lecture seule :

```powershell
python run_nyx_annual_cpu_live.py --bundle <dossier_bundle> --delivery-day 2026-09-29 --preflight
```

Le producteur prospectif des 12 matrices, des baselines NYX et des références
de rareté n'est pas encore présent dans le clone GitHub. La qualification
annuelle des trois experts CPU n'est pas encore disponible. Le contrôle
retourne donc `ready: false` aujourd'hui.
