# Captures prospectives des échanges retardés CWE

Le modèle annuel à 503 colonnes contient cinq échanges Energy Charts, chacun
retardé de **48 heures physiques** et accompagné d'un indicateur de présence.
`run_nyx_annual_exchange_source.py` collecte ces dix colonnes pour chaque
journée de livraison, puis assemble la fenêtre de 365 jours d'entraînement et
la journée prévue. Il utilise la formule historique de
`chronos2_hourly/nyx_lagged_exchange_features.py` : flux physiques DE/FR,
DE/NL, DE/BE et solde DE en GW, plus solde commercial FR converti de MW en GW.
Les signes sont conservés. Il ne produit aucune des 493 autres colonnes.

## Capture quotidienne

Pour la livraison `D`, exécuter avant **D−1 08:00 Europe/Paris** :

```powershell
python run_nyx_annual_exchange_source.py --action capture --delivery-day AAAA-MM-JJ
```

La capture interroge les endpoints publics `/v2/cbpf?country=de` et
`/v2/public_power?country=fr` sur les seules heures sources de `D−48 h`.
Elle vérifie l'identité des séries, unités, résolutions, quatre quarts par
heure quand la résolution est de 15 minutes, l'heure `generated_at` du fournisseur et
l'heure locale de récupération. Ces deux dernières doivent précéder la
coupure. Elle fige les deux corps JSON, le Parquet des dix variables et un
reçu SHA-256 sous `data/pit/nyx_annual_exchange_captures/D/`. Une nouvelle
réponse différente ne remplace jamais une capture scellée.

L'[API officielle Energy Charts](https://api.energy-charts.info/openapi.json)
précise que les timestamps des valeurs désignent le **début** des intervalles,
que `generated_at` correspond à la génération de la réponse et que l'API
limite les requêtes. Elle ne propose pas de requête de révision « as-of ».
`generated_at` **ne prouve pas l'heure de première publication** d'une valeur.
Le reçu atteste seulement que l'état retourné a été effectivement capturé
avant la coupure du jour correspondant. Un intervalle absent du corps API,
une réponse tardive ou une erreur HTTP arrête la capture sans reçu complet.
Une **valeur explicitement `null`** dans un intervalle présent reste `NaN`
avec `__available=0`, exactement comme dans la recette historique ; le reçu
compte les heures manquantes par variable, sans imputation.

## Assemblage de la fenêtre

```powershell
python run_nyx_annual_exchange_source.py --action assemble --delivery-day AAAA-MM-JJ --bundle runs/live/nyx_annual_cpu/AAAA-MM-JJ
```

L'assemblage exige **366 captures quotidiennes vérifiées** : les 365 jours
d'entraînement et `D`. Il relit et recalcule les dix variables de chaque
capture, vérifie leurs empreintes et la grille UTC avec les journées de 23/25
heures. Il publie un Parquet commun aux quatre pays, une archive ZIP immuable
des captures et un reçu `source_receipts/lagged_exchange.json` conforme au
précontrôle NYX. Le reçu lie les artefacts par SHA-256 et indique explicitement
que l'heure de première publication des observations n'est pas certifiée.

Les archives de recherche de 2024–2026 ont été récupérées après les coupures
de nombreuses journées. Elles ne peuvent pas remplacer les captures
prospectives manquantes. Sur un clone neuf, l'assemblage échoue jusqu'à ce
qu'un historique de 365 jours de captures avant coupure existe. Il faut donc
planifier la capture quotidienne sur le poste de travail ; lancer seulement
l'application après 08:00 ne permet pas de recréer la capture du jour.
L'absence de reçu complet laisse le lancement annuel bloqué par le
précontrôle. Une fois cet historique acquis, l'évaluation chronologique de la
chaîne entière reste nécessaire avant activation.
