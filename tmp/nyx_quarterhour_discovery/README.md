# Sources natives de prix day-ahead à 15 minutes — variante 2 NYX

Collecte isolée et audit du 16/09/2026. **Aucun modèle, entraînement ou calcul de score prévisionnel n'a été exécuté par cette tâche.** Les sources, caches, labels et résultats du NYX opérationnel restent inchangés.

## Archive utilisable

`data/pit/nyx_quarterhour/primary_prices_20251001_20260915/manifest.json`, avec `native_prices.parquet` et `collection_audit.json`.

- **350 jours**, du 01/10/2025 au 15/09/2026 inclus.
- **4 pays × 33 600 quarts = 134 400 lignes**, tous physiques, uniques, finis et complets.
- **8 400 heures par pays** après moyenne des quatre quarts.
- Dans chaque pays : **100 quarts le 26/10/2025**, **92 quarts le 29/03/2026** ; les deux heures du retour d'hiver restent distinctes en UTC.
- Champs : `timestamp_utc` (UTC), `zone` (BE/DE/FR/NL), `actual_15m` (EUR/MWh).
- SHA256 parquet : `79e7954cc1c69a7b67fa7cb00f458c44a7d6275d7e7a7693c5948972edb8f3c0`.

L'acquisition finale effectue **un GET de plage par pays**, au maximum deux simultanément. Les observations sont directement celles des séries primaires UTC, sans formule intermédiaire, conversion horaire, répétition ou interpolation.

## Identifiants et preuve de résolution

| Pays | Série primaire Saturn | Catalogue | Heures avec variation entre quarts / 8 400 |
|---|---|---|---:|
| BE | `60451` | `power`, primary | 8 385 |
| DE | `60452` | `power`, primary | 8 382 |
| FR | `60454` | `power`, primary | 8 215 |
| NL | `60453` | `power`, primary | 8 387 |

Les quatre métadonnées indiquent un index UTC conscient du fuseau (`tzaware=true`), et l'endpoint formule retourne `null`. Un échantillon brut couvrant les 25–27/10/2025 fournit une cadence de 900 secondes, sans doublon ni trou, pour chaque source. La collecte complète confirme ensuite la totalité des grilles physiques et leurs variations intrahoraires. Les heures de prix constants restent des observations valides ; leur présence ne constitue pas une interpolation.

Les alias nommés `power.price.be.euromwh.qh.obs.epex` et `power.price.de.euromwh.qh.obs.epex` pointent directement vers les primaires BE/DE. Pour FR, l'alias privilégie PointConnect `165188855`, puis `60454` ; pour NL, PointConnect `165546978`, puis `60453`. **Ces branches PointConnect n'ont pas été utilisées**, les primaires seuls couvrant toute la fenêtre. Les wrappers `power.nrjscan.{be,de_lu,fr,nl}.price.spot.qh.epex.obs.eurmwh` retirent le fuseau en CET ; ils ont été évités afin de conserver les deux heures physiques du pli DST. Les produits intraday/IDA ne sont pas utilisés.

Les preuves de découverte sont les fichiers `metadata*.json`, `probe_2025-10-25.json` et les échantillons CSV de ce dossier. Le catalogue local, acquis précédemment le16/09, n'a servi qu'à découvrir des identifiants existants ; les métadonnées et observations ont été effectivement relues sur Saturn. Les lectures réseau ont utilisé l'escalade autorisée, sans modifier les proxys ni exposer d'identifiants.

## Vintage et contrat du lecteur

Le manifeste annonce explicitement **`price_vintage="latest_observations"`**, `provider_revision_timestamp_available=false` et `production_pit_evidence=false`. La date de téléchargement est connue, pas la publication historique de chaque valeur. Aucun `revision_date` historique n'est inventé. Il s'agit de labels actuels pour une expérience rétrospective, pas d'une archive reçue à chaque date d'émission.

`nyx_quarterhour/sources.py` expose :

```python
frame, audit = read_native_prices(manifest_path)
hourly = hourly_means(frame)
```

Le lecteur vérifie le contrat `schema_version=1`, `artifact_type="nyx_quarterhour_price_observations"`, les quatre sources primaires autorisées, unité/résolution/fuseau/vintage, le chemin du parquet frère, son SHA256, l'absence de liens/jonctions et la grille complète annoncée. Le statut est `complete` ou `partial`. Une heure n'est agrégée que si ses quatre quarts sont uniques et finis ; les négatifs sont conservés. Les archives scellées ne sont jamais écrasées par le collecteur.

Commande d'acquisition isolée :

```text
C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe materialize_nyx_quarterhour.py --start-day 2025-10-01 --end-day 2026-09-15 --output-dir data/pit/nyx_quarterhour/primary_prices_20251001_20260915 --workers 2
```

## Comparaison des labels horaires — aucun score de modèle

La comparaison utilise le bundle NYX complet du **16/09/2026**, via le chargeur qui vérifie son identité, ses fichiers figés et ses observations report-only. Chaque moyenne native est appariée à une heure physique, sans décalage ni suppression : **8 400 heures par pays**. Les pièces `hourly_label_comparison.json` et `hourly_label_pairs.parquet` sont conservées à côté de l'archive source, avec les empreintes de la source et du bundle vérifié.

| Pays | Écart absolu moyen entre moyenne native et label report-only, EUR/MWh | Maximum, EUR/MWh | Heures différentes >0,01 |
|---|---:|---:|---:|
| BE | 0,159248 | 37,7025 | 155 |
| DE | 0,166495 | 30,8425 | 155 |
| FR | 0,149036 | 36,4125 | 155 |
| NL | 0,159764 | 33,7275 | 155 |

**Sur le test fixé du 18/06 au 15/09/2026, aucun écart supérieur à1e−9 n'est observé, tous pays confondus.** Les anciennes divergences se terminent le16/06 à21:00 UTC. L'examen français montre que les155 labels divergents correspondent exactement au quart`:45` de l'heure, au lieu de la moyenne des quatre ; les heures UTC concernées sont21,22et23. C'est un motif de bordure observé, **pas une preuve de sa cause amont**. Le diagnostic condensé se trouve dans `disagreement_localization.json` ; aucune donnée NYX n'a été réécrite.

Pour toute comparaison prévisionnelle, les prédictions du candidat et celles de NYX doivent être évaluées sur la **même moyenne native des quatre quarts**, avec les mêmes heures. Les statistiques ci-dessus sont seulement des écarts entre labels, jamais un gain de prévision.

## Vérification locale

**24 tests d'acquisition réussis en3,93s**, sans réseau ni modèle : DST92/96/100, négatifs, identité des GET sans fausse révision historique, quarts absents/NaN sans remplissage, refus du naïf/doublon/horaire/horsgrille, confidentialité des erreurs, HTTP lecture seule, checksums et schémas falsifiés, refus d'une archive prétendue complète avec un quart retiré, non-écrasement et confinement des descendants/temporaires/verrous.
