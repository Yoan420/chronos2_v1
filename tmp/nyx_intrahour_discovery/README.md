# Découverte de fondamentaux natifs à 15 minutes pour NYX horaire

Audit et lectures rétrospectives du 16 septembre 2026. NYX, ses caches et ses résultats scientifiques restent inchangés. Cette archive prépare un essai horaire enrichi ; elle ne constitue ni un modèle de prix au quart d'heure, ni une preuve de gain prévisionnel.

## Source retenue pour le premier essai

- **Alias** : `be_solar_elia_fcst` ; pays **BE**, conducteur **solar**, unité **MW**.
- **Identifiant Saturn primaire exact** : `23259`, source de catalogue `power`, type `primary`.
- Métadonnées : `tzaware=true`, index physique UTC ; endpoint formule retourne `null`.
- La formule nommée `power.stp.da_solar_production.be.mw.qh.fcst.elia` référence `(resample (series "23259") "15min" #:method "ffill")`. **L'acquisition lit directement `23259` et n'exécute pas ce rééchantillonnage.**
- Cinq échantillons physiques, consultés au cutoff D−1 à 08:00 Europe/Paris : 2025-10-15, 2026-01-15, 2026-04-15, 2026-07-15 et 2026-09-15. Chacun fournit **96 valeurs natives**, pas de valeur manquante, pas de doublon, pas d'horodatage hors grille et un espacement de **900 secondes**. Les heures présentant des variations entre quarts valent respectivement **11, 10, 15, 17 et 13**. Cela exclut une simple répétition horaire pour ces échantillons ; l'amont exact du fournisseur reste une limite de provenance.

Le catalogue est le cache lu en lecture seule `tmp/saturn_demand_deep_catalog.json`, acquis auparavant le 16/09. Les métadonnées et formules ont ensuite été interrogées auprès de Saturn. Les fichiers `metadata*.json`, `probe*.json` et `raw*.csv` conservent les éléments structurels et échantillons. Aucun identifiant d'authentification n'est enregistré. Une première requête sandbox a échoué (`ProxyError`) ; les lectures suivantes ont utilisé l'escalade réseau autorisée, sans supprimer ou modifier les variables proxy.

## Candidats examinés et non activés

| Candidat | Résultat observé | Décision |
|---|---|---|
| `power.stp.wind_production.be.mw.qh.fcst.cet.meteologica` → `38840` | Primaire réellement horaire, formule `resample 15min ffill` | Exclu : faux enrichissement intrahoraire |
| Même famille DE → `28217`, FR → `28211`, NL → `28226` | Même résultat : 24 valeurs horaires D=15/09, as-of14/09 08Paris | Exclues |
| Charge BE `38100`, référencée par `power.stp.da_total_load.be.mw.qh.fcst.cet.elia` | 96 valeurs et variation intrahoraire, mais horodatages naïfs ; fuseau non prouvé | Non activée |
| Charge NL `41271`, référencée par `power.stp.da_total_load.nl.mw.qh.fcst.cet.entsoe` | Aucune valeur à ce cutoff pour D=15/09 | Non activée |
| `power.fr.demand.peak.gw.fcst.quarter.hourly` → `20747` | Grille15min sur la veille, **aucune valeur du lendemain** à08h sur l'échantillon | Non activée |
| Vent BE onshore/offshore day-ahead Elia | Formules `pointconnect "108845449"` et `pointconnect "108845412"`; UTC déclaré, mais granularité historique/08h non vérifiées | Non activées |
| VentFR onshore day-ahead ENTSO-E | `resample (pointconnect "114740602") 15min ffill` ; source pointconnect non auditée | Non activée |
| ChargeDE50Hertz/Amprion | Pointconnect `103114898` / `101648707`, zone TSO et non total national ; données non interrogées | Non activées |
| Solaire canonique horaireFR/DE/NL | Mélange70% primaire Meteologica50262/28214/50265 +30% ECMWF ; ne prouve pas du natif15 | Non activé |

Aucune série résiduelle native15 valide pour les quatre pays n'a été établie. Le pilote a donc **un fondamental belge commun seulement**. Évaluer les quatre sorties NYX reste possible, avec conclusion Belgique séparée ; ne pas présenter l'expérience comme quatre ensembles nationaux enrichis natifs.

## Archive isolée et sémantique temporelle

Sortie autorisée : `data/pit/nyx_intrahour/be_solar_elia_20250916_20260915/manifest.json`, compagnon `native_forecasts.parquet`, audit `collection_audit.json` et pièces journalières sous `days/`.

**Collecte terminée et validée : 365/365 journées complètes, 35 040 quarts physiques = 8 760 heures, zéro absence, NaN ou erreur.** Grille UTC continue du 15/09/2025 à22:00 UTC au 15/09/2026 à21:45 UTC. Le 26/10/2025 contient100 quarts et le 29/03/2026 en contient92 ; aucune heure n'a été supprimée. **4 851 heures présentent une variation intrahoraire** ; les autres heures constantes, notamment nocturnes, ne sont pas des valeurs absentes.

Contrôle indépendant après collecte : lecteur `read_native_sources` accepté, cutoffs des365 journées exacts, empreintes des365 pièces journalières vérifiées,736 chemins contrôlés sans lien/jonction. Résumé machine : `tmp/nyx_intrahour_discovery/final_validation.json`. SHA256 données : `264ec36a410265a682f20b4c4e36a27d0b45f4497d254a9460693a8e70756ac0`. SHA256 manifeste : `ba2867345ad6904ef7c3417e90887ab6435801604173690cc78865c4ce99a00e`.

Période demandée : **365 jours, du 16/09/2025 au 15/09/2026 inclus**. Une seule source, au maximum deux requêtes simultanées. `materialize_nyx_intrahour.py` utilise `nyx_intrahour/saturn_sources.py`, qui autorise seulement GET ; le nom de série est fixe et audité. La garde de chemin refuse les jonctions/liens redirigeant le namespace de recherche ou le dossier de sortie.

Chaque requête appelle `Client.get("23259", revision_date=cutoff_utc, from_value_date=first_quarter_utc, to_value_date=last_quarter_utc, _keep_nans=True)`. Les bornes sont inclusives. Le cutoff est construit à partir de la **date civile précédente à08h Paris**, puis converti en UTC ; il ne résulte pas d'une soustraction fixe de24heures. Les journées physiques ont92,96ou100quarts selonDST. Les valeurs absentes restent absentes, les NaN restent NaN, les doublons/horodatages naïfs ou hors grille font refuser la journée ; aucun remplissage, arrondi temporel, remplacement de valeur ou conversion vers une courbe horaire répétée.

Le tableau expose `source_alias`, `value_time_utc`, `snapshot_time_utc`, `revision_time_utc`, `value`, avec les trois dates UTC. **Snapshot et revision valent le cutoff demandé ; ce ne sont pas des dates de publication du fournisseur.** Le manifeste annonce `temporal_evidence="retrospective_asof"`, `provider_revision_timestamp_available=false` et `production_pit_evidence=false`. `downloaded_at_utc` est enregistré séparément dans chaque audit. Une relecture as-of de l'archive Saturn ne prouve ni une réception réelle historique ni l'absence de modifications rétrospectives du fournisseur.

La collecte scelle le SHA256 des données finales, les pièces de découverte et son audit. Les chiffres complets de disponibilité doivent être lus dans `manifest.json` après terminaison ; les jours avec erreurs ou trous ne sont pas masqués. `--resume` est une **reprise avant gel**, pas une promesse d'immuabilité : elle réutilise les journées réussies dont le checksum concorde et retente les erreurs. Une expérience doit copier les données et le manifeste exacts, puis vérifier leurs checksums ; ne pas relancer une collecte sur les entrées figées de l'expérience.

Commande de reproduction de l'archive isolée :

```text
C:/Users/BQ6757/venvs/pricefm311/Scripts/python.exe materialize_nyx_intrahour.py --start-day 2025-09-16 --end-day 2026-09-15 --output-dir data/pit/nyx_intrahour/be_solar_elia_20250916_20260915 --workers 2
```

Vérification locale : **18 tests d'acquisition réussis en 2,65 s**. Ils couvrent les cutoffs civils et DST (92–100 quarts), les absences/NaN sans remplissage, les données naïves/dupliquées/hors grille/horaires refusées, la confidentialité des erreurs, le refus des mutations HTTP, le manifeste/checksum, la reprise et les refus de jonction avant écriture, y compris dans les sous-dossiers, fichiers finaux, fichiers temporaires et verrous. Aucune science NYX n'est exécutée par ces tests.
