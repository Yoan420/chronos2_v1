# Preuves PIT historiques et prospectives — Chronos-2 exogène

## Conclusion

Une archive fournisseur réellement versionnée, interrogée à l'état connu à
`D-1 08:00`, peut constituer une preuve PIT **scientifique pour le backtest**.
Elle ne prouve pas que les octets ont été capturés avant cette origine et ne
doit donc jamais être présentée comme une capture prospective de production.

Le pipeline distingue désormais ces deux propriétés :

| Propriété | Usage autorisé | Preuve minimale |
|---|---|---|
| reconstruction causale | exploration | horodatages d'information `<= cutoff` |
| PIT historique attesté | entraînement, OOF et backtest hors ligne | capacité fournisseur et manifeste exact approuvés dans une racine de confiance distincte, ledger complet, réponses brutes archivées et relues à l'identique |
| capture prospective | shadow et live | octets capturés avant l'origine, drapeau par ligne et sidecar SHA-lié |

Une preuve historique attestée ne positionne jamais
`production_pit_evidence=true`. Le gate prospectif existant reste inchangé.
Les SHA garantissent l'intégrité après matérialisation; ils ne prouvent pas à
eux seuls qu'un fichier existait dans le passé. Une déclaration de source, un
sidecar et un manifeste produits ensemble ne forment donc jamais une
attestation. La datation historique repose explicitement sur une revue externe
de la capacité fournisseur **et du manifeste achevé**, alors que la datation
live repose sur la capture prospective locale.

## État des sources actuellement matérialisées

### Saturn

La bibliothèque locale `tshistory_lite 0.5` documente bien
`get(revision_date=...)` comme l'état de la série connu à cette date
d'insertion. Elle expose aussi `history()` et `insertion_dates()`. Cela rend la
méthode pertinente pour un backtest causal.

La même interface montre toutefois que l'historique fait partie d'une
frontière de confiance mutable : un auteur peut fournir explicitement une
`insertion_date`, et l'API expose `strip()` et `delete()`. Les dates d'insertion
ne sont donc pas, à elles seules, une preuve cryptographique immuable.

Les matérialisateurs actuels conservent le résultat Parquet et son SHA, mais
pas les octets bruts de chaque réponse. Lorsque Saturn ne retourne pas le vrai
timestamp d'insertion sélectionné, ils inscrivent le cutoff de requête dans
`revision_time_utc`. Les sidecars le déclarent correctement avec
`provider_revision_timestamp_available=false`. Ces caches restent donc
`research_versioned_history`; aucun booléen ne peut les reclasser.

Pour devenir admissible au backtest attesté, une nouvelle matérialisation doit
conserver, pour chaque jour : la requête canonique et son SHA, le cutoff exact,
au moins deux lectures stables, les octets bruts de la réponse et leur SHA, le
statut TLS, l'heure de téléchargement et l'identité/version/hash du client. La
capacité `insertion_date = état connu au plus tard à l'instant demandé` doit
être revue et épinglée dans le manifeste.

### JAO Core

Le matérialiseur JAO est plus auditable : il conserve les enregistrements JSON
parsés dans une archive canonique compressée, leurs SHA, les filtres appliqués,
`lastModifiedOn`, l'heure de récupération et le statut TLS. Le client sait
répéter une lecture multi-page jusqu'à obtenir deux snapshots identiques, mais
retourne immédiatement après une lecture lorsque la réponse tient sur une
page. Dans le store local audité, 529 partitions déclarent une seule lecture,
202 partitions anciennes ne déclarent pas ce compteur et aucune n'en déclare
deux. Ce n'est donc pas encore le ledger à deux relectures exigé ici.

JAO ne fournit toutefois pas ici de requête historique as-of. Une récupération
postérieure n'est causalement admissible que si `lastModifiedOn <= cutoff` et
si le contrat fournisseur garantit que ce watermark couvre tout le snapshot
retourné et ne peut être réinitialisé. Cette garantie n'est pas attestée dans
les artefacts locaux. De plus, le store courant contient des jours de fallback
depuis une publication initiale antérieure; la chaîne donneuse doit être
explicitement incluse dans un futur ledger attesté. Le store actuel reste donc
une excellente preuve de recherche, mais pas une preuve historique attestée.

## Contrat fail-closed ajouté

`chronos2_exogenous/pit_evidence.py` vérifie le nouveau type
`chronos2_attested_historical_asof_archive`. Le manifeste complet doit être
SHA-épinglé par la déclaration de source, mais aussi être présent dans un
registre de confiance indépendant du bundle. La capacité fournisseur possède
son propre registre indépendant. Ces deux registres immuables sont vides par
défaut : une modification revue du code est requise pour approuver les SHA
exacts. Cela empêche de fabriquer localement les fichiers, recalculer leurs SHA
et s'auto-attribuer `historical_backtest_pit_evidence=true`. Il lie :

- les octets exacts du Parquet ;
- le contrat exact qui interprète ces octets (colonnes, aliases, famille,
  routage vers Chronos/résiduel/Kalman, fuseaux, cutoff et colonnes
  d'information) ;
- une capacité fournisseur revue et limitée à
  `offline_training_backtest_only` ;
- l'identité, la version, le fichier et le SHA du client API ;
- le fichier/hash du matérialiseur et le document/hash du contrat fournisseur ;
- un ledger canonique couvrant exactement chaque jour demandé, sans jour
  supplémentaire non vérifié ;
- le cutoff civil exact de chaque jour ;
- au moins deux empreintes de réponse identiques ;
- une archive brute locale dont le SHA est celui de la réponse ;
- TLS, zéro violation causale et, selon le mécanisme, une requête as-of exacte
  ou un `lastModified` fournisseur antérieur au cutoff.

Tous les fichiers internes référencés doivent rester sous la racine du bundle :
les chemins absolus, `..` et liens symboliques sortants sont refusés. La requête
doit être exactement égale au patron canonique approuvé, sans paramètre
additionnel. Les timestamps d'audit doivent déclarer explicitement UTC, ne
  peuvent être futurs, et chaque relecture possède un identifiant unique, son
  propre constat TLS et une archive physique distincte dont le SHA est vérifié.
  Le gate historique strict refuse aussi une couverture
horaire partielle ; les journées DST doivent contenir exactement leur timeline
physique de 23, 24 ou 25 heures.

La capacité reconnaît explicitement la mutabilité de l'historique fournisseur
et interdit les scopes `prospective_shadow` et `live_inference`. Une archive
manquante, modifiée, non couverte ou un manifeste non identique au SHA épinglé
fait échouer la construction.

`feature_bank.py` expose maintenant séparément :

- `historical_backtest_pit_evidence` par source ;
- `historical_backtest_ready` et ses blockers au niveau banque ;
- `production_pit_evidence` et `production_ready`, inchangés et réservés à la
  capture prospective.

Le panel propage les deux scopes à titre d'audit, sans les substituer au gate
`production_pit_evidence` lu par la gouvernance live. Les manifests historiques existants ne
possèdent ni le type attesté, ni le ledger, ni les réponses brutes exigées :
ils restent faux sur le gate historique attesté comme sur le gate prospectif,
par construction.

## Limites et frontière de confiance

Le fournisseur ne remet actuellement ni reçu signé ni preuve cryptographique
de l'heure de réponse. L'attestation historique dépend donc encore de la revue
humaine, du registre embarqué dans le code approuvé et des contrôles d'accès du
fournisseur. Un acteur autorisé à modifier le code de confiance peut modifier
ce registre : branche protégée et revue indépendante sont donc obligatoires.
En revanche, la seule capacité à écrire dans le dossier de données ne suffit
plus pour fabriquer une preuve acceptée.

Les champs historiques propagés dans le sidecar du panel sont informatifs. Ils
ne doivent jamais devenir un gate de promotion sans relire le manifeste exact,
son registre d'approbation et son contrat d'interprétation. Le contrat existant
de capture prospective demeure séparé et n'a pas été assoupli par cette
évolution.

## Gouvernance recommandée pour la suite

Une évolution de gouvernance pourra, après matérialisation attestée réelle,
appliquer les gates suivants sans attendre trois ans de captures live :

1. entraînement, OOF et holdout : `historical_backtest_ready=true` sur chaque
   source et chaque jour ;
2. émission shadow/live : capture prospective avant origine pour les features
   de l'horizon prédit, panel scellé et cible future absente ;
3. contexte historique du run live : source historique attestée, immuable et
   identique au schéma d'entraînement ;
4. promotion : backtest final admissible, au moins 30 émissions shadow
   prospectives, actuals attachés après gel, et chaîne de SHA intacte.

Cette séparation devra faire l'objet d'un changement de schéma explicite dans
la gouvernance. Elle ne doit pas être réalisée en remplaçant l'ancien booléen
`production_pit_evidence` ni en modifiant les sidecars existants.
