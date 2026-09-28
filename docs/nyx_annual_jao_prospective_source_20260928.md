# Capture JAO initiale prospective pour CWE annuel

Le [manuel officiel JAO Core, §5.7](https://publicationtool.jao.eu/core/CORE_PublicationHandbook)
annonce l’« Initial Computation (Virgin Domain) » à **01 h 15 D−1**. La
publication « Pre-Final » est annoncée à **08 h D−1** (§5.10) et ne remplace
pas l’entrée initiale du modèle. Cette heure planifiée ne prouve pas qu’un
jour donné était disponible. Une [communication JAO](https://www.jao.eu/news/delay-publication-initial-computation-virgin)
documente par exemple un retard d’Initial Computation.

Le collecteur `run_nyx_annual_jao_source.py` appelle exclusivement
`initialComputation` avec le filtre serveur `Presolved=true`, TLS vérifié,
pour le jour civil `Europe/Paris` et ses 23, 24 ou 25 heures physiques. Une
capture neuve n’est autorisée qu’entre **01 h 15 et 08 h D−1**. La réponse
doit être non vide, avoir été **effectivement reçue au plus tard à 08 h**,
avoir `lastModifiedOn` au plus tard à 08 h et satisfaire les contrôles de
pagination, d’identité de domaine et de bornes UTC du client JAO. Un appel
après 08 h ne peut jamais recréer cette preuve, même si `lastModifiedOn`
indique une heure plus ancienne.

```powershell
python run_nyx_annual_jao_source.py --delivery-day 2026-09-30 --bundle runs/live/nyx_annual_cpu/2026-09-30
```

La commande doit être lancée **le 29 septembre entre 01 h 15 et 08 h**, heure
de Paris. Une exécution planifiée vers 07 h peut la déclencher, avec une
nouvelle tentative avant 08 h si JAO n’a pas encore publié. Une capture déjà
archivée et validée peut ensuite être relue avec `--verify-only` après la
coupure. Le cache local par défaut est
`data/pit/nyx_annual_jao_initial_live/`; il est distinct des archives de
recherche et n’est pas livré par Git.

Chaque capture conserve le JSON brut compressé et l’audit du client JAO. Le
fichier de variables dans le bundle est reconstruit à partir de ce brut avec
les **27 descripteurs annuels** et `extra_jao__available`. Les heures où une
MTU CNEC ou une dépendance numérique manque restent `NaN` avec disponibilité
0. Aucun profil du jour précédent, publication finale, ni médiane ne remplit
ces valeurs. Le fichier de variables historique du matérialiseur JAO, qui
comporte des imputations, reste dans le cache uniquement comme audit de la
capture ; il n’alimente pas cet artefact annuel.

## Qualification de la fenêtre d’entraînement

Le modèle annuel requiert les **365 jours précédents plus la journée D**.
Le reçu `source_receipts/jao_initial.json` ne passe le précontrôle NYX que
si les 366 jours disposent chacun d’une archive faite avant **sa propre**
coupure D−1 08 h, avec les empreintes brutes et audits valides. Le fichier
`source_artifacts/jao_initial/snapshot_ledger.json` détaille ces captures ;
un Parquet de 366 jours est produit seulement quand la fenêtre est complète.

Sur un clone propre, la première exécution archive la journée courante,
mais émet un reçu `INCOMPLETE` et un code de sortie 2. Le précontrôle NYX
refuse ce reçu : les 365 captures antérieures ne peuvent être créées
rétroactivement. `lastModifiedOn` indique l’état de l’API, pas un cliché
historique vérifié. Ainsi, la présence du code dans GitHub ne rend pas la
prévision annuelle CWE immédiatement lançable sur le poste professionnel.
Il faut soit une archive indépendante prouvant réellement les 365 captures
antérieures, soit qualifier par une nouvelle évaluation une recette qui
n’exige pas cet historique prospectif JAO.

Un contrôle local explicite illustre ce point : pour la livraison du
3 septembre 2026, le cache d’étude JAO indique `pit_eligible=true` selon
`lastModifiedOn`, mais `operational_pit_eligible=false` car son téléchargement
a eu lieu le 2 septembre après 08 h. Le collecteur prospectif n’importe pas
ce cache. L’inventaire des **731 partitions** du cache d’étude donne
**0 capture opérationnelle avant sa coupure**. Il n’y a donc à ce jour aucune
preuve locale suffisante pour les 365 jours d’entraînement.

Cet adaptateur ne produit que le groupe source `jao_initial`. Les autres
groupes, la baseline NYX et les matrices ordonnées 449/503/123 colonnes
restent nécessaires avant qu’une prévision puisse être publiée.
