# POC Kalman flow-based — CNEC/RAM Core

## Objectif et statut

Le POC teste si l'erreur restante de `residual_corrected` dépend du domaine
flow-based Core disponible avant l'enchère. Il est volontairement séparé de
`Forecast.ps1 -Mode All` : aucune panne JAO ne peut bloquer les forecasts
actuels, et aucune promotion n'est automatique.

Le rapport compare exactement les 365 derniers jours. Un vrai backtest
rolling-365 exige néanmoins :

- 365 jours affichés ;
- 365 jours antérieurs pour entraîner la première origine ;
- le jour live à prévoir.

La collecte POC couvre donc **731 jours physiques**. Une collecte de seulement
365 jours suffit pour l'audit descriptif CNEC/RAM, pas pour une comparaison
FINAL365 équitable.

## Contrat causal JAO

L'entrée v1 est l'endpoint public JAO Core `initialComputation`, filtré côté
serveur avec `Presolved=true`. L'API fournit RAM, Fmax, FRM et les PTDF Core.
Pour chaque journée `D`, le snapshot est admissible uniquement si
`lastModifiedOn <= D-1 08:00` dans le fuseau civil. Le téléchargement
historique est marqué comme tel : `lastModifiedOn` permet un gate utile, mais
une archive locale réellement capturée avant 08:00 restera obligatoire avant
toute promotion opérationnelle.

JAO peut fournir un `lastModifiedOn` différent par page. Le client demande
jusqu'à 40 000 lignes pour rendre la journée atomique dans le cas usuel. Si
une pagination reste nécessaire, deux lectures complètes consécutives doivent
avoir la même empreinte SHA-256 (contenu, total et ensemble des watermarks).
L'audit retient toujours le watermark le plus tardif, choix conservateur pour
le contrôle PIT ; aucune page d'une lecture instable n'est conservée.

Ces données restent dans trois couches distinctes :

1. réponse JSON brute compressée, immuable et hashée ;
2. table CNEC normalisée, conservant les identifiants pour l'audit ;
3. agrégats horaires stables transmis au POC.

Les données post-coupling sont interdites dans les features : contraintes
actives, shadow prices, `RAM@MCP`, positions nettes/échanges réalisés et
spreads de prix. Elles pourront être collectées séparément comme labels de
diagnostic.

## Variables du premier POC

Le Kalman ne reçoit pas des milliers d'identifiants CNEC. Il utilise des
agrégats robustes : nombre moyen de CNEC, minimum et quantiles de RAM, parts de
RAM sous 500/1 000 MW, ratios RAM/Fmax, sensibilités PTDF de la France et
spreads de sensibilité FR-DE/BE/NL, compte séparé des contraintes externes et
d'égalité, marges RAM des seules contraintes externes,
sensibilité de l'interconnexion HVDC ALEGrO (`ALBE-ALDE`), stress RAM pondéré
par PTDF et concentration HHI du stress. Les variables sont invariantes au
choix du slack PTDF ; aucune pression signée dépendante du slack n'entre dans
le modèle.

Quand l'initial domain est absent sur une ou plusieurs MTU (fallback,
spanning ou incident de publication), le pipeline n'utilise jamais le domaine
final. Il conserve le taux de disponibilité et la part de MTU manquantes. Une
heure entièrement vide reçoit la médiane des autres heures de la même
publication initiale, disponible en bloc à D-1, ainsi qu'un indicateur
`flowbased_hour_imputed=1`. L'absence devient donc un signal explicite sans
fuite post-coupling et ne bloque pas le backfill 365/731 jours.

Si toute la publication `Presolved=true` du jour est vide, le profil est
recopié depuis la dernière publication initiale antérieure admissible, par
heure civile locale, avec disponibilité forcée à zéro. Le manifeste conserve
le jour source, son checksum et le nombre de jours de fallback consécutifs.
La même politique s'applique à une publication corrigée après le cutoff : le
brut tardif et son watermark restent archivés pour l'audit, mais ne deviennent
jamais des features. Seul le profil causal antérieur est utilisé.
Cette branche reste une reconstruction de recherche (`historical_previous_initial_only`),
jamais une capture opérationnelle du vintage courant.
Dans le backtest et le forecast publiés, toute heure dont la disponibilité
CNEC vaut zéro est en plus neutralisée : `kalman_flowbased` est forcé à rester
identique à `residual_corrected` sur cette heure.

La banque gouvernée compare :

- biais lent et profil harmonique ;
- charge résiduelle européenne (`linear_market`) ;
- domaine flow-based (`linear_flowbased`) ;
- charge résiduelle + domaine (`linear_market_flowbased`) ;
- correction d'échelle ;
- identité implicite, toujours disponible.

Pour éviter d'invalider les caches des quatre Kalman opérationnels, les deux
noms flow-based sont pour ce POC des alias sémantiques de familles exogènes
génériques déjà testées. Cette correspondance est écrite dans le YAML et dans
chaque audit. Le modèle mathématique et la gouvernance sont inchangés.

## Utilisation

Afficher le plan sans appel réseau :

```powershell
& '.\FlowBased.ps1' -Action Plan -DeliveryDay 2026-09-03
```

Construire le support complet rolling-365 :

```powershell
& '.\FlowBased.ps1' -Action Backfill -DeliveryDay 2026-09-03
```

Sous Windows, le pipeline choisit automatiquement une chaîne de confiance
vérifiée : `SSL_CERT_FILE`, puis `REQUESTS_CA_BUNDLE`, puis le magasin de
certificats Windows. Sur ce poste, `REQUESTS_CA_BUNDLE` est déjà configuré ; la
commande précédente ne nécessite donc aucun paramètre TLS.

Si l'équipe IT fournit explicitement un autre bundle PEM, passer son chemin
réel (et non un chemin d'exemple) :

```powershell
& '.\FlowBased.ps1' -Action Backfill -DeliveryDay 2026-09-03 -CaBundle $env:REQUESTS_CA_BUNDLE
```

Un chemin `-CaBundle` absent ou un PEM invalide est refusé avant tout appel
réseau. `-Insecure` existe seulement comme solution locale de dernier recours
pour l'audit descriptif ; le manifeste le signale et le runner POC refuse une
telle collecte. Le pipeline ne bascule jamais automatiquement en mode non sûr.

Produire l'audit CNEC/RAM puis le POC FR :

```powershell
& '.\FlowBased.ps1' -Action Audit -Country FR -DeliveryDay 2026-09-03
& '.\FlowBased.ps1' -Action Preflight -Country FR -DeliveryDay 2026-09-03
& '.\FlowBased.ps1' -Action Poc -Country FR -DeliveryDay 2026-09-03
```

Une fois le cache JAO rempli, les relances reprennent uniquement les jours
absents. Le premier rolling Kalman reste coûteux ; les refits journaliers sont
ensuite réutilisés via le cache `runs/cache/kalman_rolling/<zone>/kalman_flowbased`.

La première banque est volontairement limitée à la France : ses sensibilités
invariantes au slack comparent FR à DE/BE/NL. Une extension par zone devra
construire les spreads PTDF propres à chaque frontière avant d'activer les
autres pays.

Pour un audit descriptif de 365 jours sans POC :

```powershell
& '.\FlowBased.ps1' -Action Backfill -AuditOnly -DeliveryDay 2026-09-03
```

## Fichiers modifiables

- paramètres Kalman et liste de variables :
  `config/kalman_flowbased_experimental.yaml` ;
- acquisition, contrat PIT et agrégation :
  `chronos2_hourly/jao_flowbased.py` ;
- reconstruction reprenable : `materialize_jao_core_flowbased.py` ;
- audit descriptif : `run_cnec_ram_audit.py` ;
- POC rolling : `run_kalman_flowbased_experiment.py`.

## Critères avant promotion

La promotion ne sera envisagée que si la couverture PIT est complète, si le
gain apparié sur 365 jours est positif et stable sur les deux demi-périodes,
si les jours extrêmes/interconnexions s'améliorent sans dégrader fortement les
jours calmes, et si une capture locale pré-cutoff remplace progressivement le
backfill historique. À ce moment seulement, un mode opt-in pourra précéder une
éventuelle intégration à `Mode All`.
