# NYX Physical P50 — laboratoire indépendant

Ce laboratoire teste deux hypothèses de sévérité des pics, **sans modifier `Forecast.ps1`, ses modes, ses modèles ni ses rapports opérationnels**. Il ne promeut aucun candidat. Tous ses fichiers sont confinés à `runs/experiments/nyx_physical_p50_v1`.

## Utilisation

```powershell
# Calcul local sur les archives déjà disponibles, avec reprise des checkpoints.
& 'C:\Users\BQ6757\chronos2_v1\NyxPhysical.ps1' -Action Run

# Vérifier l'état et les empreintes des résultats/rapports.
& 'C:\Users\BQ6757\chronos2_v1\NyxPhysical.ps1' -Action Status

# Régénérer le rapport sans réentraîner les modèles.
& 'C:\Users\BQ6757\chronos2_v1\NyxPhysical.ps1' -Action Report
```

La collecte réseau est **explicite et séparée**. `Run`, `Backtest` et `Prepare` ne téléchargent aucune donnée. Une absence de données qualifiées entraîne l'abstention du candidat réseau, pas une reconstruction de capacité.

```powershell
# Facultatif : collecte publique JAO pour une plage explicite de 30 jours au maximum.
& 'C:\Users\BQ6757\chronos2_v1\NyxPhysical.ps1' -Action Collect `
    -StartDay 2026-09-04 -EndDay 2026-09-15

# Après une collecte ou une modification de configuration, créer un NOUVEAU snapshot.
& 'C:\Users\BQ6757\chronos2_v1\NyxPhysical.ps1' -Action Prepare
& 'C:\Users\BQ6757\chronos2_v1\NyxPhysical.ps1' -Action Run
```

La collecte ne rend pas automatiquement un jour utilisable à 08 h : ses publications, versions et heures doivent encore satisfaire le contrat d'entrée. Ne fournir `-CaBundle` qu'avec le chemin d'un vrai certificat local si le proxy d'entreprise l'exige ; la vérification TLS ne doit pas être désactivée. `-DryRun` affiche les arguments sans lancer Python. `-PythonExecutable` permet d'utiliser un environnement compatible.

Configuration : `config/nyx_physical_p50.yaml`. La source historique est volontairement figée. La seule option de calcul initiale est `threads: 1` ou `2`, sans grille réglée sur le pic du 14 septembre. `-RunDirectory` désigne un snapshot particulier pour Run/Backtest/Report/Status, avec sa configuration exacte.

## Modèles comparés

| Nom | Hypothèse |
|---|---|
| `fuel_transport_direct` | Transport de la sévérité de la queue positive selon le clean gas cost, sans modifier le détecteur ni sa distribution normale. |
| `fuel_transport_governed` | Même proposition avec poids choisi uniquement sur les erreurs historiques déjà connues. |
| `network_fuel_direct` | Même transport, avec CDF conditionnée en plus par des expositions directionnelles du domaine initial JAO qualifié. |
| `nyx_physical_p50` | Variante réseau + combustible gouvernée, fixée comme candidat principal avant ce replay. |

Les références sont NYX `nuclear_kalman`, Storm et les variantes archivées disponibles dans le rapport. Les comparaisons utilisent des heures communes et la même définition des KPI et de l'EVA ; le rapport indique les jours/heures réellement couverts.

## Méthodologie : une vraie médiane, pas une moyenne renommée P50

Pour chaque heure, le résiduel est `r = prix observé − NYX`. Le détecteur historique fournit la probabilité `p` de l'événement `r ≥ u`, où `u` est le seuil zonal appris sur le bloc CORE. Il est réutilisé **sans réentraînement, recalibration ni hausse arbitraire de sa probabilité**.

La distribution est un mélange :

`F(r) = (1 − p) F_normal(r < u) + p F_spike(r ≥ u)`.

La médiane est l'inverse de cette CDF en 0,5. Ce n'est pas la moyenne des médianes des deux régimes. Lorsque `p > 0,5`, le quantile utilisé dans la queue est `1 − 0,5/p`. Par exemple, une probabilité de 54 % utilise approximativement le 7,4e percentile de la queue, pas son sommet.

Dans l'ablation combustible, la distribution normale et les poids de feuilles des forêts historiques restent inchangés. Pour chaque observation CORE positive, la nouvelle amplitude à l'heure cible est :

`r_transporté = u_cible + CGC_cible × (r_CORE − u_CORE) / CGC_CORE`.

Cette transformation conserve la séparation des régimes. Seul l'excès au-dessus du seuil est redimensionné. Les CGC doivent être finis, strictement positifs et issus des informations disponibles au cutoff. Le support transporté peut changer d'ordre : les quantiles sont recalculés à partir de la CDF pondérée, jamais à partir de rangs supposés inchangés.

L'hypothèse « l'excès résiduel se redimensionne avec le coût du gaz » est **testable mais non garantie** : NYX incorpore déjà une partie de l'effet des combustibles. Il ne s'agit ni d'un calcul de dispatch ni de l'identification de la dernière centrale appelée.

La variante réseau ajoute des descripteurs directionnels JAO à la CDF, pas au détecteur figé. Ces expositions utilisent le domaine initial et ses PTDF/RAM. **Elles ne sont pas une capacité d'import physiquement réalisable** ; aucune donnée réseau post-couplage n'entre dans les variables explicatives. Les prix réalisés restent naturellement les labels historiques, disponibles seulement après leur publication.

Le transport CGC ne redimensionne pas le régime normal. En revanche, la variante réseau réapprend les poids conditionnels des deux régimes sur les heures CORE dont le réseau est qualifié. Son échantillon d'apprentissage et ses possibilités d'intervention sont donc différents. Le rapport compare les chaînes complètes, abstentions incluses ; il ne mesure pas l'effet causal isolé de l'ajout d'un PTDF.

Les propositions positives passent les gates et plafonds déclarés. Les poids gouvernés sont évalués chronologiquement ; le poids zéro conserve NYX. L'interpolation porte sur une fonction de quantiles monotone, puis une calibration historique élargit les intervalles lorsque les observations antérieures sont suffisantes. Aucune couverture future ni non-dégradation de MAE n'est garantie.

## Séparation entraînement, calibration et évaluation

- L'origine est strictement **D−1 à 08 h Europe/Paris**, avec heures UTC physiques et jours DST de 23/24/25 heures.
- Les plis et probabilités du détecteur sont figés. L'apprentissage de la CDF utilise son CORE ; les 28 jours de calibration du détecteur ne sont pas réintroduits silencieusement dans le CORE.
- La fenêtre historique est plafonnée à 365 jours. Le panel source possède 365 jours d'évaluation, mais pas 365 jours supplémentaires de chauffe avant chacun de ces jours. La phase d'abstention initiale reste incluse et explicitée.
- Les observations publiées après l'origine courante ne participent pas aux ajustements de cette origine. Les prix Storm ne servent ni à l'apprentissage ni à la gouvernance.
- L'année et les pics ayant déjà été examinés, ce replay est exploratoire. Il ne constitue pas une validation prospective indépendante.

## Pourquoi cette expérience ne promet pas de résoudre le 14/09 à elle seule

À 19 h, le détecteur archivait déjà un risque d'erreur positive importante en DE et BE, mais l'amplitude conditionnelle était faible. Son CORE se terminait au 16/08 ; le CGC du 14/09 dépassait le maximum CORE d'environ 20 %. Le transport de queue teste cette faiblesse de représentation sans changer artificiellement les probabilités.

Avec les probabilités et poids figés, ce simple transport ne peut pas reproduire une hausse arbitraire de plusieurs centaines d'euros. Il reste nécessaire de vérifier les informations réellement nouvelles : disponibilité horaire des unités, flexibilité mobilisable, stockage et contraintes réseau. La divergence de prix DE/BE/FR/NL ne prouve pas, à elle seule, quelle centrale ou quelle ligne était responsable.

Limites sources conservées : températures manquantes après le 04/09 dans le panel figé ; disponibilités Pmax journalières d'un parc partiel ; composants résiduels/éoliens/solaires susceptibles de provenir de séries ou vintages différents, notamment NL. Un bilan obtenu en combinant ces sources ne doit pas être présenté comme une réserve physique vérifiée. Les métadonnées historiques API ne remplacent pas une preuve indépendante de capture opérationnelle à 08 h.

L'EVA conserve la référence hypothétique du day-ahead connu de la veille, la même politique, les mêmes coûts et le même périmètre que le KPI actuel. **Ce n'est pas un P&L exécutable démontré.**

## Reprise et intégrité

Chaque `Prepare` fige entrées, réseau dérivé, audit, configuration, versions et SHA du code de calcul dans un nouveau snapshot. Chaque pli sauvegarde ensemble les états des deux ablations. `Run` réutilise les checkpoints vérifiés et ne relance aucun Chronos-2.

Une modification de source, configuration, code de calcul ou environnement invalide la reprise : créer un nouveau snapshot, ne pas modifier les empreintes de l'ancien. Les rapports ont leurs propres manifestes liés au résultat exact du replay. `Report` permet d'améliorer la présentation sans refaire les fits tant que le contrat de calcul reste inchangé.

Le dernier rapport se trouve via `runs/experiments/nyx_physical_p50_v1/latest.json`, et la progression via le `status.json` du snapshot. Un simple `Status` vérifie les empreintes avant d'annoncer un résultat réutilisable.

## Premier replay terminé le 15 septembre 2026

Snapshot : `20260915T133116Z_095dc0d3`. Période évaluée : du 15/09/2025 au 14/09/2026, sur 34 940 heures-pays communes et 1 452 journées-pays complètes (363 par pays). Les références archivées NYX et Storm sont inchangées. La couverture réseau qualifiée atteint 27 728 / 35 136 lignes du panel, livraison non évaluée incluse ; les autres lignes conservent NYX pour les variantes réseau.

| Modèle | MAE €/MWh | RMSE €/MWh | Gain EVA simulée vs NYX |
|---|---:|---:|---:|
| NYX opérationnel | 11,3675 | 21,8755 | 0 € |
| Réseau + CGC gouverné — candidat principal | 11,3541 | 21,7762 | +504,25 € |
| Réseau + CGC direct | 11,3376 | 21,5837 | +34 132,13 € |
| CGC direct | 11,3440 | 21,5667 | +47 259,00 € |
| Storm | 11,3176 | 20,7876 | +388 617,19 € |

L'EVA reste le diagnostic hypothétique à politique et portefeuille inchangés, pas un profit négociable. Le candidat principal améliore modestement les moyennes annuelles : 55 heures modifiées, dont 36 améliorées et 19 dégradées. Il conserve exactement NYX en France sur le support annuel commun. La variante directe intervient davantage (434 heures, dont 180 dégradées) et détériore légèrement la MAE et la RMSE françaises. Aucun de ces candidats ne dépasse Storm en MAE/RMSE globales sur cette période.

À 19 h le 14/09, en €/MWh :

| Pays | NYX | Réseau + CGC direct | Réseau + CGC gouverné | Observé |
|---|---:|---:|---:|---:|
| DE | 324,59 | 419,01 | 371,80 | 697,31 |
| BE | 286,78 | 345,42 | 345,42 | 441,74 |
| FR | 284,41 | 286,04 | 284,41 | 298,01 |
| NL | 307,18 | 323,59 | 323,59 | 400,00 |

Le pic allemand reste largement sous-estimé. Ce résultat valide seulement une amélioration descriptive limitée, pas la résolution des spikes ni la promotion du modèle.

### Diagnostic physique et priorité suivante

En Allemagne, les prévisions archivées indiquaient entre 16 h et 19 h une hausse de charge résiduelle de 26,02 GW et une baisse du solaire de 22,87 GW, avec seulement 2,41 GW d'éolien à 19 h. Le CGC de 166,888 €/MWh dépassait d'environ 20 % le maximum CORE. Les disponibilités du parc sélectionné suggéraient une tension, sans constituer un bilan exhaustif de réserve nationale.

L'audit JAO après couplage identifie surtout Ensdorf–Vigy (VIGY2 sous contingence VIGY1) et PST Gronau. Leurs prix duaux et PTDF reconstituent 398,50 des 399,30 €/MWh de spread DE−FR, et 143,44 des 143,73 €/MWh de spread BE−FR. Cela rapproche environ 99,8 % de ces **écarts zonaux**, pas 99,8 % du prix absolu ni un contrefactuel causal. Ces publications après cutoff restent exclusivement dans le dossier explicatif.

Le domaine initial signalait déjà une tension directionnelle autour de Gronau et de la paire inverse Vigy ; ce n'est pas la preuve que le CNEC final exact était prévisible à 08 h. Voir les fichiers `forensics/2026-09-14/initial_signal_audit.md` et `*/spread_diagnostic.json` dans le laboratoire.

La prochaine hypothèse à tester séparément est un **détecteur de congestion et de sévérité horaire**, alimenté par les contraintes initiales individuelles, leurs PTDF directionnels et le stress physique prévu. Les activations/prix duaux des seules journées passées peuvent servir de labels d'entraînement ; ceux du jour prédit ne doivent jamais devenir des entrées. Cela doit permettre de réestimer la probabilité d'une forte hausse, actuellement figée, avant de recalculer le P50. Prévoir un jeu futur gelé et des comparaisons chronologiques avec/sans ce nouveau détecteur, sans règles zonales choisies après examen de cette année.
