# Candidat nucléaire CWE indépendant

Deux modèles expérimentaux, sans changement de `Forecast.ps1`, `Both`, `All` ou `Complete` :

- `nuclear_cwe_autonomous` : Chronos-2 + nucléaire CWE + correcteur résiduel.
- `nuclear_cwe_kalman` : même base + Kalman gouverné.

## Utilisation

Depuis n'importe quel dossier PowerShell :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\NuclearCWE.ps1' -Action Run -DeliveryDay 2026-09-11 -Zones FR
```

`Run` synchronise seulement les nouvelles sources isolées, prépare les entrées, effectue le replay et génère les deux rapports standard. Les calculs quotidiens sont conservés dans un cache propre au candidat : la même commande reprend les checkpoints valides après interruption. Deux lancements concurrents du laboratoire sont refusés.

```powershell
# Vérification des données et gel des entrées, sans modèle
& 'C:\Users\BQ6757\chronos2_v1\NuclearCWE.ps1' -Action Prepare -DeliveryDay 2026-09-11 -Zones FR

# État local, sans écriture
& 'C:\Users\BQ6757\chronos2_v1\NuclearCWE.ps1' -Action Status

# Refaire les HTML à partir du résultat gelé, sans réentraîner
& 'C:\Users\BQ6757\chronos2_v1\NuclearCWE.ps1' -Action Report -DeliveryDay 2026-09-11 -Zones FR
```

La configuration dédiée est `config/nuclear_cwe.yaml`. Sa livraison par défaut est volontairement fixée au rapport demandé, le 11/09/2026. Pour comparer une autre livraison, passer `-DeliveryDay` ; un résultat nucléaire opérationnel **déjà terminé pour cette même date et ce même pays** est nécessaire. `-Zones FR,DE,BE,NL` étend les pays prédits lorsque leurs comparateurs sont disponibles ; le premier test demandé ici porte sur la France. `-Threads`, `-Workers`, `-Device`, `-PythonExecutable`, `-SkipAttribution` et `-DryRun` sont disponibles.

Les rapports apparaissent exclusivement dans :

```text
runs/experiments/nuclear_cwe_v1/2026-09-11/fr/reports/
  forecast_fr_2026-09-11_nuclear_cwe_autonomous.html
  forecast_fr_2026-09-11_nuclear_cwe_kalman.html
  nuclear_cwe_incumbent_comparison.json
```

Le statut « complete » et le reçu `run_result.json` confirment la fin. L'existence d'un snapshot ou de checkpoints ne signifie pas que le backtest est fini.

## Variables effectivement utilisées

| Périmètre | Entrée | Signification |
|---|---|---|
| FR | `power.fr.generation.nuclear.gw.fcst` | Prévision horaire de production, GW, identique au modèle de référence |
| BE | `power.nrjscan.be.3mv.availability.pmax.type.nuclear.gw` | Prévision quotidienne de puissance maximale disponible, GW |
| NL | `power.nrjscan.nl.3mv.availability.pmax.type.nuclear.gw` | Prévision quotidienne de puissance maximale disponible, GW |
| DE, AT, LU | Aucun canal constant ajouté | Zéro structurel de production nucléaire sur le support retenu, pas une imputation des séries manquantes |

Les séries de **production prévue** BE/NL sondées n'ont pas fourni un historique complet à 08 h. Les deux nouvelles entrées sont donc des **capacités disponibles prévues**, pas des productions : elles décrivent la disponibilité de l'offre, sans garantir le dispatch réalisé. Les libellés des rapports et les audits conservent cette distinction. La couverture nationale exacte des agrégats fournisseur n'est pas certifiée ; un pays à capacité disponible nulle n'est pas traité comme une donnée absente.

Les Pmax quotidiennes sont répétées, sans interpolation, sur les 23, 24 ou 25 heures physiques du jour. La source française conserve sa politique DST auditée d'origine. Chaque nouveau canal est transmis au contexte et au futur de Chronos-2, aux features du correcteur et aux groupes de covariables du Kalman. Les autres entrées (prix passés, charges résiduelles régionales, calendrier) restent celles du comparateur.

Les pays à zéro structurel sont documentés par [le ministère allemand](https://www.bundeswirtschaftsministerium.de/Redaktion/DE/Pressemitteilungen/2023/04/20230413-deutschland-beendet-das-zeitalter-der-atomkraft.html), [le rapport autrichien à l'AIEA](https://www.iaea.org/sites/default/files/joint_convention_8th_national_report_of_austria_at1.pdf) et [le plan national luxembourgeois](https://mint.gouvernement.lu/dam-assets/publications/guide-manuel/PNOS-final.pdf). Le support allemand est borné après le 15/04/2023. Le périmètre CWE ne doit pas être confondu avec l'ensemble de la région Core : CZ/SK/HU/RO/SI ne sont pas ajoutés ici.

## Méthode de comparaison

1. Copier et vérifier les entrées et la recette figées du `nuclear_kalman` de référence. Aucun nouveau réglage choisi sur l'année évaluée.
2. Ajouter les deux nouvelles séries, en utilisant exclusivement l'état Saturn demandé à **J−1 08 h civile Paris**. La collecte BE/NL couvre ici 09/09/2024–11/09/2026, soit 733 jours et 17 592 heures par série. Les six nouvelles journées téléchargées complètent une copie intacte des sources existantes.
3. Recalculer Chronos-2 : ses anciennes prévisions FR-seul ne constituent pas des prévisions CWE. Les poids neuronaux sont inchangés (ce candidat n'ajoute pas LoRA).
4. Réentraîner le correcteur quotidiennement sur les 365 jours passés au maximum, avec le même démarrage à froid que le comparateur ; réserver les premiers 365 jours au préchauffage. Rejouer le Kalman avec sa calibration glissante de 365 jours et sa gouvernance inchangées. Aucun label du jour prédit ne sert à entraîner ce jour-là.
5. Conserver exactement l'ancre du 09/09/2024 pour ce comparateur. Les prix de contexte antérieurs à cette ancre conservent le support de covariables historique de l'ancien modèle, sans reconstruction rétrospective supplémentaire.
6. Pour les rapports, rattacher les **mêmes observations canoniques et le même snapshot Storm** que le rapport fourni. Ces données de scoring ne révisent aucune entrée ou prévision gelée.

Le backtest moteur couvre 11/09/2025–10/09/2026. Comme les observations de la livraison du 11 sont présentes dans le rapport de référence rafraîchi, ses **Statistics** couvrent les **365 derniers jours observés : 12/09/2025–11/09/2026**, soit 8760 heures. Le candidat utilise exactement cette même fenêtre. Le trou Storm DST reste exclu seulement des métriques appariées Storm, sans interpolation. Si une livraison n'est pas encore observée, elle reste vide et Statistics conserve sa fenêtre précédente.

Les HTML réutilisent le renderer standard : Statistics, prix moyens, calendrier, performances horaires contre Storm, composants du prix et mode nuit. Une section supplémentaire compare directement CWE + Kalman et nucléaire FR + Kalman sur les mêmes heures (MAE, RMSE, biais, erreur des prix moyens journaliers et diagnostic des 24–26 juin). L'attribution est recalculée pour le candidat ; un ancien artefact FR ne peut pas servir de preuve de ses influences. Sur le rapport Kalman, cette attribution décrit l'amont autonome, pas une décomposition causale du filtre.

## Limites et protection

- « As-of » décrit l'interrogation Saturn à 08 h. Le timestamp original de publication fournisseur n'est pas retourné par cette interface : le test n'est pas une certification PIT de production.
- Pmax BE/NL n'est pas une production prévue ni réalisée. Pas de substitution par les observations pour combler des trous.
- Un gain annuel n'est pas acquis ; ce test est une ablation descriptive préspécifiée. Les journées de juin déjà connues ne sont pas un jeu de validation vierge.
- Les sources, snapshots, checkpoints et rapports du candidat sont séparés ; aucun modèle opérationnel n'est remplacé ou promu.
- Changer la recette ou le code de l'adaptateur impose un nouvel `output_root` enfant de `runs/experiments/nuclear_cwe_v1`. Les snapshots incompatibles sont refusés, jamais silencieusement écrasés.
- Le premier run complet est coûteux. `Report` n'exécute aucun modèle et ne synchronise pas les données : il conserve la comparaison gelée. Un échec d'attribution est signalé, sans inventer de poids et sans invalider la prévision déjà sauvegardée.
