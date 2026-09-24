# SolarCWE : quatre prévisions solaires dans NYX

Variante expérimentale indépendante. `Forecast.ps1`, ses modes et ses modèles
opérationnels ne changent pas. Les résultats restent sous
`runs/experiments/solar_cwe_v1` ; aucune promotion automatique.

## Ce qui change, et seulement cela

On conserve la recette nucléaire française : Chronos-2, correcteur résiduel et
Kalman, leurs paramètres et les cinq charges résiduelles déjà utilisées. On ajoute
les quatre séries Saturn `power.{fr,de,be,nl}.generation.solar.hourly.gw.fcst`, en
GW, dans le contexte et les covariables futures de Chronos-2, dans le correcteur
résiduel et dans les covariables du Kalman. Pas de détecteur, de nouvelle rampe
calculée ou de règle manuelle de correction des pics. Les transformations
usuelles des moteurs existants restent inchangées. On ne soustrait pas une
seconde fois le solaire de la charge résiduelle.

Deux rapports par pays : `solar_autonomous` et `solar_kalman`. Chacun compare la
variante à sa référence nucléaire de même famille, et à Storm dans les sections
habituelles. Il s'agit de nouveaux replays : changer les entrées de Chronos-2
exige de recalculer ses prévisions historiques. Ce n'est pas un entraînement LoRA.

## Utilisation

Dans PowerShell, depuis n'importe quel dossier :

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarCWE.ps1' -Action Run -Zones FR,DE,BE,NL
```

La date par défaut est le **19 septembre 2026**, dernière référence nucléaire
complète commune retenue pour cette expérience. `-DeliveryDay YYYY-MM-DD`
sélectionne une autre date uniquement si ses références nucléaires figées sont
déjà disponibles. Paramètres simples : `-Device auto`, `-Threads 4`, `-Workers 2`.

Actions utiles :

- `Audit` : vérifier les références, sans réseau ni entraînement.
- `Sync` : copier les caches solaires vérifiés dans un espace séparé et compléter
  seulement les jours manquants. Aucune interpolation de jours absents.
- `Prepare` : figer les entrées et préparer les quatre jeux de données, sans modèle.
- `Run` : préparer, reprendre les checkpoints compatibles, recalculer et produire
  les HTML. Relancer la même commande après une interruption.
- `Status` : consulter l'étape et les dossiers sélectionnés, sans lancer de calcul.
- `Report` : reconstruire les HTML d'un calcul terminé, sans réseau ni entraînement.

```powershell
& 'C:\Users\BQ6757\chronos2_v1\SolarCWE.ps1' -Action Status
& 'C:\Users\BQ6757\chronos2_v1\SolarCWE.ps1' -Action Report
```

Les premières exécutions sont longues : un historique de calibration puis
**365 jours évalués** sont recalculés avec les nouvelles entrées. L'ancre historique
est identique à celle de la référence nucléaire. Les fenêtres résiduelle/Kalman,
les observations et Storm sont conservés pour une comparaison appariée. Le
backtest annuel exclut la journée de livraison ; Statistics suit la convention
existante des 365 derniers jours observés. Une observation non publiée reste vide.

## Précautions méthodologiques

Les quatre prévisions sont sélectionnées par requête Saturn **as-of J−1 à 08 h,
heure civile du pays**. Les fichiers sont copiés, hachés et figés. On conserve la
convention locale CET/Amsterdam documentée pour NL. Les conventions DST sont
auditées ; le second pli automnal NL n'est reconstruit que si sa valeur nocturne
est vérifiée nulle. Les autres pays héritent de la convention de duplication du
pli manquant des archives civiles, tracée dans leurs audits.

Saturn ne renvoie pas ici l'horodatage d'insertion d'origine : une requête as-of
ne certifie donc pas à elle seule la publication contemporaine. Cette limite est
conservée dans les audits ; ces résultats restent diagnostiques. La référence
initiale inclut également son propre démarrage de calibration : aucun historique
supplémentaire ou résultat hors-échantillon n'est inventé.

Les attributions par perturbation sont désactivées par défaut pour limiter le
coût. `include_attribution: true` dans `config/solar_cwe.yaml` les active sur les
nouveaux calculs ; elles expliquent le modèle amont, pas une contribution causale
ni l'effet total du Kalman. Aucun pourcentage de poids n'est fabriqué en leur absence.
