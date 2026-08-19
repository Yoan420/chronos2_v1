# Moteur live multizone strict

`run_mkonline_live_model.py` est le moteur indépendant pour les bundles DE,
BE, NL et ES. Le runner France scellé n'est ni modifié ni réutilisé.

Un pays n'est exécutable qu'après validation complète de son
`ZoneModelContract` : identité zone/fuseaux/cible/primary/Storm, artefacts
scellés, empreintes SHA-256, recette, dépendance, poids et chemins de sortie.
Le registre doit alors pointer `runner` vers `run_mkonline_live_model.py`.

## Modes de prédiction

La recette et le YAML live doivent déclarer exactement le même mode. Un blend
MKOnline n'est permis qu'avec une dépendance auditée et deux poids strictement
positifs :

```yaml
live:
  prediction_mode: mkonline_blend
  mkonline_enabled: true
  weights:
    autonomous: 0.60
    mkonline_primary: 0.40
```

Si les gates rejettent MKOnline, le mode autonome doit être déclaré avant
l'activation du bundle, jamais découvert pendant le run :

```yaml
live:
  prediction_mode: autonomous_only
  mkonline_enabled: false
  primary_series: null
  dependency_manifest: null
  expected_hashes:
    dependency_manifest_sha256: null
  weights:
    autonomous: 1.0
    mkonline_primary: 0.0
```

La recette JSON répète `prediction_mode` (ou son alias scellé `recipe_mode`),
`mkonline_enabled`, `source_autonomous_run`, son empreinte et les poids. En
mode `autonomous_only`, elle fournit aussi un motif non vide dans
`autonomous_only_reason` ou `autonomous_validation.reason`.

Le registre répète obligatoirement :

```yaml
runner: run_mkonline_live_model.py
prediction_mode: autonomous_only
mkonline_enabled: false
primary_series: null
primary_status: not_used_autonomous_only
```

Les configurations opérationnelles utilisent les noms
`chronos2_hourly_{de,be,nl,es}_mkonline_live_v1.yaml`. Elles ne doivent être
publiées qu'après stabilisation des empreintes des runs autonomes, recettes et
benchmarks scellés.

## Commandes

Commande directe d'un bundle prêt :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_mkonline_live_model.py' `
  --config 'C:\Users\BQ6757\chronos2_v1\chronos2_hourly_de_mkonline_live_v1.yaml' `
  --registry 'C:\Users\BQ6757\chronos2_v1\chronos2_hourly_live_zones.yaml'
```

Le chemin normal reste le dispatcher, qui effectue d'abord le preflight local :

```powershell
& 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe' `
  'C:\Users\BQ6757\chronos2_v1\run_mkonline_live_zone.py' `
  --zone DE
```

Le jour physique est construit dans le fuseau de livraison (23/24/25 heures),
alors que le cutoff est construit civilement dans le fuseau d'origine déclaré.
Chaque archive est immuable et isolée sous l'`output_root` du pays.

Storm reste hors du chemin de prédiction. Le CSV candidat est écrit et son
SHA-256 est gelé avant tout chargement Storm. La série native officielle est
ensuite jointe uniquement à Statistics et au rapport. Pour ES, l'absence de
série native officielle est explicite et ne déclenche aucune substitution.
