# NYX annuel CPU sur le poste de travail — étapes du 28 septembre 2026

## Consulter les résultats sans remplacer l'installation actuelle

1. Ouvrir PowerShell dans le dossier où créer une **seconde** installation
   NYX, puis cloner la branche d'essai :

   ```powershell
   git clone --branch codex/nyx-regional-rmse-production --single-branch https://github.com/Yoan420/chronos2_v1.git NYX_CWE_CPU
   cd NYX_CWE_CPU
   ```

2. Dans **le même PowerShell**, après `cd NYX_CWE_CPU`, copier ce bloc. Il
   vérifie le Python déjà indiqué dans la configuration et le port séparé
   **8766**. Le message attendu est `Python NYX OK` :

   ```powershell
   $config = Get-Content .\config\experiment_console.json -Raw | ConvertFrom-Json
   $python = $config.python_executable
   if ($config.port -ne 8766) { throw 'Branche NYX_CWE_CPU attendue (port 8766).' }
   if (-not (Test-Path -LiteralPath $python)) { throw "Python NYX introuvable : $python" }
   $pythonw = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
   if (-not (Test-Path -LiteralPath $pythonw)) { throw "pythonw.exe introuvable : $pythonw" }
   & $python -c "import filelock, yaml; print('Python NYX OK')"
   ```

   Si le chemin configuré n'existe pas sur ce poste, il faut trouver le
   `python.exe` de l'installation NYX actuelle ; un Python quelconque ne
   garantit pas les dépendances nécessaires.

3. Toujours dans ce PowerShell, lancer :

   ```powershell
   & $python .\NYX.pyw
   ```

   Dans la fenêtre NYX, ouvrir **Modèles régionaux → Modèles annuels FR / BE /
   NL**, puis choisir le pays. Le panneau affiche séparément les scores CPU
   du replay et les scores GPU des rapports historiques, ainsi que la
   disponibilité des archives sur ce poste.

4. Lire [le rapport CPU](nyx_annual_cpu_evaluation_20260928.md) pour les
   chiffres exacts : FR, BE et NL passent les deux critères contre Storm sur
   l'année rétrospective. Les probabilités négatives y figurent également.

## Prévisions futures

Le modèle annuel n'est **pas activé** dans NYX. Le reçu scellé qualifie les
experts prix CPU et les probabilités **sur les entrées historiques archivées** ;
il porte `full_input_chain_qualified: false` et `qualified: false`. Le panneau
de bureau est en lecture seule et le précontrôle du consommateur retourne
`ready: false` sur un clone neuf. Il manque notamment la production causale
des baselines NYX, des 12 matrices et des trois références, l'orchestration
automatique de toutes les sources et des archives antérieures vérifiables.
Le client Saturn privé `tshistory_lite` doit aussi être disponible dans
l'environnement du poste et l'accès intranet vérifié.

Il n'y a donc actuellement **aucune étape de lancement de prévisions futures
CWE** à effectuer sur l'ordinateur professionnel. Le modèle en production
actuel peut continuer à fonctionner dans l'installation d'origine. La branche
d'essai a un port et un dossier distincts pour préserver cette installation.
