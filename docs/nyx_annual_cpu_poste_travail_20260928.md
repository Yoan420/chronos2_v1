# NYX annuel CPU sur le poste de travail — étapes du 28 septembre 2026

## Consulter les résultats sans remplacer l'installation actuelle

1. Ouvrir PowerShell dans le dossier où créer une **seconde** installation
   NYX, puis cloner la branche d'essai :

   ```powershell
   git clone --branch codex/nyx-regional-rmse-production --single-branch https://github.com/Yoan420/chronos2_v1.git NYX_CWE_CPU
   cd NYX_CWE_CPU
   ```

2. Vérifier que le Python indiqué dans `config/experiment_console.json` existe
   sur le poste. Cette branche utilise le port local **8766** ; l'installation
   NYX actuelle garde son port **8765**. Le chemin Python configuré est celui
   de l'installation professionnelle actuelle
   (`C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe`). Si ce chemin
   est différent sur le poste, corriger uniquement la valeur
   `python_executable` dans cette seconde installation ; le même dossier doit
   contenir `pythonw.exe`.

3. Ouvrir `NYX.pyw` dans `NYX_CWE_CPU`, puis **Modèles régionaux → Modèles
   annuels FR / BE / NL**. Choisir le pays. Le panneau affiche séparément les
   scores CPU du replay et les scores GPU des rapports historiques, ainsi que
   la disponibilité des archives sur ce poste.

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
