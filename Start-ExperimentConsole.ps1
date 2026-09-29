<#
.SYNOPSIS
Demarre ou retrouve la console locale persistante sans lancer de calcul.

.DESCRIPTION
Reutilise une instance du meme depot, du meme dossier d'etat et du meme Python.
-Restart recharge le serveur apres une mise a jour, uniquement sans calcul actif
ni en attente. Aucun processus de calcul ni fichier de verrou n'est supprime.
#>
[CmdletBinding()]
param(
    [string]$Settings = '',
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [switch]$NoOpen,
    [switch]$Restart
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Settings)) {
    $Settings = Join-Path $PSScriptRoot 'config\experiment_console.json'
}
$ResolvedSettings = (Resolve-Path -LiteralPath $Settings).Path
$ConsoleSettings = Get-Content -LiteralPath $ResolvedSettings -Raw | ConvertFrom-Json
$ConsolePython = [string]$ConsoleSettings.python_executable
if (-not [System.IO.Path]::IsPathRooted($ConsolePython) -or -not (Test-Path -LiteralPath $ConsolePython -PathType Leaf)) {
    throw "Renseignez un chemin Python absolu existant dans $ResolvedSettings."
}
if (-not $PSBoundParameters.ContainsKey('Port') -and $ConsoleSettings.port) {
    $Port = [int]$ConsoleSettings.port
}
$ConsoleArguments = @('-m', 'experiment_console.launcher', '--settings', $ResolvedSettings, '--port', [string]$Port)
if (-not $NoOpen) { $ConsoleArguments += '--open' }
if ($Restart) { $ConsoleArguments += '--restart' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $ConsolePython @ConsoleArguments
    if ($LASTEXITCODE -ne 0) { throw "La console s'est arretee avec le code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
