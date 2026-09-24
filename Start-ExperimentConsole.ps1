<#
.SYNOPSIS
Demarre la console locale persistante sans lancer de calcul.
#>
[CmdletBinding()]
param(
    [string]$Settings = (Join-Path $PSScriptRoot 'config\experiment_console.json'),
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [switch]$NoOpen
)
$ErrorActionPreference = 'Stop'
$ResolvedSettings = (Resolve-Path -LiteralPath $Settings).Path
$ConsoleSettings = Get-Content -LiteralPath $ResolvedSettings -Raw | ConvertFrom-Json
$ConsolePython = [string]$ConsoleSettings.python_executable
if (-not [System.IO.Path]::IsPathRooted($ConsolePython) -or -not (Test-Path -LiteralPath $ConsolePython -PathType Leaf)) {
    throw "Renseignez un chemin Python absolu existant dans $ResolvedSettings."
}
if (-not $PSBoundParameters.ContainsKey('Port') -and $ConsoleSettings.port) {
    $Port = [int]$ConsoleSettings.port
}
$ConsoleArguments = @('-m', 'experiment_console.server', '--settings', $ResolvedSettings, '--port', [string]$Port)
if (-not $NoOpen) { $ConsoleArguments += '--open' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $ConsolePython @ConsoleArguments
    if ($LASTEXITCODE -ne 0) { throw "La console s'est arretee avec le code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
