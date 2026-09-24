<# Creates per-user NYX shortcuts; no administrator privileges or Python file associations. #>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$NyxRoot = $PSScriptRoot
$NyxSettings = Get-Content -LiteralPath (Join-Path $NyxRoot 'config\experiment_console.json') -Raw | ConvertFrom-Json
$NyxPython = [string]$NyxSettings.python_executable
$NyxPythonW = Join-Path (Split-Path -Parent $NyxPython) 'pythonw.exe'
$NyxEntry = Join-Path $NyxRoot 'NYX.pyw'
$NyxLegacyEntry = Join-Path $NyxRoot 'Chronos.pyw'
$NyxIcon = Join-Path $NyxRoot 'experiment_console\static\chronos.ico'
foreach ($NyxRequired in @($NyxPythonW, $NyxEntry, $NyxIcon)) {
    if (-not [System.IO.Path]::IsPathRooted($NyxRequired) -or -not (Test-Path -LiteralPath $NyxRequired -PathType Leaf)) {
        throw "Fichier requis introuvable : $NyxRequired"
    }
}
$NyxShell = New-Object -ComObject WScript.Shell
$NyxFolders = @(
    [Environment]::GetFolderPath('Desktop'),
    [Environment]::GetFolderPath('Programs'),
    $NyxRoot
)
foreach ($NyxFolder in $NyxFolders) {
    if (-not [System.IO.Path]::IsPathRooted($NyxFolder) -or -not (Test-Path -LiteralPath $NyxFolder -PathType Container)) {
        throw "Dossier de raccourci introuvable : $NyxFolder"
    }
}

function Test-NyxShortcutOwnership {
    param([string]$LinkPath, [string]$ExpectedEntry)
    if (-not (Test-Path -LiteralPath $LinkPath -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $LinkPath -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) { return $false }
    $NyxPrevious = $NyxShell.CreateShortcut($LinkPath)
    return [string]::Equals($NyxPrevious.TargetPath, $NyxPythonW, [System.StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals($NyxPrevious.Arguments, ('"{0}"' -f $ExpectedEntry), [System.StringComparison]::Ordinal)
}

# Check every new destination before changing any shortcut. An unrelated NYX
# shortcut must never be overwritten, even if its name happens to match.
$NyxDestinations = @($NyxFolders | ForEach-Object { Join-Path $_ 'NYX.lnk' })
foreach ($NyxDestination in $NyxDestinations) {
    if ((Test-Path -LiteralPath $NyxDestination) -and -not (Test-NyxShortcutOwnership $NyxDestination $NyxEntry)) {
        throw "Un autre raccourci NYX existe deja : $NyxDestination. Aucun remplacement effectue."
    }
}
foreach ($NyxDestination in $NyxDestinations) {
    $NyxShortcut = $NyxShell.CreateShortcut($NyxDestination)
    $NyxShortcut.TargetPath = $NyxPythonW
    $NyxShortcut.Arguments = '"{0}"' -f $NyxEntry
    $NyxShortcut.WorkingDirectory = $NyxRoot
    $NyxShortcut.IconLocation = "$NyxIcon,0"
    $NyxShortcut.Description = 'NYX - Architecture du modele, previsions et resultats'
    $NyxShortcut.WindowStyle = 1
    $NyxShortcut.Save()
    Write-Output "Raccourci cree : $NyxDestination"
}

# Retire only a previous shortcut whose target AND exact arguments still match
# our original installation, after all three NYX shortcuts have been saved.
foreach ($NyxFolder in $NyxFolders) {
    $NyxLegacyLink = Join-Path $NyxFolder 'Chronos.lnk'
    if (Test-NyxShortcutOwnership $NyxLegacyLink $NyxLegacyEntry) {
        Remove-Item -LiteralPath $NyxLegacyLink
        Write-Output "Ancien raccourci remplace : $NyxLegacyLink"
    } elseif (Test-Path -LiteralPath $NyxLegacyLink) {
        Write-Output "Raccourci Chronos non reconnu, conserve : $NyxLegacyLink"
    }
}
