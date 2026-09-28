<# Creates only NYX CPU Preview shortcuts for this checkout. #>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'

$PreviewRoot = $PSScriptRoot
$PreviewSettings = Get-Content -LiteralPath (Join-Path $PreviewRoot 'config\experiment_console.json') -Raw | ConvertFrom-Json
$PreviewPython = [string]$PreviewSettings.python_executable
if (-not [System.IO.Path]::IsPathRooted($PreviewPython) -or -not (Test-Path -LiteralPath $PreviewPython -PathType Leaf)) {
    throw "Python NYX introuvable : $PreviewPython"
}
$PreviewPythonW = Join-Path (Split-Path -Parent $PreviewPython) 'pythonw.exe'
$PreviewEntry = Join-Path $PreviewRoot 'NYX_CPU_Preview.pyw'
$PreviewIcon = Join-Path $PreviewRoot 'experiment_console\static\chronos.ico'
foreach ($PreviewRequired in @($PreviewPythonW, $PreviewEntry, $PreviewIcon)) {
    if (-not (Test-Path -LiteralPath $PreviewRequired -PathType Leaf)) {
        throw "Fichier requis introuvable : $PreviewRequired"
    }
}

$PreviewFolders = @(
    [Environment]::GetFolderPath('Desktop'),
    [Environment]::GetFolderPath('Programs'),
    $PreviewRoot
)
foreach ($PreviewFolder in $PreviewFolders) {
    if (-not [System.IO.Path]::IsPathRooted($PreviewFolder) -or -not (Test-Path -LiteralPath $PreviewFolder -PathType Container)) {
        throw "Dossier de raccourci introuvable : $PreviewFolder"
    }
}
$PreviewShell = New-Object -ComObject WScript.Shell
$PreviewDestinations = @($PreviewFolders | ForEach-Object { Join-Path $_ 'NYX CPU Preview.lnk' })

function Test-PreviewShortcutOwnership {
    param([string]$LinkPath)
    if (-not (Test-Path -LiteralPath $LinkPath -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $LinkPath -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) { return $false }
    $Previous = $PreviewShell.CreateShortcut($LinkPath)
    return [string]::Equals($Previous.TargetPath, $PreviewPythonW, [System.StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals($Previous.Arguments, ('"{0}"' -f $PreviewEntry), [System.StringComparison]::Ordinal) -and
        [string]::Equals($Previous.WorkingDirectory, $PreviewRoot, [System.StringComparison]::OrdinalIgnoreCase)
}

# Check every destination before writing any shortcut. NYX.lnk is never read or changed.
foreach ($PreviewDestination in $PreviewDestinations) {
    if ((Test-Path -LiteralPath $PreviewDestination) -and -not (Test-PreviewShortcutOwnership $PreviewDestination)) {
        throw "Un autre raccourci NYX CPU Preview existe deja : $PreviewDestination. Aucun remplacement effectue."
    }
}
foreach ($PreviewDestination in $PreviewDestinations) {
    $PreviewShortcut = $PreviewShell.CreateShortcut($PreviewDestination)
    $PreviewShortcut.TargetPath = $PreviewPythonW
    $PreviewShortcut.Arguments = '"{0}"' -f $PreviewEntry
    $PreviewShortcut.WorkingDirectory = $PreviewRoot
    $PreviewShortcut.IconLocation = "$PreviewIcon,0"
    $PreviewShortcut.Description = 'NYX CPU Preview - copie de test isolee'
    $PreviewShortcut.WindowStyle = 1
    $PreviewShortcut.Save()
    Write-Output "Raccourci cree : $PreviewDestination"
}
