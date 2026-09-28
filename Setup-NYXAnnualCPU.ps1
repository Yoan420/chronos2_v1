<# Installe une console NYX annuelle CPU dans un environnement Python separe. #>
[CmdletBinding()]
param(
    [string]$BasePython = '',
    [string]$SaturnPython = '',
    [string]$SaturnWheel = '',
    [string]$SaturnIndexUrl = '',
    [string]$SaturnRequirement = 'tshistory_lite==0.5',
    [switch]$SkipModelDownload,
    [switch]$NoDesktopShortcut
)
$ErrorActionPreference = 'Stop'
$NyxRoot = $PSScriptRoot
$NyxEnvironment = Join-Path $NyxRoot '.venv-annual'
$NyxPython = Join-Path $NyxEnvironment 'Scripts\python.exe'
$NyxSaturnHelper = Join-Path $NyxRoot 'prepare_nyx_saturn.py'
if ($SaturnPython -and $SaturnWheel) { throw 'Choisir -SaturnPython ou -SaturnWheel.' }

function Get-NyxSaturnProbe([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $null }
    try {
        $NyxProbeText = & $Candidate -B $NyxSaturnHelper probe --requirement $SaturnRequirement 2>$null
        if ($LASTEXITCODE -eq 0) { return ($NyxProbeText | ConvertFrom-Json) }
    } catch { return $null }
    return $null
}

# Existing work environment is inspected read-only; its packages are never changed.
if (-not $SaturnPython -and -not $SaturnWheel -and -not $SaturnIndexUrl) {
    $NyxCandidates = @()
    $NyxExistingConfig = Join-Path $NyxRoot 'config\experiment_console.json'
    if (Test-Path -LiteralPath $NyxExistingConfig -PathType Leaf) {
        try { $NyxCandidates += (Get-Content -LiteralPath $NyxExistingConfig -Raw | ConvertFrom-Json).python_executable } catch { }
    }
    $NyxCandidates += (Join-Path $env:USERPROFILE 'venvs\pricefm311\Scripts\python.exe')
    $NyxCandidates += $BasePython
    $NyxCandidates += (Join-Path $NyxRoot '.venv\Scripts\python.exe')
    $NyxCandidates += (Get-Command python -ErrorAction SilentlyContinue).Source
    foreach ($NyxCandidate in ($NyxCandidates | Where-Object { $_ } | Select-Object -Unique)) {
        if ($NyxCandidate -eq $NyxPython) { continue }
        $NyxProbe = Get-NyxSaturnProbe $NyxCandidate
        if ($NyxProbe -and $NyxProbe.ok) { $SaturnPython = $NyxCandidate; break }
    }
}
if ($SaturnPython) {
    $NyxSourceProbe = Get-NyxSaturnProbe $SaturnPython
    if (-not $NyxSourceProbe -or -not $NyxSourceProbe.ok) {
        throw 'Le Python Saturn source doit etre Python 3.11 avec tshistory_lite et Client.get/block_staircase fonctionnels. Sinon fournir -SaturnWheel ou -SaturnIndexUrl.'
    }
    if (-not $BasePython) { $BasePython = $SaturnPython }
}
if (-not (Test-Path -LiteralPath $NyxPython -PathType Leaf)) {
    if ($BasePython) {
        & $BasePython -B -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 requis'"
        if ($LASTEXITCODE -ne 0) { throw 'BasePython doit etre Python 3.11.' }
    }
    if ($BasePython) { & $BasePython -m venv $NyxEnvironment }
    else { & py -3.11 -m venv $NyxEnvironment }
    if ($LASTEXITCODE -ne 0) { throw 'Creation Python 3.11 impossible. Fournir -BasePython avec un interpreteur Python 3.11.' }
}
& $NyxPython -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 requis'"
if ($LASTEXITCODE -ne 0) { throw 'Version Python incorrecte.' }
& $NyxPython -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Installation PyTorch CPU echouee.' }
& $NyxPython -m pip install -r (Join-Path $NyxRoot 'requirements_nyx_annual_cpu.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installation des dependances annuelles echouee.' }
$NyxTargetProbe = Get-NyxSaturnProbe $NyxPython
if (-not $NyxTargetProbe -or -not $NyxTargetProbe.ok) {
    $NyxConstraints = Join-Path $NyxEnvironment 'saturn-public-constraints.txt'
    $NyxRuntime = Join-Path $NyxEnvironment 'saturn-target-runtime.json'
    & $NyxPython -B $NyxSaturnHelper constraints --output $NyxConstraints
    if ($LASTEXITCODE -ne 0) { throw 'Capture des versions publiques echouee.' }
    & $NyxPython -B $NyxSaturnHelper runtime --output $NyxRuntime
    if ($LASTEXITCODE -ne 0) { throw 'Identification du Python cible echouee.' }
    if ($SaturnPython) {
        # A fresh directory excludes stale wheels from earlier installations.
        $NyxWheelhouse = Join-Path $NyxEnvironment ('saturn-wheelhouse\' + [guid]::NewGuid().ToString('N'))
        & $SaturnPython -B $NyxSaturnHelper export --requirement $SaturnRequirement --output $NyxWheelhouse --target-runtime $NyxRuntime
        if ($LASTEXITCODE -ne 0) { throw 'Transfert local Saturn echoue. Utiliser un wheel Python 3.11 autorise ou un index interne.' }
        & $NyxPython -m pip install --no-index --find-links $NyxWheelhouse --constraint $NyxConstraints $SaturnRequirement
    } elseif ($SaturnWheel) {
        $NyxResolvedWheel = (Resolve-Path -LiteralPath $SaturnWheel).Path
        & $NyxPython -B $NyxSaturnHelper check-wheel --wheel $NyxResolvedWheel
        if ($LASTEXITCODE -ne 0) { throw 'Wheel Saturn incompatible avec Python 3.11 sur ce poste.' }
        $NyxInstallArguments = @('-m', 'pip', 'install', '--constraint', $NyxConstraints, '--find-links', (Split-Path -Parent $NyxResolvedWheel))
        if ($SaturnIndexUrl) { $NyxInstallArguments += @('--index-url', $SaturnIndexUrl) }
        $NyxInstallArguments += $NyxResolvedWheel
        & $NyxPython @NyxInstallArguments
    } elseif ($SaturnIndexUrl) {
        & $NyxPython -m pip install --constraint $NyxConstraints --index-url $SaturnIndexUrl $SaturnRequirement
    } else {
        throw 'Bibliotheques publiques installees. Client Saturn prive introuvable. Relancer: .\Setup-NYXAnnualCPU.ps1 -SaturnPython "$env:USERPROFILE\venvs\pricefm311\Scripts\python.exe" ; ou -SaturnWheel "C:\chemin\tshistory_lite-0.5-py3-none-any.whl" ; ou -SaturnIndexUrl "URL_INDEX_INTERNE_AUTORISE".'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Installation Saturn echouee; les versions publiques CPU sont protegees par contraintes.' }
}
$NyxFinalProbe = Get-NyxSaturnProbe $NyxPython
if (-not $NyxFinalProbe -or -not $NyxFinalProbe.ok) { throw 'Validation hors reseau de Client.get/block_staircase Saturn echouee.' }
& $NyxPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Conflit de dependances dans .venv-annual. La console ne sera pas configuree.' }
Write-Host 'Client Saturn installe et contrat get/block_staircase verifie hors reseau.'
if (-not $SkipModelDownload) {
    & $NyxPython -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-2', revision='29ec3766d36d6f73f0696f85560a422f50e8498c')"
    if ($LASTEXITCODE -ne 0) { throw 'Telechargement du snapshot Chronos echoue. Reexecuter sur un reseau autorise.' }
}
$NyxSettings = Join-Path $NyxRoot 'runs\annual_desktop.json'
$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $NyxSettings)
$NyxConfig = @{ python_executable = $NyxPython; state_root = 'runs/.experiment_console_annual'; max_concurrency = 1; port = 8767 }
[System.IO.File]::WriteAllText($NyxSettings, ($NyxConfig | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
if (-not $NoDesktopShortcut) {
    $NyxDesktop = [Environment]::GetFolderPath('Desktop')
    $NyxShell = New-Object -ComObject WScript.Shell
    $NyxShortcut = $NyxShell.CreateShortcut((Join-Path $NyxDesktop 'NYX annuel CPU.lnk'))
    $NyxShortcut.TargetPath = Join-Path $NyxEnvironment 'Scripts\pythonw.exe'
    $NyxShortcut.Arguments = '"' + (Join-Path $NyxRoot 'NYX.pyw') + '" --settings "' + $NyxSettings + '"'
    $NyxShortcut.WorkingDirectory = $NyxRoot
    $NyxShortcut.Save()
}
Write-Host 'Installation terminee. Ouvrir NYX annuel CPU, puis Modeles regionaux > Modeles annuels CPU.'
Write-Host 'La preparation et les captures ne constituent pas une qualification de production.'
