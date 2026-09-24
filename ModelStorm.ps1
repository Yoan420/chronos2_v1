<#
.SYNOPSIS
Assemble et ouvre un seul rapport HTML Model / Storm pour BE, DE, FR et NL.

.DESCRIPTION
Lit les previsions et comparateurs de prix locaux disponibles. Model correspond
a nuclear_kalman. Rafraichit uniquement la metrique de reporting VPS de Saturn;
aucune prevision n'est lancee. -SkipVpsSync desactive cette collecte, sans cache
VPS de remplacement. Le HTML genere reste autonome et consultable hors ligne.
Les donnees manquantes restent indiquees comme indisponibles dans le rapport.
La livraison par defaut est demain, selon le fuseau Europe/Paris.
Les chemins relatifs sont resolus depuis le dossier de ce lanceur.
DryRun affiche les arguments exacts sans lancer Python ni creer de fichiers.

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\ModelStorm.ps1'

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\ModelStorm.ps1' -DeliveryDay 2026-09-10 -NoOpen
#>
[CmdletBinding()]
param(
    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',
    [string]$OutputPath = '',
    [string]$NuclearRoot = '',
    [string]$PythonExecutable = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe',
    [switch]$SkipVpsSync,
    [switch]$NoOpen,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

try {
    if ([string]::IsNullOrWhiteSpace($DeliveryDay)) {
        try {
            $ParisZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('Romance Standard Time')
        }
        catch {
            $ParisZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('Europe/Paris')
        }
        $ParisNow = [System.TimeZoneInfo]::ConvertTimeFromUtc([datetime]::UtcNow, $ParisZone)
        $DeliveryDay = $ParisNow.Date.AddDays(1).ToString('yyyy-MM-dd', [cultureinfo]::InvariantCulture)
    }
    else {
        $null = [datetime]::ParseExact($DeliveryDay, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture)
    }

    if ([string]::IsNullOrWhiteSpace($OutputPath)) {
        $OutputPath = "runs\reports\model_storm\CWE_Model_Storm_$DeliveryDay.html"
    }
    $OutputPath = Resolve-ProjectPath -Path $OutputPath
    if ([System.IO.Path]::GetExtension($OutputPath) -ine '.html') {
        throw 'Le rapport de sortie doit porter l''extension .html.'
    }

    # Prefer the same environment as Forecast.ps1. A caller-provided Python
    # path is authoritative; fallback applies only when the default is absent.
    if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) {
        $PythonExecutable = Resolve-ProjectPath -Path $PythonExecutable
    }
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        if ($PSBoundParameters.ContainsKey('PythonExecutable')) {
            throw "Python introuvable : $PythonExecutable"
        }
        $PythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $PythonCommand) {
            throw 'Python introuvable. Indiquez -PythonExecutable.'
        }
        $PythonExecutable = $PythonCommand.Source
    }
    $PythonExecutable = [System.IO.Path]::GetFullPath($PythonExecutable)
    $ReportScript = Join-Path $ProjectRoot 'run_model_storm_report.py'
    if (-not (Test-Path -LiteralPath $ReportScript -PathType Leaf)) {
        throw "Script de rapport introuvable : $ReportScript"
    }
    $Arguments = @($ReportScript, '--delivery-day', $DeliveryDay, '--output', $OutputPath)
    if (-not [string]::IsNullOrWhiteSpace($NuclearRoot)) {
        $Arguments += @('--nuclear-root', (Resolve-ProjectPath -Path $NuclearRoot))
    }
    if ($SkipVpsSync) {
        $Arguments += '--skip-vps-sync'
    }
    $DisplayCommand = @($PythonExecutable) + $Arguments
    $DisplayJson = ConvertTo-Json -InputObject $DisplayCommand -Compress
    Write-Host "Commande (argv, shell=False): $DisplayJson"
    if ($DryRun) {
        Write-Host 'DryRun : aucune commande executee.'
        exit 0
    }

    & $PythonExecutable @Arguments
    $ProcessExitCode = $LASTEXITCODE
    if ($null -eq $ProcessExitCode) {
        $ProcessExitCode = 1
    }
    if ($ProcessExitCode -ne 0) {
        exit ([int]$ProcessExitCode)
    }
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw "Rapport attendu introuvable : $OutputPath"
    }
    if (-not $NoOpen) {
        Invoke-Item -LiteralPath $OutputPath
    }
    exit 0
}
catch {
    [Console]::Error.WriteLine("[Model / Storm] $($_.Exception.Message)")
    exit 2
}
