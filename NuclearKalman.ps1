<#
.SYNOPSIS
Calcule nuclear_kalman, ses rapports par pays et un HTML CWE Model / Storm.

.DESCRIPTION
Reutilise les caches journaliers et les resultats figes. Synchronise les sources
communes une seule fois, puis execute chaque pays successivement. Aucun pipeline
ordinaire ni blend MKOnline n'est lance. Les calculs d'attribution sont facultatifs.
Les journaux et le statut du batch restent dans runs/logs/nuclear_kalman.
Le rapport groupe utilise les resultats disponibles pour BE, DE, FR et NL.
Il actualise la metrique VPS Saturn; -SkipObservedSync desactive aussi cette
collecte de reporting, sans reprendre un ancien cache VPS.
Un echec pays reste un echec du batch; les autres pays peuvent terminer.

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\NuclearKalman.ps1'

.EXAMPLE
& 'C:\Users\BQ6757\chronos2_v1\NuclearKalman.ps1' -Countries DE -DeliveryDay 2026-09-11 -NoOpen
#>
[CmdletBinding()]
param(
    [ValidateSet('BE', 'DE', 'FR', 'NL')]
    [string[]]$Countries = @('BE', 'DE', 'FR', 'NL'),
    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [ValidateRange(1, 128)]
    [int]$Threads = 4,
    [ValidateRange(1, 128)]
    [int]$Workers = 4,
    [string]$NuclearConfig = 'config\nuclear_forecast.yaml',
    [string]$OutputPath = '',
    [string]$PythonExecutable = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe',
    [switch]$WithAttribution,
    [switch]$SkipObservedSync,
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
        try { $ParisZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('Romance Standard Time') }
        catch { $ParisZone = [System.TimeZoneInfo]::FindSystemTimeZoneById('Europe/Paris') }
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
    $NuclearConfig = Resolve-ProjectPath -Path $NuclearConfig
    if (-not [System.IO.Path]::IsPathRooted($PythonExecutable)) {
        $PythonExecutable = Resolve-ProjectPath -Path $PythonExecutable
    }
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        if ($PSBoundParameters.ContainsKey('PythonExecutable')) {
            throw "Python introuvable : $PythonExecutable"
        }
        $PythonCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $PythonCommand) { throw 'Python introuvable. Indiquez -PythonExecutable.' }
        $PythonExecutable = $PythonCommand.Source
    }
    $PythonExecutable = [System.IO.Path]::GetFullPath($PythonExecutable)
    $RunScript = Join-Path $ProjectRoot 'run_nuclear_kalman.py'
    if (-not (Test-Path -LiteralPath $RunScript -PathType Leaf)) {
        throw "Script introuvable : $RunScript"
    }
    # PowerShell ValidateSet accepts mixed case; Python choices are case-sensitive.
    $Countries = @($Countries | ForEach-Object { $_.ToUpperInvariant() } | Select-Object -Unique)
    $Device = $Device.ToLowerInvariant()
    $Arguments = @($RunScript, '--zones') + @($Countries | Select-Object -Unique) + @(
        '--delivery-day', $DeliveryDay, '--device', $Device,
        '--threads', [string]$Threads, '--workers', [string]$Workers,
        '--nuclear-config', $NuclearConfig, '--output', $OutputPath
    )
    if ($WithAttribution) { $Arguments += '--with-attribution' }
    if ($SkipObservedSync) { $Arguments += '--skip-observed-sync' }
    if ($NoOpen) { $Arguments += '--no-open' }
    $DisplayJson = ConvertTo-Json -InputObject (@($PythonExecutable) + $Arguments) -Compress
    Write-Host "Commande (argv, shell=False): $DisplayJson"
    if ($DryRun) {
        Write-Host 'DryRun : aucune commande executee.'
        exit 0
    }
    & $PythonExecutable @Arguments
    $ProcessExitCode = $LASTEXITCODE
    if ($null -eq $ProcessExitCode) { $ProcessExitCode = 1 }
    exit ([int]$ProcessExitCode)
}
catch {
    [Console]::Error.WriteLine("[Nuclear Kalman] $($_.Exception.Message)")
    exit 2
}
