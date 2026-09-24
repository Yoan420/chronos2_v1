<# Isolated research only. No Forecast.ps1 activation and no paid API calls. #>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','Run','Report','Status','Validate','Demo','DryRun')][string]$Action = 'Run',
    [string]$Config,
    [string]$RunDirectory,
    [string]$PythonExecutable
)
$ErrorActionPreference = 'Stop'
$DemandRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Config) { $Config = Join-Path $DemandRoot 'config\nyx_demand_response.yaml' }
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $DemandRoot $Config }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Configuration introuvable : $Config" }
if ($RunDirectory -and $Action -notin @('Run','Report','Status')) { throw 'RunDirectory s applique uniquement a Run/Report/Status.' }
if (-not $PythonExecutable) {
    $DemandPython = Join-Path (Split-Path -Parent $DemandRoot) 'venvs\pricefm311\Scripts\python.exe'
    if (Test-Path -LiteralPath $DemandPython -PathType Leaf) { $PythonExecutable = $DemandPython }
    else {
        $DemandCommand = Get-Command python.exe, python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $DemandCommand) { throw 'Python introuvable. Precisez -PythonExecutable.' }
        $PythonExecutable = $DemandCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python introuvable : $PythonExecutable" }
$DemandArguments = @((Join-Path $DemandRoot 'run_nyx_demand_response.py'),'--action',$Action.ToLowerInvariant(),'--config',$Config)
if ($RunDirectory) { $DemandArguments += @('--run-directory',$RunDirectory) }
& $PythonExecutable @DemandArguments
if ($LASTEXITCODE -ne 0) { throw "Le laboratoire Demand Response a retourne le code $LASTEXITCODE. La production reste inchangee." }
