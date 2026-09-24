[CmdletBinding()]
param(
    [ValidateSet('Audit', 'Prepare', 'Backtest', 'Run', 'Report')]
    [string]$Action = 'Run',
    [string]$Config = 'config\marginal_cost_expert.yaml',
    [ValidateSet('FR', 'DE', 'BE', 'NL')]
    [string[]]$Countries = @('FR', 'DE', 'BE', 'NL'),
    [string]$RunDirectory = '',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Python) {
    $CandidatePython = Join-Path (Split-Path -Parent $ProjectRoot) 'venvs\pricefm311\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $CandidatePython) { $CandidatePython } else { 'python' }
}
$Arguments = @((Join-Path $ProjectRoot 'run_marginal_cost_expert.py'), '--action', $Action.ToLowerInvariant(), '--config', $Config)
if ($Countries.Count -gt 0) { $Arguments += @('--zones') + $Countries }
if ($RunDirectory) { $Arguments += @('--run-directory', $RunDirectory) }
Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "L'experimentation de cout marginal a retourne le code $LASTEXITCODE." }
}
finally { Pop-Location }
