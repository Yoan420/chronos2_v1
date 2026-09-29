<# Lance la chaine annuelle CPU FR DE NL BE depuis le depot courant. #>
[CmdletBinding()]
param(
    [ValidateSet('inspect','capture','bootstrap','prepare','forecast')]
    [string]$Action = 'forecast',
    [ValidatePattern('^$|^\d{4}-\d{2}-\d{2}$')]
    [string]$DeliveryDay = '',
    [string]$PythonExecutable = ''
)
$ErrorActionPreference = 'Stop'
$NyxRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $NyxRoot '.venv-annual\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Python annuel absent. Executer Setup-NYXAnnualCPU.ps1 ou fournir -PythonExecutable.'
}
$NyxArguments = @('-u', (Join-Path $NyxRoot 'run_nyx_annual_pipeline.py'), '--action', $Action)
if ($DeliveryDay) { $NyxArguments += @('--delivery-day', $DeliveryDay) }
Push-Location -LiteralPath $NyxRoot
try {
    & $PythonExecutable @NyxArguments
    if ($LASTEXITCODE -ne 0) { throw "La chaine annuelle est bloquee ou en erreur (code $LASTEXITCODE). Consulter le journal affiche." }
} finally { Pop-Location }
