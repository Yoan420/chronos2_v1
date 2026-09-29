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
$NyxResult = Join-Path $NyxRoot ('runs\logs\nyx_annual_cpu\interactive_' + [guid]::NewGuid().ToString('N') + '.json')
$NyxArguments = @('-u', (Join-Path $NyxRoot 'run_nyx_annual_scheduled.py'), '--action', $Action, '--console', '--result-file', $NyxResult)
if ($DeliveryDay) { $NyxArguments += @('--delivery-day', $DeliveryDay) }
Push-Location -LiteralPath $NyxRoot
try {
    & $PythonExecutable @NyxArguments
    $NyxExitCode = $LASTEXITCODE
    $NyxDiagnostic = $null
    if (Test-Path -LiteralPath $NyxResult -PathType Leaf) {
        $NyxDiagnostic = Get-Content -LiteralPath $NyxResult -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    if ($NyxExitCode -ne 0) {
        if ($Action -eq 'inspect' -and $NyxExitCode -eq 2 -and $NyxDiagnostic.state -eq 'NOT_READY') {
            Write-Host 'Verification terminee : la preparation ou la qualification reste incomplete. Consulter les causes et le diagnostic ci-dessus.'
        } elseif ($null -ne $NyxDiagnostic) {
            throw ("Calcul bloque (code {0}). {1}`nJournal : {2}`nDiagnostic a transmettre : {3}" -f $NyxExitCode, $NyxDiagnostic.failure_summary, $NyxDiagnostic.log_path, $NyxDiagnostic.diagnostic_path)
        } else {
            throw "Le lanceur annuel a echoue (code $NyxExitCode) avant de produire son diagnostic. Conserver les messages affiches ci-dessus."
        }
    }
    $global:LASTEXITCODE = $NyxExitCode
} finally { Pop-Location }
