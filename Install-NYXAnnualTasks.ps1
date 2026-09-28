<# Programme capture07h et forecast08h05 dans la session Windows courante. #>
[CmdletBinding()]
param([switch]$EnableQualifiedForecast)
$ErrorActionPreference = 'Stop'
$NyxPython = Join-Path $PSScriptRoot '.venv-annual\Scripts\python.exe'
$NyxPythonw = Join-Path $PSScriptRoot '.venv-annual\Scripts\pythonw.exe'
$NyxRunner = Join-Path $PSScriptRoot 'run_nyx_annual_scheduled.py'
if (-not (Test-Path -LiteralPath $NyxPython -PathType Leaf)) { throw 'Executer Setup-NYXAnnualCPU.ps1 auparavant.' }
if ((Get-TimeZone).Id -notin @('Romance Standard Time', 'W. Europe Standard Time')) {
    throw 'Ces horaires exigent le fuseau Windows Paris/CET-CEST. Corriger le fuseau avant installation.'
}
$NyxPrincipal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$NyxSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12)
$NyxCapture = New-ScheduledTaskAction -Execute $NyxPythonw -Argument ('"' + $NyxRunner + '" --action capture') -WorkingDirectory $PSScriptRoot
$NyxTriggers = @(foreach ($NyxMinute in 0,15,30,45,55) { New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(7).AddMinutes($NyxMinute)) })
Register-ScheduledTask -TaskName 'NYX CWE PIT' -Action $NyxCapture -Trigger $NyxTriggers -Settings $NyxSettings -Principal $NyxPrincipal -Description 'Capture JAO, hydro et echanges avant 08h Paris' -Force
if ($EnableQualifiedForecast) {
    Push-Location -LiteralPath $PSScriptRoot
    try { & $NyxPython -c "from chronos2_hourly.nyx_annual_cpu_live import verify_activation; verify_activation(); print('Qualification complete valide')" }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw 'Prevision automatique non installee : qualification complete absente. La capture reste installee.' }
    $NyxForecast = New-ScheduledTaskAction -Execute $NyxPythonw -Argument ('"' + $NyxRunner + '" --action forecast') -WorkingDirectory $PSScriptRoot
    $NyxTrigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours(8).AddMinutes(5))
    Register-ScheduledTask -TaskName 'NYX CWE CPU production' -Action $NyxForecast -Trigger $NyxTrigger -Settings $NyxSettings -Principal $NyxPrincipal -Description 'Saturn, reentrainement CPU et quatre previsions annuelles qualifiees' -Force
}
Write-Host 'Le poste doit rester allume, eveille, connecte au reseau, avec la session ouverte.'
