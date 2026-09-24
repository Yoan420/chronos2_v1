param(
    [Parameter(Mandatory=$true)][string]$PlanSha256,
    [string]$DeliveryDay = '2026-09-24',
    [string]$Reason = 'User-approved real shadow training, DE/NL then separately validated BE/FR'
)
$ErrorActionPreference = 'Stop'
$nyxRoot = 'C:\Users\BQ6757\chronos2_v1'
$nyxPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$nyxRunner = Join-Path $nyxRoot 'run_nyx_test2_live.py'
$nyxAllowed = Join-Path $nyxRoot 'runs\experiments\nyx_test2_live_v1'
if ($PlanSha256 -notmatch '^[a-fA-F0-9]{64}$' -or $DeliveryDay -notmatch '^\d{4}-\d{2}-\d{2}$') { throw 'Invalid fixed plan/day' }
$nyxPlan = Join-Path $nyxAllowed ($DeliveryDay + '\plan.json')
if ((Get-FileHash -LiteralPath $nyxPlan -Algorithm SHA256).Hash -ne $PlanSha256) { throw 'Plan SHA mismatch' }
$nyxExisting = @(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'})
if (@($nyxExisting | Where-Object {$_.CommandLine -match 'run_nyx_test2_live\.py'}).Count) { throw 'An NYX/Test2 process already exists; inspect before retry' }
$nyxLogs = Join-Path $nyxAllowed 'launcher_logs'
if (Test-Path -LiteralPath $nyxLogs) {
    foreach ($nyxOld in Get-ChildItem -LiteralPath $nyxLogs -Filter '*.launch.json') {
        $nyxRecord = Get-Content -Raw -LiteralPath $nyxOld.FullName | ConvertFrom-Json
        foreach ($nyxIdentity in $nyxRecord.verified_processes) {
            $nyxAlive = @($nyxExisting | Where-Object {$_.ProcessId -eq $nyxIdentity.pid -and $_.CreationDate.ToUniversalTime().ToString('o') -eq $nyxIdentity.created_utc})
            if ($nyxAlive.Count) { throw 'An identified prior process/descendant is still alive' }
        }
    }
}
$nyxFree = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($nyxFree -lt 3.5) { throw 'Less than3.5GiB available; no launch' }
[System.IO.Directory]::CreateDirectory($nyxLogs) | Out-Null
$nyxStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$nyxOut = Join-Path $nyxLogs ($nyxStamp + '.stdout.log')
$nyxErr = Join-Path $nyxLogs ($nyxStamp + '.stderr.log')
$nyxLaunch = Join-Path $nyxLogs ($nyxStamp + '.launch.json')
$nyxArgs = @('-B','-u',$nyxRunner,'--action','run','--delivery-day',$DeliveryDay,'--expected-plan',$PlanSha256)
$nyxStart = [DateTime]::UtcNow.ToString('o')
$nyxMeta = [ordered]@{schema_version=1;engine='nyx_test2_live_v1';delivery_day=$DeliveryDay;started_utc=$nyxStart;python=$nyxPython;arguments=$nyxArgs;command=@($nyxPython)+$nyxArgs;plan_sha256=$PlanSha256;stdout=$nyxOut;stderr=$nyxErr;reason=$Reason;status='launching';priority='BelowNormal';threads=1;workers=1;min_free_memory_gib=3.5;production_modified=$false;run_identities=@{};verified_processes=@();report_policy='One final standard HTML per country, containing the hybrid forecast and NYX comparison'}
[System.IO.File]::WriteAllText($nyxLaunch, ($nyxMeta | ConvertTo-Json -Depth 12), [System.Text.UTF8Encoding]::new($false))
# Last duplicate check immediately before launch; the runner also holds an OS lock.
if (@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'run_nyx_test2_live\.py'}).Count) { throw 'Duplicate appeared before launch; no start performed' }
$nyxProcess = Start-Process -FilePath $nyxPython -ArgumentList $nyxArgs -WorkingDirectory $nyxRoot -WindowStyle Hidden -RedirectStandardOutput $nyxOut -RedirectStandardError $nyxErr -PassThru
try { $nyxProcess.PriorityClass = 'BelowNormal' } catch { $nyxMeta['priority_assignment_warning'] = $_.Exception.Message }
$nyxMeta['venv_wrapper_pid'] = $nyxProcess.Id
$nyxMeta['venv_wrapper_created_utc'] = $nyxProcess.StartTime.ToUniversalTime().ToString('o')
Start-Sleep -Seconds 1
$nyxNow = @(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$'})
$nyxIds = [System.Collections.Generic.HashSet[int]]::new()
[void]$nyxIds.Add([int]$nyxProcess.Id)
do {
    $nyxCount = $nyxIds.Count
    foreach ($nyxRow in $nyxNow) { if ($nyxIds.Contains([int]$nyxRow.ParentProcessId)) { [void]$nyxIds.Add([int]$nyxRow.ProcessId) } }
} while ($nyxCount -ne $nyxIds.Count)
$nyxVerified = @($nyxNow | Where-Object {$nyxIds.Contains([int]$_.ProcessId)} | ForEach-Object {
    [ordered]@{pid=$_.ProcessId;ppid=$_.ParentProcessId;created_utc=$_.CreationDate.ToUniversalTime().ToString('o');command=$_.CommandLine}
})
$nyxMeta['verified_processes'] = $nyxVerified
$nyxMain = @($nyxNow | Where-Object {$_.ParentProcessId -eq $nyxProcess.Id -and $_.CommandLine -match 'run_nyx_test2_live\.py'})
if ($nyxMain.Count -eq 1) {
    $nyxMeta['python_pid'] = $nyxMain[0].ProcessId
    $nyxMeta['python_created_utc'] = $nyxMain[0].CreationDate.ToUniversalTime().ToString('o')
    try { (Get-Process -Id $nyxMain[0].ProcessId).PriorityClass = 'BelowNormal' } catch { $nyxMeta['main_priority_warning'] = $_.Exception.Message }
    $nyxMeta['status'] = 'process_verified'
} else { $nyxMeta['status'] = 'launch_uncertain_inspect_before_retry' }
[System.IO.File]::WriteAllText($nyxLaunch, ($nyxMeta | ConvertTo-Json -Depth 12), [System.Text.UTF8Encoding]::new($false))
$nyxMeta['launch_record'] = $nyxLaunch
$nyxMeta | ConvertTo-Json -Depth 12
