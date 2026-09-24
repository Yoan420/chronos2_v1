param([Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedPlan)
$ErrorActionPreference = 'Stop'
$nyxRoot = 'C:\Users\BQ6757\chronos2_v1'
$nyxPython = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$nyxRunner = Join-Path $nyxRoot 'run_nyx_test2_fast.py'
$nyxDay = '2026-09-24'
$nyxPlan = Join-Path $nyxRoot 'runs\experiments\n2\2026-09-24\plan.json'
if ((Get-FileHash -LiteralPath $nyxPlan -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedPlan) { throw 'Plan SHA mismatch' }
function Assert-NyxAbsent {
    $nyxExisting = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and (
            $_.CommandLine -match 'run_nyx_test2_(live|fast)\.py.*--action\s+run' -or
            $_.CommandLine -match 'loky\.backend\.popen_loky_win32|multiprocessing\.spawn'
        )
    })
    if ($nyxExisting.Count) { $nyxExisting | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json; throw 'Existing NYX or unclassified Python workers: inspect before launch' }
}
Assert-NyxAbsent
Push-Location $nyxRoot
try {
    $nyxValidation = & $nyxPython -B -u $nyxRunner --action validate --delivery-day $nyxDay --expected-plan $ExpectedPlan
    if ($LASTEXITCODE -ne 0) { throw 'Read-only validation failed' }
    if (($nyxValidation | ConvertFrom-Json).status -ne 'VALID') { throw 'Validation result not VALID' }
    $nyxFree = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB / 1GB
    if ($nyxFree -lt 5.5) { throw "Insufficient launch headroom: $nyxFree GiB free" }
    $nyxLogs = Join-Path $nyxRoot 'runs\experiments\n2\launcher_logs'
    New-Item -ItemType Directory -Path $nyxLogs -Force | Out-Null
    $nyxStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
    $nyxOut = Join-Path $nyxLogs ($nyxStamp + '.stdout.log')
    $nyxErr = Join-Path $nyxLogs ($nyxStamp + '.stderr.log')
    $nyxReceipt = Join-Path $nyxLogs ($nyxStamp + '.launch.json')
    $nyxArgs = @('-B','-u',$nyxRunner,'--action','run','--delivery-day',$nyxDay,'--expected-plan',$ExpectedPlan)
    $nyxRecord = [ordered]@{state='PREPARED';delivery_day=$nyxDay;engine='nyx_test2_fast_v1';plan_sha256=$ExpectedPlan;command=@($nyxPython)+$nyxArgs;reason='Explicit user authorization to maximize isolated NYX/Test2 computation';created_utc=(Get-Date).ToUniversalTime().ToString('o');free_memory_gib=$nyxFree;stdout=$nyxOut;stderr=$nyxErr;production_modified=$false;window_style='Hidden';priority='Normal'}
    if (Test-Path -LiteralPath $nyxReceipt) { throw 'Unique receipt collision' }
    $nyxRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $nyxReceipt -Encoding utf8
    Assert-NyxAbsent
    $nyxProcess = Start-Process -FilePath $nyxPython -ArgumentList $nyxArgs -WorkingDirectory $nyxRoot -WindowStyle Hidden -RedirectStandardOutput $nyxOut -RedirectStandardError $nyxErr -PassThru
    $nyxRecord['state'] = 'LAUNCHED_UNVERIFIED'
    $nyxRecord['pid'] = $nyxProcess.Id
    $nyxRecord['launch_utc'] = (Get-Date).ToUniversalTime().ToString('o')
    $nyxRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $nyxReceipt -Encoding utf8
    Start-Sleep -Seconds 2
    $nyxAll = @(Get-CimInstance Win32_Process)
    $nyxTree = [System.Collections.Generic.HashSet[int]]::new()
    [void]$nyxTree.Add([int]$nyxProcess.Id)
    do {
        $nyxAdded = $false
        foreach ($nyxEntry in $nyxAll) {
            if ($nyxTree.Contains([int]$nyxEntry.ParentProcessId) -and $nyxTree.Add([int]$nyxEntry.ProcessId)) { $nyxAdded=$true }
        }
    } while ($nyxAdded)
    $nyxVerified = @($nyxAll | Where-Object {$nyxTree.Contains([int]$_.ProcessId)} | ForEach-Object {
        [ordered]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;name=$_.Name;created_utc=$_.CreationDate.ToUniversalTime().ToString('o');command=$_.CommandLine}
    })
    $nyxRecord['processes'] = $nyxVerified
    $nyxRecord['verified_utc'] = (Get-Date).ToUniversalTime().ToString('o')
    $nyxRecord['state'] = $(if ($nyxVerified.Count) {'RUNNING_IDENTITY_CAPTURED'} else {'LAUNCH_EXITED_INSPECT_LOGS'})
    $nyxRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $nyxReceipt -Encoding utf8
    [ordered]@{launch_receipt=$nyxReceipt;launch=$nyxRecord} | ConvertTo-Json -Depth 10
} finally { Pop-Location }
