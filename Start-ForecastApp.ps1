[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$Headless
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = 'C:\Users\BQ6757\venvs\pricefm311\Scripts\python.exe'
$AppPath = Join-Path $ProjectRoot 'app_multizone.py'

function Test-IsLoopbackBlackholeProxy {
    param(
        [AllowNull()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    # Port 9 on a loopback address is the deliberate "no network" proxy used by
    # isolated tool sessions.  Keep the match deliberately narrow so that a real
    # corporate proxy (including another local proxy port) is never removed.
    return $Value.Trim() -match '^(?:(?:https?|socks5?)://)?(?:127\.0\.0\.1|localhost|\[::1\]):9/?$'
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python introuvable : $PythonExe"
}
if (-not (Test-Path -LiteralPath $AppPath -PathType Leaf)) {
    throw "Application introuvable : $AppPath"
}

$Arguments = @(
    '-m', 'streamlit', 'run', $AppPath,
    '--server.port', [string]$Port,
    '--browser.gatherUsageStats', 'false'
)
if ($Headless) {
    $Arguments += @('--server.headless', 'true')
}

$BlackholeProxyVariables = @(
    foreach ($Name in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')) {
        $Value = [System.Environment]::GetEnvironmentVariable(
            $Name,
            [System.EnvironmentVariableTarget]::Process
        )
        if (Test-IsLoopbackBlackholeProxy -Value $Value) {
            $Name
        }
    }
)

$InheritedProxyValues = [System.Collections.Generic.List[object]]::new()
if ($BlackholeProxyVariables.Count -gt 0) {
    $Names = ($BlackholeProxyVariables | Sort-Object -Unique) -join ', '
    $IsCodexSandbox = (
        $env:CODEX_SANDBOX_NETWORK_DISABLED -eq '1' -or
        -not [string]::IsNullOrWhiteSpace($env:CODEX_THREAD_ID)
    )
    if ($IsCodexSandbox) {
        throw (
            "Cette console appartient encore au sandbox Codex " +
            "(proxy 127.0.0.1:9 dans $Names). Quittez completement Codex, " +
            "puis double-cliquez sur launch_forecast_app.cmd depuis " +
            "l'Explorateur Windows."
        )
    }

    foreach ($Name in $BlackholeProxyVariables) {
        $UserValue = [System.Environment]::GetEnvironmentVariable($Name, 'User')
        $MachineValue = [System.Environment]::GetEnvironmentVariable($Name, 'Machine')
        if (
            (Test-IsLoopbackBlackholeProxy -Value $UserValue) -or
            (Test-IsLoopbackBlackholeProxy -Value $MachineValue)
        ) {
            throw (
                "Le proxy 127.0.0.1:9 est configure durablement dans Windows " +
                "pour $Name. Supprimez cette variable utilisateur/machine avant " +
                "de relancer l'application."
            )
        }
        $ProcessValue = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
        $InheritedProxyValues.Add([pscustomobject]@{Name=$Name; Value=$ProcessValue})
        [System.Environment]::SetEnvironmentVariable($Name, $null, 'Process')
    }
    Write-Warning (
        "Ancien proxy local 127.0.0.1:9 retire d'un processus Windows non-Codex " +
        "($Names). Les configurations utilisateur/machine restent intactes."
    )
}

Push-Location $ProjectRoot
try {
    & $PythonExe @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    foreach ($ProxyValue in $InheritedProxyValues) {
        [System.Environment]::SetEnvironmentVariable(
            $ProxyValue.Name,
            $ProxyValue.Value,
            'Process'
        )
    }
}
