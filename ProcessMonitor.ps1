$ErrorActionPreference = 'Stop'
$taskProject = $PSScriptRoot
$taskSettings = Get-Content -LiteralPath (Join-Path $taskProject 'config\experiment_console.json') -Raw | ConvertFrom-Json
$taskPythonw = Join-Path (Split-Path -Parent $taskSettings.python_executable) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $taskPythonw -PathType Leaf)) { throw 'Python NYX introuvable.' }
Start-Process -FilePath $taskPythonw -ArgumentList @('-B','-m','nyx_process_monitor.desktop') -WorkingDirectory $taskProject -WindowStyle Hidden
