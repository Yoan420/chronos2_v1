@echo off
setlocal

set "APP_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%APP_ROOT%Forecast.ps1" -Action App
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo L'application s'est arretee avec le code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
